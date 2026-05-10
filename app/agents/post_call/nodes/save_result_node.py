from __future__ import annotations

from typing import Literal

from app.agents.post_call.normalizer import normalize_post_call_result
from app.agents.post_call.state import PostCallAgentState
from app.repositories.call_summary_repo import CallSummaryRepository
from app.repositories.voc_analysis_repo import VOCAnalysisRepository
from app.repositories.mcp_action_log_repo import MCPActionLogRepository
from app.repositories.dashboard_repo import DashboardRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)
_summary_repo = CallSummaryRepository()
_voc_repo = VOCAnalysisRepository()
_action_log_repo = MCPActionLogRepository()
_dashboard_repo = DashboardRepository()

SaveMode = Literal["intermediate", "final"]


async def save_result_node(state: PostCallAgentState, mode: SaveMode = "final") -> dict:
    """분석 결과 / 액션 결과를 영속화한다.

    mode=intermediate (= save_reviewed_analysis)
        reviewer 통과 직후 호출. 보정된 analysis_result 를 call_summaries / voc_analyses
        로 영속화. 실패해도 다음 단계 (action_executor) 진행 가능하도록 절대 raise 안 함.
    mode=final
        executor 또는 human_queue + auto_action_executor 직후 호출.
        mcp_action_logs + dashboard 최종 upsert.

        review_verdict=='fail' 인 경우 summary / voc 는 save_reviewed_analysis 단계를
        거치지 않았으므로 final 단계에서도 저장하지 않는다 (검토 통과 분석만 저장).
        대신 mcp_action_logs (admin alert / Notion 마킹된 row) + dashboard 만 갱신.
    """
    call_id = state["call_id"]
    normalized_state = normalize_post_call_result(dict(state))
    errors: list[dict] = list(normalized_state.get("errors", []))

    save_label_prefix = f"save_{mode}"

    # fail max 도달 시 분석 본문 영속화 차단 (사용자 핵심 의도).
    fail_block_analysis = (
        mode == "final"
        and (normalized_state.get("review_verdict") or "") == "fail"
    )

    coros: list[tuple[str, object]] = []
    if not fail_block_analysis:
        coros.append(("summary", _maybe_save_summary(call_id, normalized_state)))
        coros.append(("voc", _maybe_save_voc(call_id, normalized_state)))
    if mode == "final":
        coros.append(("action_log", _maybe_save_actions(call_id, normalized_state)))

    for label, coro in coros:
        try:
            await coro
        except Exception as exc:
            errors.append({"node": f"{save_label_prefix}:{label}", "error": str(exc)})
            logger.warning(
                "save_result(%s) %s 저장 실패 call_id=%s err=%s",
                mode, label, call_id, exc,
            )

    final_partial_success = len(errors) > 0

    planner_telemetry = normalized_state.get("analysis_planner_telemetry")
    reviewer_telemetry = normalized_state.get("reviewer_telemetry")
    # B1: fail max 시 분석 본문은 dashboard 에도 노출하지 않는다.
    # action_plan / executed_actions / 텔레메트리는 그대로 — 운영자가 fail 발생 자체는 봐야 함.
    payload_summary = None if fail_block_analysis else normalized_state.get("summary")
    payload_voc = None if fail_block_analysis else normalized_state.get("voc_analysis")
    payload_priority = None if fail_block_analysis else normalized_state.get("priority_result")
    payload_analysis = None if fail_block_analysis else normalized_state.get("analysis_result")

    payload = {
        "call_id": call_id,
        "tenant_id": normalized_state["tenant_id"],
        "trigger": normalized_state["trigger"],
        "summary": payload_summary,
        "voc_analysis": payload_voc,
        "priority_result": payload_priority,
        "action_plan": normalized_state.get("action_plan"),
        "executed_actions": normalized_state.get("executed_actions", []),
        "errors": errors,
        "partial_success": final_partial_success,
        # ── Agent 출력 ──────────────────────────────────────────────────────
        "analysis_result": payload_analysis,
        "proposed_actions": normalized_state.get("proposed_actions", []),
        "review_result": normalized_state.get("review_result"),
        "review_verdict": normalized_state.get("review_verdict"),
        "approved_actions": normalized_state.get("approved_actions", []),
        "corrections_to_analysis": normalized_state.get("corrections_to_analysis", {}),
        "escalate_reason": normalized_state.get("escalate_reason"),
        "reviewer_steps": normalized_state.get("reviewer_steps", 0),
        "human_review_required": normalized_state.get("human_review_required", False),
        "save_mode": mode,
        # ── 텔레메트리 (production 모니터링용) ─────────────────────────────────
        "telemetry": {
            "analysis_planner": planner_telemetry,
            "reviewer": reviewer_telemetry,
            "analysis_retry_count": int(normalized_state.get("analysis_retry_count") or 0),
            "review_feedback": list(normalized_state.get("review_feedback") or []),
        },
        "analysis_retry_count": int(normalized_state.get("analysis_retry_count") or 0),
        "review_feedback": list(normalized_state.get("review_feedback") or []),
    }

    try:
        await _dashboard_repo.upsert_dashboard(call_id, payload)
    except Exception as exc:
        errors.append({"node": f"{save_label_prefix}:dashboard", "error": str(exc)})
        final_partial_success = True
        payload["errors"] = errors
        payload["partial_success"] = final_partial_success
        logger.warning(
            "save_result(%s) dashboard upsert 실패 call_id=%s err=%s",
            mode, call_id, exc,
        )

    logger.info(
        "save_result(%s) 완료 call_id=%s errors=%d partial_success=%s",
        mode, call_id, len(errors), final_partial_success,
    )

    # save_final 단계에서 통화당 한 줄 텔레메트리 요약 (production 모니터링용)
    if mode == "final":
        pt = planner_telemetry or {}
        rt = reviewer_telemetry or {}
        pt_tokens = (pt.get("tokens") or {}).get("total", 0)
        rt_tokens = (rt.get("tokens") or {}).get("total", 0)
        rt_steps = rt.get("steps", 0)
        rt_max = rt.get("max_steps_reached", False)
        latency_total = int((pt.get("latency_ms") or 0) + (rt.get("latency_ms") or 0))
        retry_count = int(normalized_state.get("analysis_retry_count") or 0)
        logger.info(
            "post_call telemetry call_id=%s tenant=%s verdict=%s "
            "planner_tokens=%d reviewer_tokens=%d total_tokens=%d "
            "reviewer_steps=%d max_steps_reached=%s latency_ms=%d "
            "approved=%d executed=%d errors=%d partial_success=%s "
            "analysis_retry_count=%d",
            call_id,
            normalized_state.get("tenant_id"),
            normalized_state.get("review_verdict"),
            pt_tokens, rt_tokens, pt_tokens + rt_tokens,
            rt_steps, rt_max, latency_total,
            len(normalized_state.get("approved_actions") or []),
            len(normalized_state.get("executed_actions") or []),
            len(errors), final_partial_success,
            retry_count,
        )

    out: dict = {
        "dashboard_payload": payload,
        "summary": normalized_state.get("summary"),
        "voc_analysis": normalized_state.get("voc_analysis"),
        "priority_result": normalized_state.get("priority_result"),
        "analysis_result": normalized_state.get("analysis_result"),
        "analysis_planner_telemetry": planner_telemetry,
        "reviewer_telemetry": reviewer_telemetry,
        "errors": errors,
        "partial_success": final_partial_success,
    }
    return out


async def save_intermediate_node(state: PostCallAgentState) -> dict:
    """LangGraph 노드 어댑터 — save_result_node(mode='intermediate').

    DEPRECATED: 그래프에서는 더 이상 호출되지 않는다 (reviewer 전 저장 제거).
    호출자/테스트 호환을 위해 alias 로 보존. 새 코드는 save_reviewed_analysis_node
    또는 save_final_node 를 사용한다.
    """
    return await save_result_node(state, mode="intermediate")


async def save_reviewed_analysis_node(state: PostCallAgentState) -> dict:
    """reviewer 통과 (pass / correctable) 시 보정된 분석을 영속화한다.

    state["analysis_result"] 는 reviewer 가 correct_analysis 도구로 보정한 본문
    (reviewer_agent_node 가 corrected_analysis 를 그대로 반환). 별도 적용 단계
    필요 없음.

    저장 대상:
      - call_summaries  (ON CONFLICT upsert)
      - voc_analyses    (ON CONFLICT upsert)
      - dashboards (in-memory)  — corrections_to_analysis 메타도 함께
      - mcp_action_logs 는 save_final 에서 처리 (executor 가 아직 안 돌았음)

    fail (max retry 초과) 분기에서는 절대 호출되지 않는다 — 잘못된 분석 저장 차단.
    """
    return await save_result_node(state, mode="intermediate")


async def save_final_node(state: PostCallAgentState) -> dict:
    """LangGraph 노드 어댑터 — save_result_node(mode='final')."""
    return await save_result_node(state, mode="final")


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

async def _maybe_save_summary(call_id: str, state: dict) -> None:
    summary = _summary_payload(state)
    if summary:
        await _summary_repo.save_summary(
            call_id,
            summary,
            tenant_id=state["tenant_id"],
        )


async def _maybe_save_voc(call_id: str, state: dict) -> None:
    voc = _voc_payload(state)
    if voc:
        await _voc_repo.save_voc_analysis(
            call_id,
            voc,
            tenant_id=state["tenant_id"],
            partial_success=bool(state.get("partial_success", False)),  # type: ignore[call-overload]
            failed_subagents=_failed_subagents(state),
        )


async def _maybe_save_actions(call_id: str, state: dict) -> None:
    actions = state.get("executed_actions", [])
    if actions:
        await _action_log_repo.save_action_log(
            call_id,
            actions,
            tenant_id=state["tenant_id"],
        )


def _resolve_model_used() -> str:
    """현재 LLM 모드 / 모델 환경에 따라 model_used 라벨 결정."""
    import os
    mode = (os.environ.get("POST_CALL_LLM_MODE") or "").strip().lower()
    if mode != "real":
        legacy = (os.environ.get("POST_CALL_USE_REAL_LLM") or "").strip().lower()
        if legacy not in {"1", "true", "yes", "on"}:
            return "mock"
    # real 모드 — 환경변수 우선, 없으면 GPT-4o (planner 의 default 모델)
    explicit = (os.environ.get("POST_CALL_LLM_MODEL") or "").strip()
    if explicit:
        return explicit
    return "gpt-4o"


def _summary_payload(state: dict) -> dict:
    summary = dict(state.get("summary") or {})
    analysis = state.get("analysis_result") or {}
    if not summary and isinstance(analysis, dict):
        summary = dict(analysis.get("summary") or {})
    if not summary:
        return {}

    summary.setdefault("summary_short", "")
    summary.setdefault("summary_detailed", None)
    summary.setdefault("customer_intent", summary.get("intent"))
    summary.setdefault("customer_emotion", summary.get("emotion", "neutral"))
    summary.setdefault("resolution_status", "resolved")
    summary.setdefault("keywords", [])
    summary.setdefault("handoff_notes", None)
    summary.setdefault("generation_mode", "async")
    # D-6: model_used 동적 결정 (mock / 실제 모델명)
    summary.setdefault("model_used", _resolve_model_used())
    return summary


def _voc_payload(state: dict) -> dict:
    voc = dict(state.get("voc_analysis") or {})
    analysis = state.get("analysis_result") or {}
    if not voc and isinstance(analysis, dict):
        voc = dict(analysis.get("voc_analysis") or {})
    if not voc:
        return {}

    priority = state.get("priority_result") or {}
    if not voc.get("priority_result") and priority:
        voc["priority_result"] = dict(priority)
    voc.setdefault("sentiment_result", {})
    voc.setdefault("intent_result", {})
    voc.setdefault("priority_result", {})
    return voc


def _failed_subagents(state: dict) -> list[str]:
    failed: list[str] = []
    for err in state.get("errors", []):
        if isinstance(err, dict):
            node = err.get("node")
            if node:
                failed.append(str(node))
    return failed
