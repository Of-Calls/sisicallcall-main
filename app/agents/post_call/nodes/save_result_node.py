from __future__ import annotations

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


async def save_result_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    normalized_state = normalize_post_call_result(dict(state))
    # 이전 노드들이 누적한 errors 를 수집
    errors: list[dict] = list(normalized_state.get("errors", []))

    for label, coro in [
        ("summary", _maybe_save_summary(call_id, normalized_state)),
        ("voc", _maybe_save_voc(call_id, normalized_state)),
        ("action_log", _maybe_save_actions(call_id, normalized_state)),
    ]:
        try:
            await coro
        except Exception as exc:
            errors.append({"node": f"save_result:{label}", "error": str(exc)})

    # errors 유무로 partial_success 최종 결정.
    # action_router_node 등 이전 노드가 partial_success 를 False 로 덮어썼을 수 있으므로
    # 이 노드가 authoritative setter 역할을 한다.
    final_partial_success = len(errors) > 0

    payload = {
        "call_id": call_id,
        "tenant_id": normalized_state["tenant_id"],
        "trigger": normalized_state["trigger"],
        "summary": normalized_state.get("summary"),
        "voc_analysis": normalized_state.get("voc_analysis"),
        "priority_result": normalized_state.get("priority_result"),
        "action_plan": normalized_state.get("action_plan"),
        "executed_actions": normalized_state.get("executed_actions", []),
        "errors": errors,
        "partial_success": final_partial_success,
        # ── Review Gate 필드 ─────────────────────────────────────────────
        "analysis_result": normalized_state.get("analysis_result"),
        "review_result": normalized_state.get("review_result"),
        "review_verdict": normalized_state.get("review_verdict"),
        "review_retry_count": normalized_state.get("review_retry_count", 0),
        "human_review_required": normalized_state.get("human_review_required", False),
        "blocked_actions": normalized_state.get("blocked_actions", []),
    }

    try:
        await _dashboard_repo.upsert_dashboard(call_id, payload)
    except Exception as exc:
        errors.append({"node": "save_result:dashboard", "error": str(exc)})
        final_partial_success = True
        payload["errors"] = errors
        payload["partial_success"] = final_partial_success

    logger.info("save_result 완료 call_id=%s errors=%d partial_success=%s",
                call_id, len(errors), final_partial_success)
    return {
        "dashboard_payload": payload,
        "summary": normalized_state.get("summary"),
        "voc_analysis": normalized_state.get("voc_analysis"),
        "priority_result": normalized_state.get("priority_result"),
        "analysis_result": normalized_state.get("analysis_result"),
        "errors": errors,
        "partial_success": final_partial_success,
    }


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
    summary.setdefault("model_used", "demo-mock-llm")
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
