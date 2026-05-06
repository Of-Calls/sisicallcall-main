"""
Post-call 통합 분석 노드.

기존 summary_node + voc_analysis_node + priority_node 를 LLM 1회 호출로 통합한다.
기존 3개 노드는 롤백/호환용으로 유지되며 이 노드가 그래프에서 사용된다.
"""
from __future__ import annotations

import json

from app.agents.post_call.llm_caller import PostCallLLMCaller, make_analysis_caller
from app.agents.post_call.prompts import ANALYSIS_SYSTEM, ANALYSIS_USER
from app.agents.post_call.schemas import (
    CustomerEmotion,
    PriorityLevel,
    ResolutionStatus,
)
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# lazy singleton — 테스트에서 monkeypatch.setattr 으로 교체
_caller: PostCallLLMCaller | None = None

_EMPTY_TRANSCRIPT_SUMMARY = {
    "summary_short": "통화 내용 없음",
    "summary_detailed": "녹취 데이터가 없어 분석을 생성할 수 없습니다.",
    "customer_intent": "알 수 없음",
    "customer_emotion": "neutral",
    "resolution_status": "resolved",
    "keywords": [],
    "handoff_notes": None,
}


def _get_caller() -> PostCallLLMCaller:
    global _caller
    if _caller is None:
        _caller = make_analysis_caller()
    return _caller


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    return "\n".join(f"[{t.get('role', '?')}] {t.get('text', '')}" for t in transcripts)


def _validate_summary(raw: dict) -> dict:
    emotion = raw.get("customer_emotion", "neutral")
    if emotion not in CustomerEmotion._value2member_map_:
        logger.warning("analysis: unknown customer_emotion=%r — neutral 로 대체", emotion)
        raw["customer_emotion"] = "neutral"
    status = raw.get("resolution_status", "resolved")
    if status not in ResolutionStatus._value2member_map_:
        logger.warning("analysis: unknown resolution_status=%r — resolved 로 대체", status)
        raw["resolution_status"] = "resolved"
    raw["summary_short"] = raw.get("summary_short") or ""
    raw["summary_detailed"] = raw.get("summary_detailed") or ""
    raw["customer_intent"] = raw.get("customer_intent") or ""
    if not isinstance(raw.get("keywords"), list):
        raw["keywords"] = []
    raw.setdefault("handoff_notes", None)
    return raw


def _validate_voc(raw: dict) -> dict:
    sr = raw.get("sentiment_result") or {}
    if not isinstance(sr, dict):
        sr = {}
    if sr.get("sentiment") not in CustomerEmotion._value2member_map_:
        sr["sentiment"] = "neutral"
    sr.setdefault("intensity", 0.0)
    sr.setdefault("reason", "")

    pr = raw.get("priority_result") or {}
    if not isinstance(pr, dict):
        pr = {}
    if pr.get("priority") not in PriorityLevel._value2member_map_:
        pr["priority"] = "low"
    pr.setdefault("action_required", False)
    pr.setdefault("suggested_action", None)
    pr.setdefault("reason", "")

    ir = raw.get("intent_result") or {}
    if not isinstance(ir, dict):
        ir = {}
    ir.setdefault("primary_category", "알 수 없음")
    ir.setdefault("sub_categories", [])
    ir.setdefault("is_repeat_topic", False)
    ir.setdefault("faq_candidate", False)

    return {"sentiment_result": sr, "intent_result": ir, "priority_result": pr}


def _validate_priority(raw: dict) -> dict:
    priority_val = raw.get("priority", "low")
    if priority_val not in PriorityLevel._value2member_map_:
        priority_val = "low"
    raw["priority"] = priority_val
    raw["tier"] = priority_val
    raw.setdefault("action_required", False)
    raw.setdefault("suggested_action", None)
    raw.setdefault("reason", "")
    return raw


def _validate_analysis(raw: dict) -> dict:
    """통합 분석 결과 검증. 누락 필드는 기본값으로 보정한다."""
    if raw.get("summary"):
        summary_raw = dict(raw.get("summary") or {})
    else:
        summary_raw = {
            "summary_short": raw.get("summary_short"),
            "summary_detailed": raw.get("summary_detailed"),
            "customer_intent": raw.get("customer_intent"),
            "customer_emotion": raw.get("customer_emotion"),
            "resolution_status": raw.get("resolution_status"),
            "keywords": raw.get("keywords"),
            "handoff_notes": raw.get("handoff_notes"),
        }

    if raw.get("voc_analysis"):
        voc_raw = dict(raw.get("voc_analysis") or {})
    else:
        voc_raw = {
            "sentiment_result": raw.get("sentiment_result"),
            "intent_result": raw.get("intent_result"),
            "priority_result": raw.get("priority_result"),
        }

    priority_raw = dict(raw.get("priority_result") or voc_raw.get("priority_result") or {})
    if raw.get("action_required") is not None:
        priority_raw.setdefault("action_required", raw.get("action_required"))
    if raw.get("suggested_action") is not None:
        priority_raw.setdefault("suggested_action", raw.get("suggested_action"))
    if raw.get("suggested_actions") and not priority_raw.get("suggested_action"):
        suggested_actions = raw.get("suggested_actions")
        if isinstance(suggested_actions, list) and suggested_actions:
            priority_raw["suggested_action"] = str(suggested_actions[0])

    summary = _validate_summary(summary_raw)
    voc_analysis = _validate_voc(voc_raw)
    priority_result = _validate_priority(priority_raw)

    # voc_analysis.priority_result 와 top-level priority_result 일관성 유지
    voc_pr = voc_analysis["priority_result"]
    if voc_pr.get("priority") != priority_result.get("priority"):
        voc_analysis["priority_result"]["priority"] = priority_result["priority"]
    if voc_pr.get("action_required") != priority_result.get("action_required"):
        voc_analysis["priority_result"]["action_required"] = priority_result["action_required"]

    return {
        "summary": summary,
        "voc_analysis": voc_analysis,
        "priority_result": priority_result,
    }


async def post_call_analysis_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    transcripts: list = state.get("transcripts") or []  # type: ignore[call-overload]
    errors: list = list(state.get("errors", []))  # type: ignore[call-overload]

    # 녹취 없음 — LLM 호출 없이 summary fallback 반환, voc/priority는 None
    if not transcripts:
        logger.warning("post_call_analysis: 녹취 없음 call_id=%s — fallback 사용", call_id)
        errors.append({
            "node": "post_call_analysis",
            "warning": "empty_transcript",
            "error": "transcripts 없음 — fallback 사용",
        })
        return {
            "analysis_result": None,
            "summary": dict(_EMPTY_TRANSCRIPT_SUMMARY),
            "voc_analysis": None,
            "priority_result": None,
            "errors": errors,
            "partial_success": True,
        }

    try:
        transcripts_text = _format_transcripts(transcripts)
        user_msg = ANALYSIS_USER.format(transcripts=transcripts_text)

        raw = await _get_caller().call_json(
            system_prompt=ANALYSIS_SYSTEM,
            user_message=user_msg,
            max_tokens=1800,
        )
        # _validate_analysis() strips out keys outside the analysis schema, so
        # capture LLM metadata before validation.
        llm_usage = raw.get("_llm_usage") if isinstance(raw, dict) else None
        analysis = _validate_analysis(raw)
        logger.info(
            "post_call_analysis 완료 call_id=%s emotion=%s priority=%s action_required=%s",
            call_id,
            analysis["summary"].get("customer_emotion"),
            analysis["priority_result"].get("priority"),
            analysis["priority_result"].get("action_required"),
        )
        return {
            "analysis_result": analysis,
            "summary": analysis["summary"],
            "voc_analysis": analysis["voc_analysis"],
            "priority_result": analysis["priority_result"],
            "analysis_llm_usage": llm_usage,
        }

    except Exception as exc:
        logger.error("post_call_analysis 실패 call_id=%s err=%s", call_id, exc)
        errors.append({"node": "post_call_analysis", "error": str(exc)})
        return {
            "analysis_result": None,
            "summary": None,
            "voc_analysis": None,
            "priority_result": None,
            "analysis_llm_usage": None,
            "errors": errors,
            "partial_success": True,
        }
