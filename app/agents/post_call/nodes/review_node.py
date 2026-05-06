"""
Review Gate 노드.

통화 녹취와 post_call_analysis_node 결과를 비교하여
분석이 원문에 근거하는지 검토하고 verdict를 결정한다.

verdict:
  pass        → action_planner 진행
  correctable → apply_review_corrections_node 거쳐 action_planner
  retry       → review_retry_count < 1 이면 재분석
  fail        → human_review_required=True, 외부 action 금지
"""
from __future__ import annotations

import json

from app.agents.post_call.llm_caller import PostCallLLMCaller, make_review_caller
from app.agents.post_call.prompts import REVIEW_SYSTEM, REVIEW_USER
from app.agents.post_call.schemas import ReviewVerdictValues
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# lazy singleton — 테스트에서 monkeypatch.setattr 으로 교체
_caller: PostCallLLMCaller | None = None


def _get_caller() -> PostCallLLMCaller:
    global _caller
    if _caller is None:
        _caller = make_review_caller()
    return _caller


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    return "\n".join(f"[{t.get('role', '?')}] {t.get('text', '')}" for t in transcripts)


def _make_fail_result(reason: str, issues: list | None = None) -> dict:
    return {
        "verdict": "fail",
        "confidence": 0.0,
        "confidence_missing": False,
        "confidence_parse_error": False,
        "confidence_source": "fallback",
        "issues": issues or [{"type": "review_error", "message": reason, "evidence": None}],
        "corrections": {"summary": {}, "voc_analysis": {}, "priority_result": {}},
        "blocked_actions": [],
        "reason": reason,
    }


def _normalize_confidence(raw: dict) -> tuple[float, dict]:
    """Return confidence plus parse metadata.

    Historically review_node used ``float(raw.get("confidence") or 0.0)``.
    That collapsed three different cases into the same value: LLM returned 0.0,
    LLM omitted confidence, or confidence could not be parsed. The metadata
    below keeps those cases distinguishable in logs and review_result.
    """
    if "confidence" not in raw or raw.get("confidence") in (None, ""):
        return 0.0, {
            "confidence_missing": True,
            "confidence_parse_error": False,
            "confidence_source": "default",
        }

    value = raw.get("confidence")
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0, {
            "confidence_missing": False,
            "confidence_parse_error": True,
            "confidence_source": "parse_error",
        }

    source = "fallback" if raw.get("_llm_fallback") else "llm"
    if confidence < 0.0:
        confidence = 0.0
        source = "clamped"
    elif confidence > 1.0:
        confidence = 1.0
        source = "clamped"

    return confidence, {
        "confidence_missing": False,
        "confidence_parse_error": False,
        "confidence_source": source,
    }


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _correction_keys(corrections: dict) -> list[str]:
    return [
        key
        for key in ("summary", "voc_analysis", "priority_result")
        if isinstance(corrections.get(key), dict) and corrections.get(key)
    ]


async def review_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]

    try:
        transcripts: list = state.get("transcripts") or []  # type: ignore[call-overload]
        analysis_result: dict = state.get("analysis_result") or {}  # type: ignore[call-overload]

        transcripts_text = _format_transcripts(transcripts)
        analysis_text = json.dumps(analysis_result, ensure_ascii=False, indent=2)

        user_msg = REVIEW_USER.format(
            transcripts=transcripts_text,
            analysis=analysis_text,
        )

        raw = await _get_caller().call_json(
            system_prompt=REVIEW_SYSTEM,
            user_message=user_msg,
            max_tokens=1000,
        )

        raw = raw if isinstance(raw, dict) else {}

        verdict = raw.get("verdict", "fail")
        if not ReviewVerdictValues.is_valid(verdict):
            logger.warning("review: unknown verdict=%r — fail 로 대체", verdict)
            verdict = "fail"

        confidence, confidence_meta = _normalize_confidence(raw)
        corrections = _as_dict(raw.get("corrections")) or {
            "summary": {},
            "voc_analysis": {},
            "priority_result": {},
        }
        reason = str(raw.get("reason") or "")
        llm_usage = raw.get("_llm_usage") if isinstance(raw.get("_llm_usage"), dict) else None
        review_result = {
            "verdict": verdict,
            "confidence": confidence,
            **confidence_meta,
            "issues": _as_list(raw.get("issues")),
            "corrections": corrections,
            "corrected_keys": _correction_keys(corrections),
            "blocked_actions": _as_list(raw.get("blocked_actions")),
            "reason": reason,
            "llm_fallback": bool(raw.get("_llm_fallback")),
            "llm_fallback_reason": str(raw.get("_llm_fallback_reason") or ""),
            "llm_usage": llm_usage,
        }

        blocked_actions: list[str] = review_result["blocked_actions"]
        # retry / correctable은 human_review_required=False — graph route에서 결정
        human_review_required = verdict == "fail"

        if verdict == "correctable" and confidence <= 0.0:
            warning_reason = (
                "missing_confidence"
                if review_result["confidence_missing"]
                else "confidence_parse_error"
                if review_result["confidence_parse_error"]
                else "zero_confidence"
            )
            logger.warning(
                "review confidence suspicious call_id=%s verdict=%s confidence=%.2f source=%s reason=%s",
                call_id,
                verdict,
                confidence,
                review_result["confidence_source"],
                warning_reason,
            )

        if not reason:
            logger.debug("review reason missing call_id=%s verdict=%s", call_id, verdict)

        logger.info(
            "review 완료 call_id=%s verdict=%s confidence=%.2f source=%s blocked=%s reason=%r corrected_keys=%s",
            call_id,
            verdict,
            review_result["confidence"],
            review_result["confidence_source"],
            blocked_actions,
            reason,
            review_result["corrected_keys"],
        )
        return {
            "review_result": review_result,
            "review_verdict": verdict,
            "blocked_actions": blocked_actions,
            "human_review_required": human_review_required,
            "review_llm_usage": llm_usage,
        }

    except Exception as exc:
        logger.error("review 실패 call_id=%s err=%s", call_id, exc)
        fail_result = _make_fail_result(f"review_node exception: {exc}")
        return {
            "review_result": fail_result,
            "review_verdict": "fail",
            "blocked_actions": [],
            "human_review_required": True,
            "review_llm_usage": None,
        }
