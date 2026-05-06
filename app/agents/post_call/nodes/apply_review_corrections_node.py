"""
Review 교정 적용 노드.

review_verdict == "correctable" 일 때 실행된다.
review_result.corrections 를 기존 분석 결과에 안전하게 merge 한다.
"""
from __future__ import annotations

import copy

from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_CORRECTION_KEYS = ("summary", "voc_analysis", "priority_result")
_DIFF_PATHS = {
    "summary": (
        "summary_short",
        "customer_intent",
        "customer_emotion",
        "resolution_status",
    ),
    "voc_analysis": (
        "intent_result.primary_category",
        "sentiment_result.sentiment",
        "priority_result.priority",
    ),
    "priority_result": (
        "priority",
        "action_required",
    ),
}


async def apply_review_corrections_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    review_result: dict = state.get("review_result") or {}  # type: ignore[call-overload]
    corrections: dict = review_result.get("corrections") or {}

    summary = dict(state.get("summary") or {})  # type: ignore[call-overload]
    voc_analysis = dict(state.get("voc_analysis") or {})  # type: ignore[call-overload]
    priority_result = dict(state.get("priority_result") or {})  # type: ignore[call-overload]
    analysis_result = dict(state.get("analysis_result") or {})  # type: ignore[call-overload]
    before_by_key = {
        "summary": copy.deepcopy(summary),
        "voc_analysis": copy.deepcopy(voc_analysis),
        "priority_result": copy.deepcopy(priority_result),
    }

    # 허용된 top-level key만 수정하되, nested dict는 기존 값을 보존하며 merge한다.
    for key in _ALLOWED_CORRECTION_KEYS:
        if key not in corrections or not isinstance(corrections[key], dict):
            continue
        patch = corrections[key]
        if not patch:
            continue
        if key == "summary":
            summary = _deep_merge(summary, patch)
        elif key == "voc_analysis":
            voc_analysis = _deep_merge(voc_analysis, patch)
        elif key == "priority_result":
            priority_result = _deep_merge(priority_result, patch)

    after_by_key = {
        "summary": summary,
        "voc_analysis": voc_analysis,
        "priority_result": priority_result,
    }
    correction_diff = _build_correction_diff(before_by_key, after_by_key, corrections)
    for item in correction_diff:
        logger.debug(
            "review correction diff call_id=%s key=%s before=%r after=%r",
            call_id,
            item["path"],
            _truncate(item["before"]),
            _truncate(item["after"]),
        )

    # analysis_result도 갱신 (일관성)
    analysis_result["summary"] = summary
    analysis_result["voc_analysis"] = voc_analysis
    analysis_result["priority_result"] = priority_result

    logger.info(
        "apply_review_corrections 완료 call_id=%s corrected_keys=%s",
        call_id, [k for k in _ALLOWED_CORRECTION_KEYS if corrections.get(k)],
    )
    return {
        "summary": summary,
        "voc_analysis": voc_analysis,
        "priority_result": priority_result,
        "analysis_result": analysis_result,
        "review_correction_diff": correction_diff,
    }


def _deep_merge(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _build_correction_diff(
    before_by_key: dict[str, dict],
    after_by_key: dict[str, dict],
    corrections: dict,
) -> list[dict]:
    diff: list[dict] = []
    for key in ("summary", "voc_analysis", "priority_result"):
        if not isinstance(corrections.get(key), dict) or not corrections.get(key):
            continue
        for path in _DIFF_PATHS[key]:
            before = _get_path(before_by_key.get(key, {}), path)
            after = _get_path(after_by_key.get(key, {}), path)
            if before != after:
                diff.append({
                    "path": f"{key}.{path}",
                    "before": before,
                    "after": after,
                })
    return diff


def _get_path(value: dict, path: str):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _truncate(value, limit: int = 160):
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
