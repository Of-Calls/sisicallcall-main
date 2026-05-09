"""reviewer verdict=fail 분기 — 사람 검토 대기열로 분류.

외부 액션은 실행하지 않고 escalate_reason / human_review_required 만 표시한다.
"""
from __future__ import annotations

from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def human_queue_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    review_result: dict = state.get("review_result") or {}  # type: ignore[call-overload]
    reason = (
        state.get("escalate_reason")  # type: ignore[call-overload]
        or review_result.get("escalate_reason")
        or review_result.get("finalize_reason")
        or "review_failed"
    )
    errors = list(state.get("errors", []))  # type: ignore[call-overload]
    errors.append({"node": "reviewer_agent", "warning": "human_review_required", "error": reason})

    logger.info(
        "human_queue call_id=%s reason=%r — 외부 액션 차단",
        call_id, reason,
    )
    return {
        "human_review_required": True,
        "executed_actions": [],
        "action_plan": {
            "action_required": False,
            "actions": [],
            "rationale": f"human_review_required: {reason}",
        },
        "errors": errors,
        "partial_success": True,
    }
