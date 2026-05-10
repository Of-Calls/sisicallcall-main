"""reviewer fail → analysis_planner 재시도 진입 직전 가벼운 노드.

state["analysis_retry_count"] 를 +1 하고, 직전 reviewer 가 만든
review_result.feedback_for_retry 를 state["review_feedback"] 에 누적한다.

이 누적된 feedback 은 analysis_planner_agent_node 가 system prompt 에 주입해
LLM 이 같은 실수를 반복하지 않도록 한다.
"""
from __future__ import annotations

from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def increment_analysis_retry_node(state: PostCallAgentState) -> dict:
    prev_count = int(state.get("analysis_retry_count") or 0)  # type: ignore[call-overload]
    new_count = prev_count + 1
    feedback = list(state.get("review_feedback") or [])  # type: ignore[call-overload]
    review_result = state.get("review_result") or {}  # type: ignore[call-overload]
    new_items = list(review_result.get("feedback_for_retry") or [])
    feedback.extend(new_items)

    logger.info(
        "post_call retry call_id=%s tenant=%s retry_count=%d feedback_items=%d",
        state.get("call_id"),
        state.get("tenant_id"),
        new_count,
        len(new_items),
    )
    return {
        "analysis_retry_count": new_count,
        "review_feedback": feedback,
        # 다음 분석 사이클을 위해 직전 reviewer 산출물은 비워둔다 — 새 분석에 대해
        # 다시 reviewer 가 채울 것이고, 비교용은 review_feedback 에 텍스트로 살아있다.
        "review_result": None,
        "review_verdict": None,
        "approved_actions": [],
        "corrections_to_analysis": {},
        "escalate_reason": None,
        # human_review_required 는 직전 fail 의 산물. 재시도에 들어가는 시점에서
        # 다시 평가받아야 하므로 False 로 리셋해야 다음 save_intermediate 가
        # reviewer 로 라우팅된다 (_route_after_intermediate 가드).
        "human_review_required": False,
    }
