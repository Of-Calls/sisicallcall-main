from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 해제 조건: 담당자 지정 후

logger = get_logger(__name__)


async def intent_router_llm_node(state: CallState) -> dict:
    # TODO: GPT-4o-mini 로 intent 분류 구현 (1.5초 하드컷)
    # 타임아웃 시 knn_intent 를 primary_intent 로 사용, routing_reason="knn_fallback_timeout"
    fallback_intent = state.get("knn_intent") or "intent_escalation"
    return {
        "primary_intent": fallback_intent,
        "secondary_intents": [],
        "routing_reason": "stub_fallback",
    }
