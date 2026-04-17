from app.agents.conversational.state import CallState

# TODO(신용): 연구 완료 후 구현 — architecture.md 7.3 참조
# 해제 조건: KNN Router 연구 결과 팀장 보고 후


async def knn_router_node(state: CallState) -> dict:
    # TODO(신용): BGE-M3 임베딩 기반 KNN 분류 구현
    # confidence >= KNN_CONFIDENCE_THRESHOLD 시 primary_intent 도 함께 반환
    return {
        "knn_intent": None,
        "knn_confidence": 0.0,
        "primary_intent": None,
    }
