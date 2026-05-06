"""사용자가 직전 안내를 다시 듣고 싶다고 한 경우 그대로 재생.

query_refine 이 "사용자가 직전 안내 반복 요청" 으로 재작성한 발화를
intent_router 가 repeat 으로 분류 → 이 노드가 Redis 통화 세션의
conversation_history 에서 마지막 assistant 메시지를 그대로 반환.

LLM 호출 X — 안전한 결정적 응답.
"""
from app.agents.conversational.state import CallState

_POLITE_NO_HISTORY = "죄송하지만 안내해드린 내용이 없어요. 무엇을 도와드릴까요?"


async def repeat_branch_node(state: CallState) -> dict:
    history = state.get("session_view", {}).get("conversation_history", [])
    for entry in reversed(history):
        if entry.get("role") == "assistant":
            text = entry.get("text", "").strip()
            if text:
                print(f"[repeat_branch] 직전 안내 반복 → '{text[:60]}...'")
                return {"response_text": text}

    print("[repeat_branch] history 에 assistant 메시지 없음 → no_history fallback")
    return {"response_text": _POLITE_NO_HISTORY}
