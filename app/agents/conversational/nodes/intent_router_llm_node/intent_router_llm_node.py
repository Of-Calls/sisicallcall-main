from app.agents.conversational.state import CallState
from app.agents.conversational.prompts.intent_router import build_system_prompt
from app.services.llm.gpt4o_mini import GPT4OMiniService

_llm = GPT4OMiniService()
_HISTORY_TURN_LIMIT = 6  # 직전 3턴 (user+assistant 합쳐 6개 항목)

_VALID_INTENTS = {"faq", "task", "auth", "vision", "escalation"}


def _format_history(history: list) -> str:
    if not history:
        return "(이전 대화 없음)"
    lines = []
    for entry in history[-_HISTORY_TURN_LIMIT:]:
        role = "사용자" if entry.get("role") == "user" else "AI"
        lines.append(f"{role}: {entry.get('text', '')}")
    return "\n".join(lines)


async def intent_router_llm_node(state: CallState) -> dict:
    # query_refine 이 만든 self-contained 쿼리 우선. 없으면 원본 fallback.
    query = state.get("rewritten_query") or state["user_text"]
    history = state.get("session_view", {}).get("conversation_history", [])
    tenant_name = state.get("tenant_name", "고객센터")
    tenant_industry = state.get("tenant_industry", "unknown")

    system_prompt = build_system_prompt(tenant_name, tenant_industry)
    user_message = f"[이전 대화]\n{_format_history(history)}\n\n[현재 사용자 발화]\n{query}"

    raw = await _llm.generate(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.0,
        max_tokens=15,
    )

    intent = raw.strip().lower()
    if intent not in _VALID_INTENTS:
        intent = "escalation"  # 분류 실패 시 상담원 연결

    print(f"[intent_router] query='{query}' → intent='{intent}'")
    return {"intent": intent}
