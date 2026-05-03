from app.agents.conversational.state import CallState
from app.agents.conversational.prompts.clarify import build_system_prompt
from app.services.llm.gpt4o_mini import GPT4OMiniService

_llm = GPT4OMiniService()

_FALLBACK_TEXT = "죄송하지만 다시 한 번 말씀해주시겠어요?"


async def clarify_branch_node(state: CallState) -> dict:
    user_text = state.get("user_text", "").strip()
    missing = state.get("missing_info", "").strip()
    tenant_name = state.get("tenant_name", "고객센터")
    tenant_industry = state.get("tenant_industry", "unknown")

    if not missing and not user_text:
        print("[clarify_branch] 입력 없음 → fallback")
        return {"response_text": _FALLBACK_TEXT}

    system_prompt = build_system_prompt(tenant_name, tenant_industry)
    user_message = (
        f"[사용자 발화]\n{user_text}\n\n"
        f"[부족한 정보]\n{missing or '발화가 모호함'}"
    )

    try:
        text = await _llm.generate(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.2,
            max_tokens=80,
        )
        text = text.strip().strip('"').strip("'")
        if not text:
            text = _FALLBACK_TEXT
    except Exception as exc:
        print(f"[clarify_branch] LLM 실패 → fallback: {exc}")
        text = _FALLBACK_TEXT

    print(f"[clarify_branch] missing='{missing}' generated='{text}'")
    return {"response_text": text}
