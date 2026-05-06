import json

from app.agents.conversational.state import CallState
from app.agents.conversational.prompts.query_refine import build_system_prompt
from app.services.llm.gpt4o_mini import GPT4OMiniService

_llm = GPT4OMiniService()
_HISTORY_TURN_LIMIT = 6  # 직전 3턴 (user+assistant 합쳐 6개 항목)


def _format_history(history: list) -> str:
    if not history:
        return "(이전 대화 없음)"
    lines = []
    for entry in history[-_HISTORY_TURN_LIMIT:]:
        role = "사용자" if entry.get("role") == "user" else "AI"
        lines.append(f"{role}: {entry.get('text', '')}")
    return "\n".join(lines)


async def query_refine_node(state: CallState) -> dict:
    user_text = state["user_text"]
    history = state.get("session_view", {}).get("conversation_history", [])
    tenant_name = state.get("tenant_name", "고객센터")
    tenant_industry = state.get("tenant_industry", "unknown")

    system_prompt = build_system_prompt(tenant_name, tenant_industry)
    user_message = f"[이전 대화]\n{_format_history(history)}\n\n[현재 발화]\n{user_text}"

    raw = await _llm.generate(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.0,
        max_tokens=200,
    )

    try:
        parsed = json.loads(raw.strip())
        is_clear = bool(parsed.get("is_clear", True))
        rewritten = str(parsed.get("rewritten_query", "")).strip()
        missing = str(parsed.get("missing_info", "")).strip()
    except (json.JSONDecodeError, ValueError, AttributeError):
        # 파싱 실패: 원본 그대로 통과
        is_clear = True
        rewritten = user_text
        missing = ""

    # 안전장치: is_clear=True 인데 rewritten 비어있으면 원본 사용
    if is_clear and not rewritten:
        rewritten = user_text

    print(f"[query_refine] is_clear={is_clear} rewritten='{rewritten}' missing='{missing}'")
    return {
        "rewritten_query": rewritten,
        "is_clear": is_clear,
        "missing_info": missing,
    }
