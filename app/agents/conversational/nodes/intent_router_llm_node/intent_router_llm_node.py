from app.agents.conversational.state import CallState
from app.agents.conversational.prompts.intent_router import build_system_prompt
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.session.redis_session import RedisSessionService
from app.services.vision.session import VisionSessionService

_llm = GPT4OMiniService()
_call_session_svc = RedisSessionService()
_vision_session_svc = VisionSessionService()
_HISTORY_TURN_LIMIT = 6  # 직전 3턴 (user+assistant 합쳐 6개 항목)

_VALID_INTENTS = {"faq", "task", "auth", "vision", "escalation", "repeat"}


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
    call_id = state["call_id"]
    history = state.get("session_view", {}).get("conversation_history", [])
    tenant_name = state.get("tenant_name", "고객센터")
    tenant_industry = state.get("tenant_industry", "unknown")

    # Active vision 세션 short-circuit — pending/analyzing 상태면 사용자 발화가 무엇이든
    # vision 으로 강제 라우팅. "업로드됐나요", "링크 보냈잖아요" 등 vision 컨텍스트의
    # 다양한 발화 변형을 LLM 분류에 맡기지 않고 안정 처리. analyzed/없음 시는 일반 분류.
    vision_id = await _call_session_svc.get_vision_id(call_id)
    if vision_id:
        vsession = await _vision_session_svc.get_session(vision_id)
        if vsession:
            vstatus = vsession.get("status", "")
            if vstatus in ("pending", "analyzing"):
                print(
                    f"[intent_router] active vision (status={vstatus}) → vision 강제"
                )
                return {"intent": "vision"}

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

    # 가드 — escalation 으로 분류됐지만 사용자 발화에 명시 키워드 ("상담원/사람과/답답"
    # 등) 가 없으면 LLM 비결정성으로 잘못 분류된 것으로 보고 faq 로 변경.
    # 실통화에서 LLM 이 history 영향으로 일반 FAQ 발화를 escalation 으로 라우팅하던
    # 케이스 차단. 시연 시 "5번 시도 후 escalation" 효과 (사용자가 직접 호출 X 면 X).
    if intent == "escalation":
        decision_src = f"{state.get('user_text', '')} {state.get('rewritten_query', '')}"
        explicit_kw = ("상담원", "사람과", "사람한테", "답답", "화나", "직접 연결")
        if not any(kw in decision_src for kw in explicit_kw):
            print(
                f"[intent_router] escalation 분류 → 명시 키워드 없음 → faq 강제 변경"
            )
            intent = "faq"

    print(f"[intent_router] query='{query}' → intent='{intent}'")
    return {"intent": intent}
