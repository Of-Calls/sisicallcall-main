from app.agents.conversational.state import CallState
from app.services.llm.gpt4o_mini import GPT4OMiniService

_llm = GPT4OMiniService()
_HISTORY_TURN_LIMIT = 6  # 직전 3턴 (user+assistant 합쳐 6개 항목)

_VALID_INTENTS = {"faq", "task", "auth", "vision", "escalation"}

_SYSTEM_PROMPT = """당신은 전화 상담 의도 분류기입니다. 재작성된 사용자 쿼리를 바탕으로 5가지 중 하나로 분류하세요.

- faq: 단순 정보 요청, 영업시간, 위치, 가격, 메뉴, 일반 질문, 또는 **제품명이 명시된** 제품 문의 (예: "메뉴가 뭔지 궁금해요", "몇 시까지 영업하나요", "B5 냉장고 사양 알려주세요")
- task: 업무 처리 (예약, 조회, 변경, 취소, SMS/문자 발송, 회원정보 조회/변경 등 도구 호출이 필요한 작업)
  → 예: "내일 오후 3시 예약하고 싶어요", "확인 문자 보내주세요", "내 회원정보 조회해줘", "콜백 예약해줘"
- auth: **본인 신원 확인 절차 자체** 에 대한 의사 표시 (예: "본인 인증할게요", "인증 진행해주세요", "주민번호 알려드릴게요")
  → 주의: "확인 문자 보내줘" 같은 SMS 발송이나 "회원정보 조회" 같은 데이터 작업은 task 입니다. auth 는 사용자가 인증 절차 자체를 진행하겠다는 의도일 때만.
- vision: 사용자가 **눈앞에 있는 실물/물건의 정체를 명시적으로 모른다고 표현한 경우**나, 시각적 확인이 반드시 필요한 경우만. (예: "내 눈앞에 있는 이게 뭔지 모르겠어요", "사용자가 어떤 상품인지 식별하지 못하는 상태")
  → 주의: "메뉴가 뭔지", "어떤 서비스인지" 같은 무형의 정보 요청은 시각적 확인이 아니므로 절대 vision이 아닙니다. 무조건 faq로 분류하세요.
- escalation: 상담원 연결 요청, 화남, 불만 (예: "상담원 바꿔줘요")

[상태 전이(State Transition) 및 동의/거절 처리]
쿼리 재작성기가 이전 대화 맥락을 파악하여 "사용자가 ~에 동의함/거절함" 형태로 쿼리를 넘겨준 경우, 해당 맥락에 맞춰 라우팅하세요.
- 쿼리 예시: "사용자가 본인 인증 진행에 동의함" → auth
- 쿼리 예시: "사용자가 본인 인증을 완료했음을 알림/확인함" → auth (재진입 — active 인증 세션 상태 확인)
- 쿼리 예시: "사용자가 사진 촬영/업로드에 동의함" → vision
- 쿼리 예시: "사용자가 예약 진행에 동의함" → task
- 거절 처리: 사용자가 인증이나 사진 업로드를 거절한 쿼리("사용자가 인증을 거절함")라면 일반적인 대화로 돌리기 위해 'faq'로 분류하세요.

출력 형식: 정확히 한 단어만. faq, task, auth, vision, escalation 중 하나.
다른 설명, 따옴표, 마침표 없이 단어 하나만 출력."""


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

    user_message = f"[이전 대화]\n{_format_history(history)}\n\n[현재 사용자 발화]\n{query}"

    raw = await _llm.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.0,
        max_tokens=15,
    )

    intent = raw.strip().lower()
    if intent not in _VALID_INTENTS:
        intent = "escalation"  # 분류 실패 시 상담원 연결

    print(f"[intent_router] query='{query}' → intent='{intent}'")
    return {"intent": intent}
