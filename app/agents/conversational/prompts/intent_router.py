"""intent_router 노드 system prompt — tenant industry 기반 동적 생성.

intent 분류 자체는 도메인 무관 (faq/task/auth/vision/escalation) 이지만 예시
발화가 식당/가전 가정으로 박혀있어 병원/관공서 등 다른 도메인에서 미스매치
발생. industry 별 1-2 예시 + 라벨로 컨텍스트 일관화.

state transition 섹션 (사용자가 ~에 동의함/거절함 패턴) 은 도메인 무관이라 그대로 유지.
"""
from app.agents.conversational.prompts.industry_context import get_context

_INTENT_ROUTER_EXAMPLES: dict[str, dict[str, str]] = {
    "restaurant": {
        "faq":  '"메뉴가 뭔지 궁금해요", "몇 시까지 영업하나요"',
        "task": '"내일 오후 3시 예약하고 싶어요", "확인 문자 보내주세요"',
    },
    "hospital": {
        "faq":  '"진료과 뭐가 있나요", "응급실 운영시간"',
        "task": '"내일 오전 9시 진료 예약", "진료 결과 문자로 받을게요"',
    },
    "government": {
        "faq":  '"민원 처리 시간 어떻게 되나요", "주차장 이용 가능한가요"',
        "task": '"민원 콜백 예약해주세요", "처리 결과 문자로 보내주세요"',
    },
    "finance": {
        "faq":  '"창구 운영시간 문의", "ATM 위치"',
        "task": '"상담 콜백 예약", "거래내역 문자 발송"',
    },
    "appliance": {
        "faq":  '"B5 냉장고 사양 알려주세요", "전시장 운영시간"',
        "task": '"A/S 콜백 예약", "방문 일정 문자 발송"',
    },
    "retail": {
        "faq":  '"오늘 행사 뭐가 있나요", "주차되나요"',
        "task": '"콜백 예약", "할인 안내 문자 보내주세요"',
    },
}


def _format_examples(industry: str) -> str:
    ex = _INTENT_ROUTER_EXAMPLES.get(industry, {})
    if not ex:
        return ""
    parts = []
    if ex.get("faq"):
        parts.append(f"  faq 예: {ex['faq']}")
    if ex.get("task"):
        parts.append(f"  task 예: {ex['task']}")
    return "\n".join(parts)


def build_system_prompt(tenant_name: str, tenant_industry: str) -> str:
    ctx = get_context(tenant_industry)
    label = ctx["label"]
    examples = _format_examples(tenant_industry)
    examples_block = f"\n\n[이 {label} 도메인 예시]\n{examples}" if examples else ""

    return f"""당신은 전화 상담 의도 분류기입니다. 사용자는 "{tenant_name}" ({label}) 에 전화 중입니다.
재작성된 사용자 쿼리를 바탕으로 5가지 중 하나로 분류하세요.

- faq: 단순 정보 요청, 영업시간, 위치, 가격, 메뉴/서비스/상품 안내, 일반 질문, 또는 **항목명이 명시된** 정보 문의
- task: 업무 처리 (예약, 조회, 변경, 취소, SMS/문자 발송, 회원정보 조회/변경 등 도구 호출이 필요한 작업)
- auth: **본인 신원 확인 절차 자체** 에 대한 의사 표시 ("본인 인증할게요", "인증 진행해주세요", "주민번호 알려드릴게요")
  → 주의: "확인 문자 보내줘" 같은 SMS 발송이나 "회원정보 조회" 같은 데이터 작업은 task. auth 는 사용자가 인증 절차 자체를 진행하겠다는 의도일 때만.
- vision: 사용자가 **눈앞에 있는 실물/물건의 정체를 명시적으로 모른다고 표현한 경우** 나, 시각적 확인이 반드시 필요한 경우만.
  → 주의: "메뉴", "어떤 진료", "어떤 서비스" 같은 무형의 정보 요청은 시각적 확인이 아니므로 절대 vision이 아닙니다. 무조건 faq.
- escalation: 상담원 연결 요청, 화남, 불만 ("상담원 바꿔줘요")

[분류 우선순위 — 매우 중요]
- 현재 사용자 발화 자체로 분류하세요. history 는 "사용자가 ~에 동의함/거절함" 같은
  명시적 패턴일 때만 영향. 그 외엔 history 가 분류를 좌우하지 않게 하세요.
- "예약하고싶어요", "회원정보 알려주세요" 같은 명확한 task 발화는 history 와 무관히 task.
- 직전이 faq/clarify 흐름이라도 현재 발화가 task 면 task 로 분류.{examples_block}

[상태 전이(State Transition) 및 동의/거절 처리]
쿼리 재작성기가 이전 대화 맥락을 파악하여 "사용자가 ~에 동의함/거절함" 형태로 쿼리를 넘겨준 경우, 해당 맥락에 맞춰 라우팅하세요.
- "내 회원정보 다시 알려주세요", "예약 다시 진행해주세요" 같은 task 재진입 발화는 intent='task'. task_branch 가 auth 세션 상태 보고 자동 처리.
- "사용자가 본인 인증을 완료했음을 알림/확인함" 패턴은 사용자가 명시적으로 "인증 끝났어요", "인증 됐어요" 라고 말한 경우만 → auth.
- 쿼리 예시: "사용자가 본인 인증 진행에 동의함" → auth
- 쿼리 예시: "사용자가 본인 인증을 완료했음을 알림/확인함" → auth (재진입 — active 인증 세션 상태 확인)
- 쿼리 예시: "사용자가 사진 촬영/업로드에 동의함" → vision
- 쿼리 예시: "사용자가 예약 진행에 동의함" → task
- 거절 처리: 사용자가 인증이나 사진 업로드를 거절한 쿼리 → 일반 대화로 돌리기 위해 'faq'로 분류.

출력 형식: 정확히 한 단어만. faq, task, auth, vision, escalation 중 하나.
다른 설명, 따옴표, 마침표 없이 단어 하나만 출력."""
