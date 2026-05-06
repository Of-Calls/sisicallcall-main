from typing import TypedDict


class CallState(TypedDict):
    call_id: str
    tenant_id: str
    tenant_name: str       # tenants.name (예: "한밭식당") / 미등록 시 "고객센터"
    tenant_industry: str   # tenants.industry (예: "restaurant") / 미등록 시 "unknown"
    user_text: str
    intent: str          # "faq" | "task" | "auth" | "vision" | "escalation"
    response_text: str

    # 세션 view (Redis 에서 로드한 당 턴 관점 정보)
    # 구조: {"conversation_history": [{"role": "user|assistant", "text": "...", "ts": float}, ...]}
    session_view: dict

    # query_refine 결과
    rewritten_query: str   # 맥락 보강된 self-contained 쿼리 (is_clear=False 면 빈 문자열)
    is_clear: bool         # 발화가 명확한가 (재작성 성공 여부)
    missing_info: str      # is_clear=False 일 때 무엇이 부족한지 (예: "어떤 상품")
