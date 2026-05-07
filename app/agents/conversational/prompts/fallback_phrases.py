"""industry 별 음성 fallback 표현 모음.

여러 노드 (faq/task/auth/vision) 의 hardcode "매장/메뉴판" 표현을 industry 기반
동적 표현으로 통일. finance 응답에 "매장 메뉴판" 어색하게 들어가던 문제 해결.

사용 예:
  from app.agents.conversational.prompts.fallback_phrases import get_inquiry_phrase
  msg = f"처리 중 문제가 생겼어요. 잠시 후 다시 시도해주시거나 {get_inquiry_phrase(industry)}."
"""

# "~로 직접 문의해주세요" — 일반 fallback 안내
_INQUIRY_PHRASE: dict[str, str] = {
    "restaurant": "매장으로 직접 문의해주세요",
    "finance": "영업점이나 고객센터로 직접 문의해주세요",
    "hospital": "원무과나 콜센터로 직접 문의해주세요",
    "government": "민원실로 직접 문의해주세요",
    "appliance": "매장으로 직접 문의해주세요",
    "retail": "매장으로 직접 문의해주세요",
}
_DEFAULT_INQUIRY = "고객센터로 직접 문의해주세요"

# "자세한 건 ~ 안내드려요" — faq 응답의 짧은 마무리 표현
_FALLBACK_HINT: dict[str, str] = {
    "restaurant": "매장 메뉴판으로",
    "finance": "영업점이나 고객센터로",
    "hospital": "원무과나 콜센터로",
    "government": "민원실로",
    "appliance": "매장에서",
    "retail": "매장에서",
}
_DEFAULT_HINT = "고객센터로"

# faq prompt 의 "~ 처럼 답하세요" 직원 호칭 (어조 결정)
_PERSONA: dict[str, str] = {
    "restaurant": "매장 직원",
    "finance": "은행 상담원",
    "hospital": "병원 직원",
    "government": "관공서 직원",
    "appliance": "매장 직원",
    "retail": "매장 직원",
}
_DEFAULT_PERSONA = "직원"

# SMS 본문 / 안내의 "변경 사항은 ~로 연락 주세요" 표현
_CONTACT_CHANNEL: dict[str, str] = {
    "restaurant": "매장으로",
    "finance": "고객센터로",
    "hospital": "원무과로",
    "government": "민원실로",
    "appliance": "매장으로",
    "retail": "매장으로",
}
_DEFAULT_CHANNEL = "고객센터로"


def get_inquiry_phrase(industry: str) -> str:
    """fallback 메시지의 '~로 직접 문의해주세요' 표현."""
    return _INQUIRY_PHRASE.get(industry, _DEFAULT_INQUIRY)


def get_fallback_hint(industry: str) -> str:
    """faq 응답 마무리의 '자세한 건 ~ 안내드려요' 표현."""
    return _FALLBACK_HINT.get(industry, _DEFAULT_HINT)


def get_persona(industry: str) -> str:
    """faq prompt 의 '~ 처럼 답하세요' 호칭."""
    return _PERSONA.get(industry, _DEFAULT_PERSONA)


def get_contact_channel(industry: str) -> str:
    """SMS 본문 등의 '변경 사항은 ~로 연락' 표현."""
    return _CONTACT_CHANNEL.get(industry, _DEFAULT_CHANNEL)
