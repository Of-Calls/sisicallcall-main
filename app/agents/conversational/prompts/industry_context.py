"""테넌트 industry 별 도메인 컨텍스트 — label / facility_hint 만 공유.

여러 LangGraph 노드 (query_refine, 향후 intent_router/clarify) 가 동일한
도메인 라벨을 사용하기 위한 공유 메타. 노드별 예시 (examples) 는 각 노드
prompt 모듈에서 자체 정의한다.

industry 값은 tenants.industry 컬럼 (CHECK 제약: hospital | restaurant |
finance | appliance | government | retail) 과 일치한다.
미등록/NULL → DEFAULT_INDUSTRY_CONTEXT.
"""

INDUSTRY_CONTEXT: dict[str, dict[str, str]] = {
    "restaurant": {
        "label": "식당",
        "facility_hint": "별관, 룸, 좌석, 주차장 같은 시설",
    },
    "hospital": {
        "label": "병원",
        "facility_hint": "응급실, 진료과, 검사실, 외래, 약국 같은 진료/시설",
    },
    "government": {
        "label": "관공서",
        "facility_hint": "민원실, 본관, 별관, 부서, 강당 같은 청사 시설",
    },
    "finance": {
        "label": "금융기관",
        "facility_hint": "창구, 상담실, ATM, 대기번호 같은 시설 또는 이체, 송금, 한도, 잔액, 이자, 대출, 예금, 적금, 환전, 분실 신고, 인증 같은 업무",
    },
    "appliance": {
        "label": "매장",
        "facility_hint": "전시장, A/S 센터, 제품군 같은 매장 시설",
    },
    "retail": {
        "label": "매장",
        "facility_hint": "진열대, 카운터, 주차장 같은 시설",
    },
}

DEFAULT_INDUSTRY_CONTEXT: dict[str, str] = {
    "label": "기관",
    "facility_hint": "공간/시설/부서/업무",
}


def get_context(industry: str) -> dict[str, str]:
    """industry 코드 → label/facility_hint dict. 미등록/NULL 은 DEFAULT."""
    return INDUSTRY_CONTEXT.get(industry, DEFAULT_INDUSTRY_CONTEXT)
