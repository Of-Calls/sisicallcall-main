"""통화 task 노드용 MCP 도구 카탈로그.

OpenAI Function Calling 표준 호환 형식.

각 항목 키:
- tool         : mcp_client.call_tool() 의 tool_name
- oauth_provider: file repo (.local/tenant_integrations.json) 의 provider 키.
                  None 이면 OAuth 게이트 없이 항상 노출 (mock/회사 DB 등).
- industries   : 노출 허용 industry 화이트리스트. None 이면 전체 노출.
                  industry-specific 도구 (예: 카드 정지) 가 잘못된 tenant 에
                  노출되는 회귀 방지용.
- description  : LLM 이 도구 선택 시 참고할 설명
- requires_auth: True 면 task_branch 가 호출 전 본인 인증 게이트 발동
- parameters   : JSON Schema (LLM Function Calling 표준)

도구 추가 시 새 action_type 항목만 추가. mcp_client 인프라는 재사용.
"""
from app.models.tenant_integration import IntegrationStatus
from app.repositories.tenant_integration_repo import list_integrations

# action_type → 도구 spec
TOOL_CATALOG: dict[str, dict] = {
    "lookup_member_info": {
        "tool": "company_db",
        "oauth_provider": None,
        "industries": ["finance", "hospital"],
        "description": "본인의 회원 정보 (이름, 등급, 등록 주소) 조회",
        "requires_auth": True,
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "조회할 회원의 전화번호 (010-XXXX-XXXX 형태. 사용자가 한글 숫자로 말하면 아라비아 숫자로 변환 — 예: '공일공일이삼사' → '01012345678')",
                },
            },
            "required": ["phone_number"],
        },
    },
    "schedule_callback": {
        "tool": "calendar",
        "oauth_provider": "google_calendar",
        "industries": None,
        "description": "예약/방문 일정 등록 (식당 좌석 예약, 진료 예약, 상담 일정 등 모든 일정 예약 포함)",
        "requires_auth": False,
        "parameters": {
            "type": "object",
            "properties": {
                "preferred_time": {
                    "type": "string",
                    "description": "예약 희망 일시 (예: '내일 오후 3시', '2026-05-03 15:00')",
                },
                "callback_reason": {
                    "type": "string",
                    "description": "예약 사유",
                },
            },
            "required": ["preferred_time"],
        },
    },
    "send_confirm_sms": {
        "tool": "sms",
        "oauth_provider": None,
        "industries": None,
        "description": "확정 안내 SMS 발송 (수신자는 시스템이 자동 결정 — 사용자에게 번호 묻지 마세요)",
        "requires_auth": False,
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "발송할 SMS 본문",
                },
            },
            "required": ["message"],
        },
    },
    "suspend_card": {
        "tool": "company_db",
        "oauth_provider": None,
        "industries": ["finance"],
        "description": "분실/도난 시 카드 사용 정지 처리 (등록된 회원의 카드를 즉시 정지). 시스템이 회원 전화번호를 자동 주입하므로 사용자에게 번호를 묻지 마세요.",
        "requires_auth": True,
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "정지할 카드 회원의 전화번호 (시스템 자동 주입 — LLM 은 빈 값 두세요)",
                },
            },
            "required": ["phone_number"],
        },
    },
}


async def get_available_actions(
    tenant_id: str, tenant_industry: str = ""
) -> dict[str, dict]:
    """tenant 의 OAuth 연결 + industry 화이트리스트 기준으로 노출 도구 필터링.

    필터 우선순위 (둘 다 통과해야 노출):
      1. OAuth — oauth_provider=None 이거나 tenant 가 해당 provider 연결.
      2. Industry — industries=None 이거나 tenant_industry 가 화이트리스트에 포함.

    데이터 소스: tenant_integration_repo (.local/tenant_integrations.json — file 모드).
    industry 미지정 (빈 문자열) → 화이트리스트 매칭 실패. industry-specific 도구 노출 안 됨.
    """
    integrations = list_integrations(tenant_id)
    connected = {
        i.provider for i in integrations if i.status == IntegrationStatus.connected
    }
    return {
        action_type: spec
        for action_type, spec in TOOL_CATALOG.items()
        if (spec["oauth_provider"] is None or spec["oauth_provider"] in connected)
        and (spec.get("industries") is None or tenant_industry in spec["industries"])
    }


def to_openai_tools(actions: dict[str, dict]) -> list[dict]:
    """카탈로그 → OpenAI Function Calling tools 형식 변환."""
    return [
        {
            "type": "function",
            "function": {
                "name": action_type,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for action_type, spec in actions.items()
    ]
