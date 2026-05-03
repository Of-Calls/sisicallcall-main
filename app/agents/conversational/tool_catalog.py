"""통화 task 노드용 MCP 도구 카탈로그.

OpenAI Function Calling 표준 호환 형식.

각 항목 키:
- tool         : mcp_client.call_tool() 의 tool_name
- description  : LLM 이 도구 선택 시 참고할 설명
- requires_auth: True 면 task_branch 가 호출 전 본인 인증 게이트 발동
- parameters   : JSON Schema (LLM Function Calling 표준)

도구 추가 시 새 action_type 항목만 추가. mcp_client 인프라는 재사용.
"""
import asyncpg

from app.utils.config import settings

# action_type → 도구 spec
TOOL_CATALOG: dict[str, dict] = {
    "lookup_member_info": {
        "tool": "company_db",
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
        "description": "통화/방문 일정 예약 (식당 좌석 예약, 진료 예약, 상담 콜백 등 모든 일정 예약 포함)",
        "requires_auth": False,
        "parameters": {
            "type": "object",
            "properties": {
                "preferred_time": {
                    "type": "string",
                    "description": "사용자 선호 시각 (예: '내일 오후 3시', '2026-05-03 15:00')",
                },
                "callback_reason": {
                    "type": "string",
                    "description": "콜백 사유",
                },
            },
            "required": ["preferred_time"],
        },
    },
    "send_confirm_sms": {
        "tool": "sms",
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
}


async def get_available_actions(tenant_id: str) -> dict[str, dict]:
    """tenant_integrations.status='connected' 인 도구만 필터링한 카탈로그 반환.

    예: 매장이 calendar/sms 만 가입 → schedule_callback / send_confirm_sms 만 노출.
    """
    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch(
            "SELECT provider FROM tenant_integrations "
            "WHERE tenant_id = $1::uuid AND status = 'connected'",
            tenant_id,
        )
    finally:
        await conn.close()
    connected = {r["provider"] for r in rows}
    return {
        action_type: spec
        for action_type, spec in TOOL_CATALOG.items()
        if spec["tool"] in connected
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
