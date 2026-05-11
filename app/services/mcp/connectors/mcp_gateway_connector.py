"""
MCPGatewayConnector — Post-call action 실행의 유일한 진입점.

Post-call ActionExecutor 가 호출하는 게이트웨이. 내부에서는 절대로
SlackConnector / GmailConnector 같은 direct connector 를 호출하지 않고,
``MCPProtocolClient`` 를 통해 별도 process 로 실행 중인 자체 MCP Server
의 tool 을 호출한다.

  Action Executor
  → MCPGatewayConnector.execute()
  → MCPProtocolClient.call_tool(mcp_tool_name, payload)
  → stdio transport
  → 자체 MCP Server (별도 process)
  → MCP Server tool
  → 외부 provider 실행

반환 형식은 ActionExecutor 가 그대로 action_success / action_failed /
action_skipped 로 변환할 수 있는 표준 dict — connectors/base.BaseMCPConnector
의 _success/_failed/_skipped 와 호환.
"""
from __future__ import annotations

from typing import Any

from app.services.mcp.protocol_client import (
    MCPClientTransportError,
    MCPProtocolClient,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# (tool, action_type) → MCP Server 에 등록된 tool 이름.
_TOOL_NAME_MAP: dict[tuple[str, str], str] = {
    ("slack",              "send_slack_alert"):          "slack.send_slack_alert",
    ("calendar",           "schedule_callback"):         "calendar.schedule_callback",
    ("gmail",              "send_manager_email"):        "gmail.send_manager_email",
    ("jira",               "create_jira_issue"):         "jira.create_jira_issue",
    ("jira",               "create_voc_issue"):          "jira.create_jira_issue",
    ("notion",             "create_notion_call_record"): "notion.create_notion_call_record",
    ("notion",             "create_notion_voc_record"):  "notion.create_notion_voc_record",
    ("sms",                "send_voc_receipt_sms"):      "sms.send_voc_receipt_sms",
    ("sms",                "send_callback_sms"):         "sms.send_callback_sms",
    ("sms",                "send_reservation_confirmation"): "sms.send_callback_sms",
    ("company_db",         "create_voc_issue"):          "company_db.create_voc_issue",
    ("company_db",         "add_priority_queue"):        "company_db.add_priority_queue",
    ("company_db",         "mark_faq_candidate"):        "company_db.mark_faq_candidate",
    ("internal_dashboard", "add_priority_queue"):        "internal_dashboard.add_priority_queue",
    ("internal_dashboard", "mark_faq_candidate"):        "internal_dashboard.mark_faq_candidate",
}


def resolve_mcp_tool_name(tool: str, action_type: str) -> str | None:
    """(tool, action_type) → MCP Server tool name. 알려지지 않은 조합은 None."""
    return _TOOL_NAME_MAP.get((tool, action_type))


def _flatten_to_action_result(
    *,
    tool: str,
    action_type: str,
    mcp_result: dict[str, Any],
) -> dict[str, Any]:
    """MCP Server 가 돌려준 envelope 을 ActionExecutor 가 기대하는 shape 로 정규화.

    ActionExecutor.execute_one 은 raw dict 의 status / external_id / result /
    error 키만 본다. 다만 mcp_action_logs 와 source/via_mcp 추적을 위해 result
    안에는 source=mcp_server / via_mcp=true / execution_mode=mcp / mcp_tool 을
    유지한다.
    """
    status = mcp_result.get("status") or "failed"
    external_id = mcp_result.get("external_id")
    error = mcp_result.get("error")
    inner_result = mcp_result.get("result") or {}
    if not isinstance(inner_result, dict):
        inner_result = {"value": inner_result}
    inner_result = dict(inner_result)
    inner_result.setdefault("source", mcp_result.get("source", "mcp_server"))
    inner_result.setdefault("via_mcp", mcp_result.get("via_mcp", True))
    inner_result.setdefault("execution_mode", mcp_result.get("execution_mode", "mcp"))
    if mcp_result.get("mcp_tool"):
        inner_result.setdefault("mcp_tool", mcp_result["mcp_tool"])
    if mcp_result.get("latency_ms") is not None:
        inner_result.setdefault("mcp_latency_ms", mcp_result["latency_ms"])

    return {
        "status": status,
        "external_id": external_id,
        "error": error,
        "result": inner_result,
    }


class MCPGatewayConnector:
    """Post-call action 을 MCP protocol client 로 우회 실행하는 게이트웨이."""

    def __init__(self, *, protocol_client: MCPProtocolClient | None = None) -> None:
        self._injected_client = protocol_client

    async def execute(
        self,
        action: dict,
        *,
        call_id: str,
        tenant_id: str,
    ) -> dict:
        tool = action.get("tool", "")
        action_type = action.get("action_type", "")
        mcp_tool = resolve_mcp_tool_name(tool, action_type)
        if mcp_tool is None:
            logger.warning(
                "MCPGatewayConnector: unknown (tool=%s, action_type=%s)",
                tool, action_type,
            )
            return {
                "status": "failed",
                "external_id": None,
                "error": f"unknown_mcp_tool:{tool}.{action_type}",
                "result": {
                    "source": "mcp_server",
                    "via_mcp": True,
                    "execution_mode": "mcp",
                },
            }

        payload = {
            "tenant_id": tenant_id or "",
            "call_id": call_id,
            "params": action.get("params", {}) or {},
        }

        client = self._injected_client or MCPProtocolClient()
        owns_client = self._injected_client is None
        try:
            if owns_client:
                await client.start()
            mcp_result = await client.call_tool(mcp_tool, payload)
        finally:
            if owns_client:
                try:
                    await client.close()
                except Exception:
                    logger.debug("MCPGatewayConnector: protocol client close 실패 (무시)")

        return _flatten_to_action_result(
            tool=tool, action_type=action_type, mcp_result=mcp_result,
        )


# ── 모듈 레벨 헬퍼 ────────────────────────────────────────────────────────────


_default_gateway: MCPGatewayConnector | None = None


def get_default_gateway() -> MCPGatewayConnector:
    """모듈 레벨 lazy 인스턴스. 매 호출마다 새 process 를 띄운다."""
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = MCPGatewayConnector()
    return _default_gateway


def is_transport_error(exc: BaseException) -> bool:
    return isinstance(exc, MCPClientTransportError)
