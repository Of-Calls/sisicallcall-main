"""
KDT-101: MCPGatewayConnector tool name 매핑 / payload 직렬화 테스트.

Action Executor 가 (tool, action_type) 조합을 MCP server tool 이름으로
정확히 매핑하는지, payload 가 tenant_id/call_id/params 셋으로 분리되어
gateway 에 전달되는지 확인한다.
"""
from __future__ import annotations

import asyncio

import pytest


def test_resolve_mcp_tool_name_returns_dotted_name():
    from app.services.mcp.connectors.mcp_gateway_connector import resolve_mcp_tool_name

    cases = [
        (("slack", "send_slack_alert"), "slack.send_slack_alert"),
        (("calendar", "schedule_callback"), "calendar.schedule_callback"),
        (("gmail", "send_manager_email"), "gmail.send_manager_email"),
        (("jira", "create_jira_issue"), "jira.create_jira_issue"),
        (("jira", "create_voc_issue"), "jira.create_jira_issue"),
        (("notion", "create_notion_call_record"), "notion.create_notion_call_record"),
        (("sms", "send_voc_receipt_sms"), "sms.send_voc_receipt_sms"),
        (("sms", "send_callback_sms"), "sms.send_callback_sms"),
        (("company_db", "create_voc_issue"), "company_db.create_voc_issue"),
        (("company_db", "add_priority_queue"), "company_db.add_priority_queue"),
        (("company_db", "mark_faq_candidate"), "company_db.mark_faq_candidate"),
        (("internal_dashboard", "add_priority_queue"), "internal_dashboard.add_priority_queue"),
        (("internal_dashboard", "mark_faq_candidate"), "internal_dashboard.mark_faq_candidate"),
    ]
    for key, expected in cases:
        assert resolve_mcp_tool_name(*key) == expected, key


def test_resolve_mcp_tool_name_unknown_returns_none():
    from app.services.mcp.connectors.mcp_gateway_connector import resolve_mcp_tool_name

    assert resolve_mcp_tool_name("weather", "forecast") is None
    assert resolve_mcp_tool_name("slack", "unknown_action") is None


class _FakeProtocolClient:
    """MCPProtocolClient 인터페이스를 흉내내는 fake."""

    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    async def start(self):
        return None

    async def close(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def call_tool(self, name: str, payload: dict):
        self.calls.append((name, payload))
        return self.response


def test_gateway_translates_action_to_mcp_payload():
    """gateway 가 (tool, action_type, params) 를 MCP tool name + payload 로 매핑한다."""
    from app.services.mcp.connectors.mcp_gateway_connector import MCPGatewayConnector

    fake = _FakeProtocolClient({
        "action_type": "send_slack_alert",
        "tool": "slack",
        "status": "success",
        "external_id": "C123:ts",
        "result": {"channel": "C123", "ts": "ts"},
        "error": None,
        "latency_ms": 12,
        "source": "mcp_server",
        "via_mcp": True,
        "execution_mode": "mcp",
        "mcp_tool": "slack.send_slack_alert",
    })
    gateway = MCPGatewayConnector(protocol_client=fake)

    raw = asyncio.run(gateway.execute(
        {"tool": "slack", "action_type": "send_slack_alert", "params": {"message": "x"}},
        call_id="c-1",
        tenant_id="ten-1",
    ))

    assert len(fake.calls) == 1
    name, payload = fake.calls[0]
    assert name == "slack.send_slack_alert"
    assert payload == {
        "tenant_id": "ten-1",
        "call_id": "c-1",
        "params": {"message": "x"},
    }
    assert raw["status"] == "success"
    assert raw["external_id"] == "C123:ts"
    assert raw["result"]["source"] == "mcp_server"
    assert raw["result"]["via_mcp"] is True
    assert raw["result"]["execution_mode"] == "mcp"
    assert raw["result"]["mcp_tool"] == "slack.send_slack_alert"
    assert raw["result"]["channel"] == "C123"


def test_gateway_unknown_mapping_returns_failed_envelope():
    from app.services.mcp.connectors.mcp_gateway_connector import MCPGatewayConnector

    fake = _FakeProtocolClient({})
    gateway = MCPGatewayConnector(protocol_client=fake)

    raw = asyncio.run(gateway.execute(
        {"tool": "weather", "action_type": "forecast", "params": {}},
        call_id="c-x",
        tenant_id="ten-1",
    ))

    assert raw["status"] == "failed"
    assert raw["error"].startswith("unknown_mcp_tool")
    assert len(fake.calls) == 0
    assert raw["result"]["source"] == "mcp_server"
    assert raw["result"]["via_mcp"] is True


def test_gateway_propagates_skipped_envelope():
    from app.services.mcp.connectors.mcp_gateway_connector import MCPGatewayConnector

    fake = _FakeProtocolClient({
        "action_type": "send_callback_sms",
        "tool": "sms",
        "status": "skipped",
        "error": "customer_phone_missing",
        "external_id": None,
        "result": {},
        "latency_ms": 1,
        "source": "mcp_server",
        "via_mcp": True,
        "execution_mode": "mcp",
        "mcp_tool": "sms.send_callback_sms",
    })
    gateway = MCPGatewayConnector(protocol_client=fake)

    raw = asyncio.run(gateway.execute(
        {"tool": "sms", "action_type": "send_callback_sms", "params": {}},
        call_id="c-2",
        tenant_id="ten-1",
    ))

    assert raw["status"] == "skipped"
    assert raw["error"] == "customer_phone_missing"
    assert raw["result"]["source"] == "mcp_server"
    assert raw["result"]["mcp_tool"] == "sms.send_callback_sms"


def test_executor_in_mcp_mode_routes_through_gateway(monkeypatch):
    """ActionExecutor 가 진짜로 gateway 를 통해서만 동작하는 통합 흐름.

    direct registry 모듈은 MCP-only 전환과 함께 삭제됐으므로 executor 가
    그쪽을 호출할 가능성 자체가 없다.
    """
    from app.agents.post_call.actions.executor import ActionExecutor

    fake_client = _FakeProtocolClient({
        "action_type": "send_slack_alert",
        "tool": "slack",
        "status": "success",
        "external_id": "C-X:ts",
        "result": {"channel": "C-X", "ts": "ts"},
        "error": None,
        "latency_ms": 1,
        "source": "mcp_server",
        "via_mcp": True,
        "execution_mode": "mcp",
        "mcp_tool": "slack.send_slack_alert",
    })

    from app.services.mcp.connectors.mcp_gateway_connector import MCPGatewayConnector
    gateway = MCPGatewayConnector(protocol_client=fake_client)
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: gateway,
    )

    async def _no_idem(call_id, action_type, tool):
        return None
    monkeypatch.setattr(
        "app.agents.post_call.actions.executor.find_successful_action", _no_idem,
    )

    executor = ActionExecutor()
    out = asyncio.run(executor.execute_actions(
        call_id="c-int",
        tenant_id="ten-1",
        actions=[
            {"tool": "slack", "action_type": "send_slack_alert", "params": {"message": "x"}},
        ],
    ))

    assert out[0]["status"] == "success"
    assert out[0]["external_id"] == "C-X:ts"
    assert out[0]["result"]["execution_mode"] == "mcp"
    assert out[0]["result"]["source"] == "mcp_server"
    assert out[0]["result"]["mcp_tool"] == "slack.send_slack_alert"
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][0] == "slack.send_slack_alert"
