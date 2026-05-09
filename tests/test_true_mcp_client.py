"""
KDT-101: 진짜 MCP Client (stdio transport) 테스트.

실제로 MCP Server process 를 자식 process 로 띄워서 list_tools / call_tool
을 호출하고, 표준 envelope 이 반환되는지 확인한다.

이 테스트는 외부 API 를 호출하지 않는다 — tenant 토큰이 없으므로
slack tool 은 skipped("tenant_oauth_required") 를 반환한다.

CI 환경에서 stdio process 띄우기가 제한적인 경우를 대비해
``RUN_TRUE_MCP_TRANSPORT_TEST`` 환경변수가 ``true`` 일 때만 실행하도록
한다 (기본값: 활성화). 로컬에서는 항상 실행.
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_TRUE_MCP_TRANSPORT_TEST", "true").lower() not in ("1", "true"),
    reason="set RUN_TRUE_MCP_TRANSPORT_TEST=true to enable real stdio transport test",
)


@pytest.fixture()
def mcp_env(monkeypatch):
    """자식 server process 의 동작을 결정짓는 env 를 정리."""
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("MCP_ALLOW_ENV_FALLBACK", "false")
    monkeypatch.setenv("MCP_CLIENT_TIMEOUT_SEC", "30")
    # python 인터프리터 + scripts/run_mcp_server.py 를 명시적으로 사용
    monkeypatch.setenv("MCP_SERVER_COMMAND", sys.executable or "python")
    monkeypatch.setenv("MCP_SERVER_ARGS", "scripts/run_mcp_server.py")
    yield


def test_list_tools_via_real_stdio_transport(mcp_env):
    from app.services.mcp.protocol_client import MCPProtocolClient

    async def run():
        async with MCPProtocolClient() as cli:
            return await cli.list_tools()

    tools = asyncio.run(run())
    names = {t["name"] for t in tools}
    expected = {
        "slack.send_slack_alert",
        "calendar.schedule_callback",
        "gmail.send_manager_email",
        "jira.create_jira_issue",
        "notion.create_notion_call_record",
        "sms.send_voc_receipt_sms",
        "sms.send_callback_sms",
        "company_db.create_voc_issue",
        "company_db.add_priority_queue",
        "company_db.mark_faq_candidate",
        "internal_dashboard.add_priority_queue",
        "internal_dashboard.mark_faq_candidate",
    }
    assert expected.issubset(names), f"missing tools: {expected - names}"


def test_call_slack_tool_returns_skipped_envelope(mcp_env):
    from app.services.mcp.protocol_client import MCPProtocolClient

    async def run():
        async with MCPProtocolClient() as cli:
            return await cli.call_tool(
                "slack.send_slack_alert",
                {"tenant_id": "", "call_id": "test-1", "params": {"message": "hi"}},
            )

    result = asyncio.run(run())
    assert isinstance(result, dict)
    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"
    assert result["source"] == "mcp_server"
    assert result["via_mcp"] is True
    assert result["execution_mode"] == "mcp"
    assert result["mcp_tool"] == "slack.send_slack_alert"


def test_call_company_db_internal_tool_succeeds(mcp_env):
    from app.services.mcp.protocol_client import MCPProtocolClient

    async def run():
        async with MCPProtocolClient() as cli:
            return await cli.call_tool(
                "company_db.create_voc_issue",
                {
                    "tenant_id": "ten-1",
                    "call_id": "test-2",
                    "params": {"summary": "x"},
                },
            )

    result = asyncio.run(run())
    assert result["status"] == "success"
    assert result["external_id"]
    assert result["source"] == "mcp_server"
    assert result["via_mcp"] is True
    assert result["execution_mode"] == "mcp"
