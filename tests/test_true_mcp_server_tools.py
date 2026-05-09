"""
KDT-101: 진짜 MCP Server tool 등록/실행 단위 테스트.

- ``build_server()`` 가 12 개 tool 을 등록한다.
- 각 tool 은 토큰/설정이 없을 때 표준 envelope 의 status=skipped 를 반환한다.
- envelope 에 source=mcp_server / via_mcp=true / execution_mode=mcp 가 항상 포함된다.

이 테스트는 외부 API 를 호출하지 않는다. tenant_id="" 또는 미연결
시나리오만 사용하므로 connector 는 DB 조회 없이 항상 skipped 다.
"""
from __future__ import annotations

import asyncio
import os

import pytest


REQUIRED_TOOL_NAMES = {
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


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("MCP_ALLOW_ENV_FALLBACK", "false")
    monkeypatch.delenv("SOLAPI_API_KEY", raising=False)
    monkeypatch.delenv("SOLAPI_API_SECRET", raising=False)
    monkeypatch.delenv("SOLAPI_SENDER_NUMBER", raising=False)
    monkeypatch.delenv("SOLAPI_FROM", raising=False)
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)
    monkeypatch.delenv("SMS_TEST_TO", raising=False)
    yield


def test_build_server_registers_all_required_tools():
    from app.services.mcp.server.main import build_server

    server = build_server()
    tools = asyncio.get_event_loop().run_until_complete(server.list_tools())
    names = {t.name for t in tools}
    missing = REQUIRED_TOOL_NAMES - names
    assert not missing, f"missing MCP tools: {missing}"


def test_envelope_keys_present_for_skipped_slack():
    from app.services.mcp.server.providers.slack_tools import send_slack_alert

    result = asyncio.get_event_loop().run_until_complete(
        send_slack_alert(tenant_id="", call_id="c1", params={"message": "x"})
    )
    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"
    assert result["source"] == "mcp_server"
    assert result["via_mcp"] is True
    assert result["execution_mode"] == "mcp"
    assert result["mcp_tool"] == "slack.send_slack_alert"
    assert result["tool"] == "slack"
    assert result["action_type"] == "send_slack_alert"


def test_calendar_skipped_when_no_tenant_token():
    from app.services.mcp.server.providers.calendar_tools import schedule_callback

    result = asyncio.get_event_loop().run_until_complete(
        schedule_callback(tenant_id="", call_id="c2", params={})
    )
    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"
    assert result["mcp_tool"] == "calendar.schedule_callback"


def test_gmail_skipped_when_no_tenant_token():
    from app.services.mcp.server.providers.gmail_tools import send_manager_email

    result = asyncio.get_event_loop().run_until_complete(
        send_manager_email(tenant_id="", call_id="c3", params={"to": "x@y.com"})
    )
    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"


def test_sms_skipped_when_phone_missing():
    from app.services.mcp.server.providers.sms_tools import send_callback_sms

    result = asyncio.get_event_loop().run_until_complete(
        send_callback_sms(tenant_id="", call_id="c4", params={})
    )
    assert result["status"] == "skipped"
    assert result["error"] == "customer_phone_missing"
    assert result["mcp_tool"] == "sms.send_callback_sms"


def test_sms_skipped_when_solapi_unconfigured(monkeypatch):
    from app.services.mcp.server.providers.sms_tools import send_callback_sms

    result = asyncio.get_event_loop().run_until_complete(
        send_callback_sms(
            tenant_id="",
            call_id="c5",
            params={"customer_phone": "01012345678"},
        )
    )
    assert result["status"] == "skipped"
    assert result["error"] == "sms_config_missing"


def test_notion_skipped_when_unconfigured():
    from app.services.mcp.server.providers.notion_tools import (
        create_notion_call_record,
    )

    result = asyncio.get_event_loop().run_until_complete(
        create_notion_call_record(tenant_id="", call_id="c6", params={})
    )
    assert result["status"] == "skipped"
    assert result["error"] == "notion_not_configured"


def test_company_db_internal_tool_succeeds():
    """외부 OAuth 가 필요 없는 내부 tool 은 항상 success 를 반환한다."""
    from app.services.mcp.server.providers.company_db_tools import create_voc_issue

    result = asyncio.get_event_loop().run_until_complete(
        create_voc_issue(
            tenant_id="ten-1",
            call_id="c7",
            params={"summary": "x", "primary_category": "billing"},
        )
    )
    assert result["status"] == "success"
    assert result["external_id"]
    assert result["source"] == "mcp_server"
    assert result["via_mcp"] is True
    assert result["execution_mode"] == "mcp"


def test_internal_dashboard_internal_tool_succeeds():
    from app.services.mcp.server.providers.internal_dashboard_tools import (
        add_priority_queue,
    )

    result = asyncio.get_event_loop().run_until_complete(
        add_priority_queue(
            tenant_id="ten-1",
            call_id="c8",
            params={"priority": "high", "reason": "angry customer"},
        )
    )
    assert result["status"] == "success"
    assert result["mcp_tool"] == "internal_dashboard.add_priority_queue"


def test_jira_skipped_without_workspace(monkeypatch):
    """tenant 없으면 OAuth lookup 단계에서 skipped tenant_oauth_required."""
    from app.services.mcp.server.providers.jira_tools import create_jira_issue

    result = asyncio.get_event_loop().run_until_complete(
        create_jira_issue(tenant_id="", call_id="c9", params={})
    )
    assert result["status"] == "skipped"
    # 토큰 단계에서 skipped 발생.
    assert result["error"] == "tenant_oauth_required"
