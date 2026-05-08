"""
KDT-101: 12개 MCP provider 별 mapping / 라우팅 / metadata 보존 통합 검증.

Slack 한 개만 보지 않고 Slack/Calendar/Gmail/Jira/Notion/SMS/Company DB/
Internal Dashboard 12개 tool 전부에 대해 다음을 보장한다:

  1. MCPGatewayConnector._TOOL_NAME_MAP 의 value 가 실제 MCP Server tool
     이름 목록과 정확히 일치한다.
  2. Action Planner 가 만들 수 있는 (tool, action_type) 조합 모두가
     resolve_mcp_tool_name() 으로 해석 가능하다.
  3. 각 provider 에 대해 ActionExecutor → MCPGatewayConnector 로 라우팅 시
     fake protocol client 가 정확한 dotted MCP tool name 을 받는다.
  4. mcp_action_logs.response_payload 에 source/via_mcp/execution_mode/
     mcp_tool 이 provider 별로 손실 없이 저장된다.
"""
from __future__ import annotations

import asyncio

import pytest


EXPECTED_MCP_TOOLS = {
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


# (tool, action_type, expected_mcp_tool) — Action Planner 가 만들 수 있는 모든 조합.
PROVIDER_ROUTING_CASES: list[tuple[str, str, str]] = [
    ("slack", "send_slack_alert", "slack.send_slack_alert"),
    ("calendar", "schedule_callback", "calendar.schedule_callback"),
    ("gmail", "send_manager_email", "gmail.send_manager_email"),
    ("jira", "create_jira_issue", "jira.create_jira_issue"),
    ("jira", "create_voc_issue", "jira.create_jira_issue"),
    ("notion", "create_notion_call_record", "notion.create_notion_call_record"),
    ("notion", "create_notion_voc_record", "notion.create_notion_call_record"),
    ("sms", "send_voc_receipt_sms", "sms.send_voc_receipt_sms"),
    ("sms", "send_callback_sms", "sms.send_callback_sms"),
    ("sms", "send_reservation_confirmation", "sms.send_callback_sms"),
    ("company_db", "create_voc_issue", "company_db.create_voc_issue"),
    ("company_db", "add_priority_queue", "company_db.add_priority_queue"),
    ("company_db", "mark_faq_candidate", "company_db.mark_faq_candidate"),
    ("internal_dashboard", "add_priority_queue", "internal_dashboard.add_priority_queue"),
    ("internal_dashboard", "mark_faq_candidate", "internal_dashboard.mark_faq_candidate"),
]


# ── 1. MCP Server tool 목록 ↔ Gateway mapping 일치 ────────────────────────────


def test_gateway_mapping_values_match_mcp_server_tools():
    """gateway map 의 모든 value 가 MCP Server build_server() 에 등록된다.

    역도 동일 — server 에 등록된 12개 tool 이 정확히 mapping 의 value set 과
    같아야 한다 (orphan 방지).
    """
    from app.services.mcp.connectors.mcp_gateway_connector import _TOOL_NAME_MAP
    from app.services.mcp.server.main import build_server

    server = build_server()
    server_tool_names = {
        getattr(t, "name", "") for t in asyncio.run(server.list_tools()) or []
    }

    map_values = set(_TOOL_NAME_MAP.values())

    assert map_values == EXPECTED_MCP_TOOLS, (
        f"gateway mapping value set 이 기대 12개와 다르다: {map_values}"
    )
    assert server_tool_names == EXPECTED_MCP_TOOLS, (
        f"MCP Server 등록 tool 이 기대 12개와 다르다: {server_tool_names}"
    )
    # gateway 에 mapping 되어 있는데 server 에는 없는 orphan tool 은 transport 에서
    # unknown tool 오류로 떨어진다 — 사전에 차단.
    orphan = map_values - server_tool_names
    assert not orphan, f"gateway 가 가리키지만 MCP Server 에 없는 tool: {orphan}"


# ── 2. Action Planner 가 만들 수 있는 모든 (tool, action_type) 이 mapping 됨


@pytest.mark.parametrize("tool,action_type,expected", PROVIDER_ROUTING_CASES)
def test_resolve_mcp_tool_name_covers_all_actions(tool, action_type, expected):
    from app.services.mcp.connectors.mcp_gateway_connector import resolve_mcp_tool_name

    assert resolve_mcp_tool_name(tool, action_type) == expected


# ── 3. provider 별 ActionExecutor → Gateway → 정확한 MCP tool name 라우팅 ──


class _FakeProtocolClient:
    """list_tools/call_tool 을 흉내내는 fake — call_tool 의 name 을 기록한다."""

    def __init__(self, response: dict):
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
        # echo 구조 — 실제 server envelope 과 동일한 metadata 를 채워 돌려준다.
        return {
            **self.response,
            "mcp_tool": name,
        }


@pytest.mark.parametrize("tool,action_type,expected", PROVIDER_ROUTING_CASES)
def test_executor_routes_each_provider_to_correct_mcp_tool(
    monkeypatch, tool, action_type, expected,
):
    """ActionExecutor 가 (tool, action_type) → expected MCP tool name 으로
    정확히 라우팅하고, 결과의 result.mcp_tool 이 그대로 보존됨을 확인한다."""
    from app.agents.post_call.actions.executor import ActionExecutor
    from app.services.mcp.connectors.mcp_gateway_connector import MCPGatewayConnector

    fake_client = _FakeProtocolClient({
        "action_type": action_type,
        "tool": tool,
        "status": "success",
        "external_id": f"ext-{tool}-{action_type}",
        "result": {"echo_tool": tool, "echo_action": action_type},
        "error": None,
        "latency_ms": 1,
        "source": "mcp_server",
        "via_mcp": True,
        "execution_mode": "mcp",
    })
    gateway = MCPGatewayConnector(protocol_client=fake_client)
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: gateway,
    )

    async def _no_idem(*, call_id, action_type, tool):
        return None
    monkeypatch.setattr(
        "app.agents.post_call.actions.executor.find_successful_action", _no_idem,
    )

    executor = ActionExecutor()
    out = asyncio.run(executor.execute_actions(
        call_id=f"c-{tool}-{action_type}",
        tenant_id="ten-1",
        actions=[{"tool": tool, "action_type": action_type, "params": {"k": "v"}}],
    ))

    assert len(fake_client.calls) == 1
    sent_name, sent_payload = fake_client.calls[0]
    assert sent_name == expected
    assert sent_payload["tenant_id"] == "ten-1"
    assert sent_payload["call_id"] == f"c-{tool}-{action_type}"
    assert sent_payload["params"] == {"k": "v"}

    res = out[0]["result"]
    assert out[0]["status"] == "success"
    assert res["source"] == "mcp_server"
    assert res["via_mcp"] is True
    assert res["execution_mode"] == "mcp"
    assert res["mcp_tool"] == expected


# ── 4. mcp_action_logs response_payload 에 provider 별 metadata 가 보존 ──────


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,action_type,expected", PROVIDER_ROUTING_CASES)
async def test_response_payload_carries_provider_metadata(
    monkeypatch, tmp_path, tool, action_type, expected,
):
    """provider 별 executed_action[] 을 mcp_action_logs 에 저장 후 response_payload
    의 metadata 가 손실 없이 보존됨을 검증한다."""
    monkeypatch.setenv("MCP_ACTION_LOG_STORE", "file")
    monkeypatch.setenv(
        "MCP_ACTION_LOG_FILE",
        str(tmp_path / f"action_logs_{tool}_{action_type}.json"),
    )
    import app.repositories.mcp_action_log_repo as action_mod
    action_mod._reset(remove_file=True)

    call_id = f"call-{tool}-{action_type}"
    actions = [{
        "action_type": action_type,
        "tool": tool,
        "status": "success",
        "external_id": f"ext-{tool}",
        "error": None,
        "result": {
            "source": "mcp_server",
            "via_mcp": True,
            "execution_mode": "mcp",
            "mcp_tool": expected,
            "mcp_latency_ms": 7,
        },
        "params": {},
    }]
    await action_mod.save_action_logs(
        call_id=call_id,
        tenant_id="ten-1",
        executed_actions=actions,
    )

    logs = await action_mod.get_action_logs_by_call_id(call_id)
    assert len(logs) == 1
    payload = logs[0]["response_payload"]
    assert payload["source"] == "mcp_server"
    assert payload["via_mcp"] is True
    assert payload["execution_mode"] == "mcp"
    assert payload["mcp_tool"] == expected
    assert payload["mcp_latency_ms"] == 7

    action_mod._reset(remove_file=True)


# ── 5. Post-call action package 에서 direct handler/registry 자취가 없다 ────


def test_post_call_actions_package_has_no_direct_handlers():
    """app/agents/post_call/actions/ 디렉터리가 MCP-only 로 정리됐는지 확인한다.

    {provider}_action.py / registry.py 가 모두 삭제되어 있어야 한다.
    """
    from importlib import import_module

    forbidden = [
        "app.agents.post_call.actions.registry",
        "app.agents.post_call.actions.slack_action",
        "app.agents.post_call.actions.gmail_action",
        "app.agents.post_call.actions.jira_action",
        "app.agents.post_call.actions.calendar_action",
        "app.agents.post_call.actions.sms_action",
        "app.agents.post_call.actions.notion_action",
        "app.agents.post_call.actions.company_db_action",
    ]
    for mod in forbidden:
        with pytest.raises(ModuleNotFoundError):
            import_module(mod)
