"""
KDT-101: Post-call ActionExecutor MCP-only 라우팅 테스트.

ActionExecutor 는 항상 MCPGatewayConnector 만 호출하고, registry direct
handler 는 절대 호출하지 않는다. transport 오류라도 direct fallback 하지
않으며, status=failed 로 기록된다.

외부 API 는 호출하지 않는다 — gateway 는 fake 로 주입한다.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────


def _action(tool="slack", action_type="send_slack_alert", **extra) -> dict:
    base = {"tool": tool, "action_type": action_type, "params": {}}
    base.update(extra)
    return base


class FakeGateway:
    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        raise_transport_err: bool = False,
        raise_other: bool = False,
    ):
        self.result = result or {
            "status": "success",
            "external_id": "ext-mcp",
            "result": {
                "source": "mcp_server",
                "via_mcp": True,
                "execution_mode": "mcp",
                "mcp_tool": "slack.send_slack_alert",
            },
        }
        self.raise_transport_err = raise_transport_err
        self.raise_other = raise_other
        self.calls: list[dict] = []

    async def execute(self, action, *, call_id, tenant_id):
        self.calls.append({"action": action, "call_id": call_id, "tenant_id": tenant_id})
        if self.raise_transport_err:
            from app.services.mcp.protocol_client import MCPClientTransportError
            raise MCPClientTransportError("simulated_transport_failure")
        if self.raise_other:
            raise RuntimeError("non_transport")
        return self.result


@pytest.fixture(autouse=True)
def _no_idempotency(monkeypatch):
    """idempotency lookup 이 항상 None 을 반환하도록 패치 — 다른 테스트
    상태가 누설되지 않게 격리."""
    async def _none(call_id, action_type, tool):
        return None
    monkeypatch.setattr(
        "app.agents.post_call.actions.executor.find_successful_action", _none,
    )


# ── 1. ActionExecutor 는 MCPGatewayConnector 만 호출한다 ─────────────────────


def test_executor_always_routes_through_gateway(monkeypatch):
    """ActionExecutor 가 direct registry / direct connector 를 손대지 않고
    MCPGatewayConnector 만 호출함을 확인한다. direct registry 자체가 삭제된
    상태이므로 import 가능성도 함께 검증한다."""
    from app.agents.post_call.actions.executor import ActionExecutor

    fake_gateway = FakeGateway(
        result={
            "status": "success",
            "external_id": "C12345:1700000000.000100",
            "result": {
                "channel": "C12345",
                "ts": "1700000000.000100",
                "source": "mcp_server",
                "via_mcp": True,
                "execution_mode": "mcp",
                "mcp_tool": "slack.send_slack_alert",
            },
        }
    )
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: fake_gateway,
    )

    executor = ActionExecutor()
    result = asyncio.run(executor.execute_actions(
        call_id="c-mcp",
        tenant_id="ten-1",
        actions=[_action()],
    ))

    assert len(fake_gateway.calls) == 1
    out = result[0]
    assert out["status"] == "success"
    assert out["external_id"] == "C12345:1700000000.000100"
    assert out["result"]["source"] == "mcp_server"
    assert out["result"]["via_mcp"] is True
    assert out["result"]["execution_mode"] == "mcp"
    assert out["result"]["mcp_tool"] == "slack.send_slack_alert"


def test_executor_does_not_import_registry_get_handler():
    """executor 모듈은 더 이상 registry.get_handler 를 import 하지 않는다.
    registry 모듈 자체가 MCP-only 전환과 함께 삭제되어 import 가 실패해야 한다.
    """
    import app.agents.post_call.actions.executor as ex_mod

    assert not hasattr(ex_mod, "get_handler"), (
        "executor 가 get_handler 를 import 하면 direct mode 잔재 — MCP-only 위반"
    )

    with pytest.raises(ModuleNotFoundError):
        import app.agents.post_call.actions.registry  # noqa: F401


# ── 2. MCP transport error 는 direct fallback 하지 않는다 ────────────────────


def test_transport_error_returns_failed_without_fallback(monkeypatch):
    from app.agents.post_call.actions.executor import ActionExecutor

    fake_gateway = FakeGateway(raise_transport_err=True)
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: fake_gateway,
    )

    executor = ActionExecutor()
    result = asyncio.run(executor.execute_actions(
        call_id="c-mcp-tx",
        tenant_id="ten-1",
        actions=[_action()],
    ))

    assert len(fake_gateway.calls) == 1
    assert result[0]["status"] == "failed"
    assert result[0]["error"].startswith("mcp_transport_failed")
    res = result[0]["result"]
    assert res["source"] == "mcp_server"
    assert res["via_mcp"] is True
    assert res["execution_mode"] == "mcp"
    assert "transport_error" in res


# ── 3. unknown MCP tool 은 즉시 failed ───────────────────────────────────────


def test_unknown_tool_returns_failed_without_calling_gateway(monkeypatch):
    from app.agents.post_call.actions.executor import ActionExecutor

    fake_gateway = FakeGateway()
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: fake_gateway,
    )

    executor = ActionExecutor()
    result = asyncio.run(executor.execute_actions(
        call_id="c-mcp-unknown",
        tenant_id="ten-1",
        actions=[_action(tool="weather", action_type="forecast")],
    ))

    assert result[0]["status"] == "failed"
    assert result[0]["error"].startswith("unknown_mcp_tool")
    assert len(fake_gateway.calls) == 0
    res = result[0]["result"]
    assert res["source"] == "mcp_server"
    assert res["via_mcp"] is True
    assert res["execution_mode"] == "mcp"


# ── 4. MCP tool-level failure 는 그대로 failed 전달 ──────────────────────────


def test_tool_level_failure_propagates_without_fallback(monkeypatch):
    """gateway 가 status=failed envelope 을 돌려주면 executor 도 그대로 failed —
    direct fallback 같은 대체 경로 없음."""
    from app.agents.post_call.actions.executor import ActionExecutor

    fake_gateway = FakeGateway(
        result={
            "status": "failed",
            "external_id": None,
            "error": "slack_http_error:500",
            "result": {
                "source": "mcp_server",
                "via_mcp": True,
                "execution_mode": "mcp",
                "mcp_tool": "slack.send_slack_alert",
            },
        }
    )
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: fake_gateway,
    )

    executor = ActionExecutor()
    result = asyncio.run(executor.execute_actions(
        call_id="c-tool-fail",
        tenant_id="ten-1",
        actions=[_action()],
    ))

    assert len(fake_gateway.calls) == 1
    assert result[0]["status"] == "failed"
    assert result[0]["error"] == "slack_http_error:500"
    assert result[0]["result"]["source"] == "mcp_server"
    assert result[0]["result"]["mcp_tool"] == "slack.send_slack_alert"


# ── 5. idempotency skip 은 유지 + MCP metadata 태깅 ─────────────────────────


def test_idempotency_skip_short_circuits_gateway(monkeypatch):
    """이미 성공한 action 은 gateway 호출 없이 skipped already_succeeded."""
    from app.agents.post_call.actions.executor import ActionExecutor

    async def already_done(call_id, action_type, tool):
        return {"external_id": "prev-id", "status": "success"}
    monkeypatch.setattr(
        "app.agents.post_call.actions.executor.find_successful_action",
        already_done,
    )

    fake_gateway = FakeGateway()
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: fake_gateway,
    )

    executor = ActionExecutor()
    result = asyncio.run(executor.execute_actions(
        call_id="c-idem",
        tenant_id="ten-1",
        actions=[_action()],
    ))

    assert result[0]["status"] == "skipped"
    assert result[0]["error"] == "already_succeeded"
    assert len(fake_gateway.calls) == 0
    # idempotency-skip row 도 mcp_action_logs.response_payload 에 MCP metadata
    # 가 남아 발표용 SQL (source=mcp_server) 에서 일관되게 잡힌다.
    res = result[0]["result"]
    assert res["idempotency"] == "already_succeeded"
    assert res["previous_external_id"] == "prev-id"
    assert res["source"] == "mcp_server"
    assert res["via_mcp"] is True
    assert res["execution_mode"] == "mcp"
    assert res["mcp_tool"] == "slack.send_slack_alert"


# ── 6. MCP_EXECUTION_MODE 환경변수는 ActionExecutor 동작에 영향이 없다 ─────


def test_mcp_execution_mode_env_is_ignored(monkeypatch):
    """과거 direct/mcp_with_fallback 분기를 위해 쓰던 env 가 set 돼 있어도
    ActionExecutor 는 항상 MCPGatewayConnector 로만 라우팅한다."""
    monkeypatch.setenv("MCP_EXECUTION_MODE", "direct")

    from app.agents.post_call.actions.executor import ActionExecutor

    fake_gateway = FakeGateway()
    monkeypatch.setattr(
        "app.services.mcp.connectors.mcp_gateway_connector.get_default_gateway",
        lambda: fake_gateway,
    )

    executor = ActionExecutor()
    result = asyncio.run(executor.execute_actions(
        call_id="c-env-direct",
        tenant_id="ten-1",
        actions=[_action()],
    ))

    assert len(fake_gateway.calls) == 1, "MCP-only: env 와 무관하게 gateway 만 호출"
    assert result[0]["status"] == "success"
