"""
MCP Server tool: slack.send_slack_alert

tenant_integrations(provider=slack) 의 OAuth bot token 을 조회해
Slack chat.postMessage 를 호출한다. SlackConnector.execute() 는 호출하지
않는다 — token 추출/메시지 전송 helper 를 그대로 쓰지만, MCP mode 의
실행 진입점은 이 tool 이다.
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from app.services.mcp.server import result as R
from app.services.mcp.server.provider_lookup import (
    TenantOAuthLookupError,
    get_tenant_token,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


_TOOL = "slack"
_ACTION = "send_slack_alert"
_MCP_TOOL = f"{_TOOL}.{_ACTION}"


def _extract_bot_token(integration, decrypted: str) -> str:
    meta: dict = getattr(integration, "metadata", None) or {}
    v1_bot = (meta.get("bot") or {}).get("bot_access_token") or ""
    if v1_bot:
        return v1_bot
    meta_access = meta.get("access_token") or ""
    if isinstance(meta_access, str) and meta_access.startswith("xoxb-"):
        return meta_access
    return decrypted


def _resolve_channel(params: dict) -> str:
    return (
        params.get("channel")
        or params.get("channel_id")
        or (
            os.getenv("SLACK_CRITICAL_CHANNEL")
            if params.get("channel_type") == "critical"
            else None
        )
        or os.getenv("SLACK_ALERT_CHANNEL", "#alerts")
    )


def _resolve_text(params: dict) -> str:
    return (
        params.get("message")
        or params.get("text")
        or params.get("summary")
        or params.get("summary_short")
        or "Post-call alert"
    )


async def send_slack_alert(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        access_token, integration, _src = await get_tenant_token(
            tenant_id=tenant_id, provider="slack",
        )
    except TenantOAuthLookupError as exc:
        return R.skipped(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            reason=exc.reason,
            latency_ms=int((time.perf_counter() - started) * 1000),
        ) if exc.status == "skipped" else R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=exc.reason,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    bot_token = _extract_bot_token(integration, access_token)
    channel = _resolve_channel(params)
    text = _resolve_text(params)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                json={"channel": channel, "text": text},
                headers={"Authorization": f"Bearer {bot_token}"},
                timeout=15.0,
            )
    except Exception as exc:
        logger.error(
            "slack_tools: 예외 call_id=%s err=%s", call_id, type(exc).__name__,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"slack_exception:{type(exc).__name__}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    if resp.status_code != 200:
        logger.error(
            "slack_tools: HTTP 오류 call_id=%s status=%d", call_id, resp.status_code,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"slack_http_error:{resp.status_code}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    data = resp.json()
    if not data.get("ok"):
        err_code = data.get("error", "unknown_slack_error")
        logger.error(
            "slack_tools: API ok=false call_id=%s error=%s", call_id, err_code,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"slack_api_error:{err_code}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    ts = data.get("ts", "")
    ch = data.get("channel", channel)
    return R.success(
        tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
        external_id=f"{ch}:{ts}",
        result={"channel": ch, "ts": ts, "message": data.get("message", {})},
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def register(mcp) -> None:
    """FastMCP 인스턴스에 Slack tool 을 등록한다."""

    @mcp.tool(
        name=_MCP_TOOL,
        description="Slack chat.postMessage — tenant Slack OAuth bot token 사용",
    )
    async def slack_send_slack_alert(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await send_slack_alert(
            tenant_id=tenant_id,
            call_id=call_id,
            params=params or {},
        )
