"""
MCP Server tool: gmail.send_manager_email

tenant_integrations(provider=google_gmail / gmail) OAuth access token 으로
Gmail users.messages.send 호출.
"""
from __future__ import annotations

import base64
import email.message
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

_TOOL = "gmail"
_ACTION = "send_manager_email"
_MCP_TOOL = f"{_TOOL}.{_ACTION}"

_API = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"


def _build_raw(to: str, subject: str, body: str) -> str:
    msg = email.message.EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body or "")
    return base64.urlsafe_b64encode(msg.as_bytes()).decode().rstrip("=")


async def send_manager_email(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        access_token, _integration, _src = await get_tenant_token(
            tenant_id=tenant_id, provider="google_gmail",
        )
    except TenantOAuthLookupError as exc:
        latency = int((time.perf_counter() - started) * 1000)
        if exc.status == "skipped":
            return R.skipped(
                tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
                reason=exc.reason, latency_ms=latency,
            )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=exc.reason, latency_ms=latency,
        )

    to = params.get("to") or os.getenv("GMAIL_MANAGER_TO", "")
    if not to:
        latency = int((time.perf_counter() - started) * 1000)
        return R.skipped(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            reason="gmail_recipient_not_configured", latency_ms=latency,
        )

    subject = params.get("subject") or "[시시콜콜] 상담 후속 조치 알림"
    body = (
        params.get("body")
        or params.get("summary")
        or params.get("summary_short")
        or params.get("reason")
        or ""
    )
    raw = _build_raw(to, subject, body)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _API,
                json={"raw": raw},
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.error("gmail_tools: 예외 call_id=%s err=%s", call_id, type(exc).__name__)
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"gmail_exception:{type(exc).__name__}", latency_ms=latency,
        )

    latency = int((time.perf_counter() - started) * 1000)
    if resp.status_code not in (200, 201):
        logger.error(
            "gmail_tools: HTTP 오류 call_id=%s status=%d", call_id, resp.status_code,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"gmail_http_error:{resp.status_code}", latency_ms=latency,
        )

    data = resp.json()
    message_id = data.get("id", "")
    return R.success(
        tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
        external_id=message_id,
        result={"message_id": message_id, "to": to, "subject": subject},
        latency_ms=latency,
    )


def register(mcp) -> None:
    @mcp.tool(
        name=_MCP_TOOL,
        description="Gmail users.messages.send — tenant google_gmail OAuth token 사용",
    )
    async def gmail_send_manager_email(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await send_manager_email(
            tenant_id=tenant_id,
            call_id=call_id,
            params=params or {},
        )
