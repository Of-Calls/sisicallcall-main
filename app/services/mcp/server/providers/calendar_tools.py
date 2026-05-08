"""
MCP Server tool: calendar.schedule_callback

tenant_integrations(provider=google_calendar / calendar) OAuth access token 으로
Google Calendar events.insert 호출.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any

import httpx

from app.services.mcp.server import result as R
from app.services.mcp.server.provider_lookup import (
    TenantOAuthLookupError,
    get_tenant_token,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOOL = "calendar"
_ACTION = "schedule_callback"
_MCP_TOOL = f"{_TOOL}.{_ACTION}"

_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3/calendars"
_DEFAULT_TZ = "Asia/Seoul"
_DEFAULT_DURATION_MIN = 30


def _build_event_body(params: dict) -> dict:
    title = params.get("title") or "콜백 예약"
    description = ""
    for key in ("description", "reason", "callback_reason", "summary", "summary_short"):
        val = params.get(key)
        if val:
            description = str(val)
            break

    tz = params.get("timezone", _DEFAULT_TZ)
    duration = int(os.getenv("CALENDAR_DEFAULT_DURATION_MIN", str(_DEFAULT_DURATION_MIN)))

    start_str = params.get("start_time") or params.get("preferred_time")
    end_str = params.get("end_time")

    if start_str:
        try:
            start_dt = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            start_dt = datetime.utcnow() + timedelta(hours=1)
    else:
        start_dt = datetime.utcnow() + timedelta(hours=1)

    if end_str:
        try:
            end_dt = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            end_dt = start_dt + timedelta(minutes=duration)
    else:
        end_dt = start_dt + timedelta(minutes=duration)

    return {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": tz},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": tz},
    }


async def schedule_callback(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        access_token, _integration, _src = await get_tenant_token(
            tenant_id=tenant_id, provider="google_calendar",
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

    calendar_id = params.get("calendar_id") or os.getenv("GOOGLE_CALENDAR_ID", "primary")
    url = f"{_CALENDAR_API_BASE}/{calendar_id}/events"
    body = _build_event_body(params)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=15.0,
            )
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.error("calendar_tools: 예외 call_id=%s err=%s", call_id, type(exc).__name__)
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"calendar_insert_exception:{type(exc).__name__}",
            latency_ms=latency,
        )

    latency = int((time.perf_counter() - started) * 1000)
    if resp.status_code not in (200, 201):
        preview = (getattr(resp, "text", "") or "")[:200]
        logger.error(
            "calendar_tools: API 오류 call_id=%s status=%d preview=%s",
            call_id, resp.status_code, preview,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"google_calendar_api_error:{resp.status_code}",
            latency_ms=latency,
        )

    data = resp.json()
    return R.success(
        tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
        external_id=data.get("id"),
        result={
            "event_id": data.get("id"),
            "html_link": data.get("htmlLink"),
            "start": (data.get("start") or {}).get("dateTime"),
            "end": (data.get("end") or {}).get("dateTime"),
        },
        latency_ms=latency,
    )


def register(mcp) -> None:
    @mcp.tool(
        name=_MCP_TOOL,
        description="Google Calendar events.insert — tenant google_calendar OAuth token 사용",
    )
    async def calendar_schedule_callback(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await schedule_callback(
            tenant_id=tenant_id,
            call_id=call_id,
            params=params or {},
        )
