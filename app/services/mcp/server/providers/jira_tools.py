"""
MCP Server tool: jira.create_jira_issue

tenant_integrations(provider=jira) OAuth access token + cloud_id 로
Jira Cloud REST v3 issue create 호출.
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

_TOOL = "jira"
_ACTION = "create_jira_issue"
_MCP_TOOL = f"{_TOOL}.{_ACTION}"


def _resolve_cloud_id(integration) -> str | None:
    meta: dict = getattr(integration, "metadata", None) or {}
    if meta.get("workspace_selection_required") is True:
        return None
    return (
        meta.get("cloud_id")
        or meta.get("cloudId")
        or getattr(integration, "external_workspace_id", None)
        or None
    )


async def create_jira_issue(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        access_token, integration, _src = await get_tenant_token(
            tenant_id=tenant_id, provider="jira",
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

    cloud_id = _resolve_cloud_id(integration)
    if not cloud_id:
        latency = int((time.perf_counter() - started) * 1000)
        return R.skipped(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            reason="jira_workspace_not_selected", latency_ms=latency,
        )

    api_url = f"https://api.atlassian.com/ex/jira/{cloud_id}/rest/api/3/issue"
    project_key = (
        params.get("project_key")
        or os.getenv("JIRA_PROJECT_KEY", "VOC")
    )
    issue_type = params.get("issue_type") or os.getenv("JIRA_ISSUE_TYPE", "Task")
    issue_type_id = (params.get("issue_type_id") or os.getenv("JIRA_ISSUE_TYPE_ID", "")).strip()
    issuetype_field: dict[str, Any] = (
        {"id": issue_type_id} if issue_type_id else {"name": issue_type}
    )

    summary = (
        params.get("summary")
        or params.get("title")
        or params.get("summary_short")
        or "[시시콜콜] VOC 후속 이슈"
    )
    description_text = (
        params.get("description")
        or params.get("reason")
        or params.get("summary_short")
        or ""
    )
    labels = params.get("labels") or ["sisicallcall", "post-call"]

    body = {
        "fields": {
            "project": {"key": project_key},
            "issuetype": issuetype_field,
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description_text}],
                    }
                ],
            },
            "labels": labels,
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                api_url,
                json=body,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.error("jira_tools: 예외 call_id=%s err=%s", call_id, type(exc).__name__)
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"jira_exception:{type(exc).__name__}", latency_ms=latency,
        )

    latency = int((time.perf_counter() - started) * 1000)
    if resp.status_code not in (200, 201):
        body_preview = (getattr(resp, "text", "") or "")[:500]
        logger.error(
            "jira_tools: HTTP 오류 call_id=%s status=%d body=%s",
            call_id, resp.status_code, body_preview,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"jira_http_error:{resp.status_code}", latency_ms=latency,
        )

    data = resp.json()
    issue_key = data.get("key", "")
    return R.success(
        tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
        external_id=issue_key,
        result={
            "issue_key": issue_key,
            "issue_id": data.get("id", ""),
            "self": data.get("self", ""),
        },
        latency_ms=latency,
    )


def register(mcp) -> None:
    @mcp.tool(
        name=_MCP_TOOL,
        description="Jira Cloud REST v3 issue create — tenant jira OAuth token 사용",
    )
    async def jira_create_jira_issue(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await create_jira_issue(
            tenant_id=tenant_id,
            call_id=call_id,
            params=params or {},
        )
