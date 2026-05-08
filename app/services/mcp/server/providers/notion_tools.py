"""
MCP Server tool: notion.create_notion_call_record

Notion Internal Integration Token + Database ID 방식 사용.
tenant_integrations(provider=notion) metadata 또는 env 의 token/database_id
를 사용한다 (env fallback 은 MCP_ALLOW_ENV_FALLBACK=true 일 때만).
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import httpx

from app.services.mcp.server import result as R
from app.services.mcp.server.provider_lookup import allow_env_fallback
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOOL = "notion"
_ACTION = "create_notion_call_record"
_MCP_TOOL = f"{_TOOL}.{_ACTION}"

_PAGES_URL = "https://api.notion.com/v1/pages"
_NOTION_VERSION = "2022-06-28"


def _resolve_token_and_db(tenant_id: str, params: dict) -> tuple[str, str, str]:
    """(token, database_id, source) 를 반환. 없으면 빈 문자열."""
    from app.repositories.tenant_integration_repo import get_integration

    db_id = params.get("database_id") or ""
    token = ""
    source = "tenant_integration"

    if tenant_id:
        integration = get_integration(tenant_id, "notion")
        if integration is not None:
            meta = getattr(integration, "metadata", None) or {}
            db_id = db_id or meta.get("database_id") or ""
            # 평문 token 은 metadata 에 평문으로 저장하지 않는 게 원칙이므로
            # access_token_encrypted 우선 — 복호화 실패는 env fallback 으로 양보.
            try:
                from app.services.oauth.token_crypto import decrypt_token
                token = decrypt_token(integration.access_token_encrypted or "")
            except Exception:
                token = meta.get("integration_token") or ""

    if (not token or not db_id) and allow_env_fallback():
        token = token or os.getenv("NOTION_API_TOKEN", "")
        db_id = db_id or os.getenv("NOTION_DATABASE_ID", "")
        source = "env_fallback"

    return token, db_id, source


def _build_properties(action_type: str, params: dict, call_id: str) -> dict:
    call_id_val = params.get("call_id", call_id)
    tenant_id = params.get("tenant_id", "")
    emotion = params.get("customer_emotion") or params.get("emotion") or ""
    priority = params.get("priority") or ""
    resolution = params.get("resolution_status") or ""
    summary = (params.get("summary_short") or params.get("summary") or "")[:2000]
    voc_cat = params.get("primary_category") or params.get("voc_category") or ""
    action_req = bool(params.get("action_required", False))
    name = f"[{action_type.replace('_', '-')}] {call_id_val}"

    props: dict = {
        "Name": {"title": [{"text": {"content": name}}]},
        "Call ID": {"rich_text": [{"text": {"content": call_id_val}}]},
        "Tenant ID": {"rich_text": [{"text": {"content": tenant_id}}]},
        "Summary": {"rich_text": [{"text": {"content": summary}}]},
        "VOC Category": {"rich_text": [{"text": {"content": voc_cat}}]},
        "Action Required": {"checkbox": action_req},
        "Created At": {"date": {"start": datetime.utcnow().isoformat()}},
    }
    if emotion:
        props["Customer Emotion"] = {"select": {"name": emotion}}
    if priority:
        props["Priority"] = {"select": {"name": priority}}
    if resolution:
        props["Resolution Status"] = {"select": {"name": resolution}}
    return props


async def create_notion_call_record(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    started = time.perf_counter()

    token, db_id, _source = _resolve_token_and_db(tenant_id, params)
    if not token or not db_id:
        latency = int((time.perf_counter() - started) * 1000)
        return R.skipped(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            reason="notion_not_configured", latency_ms=latency,
        )

    payload = {
        "parent": {"database_id": db_id},
        "properties": _build_properties(_ACTION, params, call_id),
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _PAGES_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Notion-Version": _NOTION_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.error("notion_tools: 예외 call_id=%s err=%s", call_id, type(exc).__name__)
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"notion_exception:{type(exc).__name__}", latency_ms=latency,
        )

    latency = int((time.perf_counter() - started) * 1000)
    if resp.status_code not in (200, 201):
        preview = (getattr(resp, "text", "") or "")[:200]
        logger.error(
            "notion_tools: API 오류 call_id=%s status=%d preview=%s",
            call_id, resp.status_code, preview,
        )
        return R.failed(
            tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
            error=f"notion_api_error:{resp.status_code}", latency_ms=latency,
        )

    data = resp.json()
    page_id = data.get("id")
    return R.success(
        tool=_TOOL, action_type=_ACTION, mcp_tool=_MCP_TOOL,
        external_id=page_id,
        result={"page_id": page_id, "url": data.get("url")},
        latency_ms=latency,
    )


def register(mcp) -> None:
    @mcp.tool(
        name=_MCP_TOOL,
        description="Notion pages.create — Internal Integration token + database_id 사용",
    )
    async def notion_create_notion_call_record(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await create_notion_call_record(
            tenant_id=tenant_id,
            call_id=call_id,
            params=params or {},
        )
