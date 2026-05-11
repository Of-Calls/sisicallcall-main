"""
MCP Server tools: notion.create_notion_call_record / notion.create_notion_voc_record

Notion Internal Integration Token + Database ID 방식 사용.
tenant_integrations(provider=notion) metadata 또는 env 의 token/database_id
를 사용한다 (env fallback 은 MCP_ALLOW_ENV_FALLBACK=true 일 때만).

call_record / voc_record 는 같은 Notion DB 를 공유하며 'Record Type' (select)
컬럼으로 구분된다. 운영자가 Notion DB 의 view filter 로 분류한다.

  CALL row:
    properties: Record Type=call / Call ID / Caller / Started At / Duration / Branch Stats
    page body : heading_1 "통화 내역" + 각 turn paragraph block ([speaker] text)
  VOC row:
    properties: Record Type=voc / Call ID / Summary / Customer Emotion / Priority
                / VOC Category / Action Required / Suggested Action
    page body : 없음 (분석 인사이트는 properties 에 모두 들어감)
"""
from __future__ import annotations

import json
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
_ACTION_CALL = "create_notion_call_record"
_ACTION_VOC = "create_notion_voc_record"
_MCP_TOOL_CALL = f"{_TOOL}.{_ACTION_CALL}"
_MCP_TOOL_VOC = f"{_TOOL}.{_ACTION_VOC}"

_PAGES_URL = "https://api.notion.com/v1/pages"
_NOTION_VERSION = "2022-06-28"

# Notion paragraph rich_text content max length is 2000 chars per text run.
_NOTION_RICH_TEXT_MAX = 2000

# notify_admin_review_failed_node 가 분석 부적합 통화에 prefix 로 주입하는 marker.
# call_record / voc_record 의 Name 에도 동일 marker 를 prefix 해서 운영자가
# Notion 에서 한눈에 식별 가능하게 한다.
_REVIEW_FAILED_MARKER = "[REVIEW_FAILED] "


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
            try:
                from app.services.oauth.token_crypto import decrypt_token
                token = decrypt_token(integration.access_token_encrypted or "")
            except Exception:
                token = meta.get("integration_token") or ""

    if not token:
        token = os.getenv("NOTION_API_TOKEN", "") or token
    if not db_id:
        db_id = os.getenv("NOTION_DATABASE_ID", "") or db_id
    if token and not source:
        source = "env"

    return token, db_id, source


def _truncate(s: str, n: int = _NOTION_RICH_TEXT_MAX) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _build_call_properties(params: dict, call_id: str) -> dict:
    """call_record 전용 properties — LLM 가공 필드 없이 원본 메타만."""
    call_id_val = params.get("call_id", call_id)
    tenant_id = params.get("tenant_id", "")
    caller = str(params.get("caller_number") or "")
    started_at = str(params.get("started_at") or "")
    ended_at = str(params.get("ended_at") or "")
    duration = params.get("duration_sec")
    branch_stats = params.get("branch_stats") or {}
    branch_stats_json = _truncate(
        json.dumps(branch_stats, ensure_ascii=False, sort_keys=True)
    )
    name = f"통화 기록 {str(call_id_val)[:8]} ({caller or 'no-phone'})"
    # notify_admin_review_failed_node 가 분석 부적합 마킹 → Name 에도 표시
    if str(params.get("title") or "").startswith(_REVIEW_FAILED_MARKER):
        name = f"{_REVIEW_FAILED_MARKER}{name}"

    props: dict = {
        "Name": {"title": [{"text": {"content": name}}]},
        "Record Type": {"select": {"name": "call"}},
        "Call ID": {"rich_text": [{"text": {"content": str(call_id_val)}}]},
        "Tenant ID": {"rich_text": [{"text": {"content": str(tenant_id)}}]},
        "Caller Number": {"rich_text": [{"text": {"content": caller}}]},
        "Branch Stats": {"rich_text": [{"text": {"content": branch_stats_json}}]},
        "Created At": {"date": {"start": datetime.utcnow().isoformat()}},
    }
    if started_at:
        try:
            props["Started At"] = {"date": {"start": started_at}}
        except Exception:
            pass
    if isinstance(duration, (int, float)):
        props["Duration Sec"] = {"number": int(duration)}
    return props


def _build_voc_properties(params: dict, call_id: str) -> dict:
    """voc_record 전용 properties — LLM 분석 인사이트."""
    call_id_val = params.get("call_id", call_id)
    tenant_id = params.get("tenant_id", "")
    emotion = params.get("customer_emotion") or params.get("emotion") or ""
    priority = params.get("priority") or ""
    resolution = params.get("resolution_status") or ""
    summary = _truncate(
        str(params.get("summary_short") or params.get("summary") or ""),
        _NOTION_RICH_TEXT_MAX,
    )
    voc_cat = str(params.get("primary_category") or params.get("voc_category") or "")
    suggested_action = _truncate(str(params.get("suggested_action") or ""))
    action_req = bool(params.get("action_required", False))
    name = f"[VOC] {summary[:60] or str(call_id_val)[:8]}"
    if str(params.get("title") or "").startswith(_REVIEW_FAILED_MARKER):
        name = f"{_REVIEW_FAILED_MARKER}{name}"

    props: dict = {
        "Name": {"title": [{"text": {"content": name}}]},
        "Record Type": {"select": {"name": "voc"}},
        "Call ID": {"rich_text": [{"text": {"content": str(call_id_val)}}]},
        "Tenant ID": {"rich_text": [{"text": {"content": str(tenant_id)}}]},
        "Summary": {"rich_text": [{"text": {"content": summary}}]},
        "VOC Category": {"rich_text": [{"text": {"content": voc_cat}}]},
        "Suggested Action": {"rich_text": [{"text": {"content": suggested_action}}]},
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


def _build_call_children(params: dict) -> list[dict]:
    """call_record page body — heading + 각 turn paragraph."""
    transcripts = params.get("transcript_full") or []
    if not isinstance(transcripts, list):
        return []
    children: list[dict] = [
        {
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"type": "text", "text": {"content": "통화 내역"}}],
            },
        }
    ]
    for turn in transcripts:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker") or "?")
        prefix = "[고객] " if speaker == "customer" else (
            "[상담원] " if speaker == "agent" else f"[{speaker}] "
        )
        text = _truncate(str(turn.get("text") or ""))
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": prefix},
                        "annotations": {"bold": True},
                    },
                    {"type": "text", "text": {"content": text}},
                ],
            },
        })
    return children


async def _create_notion_page(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
    action: str,
    mcp_tool: str,
    properties: dict,
    children: list[dict] | None,
) -> dict[str, Any]:
    started = time.perf_counter()

    token, db_id, _source = _resolve_token_and_db(tenant_id, params)
    if not token or not db_id:
        latency = int((time.perf_counter() - started) * 1000)
        return R.skipped(
            tool=_TOOL, action_type=action, mcp_tool=mcp_tool,
            reason="notion_not_configured", latency_ms=latency,
        )

    payload: dict[str, Any] = {
        "parent": {"database_id": db_id},
        "properties": properties,
    }
    if children:
        payload["children"] = children

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
        logger.error(
            "notion_tools: 예외 call_id=%s action=%s err=%s",
            call_id, action, type(exc).__name__,
        )
        return R.failed(
            tool=_TOOL, action_type=action, mcp_tool=mcp_tool,
            error=f"notion_exception:{type(exc).__name__}", latency_ms=latency,
        )

    latency = int((time.perf_counter() - started) * 1000)
    if resp.status_code not in (200, 201):
        preview = (getattr(resp, "text", "") or "")[:200]
        logger.error(
            "notion_tools: API 오류 call_id=%s action=%s status=%d preview=%s",
            call_id, action, resp.status_code, preview,
        )
        return R.failed(
            tool=_TOOL, action_type=action, mcp_tool=mcp_tool,
            error=f"notion_api_error:{resp.status_code}", latency_ms=latency,
        )

    data = resp.json()
    page_id = data.get("id")
    return R.success(
        tool=_TOOL, action_type=action, mcp_tool=mcp_tool,
        external_id=page_id,
        result={"page_id": page_id, "url": data.get("url")},
        latency_ms=latency,
    )


async def create_notion_call_record(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    return await _create_notion_page(
        tenant_id=tenant_id,
        call_id=call_id,
        params=params,
        action=_ACTION_CALL,
        mcp_tool=_MCP_TOOL_CALL,
        properties=_build_call_properties(params, call_id),
        children=_build_call_children(params),
    )


async def create_notion_voc_record(
    *,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    return await _create_notion_page(
        tenant_id=tenant_id,
        call_id=call_id,
        params=params,
        action=_ACTION_VOC,
        mcp_tool=_MCP_TOOL_VOC,
        properties=_build_voc_properties(params, call_id),
        children=None,
    )


def register(mcp) -> None:
    @mcp.tool(
        name=_MCP_TOOL_CALL,
        description="Notion pages.create — 통화 보관소 (call_record): 원본 transcript + 메타",
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

    @mcp.tool(
        name=_MCP_TOOL_VOC,
        description="Notion pages.create — VOC 분석 (voc_record): LLM 요약/감정/우선순위",
    )
    async def notion_create_notion_voc_record(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await create_notion_voc_record(
            tenant_id=tenant_id,
            call_id=call_id,
            params=params or {},
        )
