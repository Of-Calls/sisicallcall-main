"""
MCP Server tool: company_db.* — 내부 tool, 외부 OAuth 불필요.

기존 CompanyDBConnector 가 mock 으로 처리하던 흐름을 동일하게 표준 MCP
result shape 으로 반환한다. 향후 실제 Company DB API 연동이 추가되면
이 모듈에서 분기하면 된다.
"""
from __future__ import annotations

import time
from typing import Any

from app.services.mcp.server import result as R
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOOL = "company_db"


def _ok(action_type: str, *, external_id: str, result: dict[str, Any], latency_ms: int) -> dict:
    return R.success(
        tool=_TOOL,
        action_type=action_type,
        mcp_tool=f"{_TOOL}.{action_type}",
        external_id=external_id,
        result=result,
        latency_ms=latency_ms,
    )


async def create_voc_issue(*, tenant_id: str, call_id: str, params: dict) -> dict[str, Any]:
    started = time.perf_counter()
    issue_id = params.get("issue_id") or f"VOC-MCP-{call_id}"
    latency = int((time.perf_counter() - started) * 1000)
    return _ok(
        "create_voc_issue",
        external_id=issue_id,
        result={
            "created": True,
            "issue_id": issue_id,
            "tenant_id": tenant_id,
            "tier": params.get("tier", "medium"),
            "priority": params.get("priority", "medium"),
            "primary_category": params.get("primary_category", ""),
            "summary": params.get("summary") or params.get("summary_short", ""),
        },
        latency_ms=latency,
    )


async def add_priority_queue(*, tenant_id: str, call_id: str, params: dict) -> dict[str, Any]:
    started = time.perf_counter()
    queue_entry_id = f"PQ-{call_id}"
    latency = int((time.perf_counter() - started) * 1000)
    return _ok(
        "add_priority_queue",
        external_id=queue_entry_id,
        result={
            "queued": True,
            "queue_entry_id": queue_entry_id,
            "tenant_id": tenant_id,
            "priority": params.get("priority", "high"),
            "reason": params.get("reason", ""),
        },
        latency_ms=latency,
    )


async def mark_faq_candidate(*, tenant_id: str, call_id: str, params: dict) -> dict[str, Any]:
    started = time.perf_counter()
    faq_candidate_id = f"FAQ-{call_id}"
    latency = int((time.perf_counter() - started) * 1000)
    return _ok(
        "mark_faq_candidate",
        external_id=faq_candidate_id,
        result={
            "faq_candidate_id": faq_candidate_id,
            "tenant_id": tenant_id,
            "category": params.get("primary_category", ""),
            "summary": params.get("summary", ""),
        },
        latency_ms=latency,
    )


def register(mcp) -> None:
    @mcp.tool(
        name="company_db.create_voc_issue",
        description="Company DB VOC issue 등록 (내부 tool)",
    )
    async def company_db_create_voc_issue(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await create_voc_issue(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )

    @mcp.tool(
        name="company_db.add_priority_queue",
        description="Company DB priority queue 등록 (내부 tool)",
    )
    async def company_db_add_priority_queue(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await add_priority_queue(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )

    @mcp.tool(
        name="company_db.mark_faq_candidate",
        description="Company DB FAQ 후보 등록 (내부 tool)",
    )
    async def company_db_mark_faq_candidate(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await mark_faq_candidate(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )
