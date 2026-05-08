"""
MCP Server tool: internal_dashboard.* — 내부 tool.

대시보드 우선순위 큐 / FAQ 후보 등록을 표준 MCP result shape 으로 반환한다.
외부 API 호출이 없는 in-process 동작이지만, MCP mode 에서는 반드시 이
tool 을 거쳐야 표준 source=mcp_server / via_mcp=true / execution_mode=mcp
가 result 에 남는다.
"""
from __future__ import annotations

import time
from typing import Any

from app.services.mcp.server import result as R

_TOOL = "internal_dashboard"


def _ok(action_type: str, *, external_id: str, result: dict[str, Any], latency_ms: int) -> dict:
    return R.success(
        tool=_TOOL,
        action_type=action_type,
        mcp_tool=f"{_TOOL}.{action_type}",
        external_id=external_id,
        result=result,
        latency_ms=latency_ms,
    )


async def add_priority_queue(*, tenant_id: str, call_id: str, params: dict) -> dict[str, Any]:
    started = time.perf_counter()
    queue_entry_id = f"DASH-PQ-{call_id}"
    latency = int((time.perf_counter() - started) * 1000)
    return _ok(
        "add_priority_queue",
        external_id=queue_entry_id,
        result={
            "queued": True,
            "tenant_id": tenant_id,
            "priority": params.get("priority", "high"),
            "reason": params.get("reason", ""),
        },
        latency_ms=latency,
    )


async def mark_faq_candidate(*, tenant_id: str, call_id: str, params: dict) -> dict[str, Any]:
    started = time.perf_counter()
    faq_candidate_id = f"DASH-FAQ-{call_id}"
    latency = int((time.perf_counter() - started) * 1000)
    return _ok(
        "mark_faq_candidate",
        external_id=faq_candidate_id,
        result={
            "tenant_id": tenant_id,
            "category": params.get("primary_category", ""),
            "summary": params.get("summary", ""),
        },
        latency_ms=latency,
    )


def register(mcp) -> None:
    @mcp.tool(
        name="internal_dashboard.add_priority_queue",
        description="내부 대시보드 priority queue 등록 (내부 tool)",
    )
    async def internal_dashboard_add_priority_queue(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await add_priority_queue(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )

    @mcp.tool(
        name="internal_dashboard.mark_faq_candidate",
        description="내부 대시보드 FAQ 후보 등록 (내부 tool)",
    )
    async def internal_dashboard_mark_faq_candidate(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await mark_faq_candidate(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )
