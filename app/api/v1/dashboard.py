from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.v1.admin_auth import get_current_admin_user
from app.repositories.dashboard_repo import (
    get_dashboard_overview as _legacy_get_dashboard_overview,
    get_emotion_distribution as _legacy_get_emotion_distribution,
    get_priority_queue,
)
from app.repositories.mcp_action_log_repo import get_action_logs
from app.repositories.post_call_dashboard_repo import (
    fetch_action_status_distribution,
    fetch_dashboard_emotion_distribution_counts,
    fetch_dashboard_intent_distribution as _repo_fetch_dashboard_intent_distribution,
    fetch_dashboard_keyword_stats as _repo_fetch_dashboard_keyword_stats,
    fetch_dashboard_priority_distribution as _repo_fetch_dashboard_priority_distribution,
    fetch_dashboard_recent_calls as _repo_fetch_dashboard_recent_calls,
    fetch_dashboard_stats as _repo_fetch_dashboard_stats,
)
from app.utils.config import settings

router = APIRouter()


def _request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _connect():
    return await asyncpg.connect(_database_url())


def _is_uuid(value: str | None) -> bool:
    try:
        UUID(str(value))
        return True
    except Exception:
        return False


def _resolve_dashboard_tenant_id(
    query_tenant_id: Optional[str],
    current_admin: dict[str, Any],
) -> str:
    user = current_admin.get("user") or {}
    jwt_tenant_id = str(user.get("tenant_id") or "").strip()
    if not jwt_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin tenant",
        )

    query_tenant = str(query_tenant_id).strip() if query_tenant_id else None
    if query_tenant and query_tenant.lower() != jwt_tenant_id.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="tenant 정보가 일치하지 않습니다.",
        )

    return jwt_tenant_id


async def get_dashboard_overview(
    tenant_id: str,
    *,
    started_from: Optional[datetime] = None,
    started_to: Optional[datetime] = None,
) -> dict:
    if not _is_uuid(tenant_id):
        legacy = await _legacy_get_dashboard_overview(
            tenant_id=tenant_id,
            started_from=started_from.isoformat() if started_from else None,
            started_to=started_to.isoformat() if started_to else None,
        )
        return legacy

    try:
        conn = await _connect()
    except Exception:
        legacy = await _legacy_get_dashboard_overview(
            tenant_id=tenant_id,
            started_from=started_from.isoformat() if started_from else None,
            started_to=started_to.isoformat() if started_to else None,
        )
        return {"tenant_id": tenant_id, **legacy}

    try:
        stats = await _repo_fetch_dashboard_stats(
            conn,
            tenant_id,
            date_from=started_from.isoformat() if started_from else None,
            date_to=started_to.isoformat() if started_to else None,
        )
        if stats is not None:
            return stats

        legacy = await _legacy_get_dashboard_overview(
            tenant_id=tenant_id,
            started_from=started_from.isoformat() if started_from else None,
            started_to=started_to.isoformat() if started_to else None,
        )
        return legacy
    finally:
        await conn.close()


async def fetch_dashboard_recent_calls(
    *,
    tenant_id: str,
    limit: int,
    offset: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> dict | None:
    if not _is_uuid(tenant_id):
        return None

    try:
        conn = await _connect()
    except Exception:
        return None

    try:
        return await _repo_fetch_dashboard_recent_calls(
            tenant_id,
            limit=limit,
            offset=offset,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            conn=conn,
        )
    finally:
        await conn.close()


async def fetch_dashboard_intent_distribution(
    *,
    tenant_id: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict] | None:
    if not _is_uuid(tenant_id):
        return None

    try:
        conn = await _connect()
    except Exception:
        return None

    try:
        return await _repo_fetch_dashboard_intent_distribution(
            tenant_id,
            limit=limit,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            conn=conn,
        )
    finally:
        await conn.close()


async def fetch_dashboard_keyword_stats(
    *,
    tenant_id: str,
    limit: int,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict] | None:
    if not _is_uuid(tenant_id):
        return None

    try:
        conn = await _connect()
    except Exception:
        return None

    try:
        return await _repo_fetch_dashboard_keyword_stats(
            tenant_id,
            limit=limit,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            conn=conn,
        )
    finally:
        await conn.close()


async def fetch_dashboard_priority_distribution(
    *,
    tenant_id: str,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> dict[str, int] | None:
    if not _is_uuid(tenant_id):
        return None

    try:
        conn = await _connect()
    except Exception:
        return None

    try:
        return await _repo_fetch_dashboard_priority_distribution(
            tenant_id,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            conn=conn,
        )
    finally:
        await conn.close()


async def get_emotion_distribution(
    *,
    tenant_id: str,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> dict:
    if not _is_uuid(tenant_id):
        return await _legacy_get_emotion_distribution(
            tenant_id=tenant_id,
            started_from=date_from.isoformat() if date_from else None,
            started_to=date_to.isoformat() if date_to else None,
        )

    try:
        conn = await _connect()
    except Exception:
        return await _legacy_get_emotion_distribution(
            tenant_id=tenant_id,
            started_from=date_from.isoformat() if date_from else None,
            started_to=date_to.isoformat() if date_to else None,
        )

    try:
        result = await fetch_dashboard_emotion_distribution_counts(
            tenant_id,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat() if date_to else None,
            conn=conn,
        )
        if result is not None:
            return result
        return await _legacy_get_emotion_distribution(
            tenant_id=tenant_id,
            started_from=date_from.isoformat() if date_from else None,
            started_to=date_to.isoformat() if date_to else None,
        )
    finally:
        await conn.close()


@router.get("/stats")
@router.get("/overview")
async def get_stats(
    tenant_id: Optional[str] = Query(None, description="tenant filter for legacy clients"),
    started_from: Optional[datetime] = Query(None, description="start datetime, inclusive"),
    started_to: Optional[datetime] = Query(None, description="end datetime, inclusive"),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    return await get_dashboard_overview(
        tenant_id=dashboard_tenant_id,
        started_from=started_from,
        started_to=started_to,
    )


@router.get("/emotion-distribution")
async def get_emotion_dist(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[datetime] = Query(None),
    started_to: Optional[datetime] = Query(None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    return await get_emotion_distribution(
        tenant_id=dashboard_tenant_id,
        date_from=started_from,
        date_to=started_to,
    )


@router.get("/intent-distribution")
async def get_intent_distribution(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[datetime] = Query(None),
    started_to: Optional[datetime] = Query(None),
    limit: int = Query(default=10, ge=1, le=100),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    items = await fetch_dashboard_intent_distribution(
        tenant_id=dashboard_tenant_id,
        limit=limit,
        date_from=started_from,
        date_to=started_to,
    )
    raw_items = items or []
    items = [
        {"category": item.get("label") or item.get("category"), "count": item.get("count", 0)}
        for item in raw_items
    ]
    return {
        "items": items,
        "data": raw_items,
        "request_id": _request_id(),
    }


@router.get("/keyword-stats")
async def get_keyword_stats(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[datetime] = Query(None),
    started_to: Optional[datetime] = Query(None),
    limit: int = Query(default=10, ge=1, le=50),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    items = await fetch_dashboard_keyword_stats(
        tenant_id=dashboard_tenant_id,
        limit=limit,
        date_from=started_from,
        date_to=started_to,
    )
    raw_items = items or []
    mapped_items = [
        {"keyword": item.get("keyword") or item.get("label"), "count": item.get("count", 0)}
        for item in raw_items
    ]
    return {
        "items": mapped_items,
        "data": raw_items,
        "request_id": _request_id(),
    }


@router.get("/priority-distribution")
async def get_priority_distribution(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[datetime] = Query(None),
    started_to: Optional[datetime] = Query(None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    result = await fetch_dashboard_priority_distribution(
        tenant_id=dashboard_tenant_id,
        date_from=started_from,
        date_to=started_to,
    )
    return result or {"critical": 0, "high": 0, "medium": 0, "low": 0}


@router.get("/recent-calls")
async def list_recent_calls(
    tenant_id: Optional[str] = Query(None),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    started_from: Optional[datetime] = Query(None),
    started_to: Optional[datetime] = Query(None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    result = await fetch_dashboard_recent_calls(
        tenant_id=dashboard_tenant_id,
        limit=limit,
        offset=offset,
        date_from=started_from,
        date_to=started_to,
    )
    if result is None:
        result = {
            "items": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
        }

    return {
        **result,
        "data": result,
        "request_id": _request_id(),
    }


@router.get("/priority-queue")
async def list_priority_queue(
    tenant_id: Optional[str] = Query(None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    return await get_priority_queue(tenant_id=dashboard_tenant_id)


@router.get("/action-logs")
async def list_action_logs(
    tenant_id: Optional[str] = Query(None),
    started_from: Optional[str] = Query(None),
    started_to: Optional[str] = Query(None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    return await get_action_logs(
        tenant_id=dashboard_tenant_id,
        started_from=started_from,
        started_to=started_to,
    )


