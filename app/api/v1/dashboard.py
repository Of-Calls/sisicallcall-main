from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.v1.admin_auth import get_current_admin_user
from app.repositories import (
    get_action_logs,
    get_dashboard_overview,
    get_emotion_distribution,
    get_priority_queue,
)

router = APIRouter()


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


@router.get("/stats")
async def get_stats(
    tenant_id: Optional[str] = Query(None, description="tenant filter for legacy clients"),
    started_from: Optional[str] = Query(None, description="start datetime, inclusive"),
    started_to: Optional[str] = Query(None, description="end datetime, inclusive"),
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
    started_from: Optional[str] = Query(None),
    started_to: Optional[str] = Query(None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    dashboard_tenant_id = _resolve_dashboard_tenant_id(tenant_id, current_admin)
    return await get_emotion_distribution(
        tenant_id=dashboard_tenant_id,
        started_from=started_from,
        started_to=started_to,
    )


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
