from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.admin_auth import get_current_admin_user
from app.repositories import get_summary_by_call_id

router = APIRouter()


def _current_admin_tenant_id(current_admin: dict[str, Any]) -> str:
    user = current_admin.get("user") or {}
    tenant_id = str(user.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin tenant",
        )
    return tenant_id


@router.get("/{call_id}")
async def get_summary(
    call_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    tenant_id = _current_admin_tenant_id(current_admin)
    record = await get_summary_by_call_id(call_id, tenant_id=tenant_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"summary not found: {call_id!r}",
        )
    return record
