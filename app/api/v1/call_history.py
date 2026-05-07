import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.v1.admin_auth import get_current_admin_user
from app.repositories.call_repo import (
    get_call_by_id_for_tenant,
    list_calls_for_tenant,
)
from app.repositories.transcript_repo import get_transcripts_by_call_id

router = APIRouter()


def _request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def _current_admin_tenant_id(current_admin: dict[str, Any]) -> str:
    user = current_admin.get("user") or {}
    tenant_id = str(user.get("tenant_id") or "").strip()

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin tenant",
        )

    return tenant_id


def _validate_query_tenant_id(query_tenant_id: str | None, jwt_tenant_id: str) -> None:
    if not query_tenant_id:
        return

    if query_tenant_id.strip().lower() != jwt_tenant_id.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Query tenant mismatch",
        )


def _normalize_status_filter(status_filter: str | None) -> str | None:
    if status_filter is None:
        return None

    normalized = status_filter.strip()
    if not normalized or normalized.lower() == "all":
        return None

    return normalized


@router.get("")
async def list_calls(
    status_filter: str | None = Query(default=None, alias="status"),
    started_from: datetime | None = Query(default=None),
    started_to: datetime | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1),
    tenant_id: str | None = Query(default=None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _current_admin_tenant_id(current_admin)
    _validate_query_tenant_id(tenant_id, jwt_tenant_id)

    effective_limit = min(limit, 100)

    result = await list_calls_for_tenant(
        tenant_id=jwt_tenant_id,
        status=_normalize_status_filter(status_filter),
        started_from=started_from,
        started_to=started_to,
        offset=offset,
        limit=effective_limit,
    )

    return {
        "data": {
            "items": result["items"],
            "total": result["total"],
            "offset": offset,
            "limit": effective_limit,
        },
        "request_id": _request_id(),
    }


@router.get("/{call_id}/transcripts")
async def get_call_transcripts(
    call_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    tenant_id = _current_admin_tenant_id(current_admin)

    transcripts = await get_transcripts_by_call_id(call_id, tenant_id)

    if transcripts is None:
        raise HTTPException(
            status_code=404,
            detail=f"transcripts not found: {call_id!r}",
        )

    return {
        "data": {
            "items": transcripts,
            "total": len(transcripts),
        },
        "request_id": _request_id(),
    }


@router.get("/{call_id}")
async def get_call_detail(
    call_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    tenant_id = _current_admin_tenant_id(current_admin)

    record = await get_call_by_id_for_tenant(call_id, tenant_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"call not found: {call_id!r}",
        )

    return {
        "data": record,
        "request_id": _request_id(),
    }
