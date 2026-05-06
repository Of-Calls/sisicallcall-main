"""
POST-CALL API — 통화 후처리 결과 조회 및 수동 실행.

등록 (app/main.py):
    app.include_router(post_call_router, prefix="/post-call", tags=["post-call"])
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.agents.post_call.completed_call_runner import (
    _CALL_CONTEXT_NOT_FOUND,
    run_post_call_for_completed_call,
)
from app.api.v1.admin_auth import get_current_admin_user
from app.repositories import (
    get_action_logs_by_call_id_for_tenant,
    get_dashboard_payload,
    get_post_call_detail,
)
from app.repositories.call_repo import get_call_by_id_for_tenant
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_VALID_TRIGGERS = frozenset({"call_ended", "manual", "escalation_immediate"})


def _current_admin_tenant_id(current_admin: dict[str, Any]) -> str:
    user = current_admin.get("user") or {}
    tenant_id = str(user.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin tenant",
        )
    return tenant_id


def _api_action_status(raw_status: str | None) -> str:
    if raw_status in ("failed", "fail"):
        return "fail"
    return raw_status or "pending"


def _action_log_response_item(log: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(log.get("id") or ""),
        "call_id": str(log.get("call_id") or ""),
        "tenant_id": str(log.get("tenant_id") or ""),
        "action_type": str(log.get("action_type") or ""),
        "action_detail": str(
            log.get("action_detail")
            or log.get("tool_name")
            or log.get("action_type")
            or ""
        ),
        "status": _api_action_status(log.get("status")),
        "request_payload": log.get("request_payload") or {},
        "response_payload": log.get("response_payload") or {},
        "error_message": log.get("error_message"),
        "executed_at": log.get("executed_at") or log.get("created_at"),
    }


@router.get("/{call_id}/actions")
async def get_call_actions(
    call_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    """call_id 에 해당하는 MCP action log list 를 반환한다."""
    tenant_id = _current_admin_tenant_id(current_admin)
    call_record = await get_call_by_id_for_tenant(call_id, tenant_id)
    if call_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"call not found: {call_id!r}",
        )

    logs = await get_action_logs_by_call_id_for_tenant(call_id, tenant_id)
    items = [_action_log_response_item(log) for log in logs]
    return {"items": items, "total": len(items)}


@router.get("/{call_id}")
async def get_post_call(call_id: str):
    """통화 후처리 전체 결과를 반환한다.

    summary, voc_analysis, priority_result, action_plan,
    executed_actions, errors, partial_success 포함.
    저장된 결과가 없으면 404 를 반환한다.
    """
    payload = await get_dashboard_payload(call_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"post-call result not found: {call_id!r}",
        )
    detail = await get_post_call_detail(call_id)
    return {"call_id": call_id, **detail}


@router.post("/{call_id}/run")
async def run_post_call(
    call_id: str,
    trigger: str = Query(default="call_ended"),
    tenant_id: str = Query(default="default"),
):
    """종료된 통화 데이터를 기반으로 후처리를 수동 실행한다.

    trigger: call_ended(기본) | manual | escalation_immediate
    - 통화 context가 없으면 404를 반환한다.
    - LLM은 POST_CALL_USE_REAL_LLM=true 가 아니면 mock을 사용한다.
    """
    if trigger not in _VALID_TRIGGERS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown trigger: {trigger!r}. valid: {sorted(_VALID_TRIGGERS)}",
        )

    logger.info(
        "run_post_call call_id=%s trigger=%s tenant_id=%s",
        call_id, trigger, tenant_id,
    )

    result = await run_post_call_for_completed_call(
        call_id=call_id,
        tenant_id=tenant_id,
        trigger=trigger,
    )

    if not result["ok"] and result.get("error") == _CALL_CONTEXT_NOT_FOUND:
        raise HTTPException(
            status_code=404,
            detail=f"call context not found: {call_id!r}",
        )

    return result
