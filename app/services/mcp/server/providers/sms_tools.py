"""
MCP Server tool: sms.send_voc_receipt_sms / sms.send_callback_sms

기존 SolapiSMSService 를 재사용해 Solapi API 로 SMS 를 발송한다.
SMSConnector.execute() 는 호출하지 않는다 — 같은 Solapi SDK 만 공유한다.
"""
from __future__ import annotations

import os
import time
from typing import Any

from app.services.mcp.server import result as R
from app.utils.logger import get_logger
from app.utils.phone import normalize_korean_phone

logger = get_logger(__name__)

_TOOL = "sms"

_TEMPLATES: dict[str, str] = {
    "send_callback_sms": (
        "[시시콜콜] 상담 요청이 접수되었습니다. "
        "담당자가 확인 후 다시 연락드리겠습니다."
    ),
    "send_voc_receipt_sms": (
        "[시시콜콜] 문의가 접수되었습니다. 처리 후 안내드리겠습니다. "
        "접수번호: {call_id}"
    ),
    "send_reservation_confirmation": (
        "[시시콜콜] 예약/콜백 일정이 접수되었습니다. "
        "담당자가 확인 후 안내드리겠습니다."
    ),
}


def _render_template(action_type: str, call_id: str) -> str:
    template = _TEMPLATES.get(action_type, "[시시콜콜] 후속 안내 메시지입니다.")
    return template.format(call_id=call_id)


def _resolve_phone(params: dict) -> str:
    to = params.get("to") or params.get("customer_phone") or ""
    if not to:
        test_to = (os.getenv("SMS_TEST_TO") or "").strip()
        if test_to:
            to = normalize_korean_phone(test_to) or test_to
            logger.warning(
                "sms_tools: customer_phone 없음 — SMS_TEST_TO fallback 사용",
            )
    return to


def _solapi_configured() -> bool:
    return bool(
        os.getenv("SOLAPI_API_KEY")
        and os.getenv("SOLAPI_API_SECRET")
        and (os.getenv("SOLAPI_SENDER_NUMBER") or os.getenv("SOLAPI_FROM"))
    )


async def _send(
    *,
    action_type: str,
    tenant_id: str,
    call_id: str,
    params: dict,
) -> dict[str, Any]:
    started = time.perf_counter()
    mcp_tool = f"{_TOOL}.{action_type}"

    to = _resolve_phone(params)
    if not to:
        latency = int((time.perf_counter() - started) * 1000)
        return R.skipped(
            tool=_TOOL, action_type=action_type, mcp_tool=mcp_tool,
            reason="customer_phone_missing", latency_ms=latency,
        )

    if not _solapi_configured():
        latency = int((time.perf_counter() - started) * 1000)
        return R.skipped(
            tool=_TOOL, action_type=action_type, mcp_tool=mcp_tool,
            reason="sms_config_missing", latency_ms=latency,
        )

    message = params.get("message") or _render_template(action_type, call_id)

    try:
        from app.services.sms.solapi import SolapiSMSService
        svc = SolapiSMSService()
        ok = await svc.send_sms(to, message)
    except Exception as exc:
        latency = int((time.perf_counter() - started) * 1000)
        logger.error("sms_tools: 예외 call_id=%s err=%s", call_id, type(exc).__name__)
        return R.failed(
            tool=_TOOL, action_type=action_type, mcp_tool=mcp_tool,
            error=f"sms_exception:{type(exc).__name__}", latency_ms=latency,
        )

    latency = int((time.perf_counter() - started) * 1000)
    if not ok:
        return R.failed(
            tool=_TOOL, action_type=action_type, mcp_tool=mcp_tool,
            error="sms_send_failed", latency_ms=latency,
        )

    return R.success(
        tool=_TOOL, action_type=action_type, mcp_tool=mcp_tool,
        external_id=f"sms-solapi-{call_id}",
        result={"to": to, "sent": True, "message_preview": message[:80]},
        latency_ms=latency,
    )


async def send_voc_receipt_sms(*, tenant_id: str, call_id: str, params: dict):
    return await _send(
        action_type="send_voc_receipt_sms",
        tenant_id=tenant_id, call_id=call_id, params=params,
    )


async def send_callback_sms(*, tenant_id: str, call_id: str, params: dict):
    return await _send(
        action_type="send_callback_sms",
        tenant_id=tenant_id, call_id=call_id, params=params,
    )


def register(mcp) -> None:
    @mcp.tool(
        name="sms.send_voc_receipt_sms",
        description="Solapi SMS — VOC 접수 안내 메시지",
    )
    async def sms_send_voc_receipt_sms(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await send_voc_receipt_sms(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )

    @mcp.tool(
        name="sms.send_callback_sms",
        description="Solapi SMS — 콜백 안내 메시지",
    )
    async def sms_send_callback_sms(
        tenant_id: str,
        call_id: str,
        params: dict | None = None,
    ) -> dict:
        return await send_callback_sms(
            tenant_id=tenant_id, call_id=call_id, params=params or {},
        )
