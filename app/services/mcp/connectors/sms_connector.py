"""
SMS MCP Connector.

기존 app.services.sms.solapi.SolapiSMSService를 재사용해 Post-call SMS를 발송한다.

지원 action_type:
  - send_callback_sms
  - send_voc_receipt_sms
  - send_reservation_confirmation

── real mode (SMS_MCP_REAL=true) ────────────────────────────────────────────
  SolapiSMSService를 통해 실제 Solapi API 발송.
  SOLAPI_API_KEY / SOLAPI_API_SECRET / SOLAPI_SENDER_NUMBER 필요.

── mock mode (SMS_MCP_REAL=false, 기본) ─────────────────────────────────────
  status: success
  external_id: sms-mock-{call_id}
  result: {to, message, sent, mock}

── 고객 전화번호 없음 ───────────────────────────────────────────────────────
  수신번호 결정 우선순위:
    1) params.to
    2) params.customer_phone
    3) os.getenv("SMS_TEST_TO")  ← 운영 외 환경 fallback
  세 가지 모두 비어있을 때만 skipped("customer_phone_missing") 반환.

  SMS_TEST_TO fallback 이 사용되면 logger.warning 으로 명시한다 — 운영
  배포 시 이 변수가 남아 있으면 의도치 않은 발송이 일어날 수 있으므로
  로그에서 가시적으로 추적 가능해야 한다. 운영에서는 unset 또는 빈 값.

── API key/secret 원문은 로그/result/error에 절대 출력하지 않는다. ─────────
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger
from app.utils.phone import normalize_korean_phone

logger = get_logger(__name__)

_TEMPLATES: dict[str, str] = {
    "send_callback_sms": "[시시콜콜] 상담 요청이 접수되었습니다. 담당자가 확인 후 다시 연락드리겠습니다.",
    "send_voc_receipt_sms": "[시시콜콜] 문의가 접수되었습니다. 처리 후 안내드리겠습니다. 접수번호: {call_id}",
    "send_reservation_confirmation": "[시시콜콜] 예약/콜백 일정이 접수되었습니다. 담당자가 확인 후 안내드리겠습니다.",
}


class SMSConnector(BaseMCPConnector):
    connector_name = "sms"
    _real_mode_env = "SMS_MCP_REAL"
    _required_config = ("SOLAPI_API_KEY", "SOLAPI_API_SECRET", "SOLAPI_SENDER_NUMBER")
    _oauth_provider_name = ""

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "SMSConnector call_id=%s action_type=%s real_mode=%s",
            call_id, action_type, self.is_real_mode(),
        )

        to = params.get("to") or params.get("customer_phone")
        if not to:
            test_to = (os.getenv("SMS_TEST_TO") or "").strip()
            if test_to:
                to = normalize_korean_phone(test_to) or test_to
                logger.warning(
                    "SMSConnector: customer_phone 없음 — SMS_TEST_TO fallback 사용 call_id=%s",
                    call_id,
                )

        if not to:
            logger.warning("SMSConnector: 고객 전화번호 없음 call_id=%s", call_id)
            return self._skipped("customer_phone_missing")

        message = params.get("message") or self._render_template(action_type, call_id)

        if not self.is_real_mode():
            return self._mock(to, message, call_id)

        return await self._send_real(to, message, call_id)

    def _render_template(self, action_type: str, call_id: str) -> str:
        template = _TEMPLATES.get(action_type, "[시시콜콜] 후속 안내 메시지입니다.")
        return template.format(call_id=call_id)

    async def _send_real(self, to: str, message: str, call_id: str) -> dict:
        try:
            from app.services.sms.solapi import SolapiSMSService
            svc = SolapiSMSService()
            success = await svc.send_sms(to, message)
            if success:
                logger.info("SMSConnector: 발송 완료 call_id=%s", call_id)
                return self._success(
                    external_id=f"sms-solapi-{call_id}",
                    result={"to": to, "sent": True},
                )
            logger.error("SMSConnector: 발송 실패(False) call_id=%s", call_id)
            return self._failed("sms_send_failed")
        except Exception as exc:
            logger.error(
                "SMSConnector: 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"sms_exception:{type(exc).__name__}")

    def _mock(self, to: str, message: str, call_id: str) -> dict:
        return self._success(
            external_id=f"sms-mock-{call_id}",
            result={"to": to, "message": message, "sent": True, "mock": True},
        )
