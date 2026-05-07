"""
Company DB MCP Connector.

지원 action_type:
  - create_voc_issue        (VOC 시스템 등록 — post_call 영역, mock/real)
  - add_priority_queue      (우선순위 큐 등록 — post_call 영역, mock/real)
  - lookup_member           (전화번호 → 회원 정보 조회 — finance 시연, in-memory)
  - suspend_card            (전화번호 → 카드 정지 처리 — finance 시연, in-memory)

── real mode env ─────────────────────────────────────────────────────────────
  COMPANY_DB_MCP_REAL=true   OR  MCP_COMPANY_DB_REAL=true  (둘 중 하나)
  COMPANY_DB_BASE_URL        API 서버 URL (선택)
  COMPANY_DB_MCP_SERVER_URL  MCP 서버 URL (선택)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: VOC-MOCK-{call_id}
  result: {created, issue_id, tier, priority, primary_category, reason, summary_short, mock}

── real mode 설정 부족 ────────────────────────────────────────────────────────
  status: skipped
  error: "company_db_connector_not_configured"

── finance 시연용 in-memory 회원 (lookup_member / suspend_card) ─────────────
  _FINANCE_MEMBERS dict 가 진실의 원천. mode 무관 (mock/real 둘 다 동일).
  process restart 시 초기화 — 시연 1회 안에서만 의미 있는 상태.
  customer_ref = phone 패턴 (face_embeddings 의 customer_ref 와 동일).
"""
from __future__ import annotations

import os

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Finance tenant mock 회원 (시연용) ─────────────────────────────────────────
# key = phone (= face_embeddings.customer_ref 와 동일 패턴)
_FINANCE_MEMBERS: dict[str, dict] = {
    "01047722480": {  # 희원 (실통화 SMS 수신 번호)
        "name": "이희원",
        "card_number_masked": "신한카드 *5678",
        "card_status": "normal",
    },
    "01012345678": {  # 더미 (대조용)
        "name": "김철수",
        "card_number_masked": "국민카드 *1234",
        "card_status": "normal",
    },
}


class CompanyDBConnector(BaseMCPConnector):
    connector_name = "company_db"
    # 두 가지 env var 중 하나라도 true이면 real mode
    _real_mode_env = "COMPANY_DB_MCP_REAL"

    def is_real_mode(self) -> bool:
        """COMPANY_DB_MCP_REAL 또는 MCP_COMPANY_DB_REAL 중 하나가 true이면 real mode."""
        v1 = os.getenv("COMPANY_DB_MCP_REAL", "").lower()
        v2 = os.getenv("MCP_COMPANY_DB_REAL", "").lower()
        return v1 in ("1", "true") or v2 in ("1", "true")

    def validate_config(self) -> tuple[bool, str | None]:
        # 필수 env 없음 — URL은 선택이므로 항상 통과
        return True, None

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "CompanyDBConnector call_id=%s action_type=%s real_mode=%s",
            call_id, action_type, self.is_real_mode(),
        )

        # finance 시연용 — mode 무관 (in-memory dict 가 진실)
        if action_type == "lookup_member":
            return self._lookup_member(params, call_id)
        if action_type == "suspend_card":
            return self._suspend_card(params, call_id)

        if not self.is_real_mode():
            return self._mock(params, call_id)

        ok, err = self.validate_config()
        if not ok:
            logger.warning("CompanyDBConnector: config 부족 call_id=%s err=%s", call_id, err)
            return self._skipped("company_db_connector_not_configured")

        return await self._execute_real(action_type, params, call_id=call_id)

    # ── finance 시연용 in-memory 회원 처리 ────────────────────────────────────

    def _lookup_member(self, params: dict, call_id: str) -> dict:
        phone = (params.get("phone_number") or "").strip()
        if not phone:
            return self._failed("missing_phone_number")
        member = _FINANCE_MEMBERS.get(phone)
        if member is None:
            logger.info(
                "CompanyDBConnector: 회원 미발견 call_id=%s phone=%s", call_id, phone,
            )
            return self._failed(f"member_not_found:{phone}")
        return self._success(
            external_id=f"member-{phone}",
            result={
                "phone_number": phone,
                "name": member["name"],
                "card_number_masked": member["card_number_masked"],
                "card_status": member["card_status"],
            },
        )

    def _suspend_card(self, params: dict, call_id: str) -> dict:
        phone = (params.get("phone_number") or "").strip()
        if not phone:
            return self._failed("missing_phone_number")
        member = _FINANCE_MEMBERS.get(phone)
        if member is None:
            logger.info(
                "CompanyDBConnector: 회원 미발견 call_id=%s phone=%s", call_id, phone,
            )
            return self._failed(f"member_not_found:{phone}")

        already = member["card_status"] == "suspended"
        member["card_status"] = "suspended"
        if not already:
            logger.info(
                "CompanyDBConnector: 카드 정지 처리 call_id=%s phone=%s name=%s",
                call_id, phone, member["name"],
            )
        return self._success(
            external_id=f"suspend-{phone}",
            result={
                "phone_number": phone,
                "name": member["name"],
                "card_number_masked": member["card_number_masked"],
                "card_status": "suspended",
                "already_suspended": already,
                # humanize 용 한국어 동사 hint — LLM 의 "취소" 환각 차단
                "action_label_kr": "이미 정지된 상태" if already else "정지 처리 완료",
            },
        )

    def _mock(self, params: dict, call_id: str) -> dict:
        issue_id = f"VOC-MOCK-{call_id}"
        return self._success(
            external_id=issue_id,
            result={
                "created": True,
                "issue_id": issue_id,
                "tier": params.get("tier", "medium"),
                "priority": params.get("priority", "medium"),
                "primary_category": params.get("primary_category", ""),
                "reason": params.get("reason", ""),
                "summary_short": params.get("summary_short", ""),
                "mock": True,
            },
        )

    async def _execute_real(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        # TODO: 실제 Company DB API / MCP 서버 연동 구현
        # base_url = os.getenv("COMPANY_DB_BASE_URL")
        # server_url = os.getenv("COMPANY_DB_MCP_SERVER_URL")
        logger.warning(
            "CompanyDBConnector: real mode TODO — skipped call_id=%s action_type=%s",
            call_id, action_type,
        )
        return self._skipped("company_db_mcp_real_not_implemented")
