"""
BaseMCPConnector — MCP Connector 계층 공통 기반.

각 Connector는 이 클래스를 상속하고 execute()를 구현한다.
import 시점에 외부 API / MCP 서버 연결을 만들지 않는다.

── 표준 반환 형식 ────────────────────────────────────────────────────────────
{
  "status":      "success" | "failed" | "skipped",
  "external_id": "...",          # 외부 시스템이 발급한 ID (없으면 None)
  "result":      {...},          # connector별 상세 결과
  "error":       None | "..."    # 오류 메시지 (success면 None)
}

── real mode 분기 원칙 ──────────────────────────────────────────────────────
- is_real_mode() == False  → mock result (status=success)
- is_real_mode() == True, validate_config() 실패  → skipped + error
- is_real_mode() == True, validate_config() 성공  → 실제 통합 (미구현 시 skipped)

── tenant OAuth 분기 원칙 (MCP_USE_TENANT_OAUTH=true) ───────────────────────
- _use_tenant_oauth() == True AND tenant_id AND _oauth_provider_name:
    _try_tenant_token() 호출 — DB / file / memory 백엔드 무관
    → connected:     skipped("tenant_token_found_but_real_execute_not_implemented")
    → not connected: skipped("tenant_integration_not_connected")
      MCP_ALLOW_ENV_FALLBACK=true 일 때만 .env 방식으로 폴백 가능 (실서비스
      에서는 false 권장 — env fallback 은 dev/demo 편의 기능).

── provider alias ──────────────────────────────────────────────────────────
DB / file 에 저장된 row 의 provider 이름이 실제 connector 의
``_oauth_provider_name`` 과 일치하는 게 원칙이지만, 과거 OAuth route 가
``google_gmail`` / ``google_calendar`` 로 저장한 row 가 있을 수 있다.
``_PROVIDER_ALIASES`` 가 후보 provider 를 순서대로 조회한다.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime

from app.utils.logger import get_logger

logger = get_logger(__name__)


# Connector 의 ``_oauth_provider_name`` 과 tenant_integrations.provider 가
# 다른 이름으로 저장됐을 때 후보를 순서대로 조회한다. 첫 번째가 canonical.
_PROVIDER_ALIASES: dict[str, list[str]] = {
    "google_gmail":    ["google_gmail", "gmail"],
    "gmail":           ["gmail", "google_gmail"],
    "google_calendar": ["google_calendar", "calendar"],
    "calendar":        ["calendar", "google_calendar"],
    "slack":           ["slack"],
    "jira":            ["jira"],
}


def _alias_candidates(provider: str) -> list[str]:
    return _PROVIDER_ALIASES.get(provider, [provider])


class BaseMCPConnector(ABC):
    """모든 MCP Connector의 공통 추상 기반 클래스."""

    connector_name: str = "base"
    _real_mode_env: str = ""                # 예: "GMAIL_MCP_REAL"
    _required_config: tuple[str, ...] = ()  # real mode 필수 env var
    _oauth_provider_name: str = ""          # tenant OAuth provider 이름 (예: "google_gmail")

    # ── mode 판단 ─────────────────────────────────────────────────────────────

    def is_real_mode(self) -> bool:
        """환경변수 {_real_mode_env}=true|1 이면 True."""
        if not self._real_mode_env:
            return False
        return os.getenv(self._real_mode_env, "").lower() in ("1", "true")

    def _use_tenant_oauth(self) -> bool:
        """MCP_USE_TENANT_OAUTH=true 이면 SaaS 테넌트 OAuth 흐름 사용."""
        return os.getenv("MCP_USE_TENANT_OAUTH", "").lower() in ("1", "true")

    def _allow_env_fallback(self) -> bool:
        """MCP_ALLOW_ENV_FALLBACK=true 이면 테넌트 미연결 시 .env 방식으로 폴백."""
        return os.getenv("MCP_ALLOW_ENV_FALLBACK", "").lower() in ("1", "true")

    def validate_config(self) -> tuple[bool, str | None]:
        """real mode에 필요한 env var이 모두 설정되었는지 확인한다.

        Returns:
            (True, None)        → 설정 완료
            (False, error_msg)  → 누락 env var 목록 포함 메시지
        """
        missing = [k for k in self._required_config if not os.getenv(k)]
        if missing:
            return False, f"missing env: {', '.join(missing)}"
        return True, None

    # ── tenant OAuth 헬퍼 ─────────────────────────────────────────────────────

    def _resolve_tenant_integration(self, tenant_id: str):
        """
        ``_oauth_provider_name`` + alias 후보를 순서대로 DB / file / memory 에서
        조회한다. 첫 번째 connected row 를 우선 반환하고, 없으면 마지막으로
        매칭된 row (disconnected/expired/error) 를 반환한다.

        Returns:
            (integration, source_provider) — 둘 다 None 가능.
        """
        from app.models.tenant_integration import IntegrationStatus
        from app.repositories.tenant_integration_repo import get_integration

        connected = None
        connected_source = None
        fallback = None
        fallback_source = None

        for cand in _alias_candidates(self._oauth_provider_name):
            integration = get_integration(tenant_id, cand)
            if integration is None:
                continue
            if integration.status == IntegrationStatus.connected and connected is None:
                connected = integration
                connected_source = cand
            elif fallback is None:
                fallback = integration
                fallback_source = cand

        if connected is not None:
            return connected, connected_source
        return fallback, fallback_source

    async def _try_tenant_token(self, tenant_id: str) -> dict:
        """
        테넌트 OAuth 토큰을 조회·검증한다.

        Returns:
            skipped("tenant_token_found_but_real_execute_not_implemented")
                — 연동됨, 토큰 유효, 실행 구현 대기
            skipped("tenant_token_expired_no_refresh")
                — 연동됨, 토큰 만료, refresh_token 없음
            skipped("tenant_integration_not_connected")
                — 연동 정보 없거나 disconnected 상태
        """
        from app.models.tenant_integration import IntegrationStatus
        from app.services.oauth.token_crypto import decrypt_token

        provider = self._oauth_provider_name
        integration, source_provider = self._resolve_tenant_integration(tenant_id)

        if integration is None or integration.status == IntegrationStatus.disconnected:
            return self._skipped("tenant_integration_not_connected")

        # 만료 체크
        if integration.expires_at and integration.expires_at < datetime.utcnow():
            if integration.refresh_token_encrypted:
                refreshed = await self._refresh_tenant_token(integration)
                if refreshed:
                    integration = refreshed
                else:
                    return self._skipped("tenant_token_expired_refresh_failed")
            else:
                return self._skipped("tenant_token_expired_no_refresh")

        # 토큰 복호화 확인 (키 이상 시 실패 반환)
        try:
            decrypt_token(integration.access_token_encrypted or "")
        except Exception as exc:
            logger.error(
                "_try_tenant_token 복호화 실패 tenant_id=%s provider=%s err=%s",
                tenant_id, provider, exc,
            )
            return self._failed("tenant_token_decryption_failed")

        logger.info(
            "_try_tenant_token OK tenant_id=%s provider=%s source=%s — real execute TODO",
            tenant_id, provider, source_provider or provider,
        )
        return self._skipped(
            "tenant_token_found_but_real_execute_not_implemented",
            {
                "tenant_id": tenant_id,
                "provider": provider,
                "source_provider": source_provider or provider,
            },
        )

    async def _refresh_tenant_token(self, integration) -> object | None:
        """refresh token으로 access token 갱신. 실패 시 None."""
        from app.models.tenant_integration import IntegrationStatus
        from app.repositories.tenant_integration_repo import update_tokens
        from app.services.oauth.token_crypto import decrypt_token, encrypt_token

        provider = self._oauth_provider_name
        try:
            raw_refresh = decrypt_token(integration.refresh_token_encrypted or "")
            oauth = self._get_oauth_provider()
            if oauth is None:
                return None

            token_result = await oauth.refresh_token(raw_refresh)
            new_enc = encrypt_token(token_result.access_token)
            from datetime import timedelta
            expires_at = None
            if token_result.expires_in:
                expires_at = datetime.utcnow() + timedelta(seconds=token_result.expires_in)

            update_tokens(
                integration.tenant_id, provider,
                access_token_encrypted=new_enc,
                expires_at=expires_at,
                status=IntegrationStatus.connected,
            )
            from app.repositories.tenant_integration_repo import get_integration
            return get_integration(integration.tenant_id, provider)
        except Exception as exc:
            logger.error(
                "_refresh_tenant_token 실패 tenant_id=%s provider=%s err=%s",
                integration.tenant_id, provider, exc,
            )
            return None

    def _get_oauth_provider(self):
        """_oauth_provider_name에 맞는 OAuth 프로바이더 인스턴스 반환."""
        name = self._oauth_provider_name
        if name in ("google_gmail", "google_calendar"):
            from app.services.oauth.google_oauth import GoogleGmailOAuth, GoogleCalendarOAuth
            return GoogleGmailOAuth() if name == "google_gmail" else GoogleCalendarOAuth()
        if name == "slack":
            from app.services.oauth.slack_oauth import SlackOAuth
            return SlackOAuth()
        if name == "jira":
            from app.services.oauth.jira_oauth import JiraOAuth
            return JiraOAuth()
        return None

    # ── 추상 메서드 ───────────────────────────────────────────────────────────

    @abstractmethod
    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        """action_type에 따라 외부 도구를 호출하고 표준 결과를 반환한다."""

    # ── 결과 헬퍼 ─────────────────────────────────────────────────────────────

    def _success(self, external_id: str | None, result: dict) -> dict:
        return {
            "status": "success",
            "external_id": external_id,
            "result": result,
            "error": None,
        }

    def _skipped(self, error: str, result: dict | None = None) -> dict:
        return {
            "status": "skipped",
            "external_id": None,
            "result": result or {},
            "error": error,
        }

    def _failed(self, error: str, result: dict | None = None) -> dict:
        return {
            "status": "failed",
            "external_id": None,
            "result": result or {},
            "error": error,
        }
