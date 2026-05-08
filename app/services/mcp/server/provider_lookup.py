"""
MCP Server tool 내부에서 tenant OAuth token / metadata 를 조회하는 헬퍼.

직접 connector 를 호출하지 않는다. 토큰 lookup / 만료/refresh / 복호화만 담당.
provider alias 정책은 BaseMCPConnector 와 동일하다 (google_gmail/gmail,
google_calendar/calendar 후보 순서).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)


_PROVIDER_ALIASES: dict[str, list[str]] = {
    "google_gmail":    ["google_gmail", "gmail"],
    "gmail":           ["gmail", "google_gmail"],
    "google_calendar": ["google_calendar", "calendar"],
    "calendar":        ["calendar", "google_calendar"],
    "slack":           ["slack"],
    "jira":            ["jira"],
    "notion":          ["notion"],
}


class TenantOAuthLookupError(RuntimeError):
    """tenant OAuth 조회 / 토큰 처리 중 발생한 예외 (tool 안에서 skipped/failed 로 변환)."""

    def __init__(self, *, status: str, reason: str, detail: dict[str, Any] | None = None):
        super().__init__(reason)
        self.status = status            # "skipped" | "failed"
        self.reason = reason
        self.detail = detail or {}


def _alias_candidates(provider: str) -> list[str]:
    return _PROVIDER_ALIASES.get(provider, [provider])


def _as_utc_aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def use_tenant_oauth() -> bool:
    return os.getenv("MCP_USE_TENANT_OAUTH", "").lower() in ("1", "true")


def allow_env_fallback() -> bool:
    return os.getenv("MCP_ALLOW_ENV_FALLBACK", "").lower() in ("1", "true")


def _resolve_integration(tenant_id: str, provider: str):
    from app.models.tenant_integration import IntegrationStatus
    from app.repositories.tenant_integration_repo import get_integration

    connected = None
    connected_source = None
    fallback = None
    fallback_source = None

    for cand in _alias_candidates(provider):
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


def _is_expired(integration) -> bool:
    expires_at = _as_utc_aware(getattr(integration, "expires_at", None))
    if expires_at is None:
        return False
    return expires_at < _utc_now()


async def _refresh_access_token(integration, provider: str):
    """refresh_token 으로 access token 을 갱신해 새 integration row 를 반환한다.

    실패 시 None.
    """
    from app.models.tenant_integration import IntegrationStatus
    from app.repositories.tenant_integration_repo import get_integration, update_tokens
    from app.services.oauth.token_crypto import decrypt_token, encrypt_token

    try:
        raw_refresh = decrypt_token(integration.refresh_token_encrypted or "")
    except Exception as exc:
        logger.error("refresh decrypt 실패 provider=%s err=%s", provider, exc)
        return None

    oauth = _get_oauth_client(provider)
    if oauth is None:
        return None
    try:
        token_result = await oauth.refresh_token(raw_refresh)
    except Exception as exc:
        logger.error("refresh 호출 실패 provider=%s err=%s", provider, exc)
        return None

    new_enc = encrypt_token(token_result.access_token)
    expires_at = None
    if token_result.expires_in:
        expires_at = _utc_now() + timedelta(seconds=token_result.expires_in)
    update_tokens(
        integration.tenant_id,
        provider,
        access_token_encrypted=new_enc,
        expires_at=expires_at,
        status=IntegrationStatus.connected,
    )
    return get_integration(integration.tenant_id, provider)


def _get_oauth_client(provider: str):
    if provider in ("google_gmail", "google_calendar"):
        from app.services.oauth.google_oauth import GoogleCalendarOAuth, GoogleGmailOAuth
        return GoogleGmailOAuth() if provider == "google_gmail" else GoogleCalendarOAuth()
    if provider == "slack":
        from app.services.oauth.slack_oauth import SlackOAuth
        return SlackOAuth()
    if provider == "jira":
        from app.services.oauth.jira_oauth import JiraOAuth
        return JiraOAuth()
    return None


async def get_tenant_token(
    *,
    tenant_id: str,
    provider: str,
) -> tuple[str, Any, str]:
    """tenant_integrations 에서 (access_token, integration, source_provider) 를 조회.

    tenant_id 가 비어 있거나 integration 이 없으면 TenantOAuthLookupError 를 던진다.

    Raises:
        TenantOAuthLookupError(status="skipped", reason="tenant_oauth_required")
            tenant_id 가 비어있거나 MCP_USE_TENANT_OAUTH=false
        TenantOAuthLookupError(status="skipped", reason="tenant_integration_not_connected")
        TenantOAuthLookupError(status="skipped", reason="tenant_token_expired_no_refresh")
        TenantOAuthLookupError(status="skipped", reason="tenant_token_expired_refresh_failed")
        TenantOAuthLookupError(status="failed",  reason="tenant_token_decryption_failed")
    """
    from app.models.tenant_integration import IntegrationStatus
    from app.services.oauth.token_crypto import decrypt_token

    if not tenant_id:
        raise TenantOAuthLookupError(status="skipped", reason="tenant_oauth_required")
    if not use_tenant_oauth():
        raise TenantOAuthLookupError(status="skipped", reason="tenant_oauth_required")

    integration, source_provider = _resolve_integration(tenant_id, provider)
    if integration is None or integration.status == IntegrationStatus.disconnected:
        raise TenantOAuthLookupError(status="skipped", reason="tenant_integration_not_connected")

    if _is_expired(integration):
        if integration.refresh_token_encrypted:
            refreshed = await _refresh_access_token(integration, source_provider or provider)
            if refreshed is None:
                raise TenantOAuthLookupError(
                    status="skipped",
                    reason="tenant_token_expired_refresh_failed",
                )
            integration = refreshed
        else:
            raise TenantOAuthLookupError(status="skipped", reason="tenant_token_expired_no_refresh")

    try:
        access_token = decrypt_token(integration.access_token_encrypted or "")
    except Exception as exc:
        logger.error(
            "tenant token 복호화 실패 tenant_id=%s provider=%s err=%s",
            tenant_id, provider, exc,
        )
        raise TenantOAuthLookupError(status="failed", reason="tenant_token_decryption_failed")

    return access_token, integration, source_provider or provider
