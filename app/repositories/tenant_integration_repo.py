"""
TenantIntegrationRepository — Postgres `tenant_integrations` 백엔드.

실서비스 SaaS mode — tenant_id 기준으로 DB 에 token 보관/조회. 연결 정보는
`settings.database_url` 사용 (asyncpg). 다른 백엔드 (memory / file) 는
없다 — 운영 데이터의 단일 진실 소스는 Postgres 다.

── 저장 정책 ──────────────────────────────────────────────────────────────────
access_token_encrypted / refresh_token_encrypted 는 이미 Fernet 암호화된
문자열만 저장한다. 평문 토큰은 절대 DB 에 기록하지 않는다.

── DB 스키마 ──────────────────────────────────────────────────────────────────
실제 DB 컬럼명을 그대로 사용한다 (db/init/11_tenant_integrations.sql 참조):
  id, tenant_id, provider, status, scopes,
  access_token_encrypted, refresh_token_encrypted, token_type, expires_at,
  external_account_id, external_account_email,
  external_workspace_id, external_workspace_name,
  metadata, created_at, updated_at

UNIQUE (tenant_id, provider) — upsert 는 ON CONFLICT 로 처리.
status CHECK ('connected' | 'disconnected' | 'expired' | 'error').

── async <-> sync 다리 ────────────────────────────────────────────────────────
repository 의 sync 인터페이스(get_integration, upsert_integration, ...)는
내부적으로 asyncpg 코루틴을 ``_run_async_blocking`` 으로 실행한다 — 이미
event loop 안에 있으면 별도 thread 에서 새 loop 를 만들어 잠시 블로킹한다.
후처리 액션 정도의 저빈도 호출이라 thread-per-call 비용은 허용 범위 안.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
from datetime import datetime
from typing import Any

import asyncpg

from app.models.tenant_integration import IntegrationStatus, TenantIntegration
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── async <-> sync bridge ─────────────────────────────────────────────────────

def _run_async_blocking(coro):
    """sync context 에서 async coroutine 을 동기적으로 실행한다.

    - event loop 밖이면 ``asyncio.run`` 으로 직접 실행.
    - event loop 안 (FastAPI 핸들러, 비동기 connector) 이면 별도 thread 에서
      새 loop 를 만들어 실행하고 결과를 기다린다. 짧은 DB I/O 한 번이라 잠깐
      현재 thread 가 블로킹되지만, 호출 빈도가 낮아 허용 범위.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


# ── DB I/O ────────────────────────────────────────────────────────────────────

def _database_url() -> str:
    """asyncpg 가 받아들이는 형식으로 정규화."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


_DB_COLUMNS = (
    "id, tenant_id::text AS tenant_id, provider, status, scopes, "
    "access_token_encrypted, refresh_token_encrypted, token_type, expires_at, "
    "external_account_id, external_account_email, "
    "external_workspace_id, external_workspace_name, "
    "metadata, created_at, updated_at"
)


def _coerce_jsonb(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return value


def _row_to_integration(row) -> TenantIntegration | None:
    if row is None:
        return None
    scopes = _coerce_jsonb(row["scopes"], [])
    if not isinstance(scopes, list):
        scopes = []
    metadata = _coerce_jsonb(row["metadata"], {})
    if not isinstance(metadata, dict):
        metadata = {}
    return TenantIntegration(
        id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        provider=row["provider"],
        status=row["status"] or IntegrationStatus.connected.value,
        scopes=list(scopes),
        access_token_encrypted=row["access_token_encrypted"],
        refresh_token_encrypted=row["refresh_token_encrypted"],
        token_type=row["token_type"] or "Bearer",
        expires_at=row["expires_at"],
        external_account_id=row["external_account_id"],
        external_account_email=row["external_account_email"],
        external_workspace_id=row["external_workspace_id"],
        external_workspace_name=row["external_workspace_name"],
        metadata=metadata,
        created_at=row["created_at"] or datetime.utcnow(),
        updated_at=row["updated_at"] or datetime.utcnow(),
    )


async def _db_connect():
    return await asyncpg.connect(_database_url())


async def _db_get(tenant_id: str, provider: str) -> TenantIntegration | None:
    conn = await _db_connect()
    try:
        row = await conn.fetchrow(
            f"SELECT {_DB_COLUMNS} FROM tenant_integrations "
            "WHERE tenant_id = $1::uuid AND provider = $2",
            tenant_id, provider,
        )
        return _row_to_integration(row)
    finally:
        await conn.close()


async def _db_list(tenant_id: str) -> list[TenantIntegration]:
    conn = await _db_connect()
    try:
        rows = await conn.fetch(
            f"SELECT {_DB_COLUMNS} FROM tenant_integrations "
            "WHERE tenant_id = $1::uuid ORDER BY provider",
            tenant_id,
        )
        return [_row_to_integration(r) for r in rows if r is not None]
    finally:
        await conn.close()


async def _db_upsert(integration: TenantIntegration) -> TenantIntegration:
    conn = await _db_connect()
    try:
        scopes_json = json.dumps(list(integration.scopes or []), ensure_ascii=False)
        metadata_json = json.dumps(integration.metadata or {}, ensure_ascii=False)
        status_value = (
            integration.status.value
            if hasattr(integration.status, "value")
            else str(integration.status)
        )
        now = datetime.utcnow()
        row = await conn.fetchrow(
            f"""
            INSERT INTO tenant_integrations (
                tenant_id, provider, status, scopes,
                access_token_encrypted, refresh_token_encrypted,
                token_type, expires_at,
                external_account_id, external_account_email,
                external_workspace_id, external_workspace_name,
                metadata, updated_at
            )
            VALUES (
                $1::uuid, $2, $3, $4::jsonb,
                $5, $6, $7, $8,
                $9, $10, $11, $12,
                $13::jsonb, $14
            )
            ON CONFLICT (tenant_id, provider) DO UPDATE SET
                status = EXCLUDED.status,
                scopes = EXCLUDED.scopes,
                access_token_encrypted = EXCLUDED.access_token_encrypted,
                refresh_token_encrypted = EXCLUDED.refresh_token_encrypted,
                token_type = EXCLUDED.token_type,
                expires_at = EXCLUDED.expires_at,
                external_account_id = EXCLUDED.external_account_id,
                external_account_email = EXCLUDED.external_account_email,
                external_workspace_id = EXCLUDED.external_workspace_id,
                external_workspace_name = EXCLUDED.external_workspace_name,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            RETURNING {_DB_COLUMNS}
            """,
            integration.tenant_id, integration.provider, status_value, scopes_json,
            integration.access_token_encrypted, integration.refresh_token_encrypted,
            integration.token_type, integration.expires_at,
            integration.external_account_id, integration.external_account_email,
            integration.external_workspace_id, integration.external_workspace_name,
            metadata_json, now,
        )
        return _row_to_integration(row) or integration
    finally:
        await conn.close()


async def _db_mark_disconnected(tenant_id: str, provider: str) -> bool:
    conn = await _db_connect()
    try:
        result = await conn.execute(
            "UPDATE tenant_integrations SET status = 'disconnected', updated_at = now() "
            "WHERE tenant_id = $1::uuid AND provider = $2",
            tenant_id, provider,
        )
        # asyncpg execute() returns "UPDATE n"
        parts = result.split()
        return parts[-1] != "0"
    finally:
        await conn.close()


async def _db_update_tokens(
    tenant_id: str,
    provider: str,
    *,
    access_token_encrypted: str,
    refresh_token_encrypted: str | None,
    expires_at: datetime | None,
    status: IntegrationStatus,
) -> bool:
    conn = await _db_connect()
    try:
        status_value = status.value if hasattr(status, "value") else str(status)
        result = await conn.execute(
            """
            UPDATE tenant_integrations SET
                access_token_encrypted = $3,
                refresh_token_encrypted = COALESCE($4, refresh_token_encrypted),
                expires_at = COALESCE($5, expires_at),
                status = $6,
                updated_at = now()
            WHERE tenant_id = $1::uuid AND provider = $2
            """,
            tenant_id, provider, access_token_encrypted,
            refresh_token_encrypted, expires_at, status_value,
        )
        parts = result.split()
        return parts[-1] != "0"
    finally:
        await conn.close()


# ── Repository ────────────────────────────────────────────────────────────────

class TenantIntegrationRepository:
    """tenant_integrations 테이블 기반 단일 백엔드 repository.

    인스턴스 생성에 별도 인자가 필요 없다. ``settings.database_url`` 만
    유효하면 동작한다.
    """

    def __init__(self) -> None:
        logger.info("TenantIntegrationRepo db mode")

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def upsert_integration(self, integration: TenantIntegration) -> TenantIntegration:
        saved = _run_async_blocking(_db_upsert(integration))
        logger.debug(
            "TenantIntegration upserted (db) tenant_id=%s provider=%s status=%s",
            integration.tenant_id, integration.provider, integration.status,
        )
        return saved or integration

    def get_integration(self, tenant_id: str, provider: str) -> TenantIntegration | None:
        return _run_async_blocking(_db_get(tenant_id, provider))

    def list_integrations(self, tenant_id: str) -> list[TenantIntegration]:
        return _run_async_blocking(_db_list(tenant_id))

    def mark_disconnected(self, tenant_id: str, provider: str) -> bool:
        return _run_async_blocking(_db_mark_disconnected(tenant_id, provider))

    def update_tokens(
        self,
        tenant_id: str,
        provider: str,
        *,
        access_token_encrypted: str,
        refresh_token_encrypted: str | None = None,
        expires_at: datetime | None = None,
        status: IntegrationStatus = IntegrationStatus.connected,
    ) -> bool:
        return _run_async_blocking(
            _db_update_tokens(
                tenant_id, provider,
                access_token_encrypted=access_token_encrypted,
                refresh_token_encrypted=refresh_token_encrypted,
                expires_at=expires_at,
                status=status,
            )
        )

    def clear_integrations(self) -> None:
        # db mode: 운영 데이터를 일괄 삭제하지 않는다. no-op.
        # 테스트는 asyncpg 를 mock 하거나 격리된 DB 를 사용해 isolation 한다.
        return


# ── 모듈 레벨 싱글턴 ──────────────────────────────────────────────────────────

tenant_integration_repo = TenantIntegrationRepository()


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def upsert_integration(integration: TenantIntegration) -> TenantIntegration:
    return tenant_integration_repo.upsert_integration(integration)


def get_integration(tenant_id: str, provider: str) -> TenantIntegration | None:
    return tenant_integration_repo.get_integration(tenant_id, provider)


def list_integrations(tenant_id: str) -> list[TenantIntegration]:
    return tenant_integration_repo.list_integrations(tenant_id)


def mark_disconnected(tenant_id: str, provider: str) -> bool:
    return tenant_integration_repo.mark_disconnected(tenant_id, provider)


def update_tokens(
    tenant_id: str,
    provider: str,
    *,
    access_token_encrypted: str,
    refresh_token_encrypted: str | None = None,
    expires_at: datetime | None = None,
    status: IntegrationStatus = IntegrationStatus.connected,
) -> bool:
    return tenant_integration_repo.update_tokens(
        tenant_id, provider,
        access_token_encrypted=access_token_encrypted,
        refresh_token_encrypted=refresh_token_encrypted,
        expires_at=expires_at,
        status=status,
    )


def clear_integrations() -> None:
    tenant_integration_repo.clear_integrations()
