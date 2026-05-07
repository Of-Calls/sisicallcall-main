"""
TenantIntegrationRepository — in-memory + file + Postgres(`db`) 백엔드.

저장 백엔드는 TENANT_INTEGRATION_STORAGE 환경변수로 선택한다:
  memory  (기본): in-memory dict. 서버 재시작 시 데이터 소멸.
  file:           JSON 파일 persistence. 서버 재시작 후에도 유지.
                  TENANT_INTEGRATION_FILE_PATH로 경로 지정
                  (기본: .local/tenant_integrations.json)
  db:             Postgres `tenant_integrations` 테이블 사용.
                  실서비스 SaaS mode — tenant_id 기준으로 DB에 token 보관/조회.
                  연결 정보는 settings.database_url 사용 (asyncpg).

── 파일 저장 형식 ─────────────────────────────────────────────────────────────
{
  "{tenant_id}::{provider}": { ...TenantIntegration 필드... }
}
access_token_encrypted / refresh_token_encrypted 는 이미 Fernet 암호화된
문자열만 저장한다. 평문 토큰은 절대 파일/DB 에 기록하지 않는다.

── DB 백엔드 ──────────────────────────────────────────────────────────────────
실제 DB 컬럼명을 그대로 사용한다 (db/init/11_tenant_integrations.sql 참조):
  id, tenant_id, provider, status, scopes,
  access_token_encrypted, refresh_token_encrypted, token_type, expires_at,
  external_account_id, external_account_email,
  external_workspace_id, external_workspace_name,
  metadata, created_at, updated_at

UNIQUE (tenant_id, provider) — upsert 는 ON CONFLICT 로 처리.
status CHECK ('connected' | 'disconnected' | 'expired' | 'error').

repository 의 sync 인터페이스(get_integration, upsert_integration, ...)는
db mode 에서도 그대로 동작한다. 내부적으로 asyncpg 코루틴을
``_run_async_blocking`` 으로 실행한다 — 이미 event loop 안에 있으면
별도 thread 에서 새 loop 를 만들어 잠시 블로킹한다. 후처리 액션 정도의
저빈도 호출이라 thread-per-call 비용은 허용 범위 안.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from app.models.tenant_integration import IntegrationStatus, TenantIntegration
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"

_STORAGE_DB = "db"
_STORAGE_FILE = "file"
_STORAGE_MEMORY = "memory"


# ── 직렬화 헬퍼 (file mode) ───────────────────────────────────────────────────

def _to_dict(ti: TenantIntegration) -> dict[str, Any]:
    d = ti.model_dump()
    for k in ("expires_at", "created_at", "updated_at"):
        if d[k] is not None:
            d[k] = d[k].strftime(_DATETIME_FMT)
    d["status"] = d["status"].value if hasattr(d["status"], "value") else d["status"]
    return d


def _from_dict(d: dict[str, Any]) -> TenantIntegration:
    for k in ("expires_at", "created_at", "updated_at"):
        if d.get(k):
            d[k] = datetime.strptime(d[k], _DATETIME_FMT)
    return TenantIntegration(**d)


# ── 파일 I/O ──────────────────────────────────────────────────────────────────

def _file_path() -> Path:
    raw = os.getenv("TENANT_INTEGRATION_FILE_PATH", ".local/tenant_integrations.json")
    return Path(raw)


def _load_file(path: Path) -> dict[str, TenantIntegration]:
    if not path.exists():
        return {}
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return {k: _from_dict(v) for k, v in raw.items()}
    except Exception as exc:
        logger.error("tenant_integration 파일 로드 실패 path=%s err=%s", path, exc)
        return {}


def _save_file(path: Path, store: dict[str, TenantIntegration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        raw = {k: _to_dict(v) for k, v in store.items()}
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("tenant_integration 파일 저장 실패 path=%s err=%s", path, exc)


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
    """
    기본 백엔드: in-memory.
    TENANT_INTEGRATION_STORAGE=file 설정 시 JSON 파일로 persist.
    TENANT_INTEGRATION_STORAGE=db   설정 시 Postgres tenant_integrations 사용.
    """

    def __init__(self, *, storage: str | None = None) -> None:
        raw = (storage or os.getenv("TENANT_INTEGRATION_STORAGE") or _STORAGE_MEMORY).strip().lower()
        if raw == _STORAGE_DB:
            self._storage = _STORAGE_DB
            self._store: dict[str, TenantIntegration] = {}  # 미사용 (db mode)
            logger.info("TenantIntegrationRepo db mode")
        elif raw == _STORAGE_FILE:
            self._storage = _STORAGE_FILE
            self._store = _load_file(_file_path())
            logger.info(
                "TenantIntegrationRepo file mode path=%s loaded=%d",
                _file_path(), len(self._store),
            )
        else:
            if raw != _STORAGE_MEMORY:
                logger.warning(
                    "unknown TENANT_INTEGRATION_STORAGE=%s; falling back to memory", raw,
                )
            self._storage = _STORAGE_MEMORY
            self._store = {}

    # ── 내부 ─────────────────────────────────────────────────────────────────

    def _key(self, tenant_id: str, provider: str) -> str:
        return f"{tenant_id}::{provider}"

    def _persist(self) -> None:
        if self._storage == _STORAGE_FILE:
            _save_file(_file_path(), self._store)

    @property
    def storage(self) -> str:
        return self._storage

    # ── 공개 인터페이스 ────────────────────────────────────────────────────────

    def upsert_integration(self, integration: TenantIntegration) -> TenantIntegration:
        if self._storage == _STORAGE_DB:
            saved = _run_async_blocking(_db_upsert(integration))
            logger.debug(
                "TenantIntegration upserted (db) tenant_id=%s provider=%s status=%s",
                integration.tenant_id, integration.provider, integration.status,
            )
            return saved or integration

        key = self._key(integration.tenant_id, integration.provider)
        integration.updated_at = datetime.utcnow()
        self._store[key] = integration
        self._persist()
        logger.debug(
            "TenantIntegration upserted tenant_id=%s provider=%s status=%s",
            integration.tenant_id, integration.provider, integration.status,
        )
        return integration

    def get_integration(self, tenant_id: str, provider: str) -> TenantIntegration | None:
        if self._storage == _STORAGE_DB:
            return _run_async_blocking(_db_get(tenant_id, provider))
        return self._store.get(self._key(tenant_id, provider))

    def list_integrations(self, tenant_id: str) -> list[TenantIntegration]:
        if self._storage == _STORAGE_DB:
            return _run_async_blocking(_db_list(tenant_id))
        prefix = f"{tenant_id}::"
        return [v for k, v in self._store.items() if k.startswith(prefix)]

    def mark_disconnected(self, tenant_id: str, provider: str) -> bool:
        if self._storage == _STORAGE_DB:
            return _run_async_blocking(_db_mark_disconnected(tenant_id, provider))

        key = self._key(tenant_id, provider)
        integration = self._store.get(key)
        if integration is None:
            return False
        integration.status = IntegrationStatus.disconnected
        integration.updated_at = datetime.utcnow()
        self._persist()
        return True

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
        if self._storage == _STORAGE_DB:
            return _run_async_blocking(
                _db_update_tokens(
                    tenant_id, provider,
                    access_token_encrypted=access_token_encrypted,
                    refresh_token_encrypted=refresh_token_encrypted,
                    expires_at=expires_at,
                    status=status,
                )
            )

        key = self._key(tenant_id, provider)
        integration = self._store.get(key)
        if integration is None:
            return False
        integration.access_token_encrypted = access_token_encrypted
        if refresh_token_encrypted is not None:
            integration.refresh_token_encrypted = refresh_token_encrypted
        if expires_at is not None:
            integration.expires_at = expires_at
        integration.status = status
        integration.updated_at = datetime.utcnow()
        self._persist()
        return True

    def clear_integrations(self) -> None:
        # db mode 에서는 안전을 위해 no-op. 운영 데이터를 절대 일괄 삭제하지 않는다.
        # 테스트는 storage="memory" 또는 "file" 인스턴스를 직접 만들어 격리한다.
        if self._storage == _STORAGE_DB:
            return
        self._store.clear()
        if self._storage == _STORAGE_FILE:
            path = _file_path()
            if path.exists():
                path.write_text("{}", encoding="utf-8")


# ── 모듈 레벨 싱글턴 ──────────────────────────────────────────────────────────
# TENANT_INTEGRATION_STORAGE 환경변수를 읽어 백엔드 결정.
# 테스트에서는 각 테스트가 직접 TenantIntegrationRepository(storage="memory") 인스턴스를
# 생성하거나, clear_integrations()로 격리하면 된다.

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
