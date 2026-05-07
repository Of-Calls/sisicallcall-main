"""TenantIntegrationRepository — DB mode + storage routing 테스트.

DB mode 는 실제 Postgres 연결을 만들지 않는다. ``asyncpg.connect`` 를 mock 해서
SQL 인자/쿼리 모양만 검증한다. 통합 테스트는 운영 DB 가 살아있는 환경에서
별도로 수행 (수동 검증 명령 참고).
"""
from __future__ import annotations
import re
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.tenant_integration import IntegrationStatus, TenantIntegration  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _row(**overrides) -> dict:
    """tenant_integrations row 의 dict 형태 더미. asyncpg.fetchrow 결과를 흉내.

    asyncpg.Record 는 dict 처럼 ``row["col"]`` 인덱싱이 가능해서 dict 로 충분.
    """
    base = {
        "id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "tid-uuid",
        "provider": "slack",
        "status": "connected",
        "scopes": json.dumps(["chat:write"]),
        "access_token_encrypted": "enc-access",
        "refresh_token_encrypted": None,
        "token_type": "Bearer",
        "expires_at": None,
        "external_account_id": None,
        "external_account_email": "u@co.com",
        "external_workspace_id": None,
        "external_workspace_name": None,
        "metadata": json.dumps({}),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    base.update(overrides)
    return base


def _mock_conn():
    """asyncpg connection 의 async API 를 흉내내는 MagicMock."""
    conn = MagicMock()
    conn.fetchrow = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="UPDATE 1")
    conn.close = AsyncMock()
    return conn


def _patch_asyncpg(conn):
    """``asyncpg.connect`` 를 주어진 conn 으로 대체."""
    return patch(
        "app.repositories.tenant_integration_repo.asyncpg.connect",
        new=AsyncMock(return_value=conn),
    )


# ── 1. storage routing ───────────────────────────────────────────────────────

class TestStorageRouting:
    def test_unknown_storage_falls_back_to_memory(self, monkeypatch):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        repo = TenantIntegrationRepository(storage="totally-unknown")
        assert repo.storage == "memory"

    def test_explicit_db_storage(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        repo = TenantIntegrationRepository(storage="db")
        assert repo.storage == "db"

    def test_env_db_storage(self, monkeypatch):
        monkeypatch.setenv("TENANT_INTEGRATION_STORAGE", "db")
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        repo = TenantIntegrationRepository()
        assert repo.storage == "db"

    def test_clear_integrations_is_noop_in_db_mode(self):
        """db mode 에서 운영 데이터를 일괄 삭제하지 않는다."""
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        repo = TenantIntegrationRepository(storage="db")
        repo.clear_integrations()  # raises 없이 끝나야 함


# ── 2. DB mode upsert ─────────────────────────────────────────────────────────

class TestDbUpsert:
    def test_upsert_calls_insert_on_conflict_with_jsonb_casts(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.fetchrow.return_value = _row(provider="slack", scopes=json.dumps(["chat:write"]))
        repo = TenantIntegrationRepository(storage="db")

        ti = TenantIntegration(
            tenant_id="00000000-0000-0000-0000-000000000010",
            provider="slack",
            status=IntegrationStatus.connected,
            scopes=["chat:write"],
            access_token_encrypted="enc-access",
            refresh_token_encrypted="enc-refresh",
            external_account_email="u@co.com",
            metadata={"workspace_id": "T1"},
        )

        with _patch_asyncpg(conn):
            saved = repo.upsert_integration(ti)

        assert saved is not None
        # 정확한 SQL 모양 검증
        sql_arg = conn.fetchrow.await_args.args[0]
        assert "INSERT INTO tenant_integrations" in sql_arg
        assert "ON CONFLICT (tenant_id, provider) DO UPDATE" in sql_arg
        assert "$1::uuid" in sql_arg
        assert "$4::jsonb" in sql_arg
        assert "$13::jsonb" in sql_arg

        # bound 인자 검증 — 평문 토큰이 아닌 암호화 문자열만 들어가야 함
        bound = conn.fetchrow.await_args.args[1:]
        assert bound[0] == "00000000-0000-0000-0000-000000000010"
        assert bound[1] == "slack"
        assert bound[2] == "connected"
        # scopes 는 jsonb 문자열 (json.dumps)
        assert json.loads(bound[3]) == ["chat:write"]
        assert bound[4] == "enc-access"
        assert bound[5] == "enc-refresh"
        # metadata 는 jsonb 문자열
        assert json.loads(bound[12]) == {"workspace_id": "T1"}

    def test_upsert_updates_status_on_conflict_branch(self):
        """동일 (tenant_id, provider) 재호출 시 ON CONFLICT 가지가 status 를 업데이트."""
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.fetchrow.return_value = _row(provider="slack", status="disconnected")
        repo = TenantIntegrationRepository(storage="db")

        ti = TenantIntegration(
            tenant_id="00000000-0000-0000-0000-000000000020",
            provider="slack",
            status=IntegrationStatus.disconnected,
        )
        with _patch_asyncpg(conn):
            saved = repo.upsert_integration(ti)

        assert saved.status == IntegrationStatus.disconnected
        sql_arg = conn.fetchrow.await_args.args[0]
        assert "status = EXCLUDED.status" in sql_arg
        assert "metadata = EXCLUDED.metadata" in sql_arg


# ── 3. DB mode get / list ─────────────────────────────────────────────────────

class TestDbGetAndList:
    def test_get_filters_on_tenant_and_provider(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.fetchrow.return_value = _row(provider="slack")
        repo = TenantIntegrationRepository(storage="db")

        with _patch_asyncpg(conn):
            integration = repo.get_integration(
                "00000000-0000-0000-0000-000000000030", "slack",
            )

        assert integration is not None
        assert integration.provider == "slack"
        sql_arg = conn.fetchrow.await_args.args[0]
        assert "WHERE tenant_id = $1::uuid AND provider = $2" in sql_arg
        bound = conn.fetchrow.await_args.args[1:]
        assert bound == ("00000000-0000-0000-0000-000000000030", "slack")

    def test_get_returns_none_when_no_row(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.fetchrow.return_value = None
        repo = TenantIntegrationRepository(storage="db")

        with _patch_asyncpg(conn):
            integration = repo.get_integration("tid-x", "slack")

        assert integration is None

    def test_list_uses_tenant_only_filter(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.fetch.return_value = [
            _row(provider="slack"),
            _row(provider="google_gmail"),
        ]
        repo = TenantIntegrationRepository(storage="db")

        with _patch_asyncpg(conn):
            rows = repo.list_integrations("00000000-0000-0000-0000-000000000040")

        assert {r.provider for r in rows} == {"slack", "google_gmail"}
        sql_arg = conn.fetch.await_args.args[0]
        assert "WHERE tenant_id = $1::uuid" in sql_arg
        # provider 조건이 들어가면 안 된다
        assert "AND provider" not in sql_arg


# ── 4. DB mode mark_disconnected ──────────────────────────────────────────────

class TestDbMarkDisconnected:
    def test_disconnect_returns_true_when_row_updated(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.execute.return_value = "UPDATE 1"
        repo = TenantIntegrationRepository(storage="db")

        with _patch_asyncpg(conn):
            ok = repo.mark_disconnected(
                "00000000-0000-0000-0000-000000000050", "slack",
            )

        assert ok is True
        sql_arg = conn.execute.await_args.args[0]
        assert "UPDATE tenant_integrations" in sql_arg
        assert "SET status = 'disconnected'" in sql_arg
        assert "WHERE tenant_id = $1::uuid AND provider = $2" in sql_arg

    def test_disconnect_returns_false_when_no_row(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.execute.return_value = "UPDATE 0"
        repo = TenantIntegrationRepository(storage="db")

        with _patch_asyncpg(conn):
            ok = repo.mark_disconnected("tid-missing", "slack")

        assert ok is False


# ── 5. file mode 기존 동작 보존 ────────────────────────────────────────────────

class TestFileModeStillWorks:
    def test_file_mode_loads_existing_file(self, tmp_path, monkeypatch):
        from app.models.tenant_integration import TenantIntegration as _TI
        from app.repositories.tenant_integration_repo import (
            TenantIntegrationRepository,
            _to_dict,
        )

        path = tmp_path / "ti.json"
        ti = _TI(
            tenant_id="tid-file", provider="slack",
            access_token_encrypted="enc-x",
        )
        path.write_text(json.dumps({"tid-file::slack": _to_dict(ti)}), encoding="utf-8")

        monkeypatch.setenv("TENANT_INTEGRATION_FILE_PATH", str(path))
        repo = TenantIntegrationRepository(storage="file")

        result = repo.get_integration("tid-file", "slack")
        assert result is not None
        assert result.access_token_encrypted == "enc-x"


# ── 6. logical field <-> DB column mapping ───────────────────────────────────

class TestLogicalFieldMapping:
    """현재 DB 컬럼명 (scopes / expires_at / external_account_email / metadata)
    그대로 SELECT/INSERT 에 들어가는지 검증한다. 잘못된 alias (granted_scopes,
    token_expires_at, account_email, config) 가 SQL 에 끼어들면 안 된다."""

    def test_select_uses_actual_column_names(self):
        from app.repositories.tenant_integration_repo import _DB_COLUMNS

        assert "scopes" in _DB_COLUMNS
        assert "expires_at" in _DB_COLUMNS
        assert "external_account_email" in _DB_COLUMNS
        assert "metadata" in _DB_COLUMNS

        # 잘못된 alias 가 SELECT 에 들어가면 안 된다
        assert "granted_scopes" not in _DB_COLUMNS
        assert "token_expires_at" not in _DB_COLUMNS
        columns = [c.strip() for c in _DB_COLUMNS.split(",")]
        assert "account_email" not in columns
        # `config` 단독으로 등장하면 안 된다 (단어 경계 검사)
        for token in _DB_COLUMNS.replace(",", " ").split():
            assert token.strip() != "config"

    def test_upsert_writes_actual_column_names(self):
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        conn = _mock_conn()
        conn.fetchrow.return_value = _row()
        repo = TenantIntegrationRepository(storage="db")

        ti = TenantIntegration(
            tenant_id="00000000-0000-0000-0000-000000000099",
            provider="google_gmail",
            scopes=["https://www.googleapis.com/auth/gmail.send"],
            external_account_email="u@co.com",
            metadata={"workspace_name": "ACME"},
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )

        with _patch_asyncpg(conn):
            repo.upsert_integration(ti)

        sql_arg = conn.fetchrow.await_args.args[0]
        # 실제 컬럼명이 들어가야 한다
        assert "scopes" in sql_arg
        assert "expires_at" in sql_arg
        assert "external_account_email" in sql_arg
        assert "metadata" in sql_arg
        # 잘못된 컬럼명은 절대 안 들어간다
        assert "granted_scopes" not in sql_arg
        assert "token_expires_at" not in sql_arg
        assert not re.search(r"(?<!external_)account_email\b", sql_arg)
        # `config` 라는 단독 컬럼은 없어야 한다
        # (substring 검사가 아니라 토큰 단위. _DB_COLUMNS 검사로 충분)


# ── 7. Repository singleton routing ──────────────────────────────────────────

class TestSingletonRouting:
    def test_module_singleton_respects_env_at_construction(self, monkeypatch):
        """import 된 싱글턴은 import 시점의 env 를 기준으로 storage 를 결정한다.

        새 storage 모드를 적용하려면 ``TenantIntegrationRepository(storage=...)`` 로
        직접 인스턴스를 만들어 사용한다."""
        from app.repositories.tenant_integration_repo import TenantIntegrationRepository

        monkeypatch.setenv("TENANT_INTEGRATION_STORAGE", "db")
        new_repo = TenantIntegrationRepository()
        assert new_repo.storage == "db"

        monkeypatch.setenv("TENANT_INTEGRATION_STORAGE", "memory")
        mem_repo = TenantIntegrationRepository()
        assert mem_repo.storage == "memory"
