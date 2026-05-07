from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.repositories.call_summary_repo as summary_repo
from app.api.v1 import admin_auth, summary
from app.core.security import create_access_token


USER_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "tenant-a"
OTHER_TENANT_ID = "tenant-b"
DB_TENANT_ID = "22222222-2222-2222-2222-222222222222"
DB_CALL_ID = "44444444-4444-4444-4444-444444444444"
EMAIL = "admin@example.test"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(summary.router, prefix="/summary")
    return TestClient(app)


def _token(tenant_id: str = TENANT_ID) -> str:
    return create_access_token(
        user_id=USER_ID,
        tenant_id=tenant_id,
        role="owner",
        email=EMAIL,
    )


def _auth_headers(token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token or _token()}"}


def _admin_context(tenant_id: str = TENANT_ID) -> dict:
    return {
        "user": {
            "id": USER_ID,
            "tenant_id": tenant_id,
            "email": EMAIL,
            "name": "Test Admin",
            "role": "owner",
            "is_active": True,
            "password_hash": "unused",
            "last_login_at": None,
        },
        "tenant": {
            "id": tenant_id,
            "name": "Test Tenant",
            "industry": "test",
            "plan": "basic",
            "twilio_number": "+821000000000",
            "is_active": True,
        },
    }


def _patch_admin_lookup(monkeypatch, tenant_id: str = TENANT_ID) -> None:
    async def fake_find_admin_user_by_id(user_id: str):
        assert user_id == USER_ID
        return _admin_context(tenant_id)

    monkeypatch.setattr(admin_auth, "find_admin_user_by_id", fake_find_admin_user_by_id)


async def _seed_summary(call_id: str, tenant_id: str, summary_data: dict) -> None:
    await summary_repo.save_summary(
        call_id=call_id,
        tenant_id=tenant_id,
        summary=copy.deepcopy(summary_data),
    )


@pytest.fixture(autouse=True)
def reset_summary_store():
    summary_repo._reset()
    yield
    summary_repo._reset()


def test_summary_without_token_returns_401():
    resp = _client().get("/summary/call-001")

    assert resp.status_code == 401


def test_summary_with_valid_token_returns_200(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    call_id = "call-tenant-a-001"
    summary_data = {
        "summary_short": "tenant a summary",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
    }
    import anyio

    anyio.run(_seed_summary, call_id, TENANT_ID, summary_data)

    resp = _client().get(
        f"/summary/{call_id}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == call_id
    assert data["tenant_id"] == TENANT_ID
    assert data["summary"]["summary_short"] == "tenant a summary"


def test_summary_other_tenant_returns_404(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    call_id = "call-tenant-b-001"
    import anyio

    anyio.run(
        _seed_summary,
        call_id,
        OTHER_TENANT_ID,
        {"summary_short": "other tenant summary"},
    )

    resp = _client().get(
        f"/summary/{call_id}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 404


def test_summary_uses_jwt_tenant_not_query_tenant(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[tuple[str, str | None]] = []

    async def fake_get_summary_by_call_id(call_id: str, tenant_id: str | None = None):
        captured.append((call_id, tenant_id))
        return {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "summary": {"summary_short": "jwt tenant"},
            "created_at": "2026-05-04T00:00:00Z",
            "updated_at": "2026-05-04T00:00:00Z",
        }

    monkeypatch.setattr(summary, "get_summary_by_call_id", fake_get_summary_by_call_id)

    resp = _client().get(
        f"/summary/call-001?tenant_id={OTHER_TENANT_ID}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert captured == [("call-001", TENANT_ID)]


class FakeSummaryRecord(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class FakeSummaryConn:
    def __init__(self, row):
        self.row = row
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.closed = False

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        return self.row

    async def close(self):
        self.closed = True


def _db_summary_record() -> FakeSummaryRecord:
    now = datetime(2026, 5, 5, 1, 0, tzinfo=timezone.utc)
    return FakeSummaryRecord(
        {
            "call_id": DB_CALL_ID,
            "tenant_id": DB_TENANT_ID,
            "summary_short": "db summary",
            "summary_detailed": "db detailed summary",
            "customer_intent": "reservation",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "keywords": ["reservation", "time"],
            "handoff_notes": "none",
            "generation_mode": "async",
            "model_used": "demo-mock-llm",
            "created_at": now,
            "updated_at": now,
        }
    )


def test_summary_db_fallback_returns_record_when_memory_missing(monkeypatch):
    import anyio

    conn = FakeSummaryConn(_db_summary_record())

    async def fake_connect(dsn):
        return conn

    monkeypatch.setattr(summary_repo.asyncpg, "connect", fake_connect)

    result = anyio.run(
        summary_repo.get_summary_by_call_id,
        DB_CALL_ID,
        DB_TENANT_ID,
    )

    query, args = conn.fetchrow_calls[0]
    assert "FROM call_summaries" in query
    assert "call_id = $1::uuid" in query
    assert "tenant_id = $2::uuid" in query
    assert args == (DB_CALL_ID, DB_TENANT_ID)
    assert conn.closed is True
    assert result is not None
    assert result["call_id"] == DB_CALL_ID
    assert result["tenant_id"] == DB_TENANT_ID
    assert result["summary"]["summary_short"] == "db summary"
    assert result["summary"]["keywords"] == ["reservation", "time"]


def test_summary_endpoint_uses_db_fallback_with_jwt_tenant(monkeypatch):
    _patch_admin_lookup(monkeypatch, tenant_id=DB_TENANT_ID)
    conn = FakeSummaryConn(_db_summary_record())

    async def fake_connect(dsn):
        return conn

    monkeypatch.setattr(summary_repo.asyncpg, "connect", fake_connect)

    resp = _client().get(f"/summary/{DB_CALL_ID}", headers=_auth_headers(_token(DB_TENANT_ID)))

    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == DB_CALL_ID
    assert data["tenant_id"] == DB_TENANT_ID
    assert data["summary"]["summary_short"] == "db summary"


def test_summary_db_fallback_missing_record_returns_none(monkeypatch):
    import anyio

    conn = FakeSummaryConn(None)

    async def fake_connect(dsn):
        return conn

    monkeypatch.setattr(summary_repo.asyncpg, "connect", fake_connect)

    result = anyio.run(
        summary_repo.get_summary_by_call_id,
        DB_CALL_ID,
        DB_TENANT_ID,
    )

    query, args = conn.fetchrow_calls[0]
    assert "tenant_id = $2::uuid" in query
    assert args == (DB_CALL_ID, DB_TENANT_ID)
    assert result is None
