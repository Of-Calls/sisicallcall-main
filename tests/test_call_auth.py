from __future__ import annotations

import inspect
import sys
import types
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

graph_stub = types.ModuleType("app.agents.conversational.graph")


class DummyGraph:
    async def ainvoke(self, state):
        return {}


def build_call_graph():
    return DummyGraph()


graph_stub.build_call_graph = build_call_graph
sys.modules.setdefault("app.agents.conversational.graph", graph_stub)

enrollment_stub = types.ModuleType("app.services.speaker_verify.enrollment")


async def accumulate(call_id: str, audio_chunk: bytes, transcript: str) -> bool:
    return False


def cleanup(call_id: str) -> None:
    return None


enrollment_stub.accumulate = accumulate
enrollment_stub.cleanup = cleanup
sys.modules.setdefault("app.services.speaker_verify.enrollment", enrollment_stub)

titanet_stub = types.ModuleType("app.services.speaker_verify.titanet")


class DummyTitaNetService:
    async def verify(self, audio_chunk: bytes, call_id: str):
        return False, 0.0

    def cleanup(self, call_id: str) -> None:
        return None


def get_titanet_service():
    return DummyTitaNetService()


titanet_stub.get_titanet_service = get_titanet_service
sys.modules.setdefault("app.services.speaker_verify.titanet", titanet_stub)

silero_stub = types.ModuleType("app.services.vad.silero_vad")


class SileroVADService:
    async def detect(self, audio_chunk: bytes) -> bool:
        return False


silero_stub.SileroVADService = SileroVADService
sys.modules.setdefault("app.services.vad.silero_vad", silero_stub)

from app.api.v1 import admin_auth, call_history
from app.core.security import create_access_token
from app.repositories import call_repo


USER_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "22222222-2222-2222-2222-222222222222"
OTHER_TENANT_ID = "33333333-3333-3333-3333-333333333333"
CALL_ID = "44444444-4444-4444-4444-444444444444"
EMAIL = "admin@example.test"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(call_history.router, prefix="/call")
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


def _call_payload(call_id: str = CALL_ID, tenant_id: str = TENANT_ID) -> dict:
    return {
        "id": call_id,
        "tenant_id": tenant_id,
        "twilio_call_sid": "CA123",
        "caller_number": "+821012345678",
        "status": "completed",
        "started_at": "2026-05-05T01:00:00+00:00",
        "ended_at": "2026-05-05T01:03:00+00:00",
        "duration_sec": 180,
        "latency_log": {},
        "branch_stats": {},
        "created_at": "2026-05-05T01:00:00+00:00",
    }


def test_list_calls_without_token_returns_401():
    resp = _client().get("/call")

    assert resp.status_code == 401


def test_list_calls_with_valid_token_returns_200_and_uses_jwt_tenant(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_list_calls_for_tenant(**kwargs):
        captured.append(kwargs)
        return {"items": [_call_payload()], "total": 1}

    monkeypatch.setattr(call_history, "list_calls_for_tenant", fake_list_calls_for_tenant)

    resp = _client().get(
        "/call?status=completed&offset=5&limit=200",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert captured[0]["tenant_id"] == TENANT_ID
    assert captured[0]["status"] == "completed"
    assert captured[0]["offset"] == 5
    assert captured[0]["limit"] == 100
    assert resp.json()["data"]["limit"] == 100


def test_list_calls_status_all_omits_status_filter(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_list_calls_for_tenant(**kwargs):
        captured.append(kwargs)
        return {"items": [_call_payload()], "total": 1}

    monkeypatch.setattr(call_history, "list_calls_for_tenant", fake_list_calls_for_tenant)

    resp = _client().get("/call?status=all", headers=_auth_headers())

    assert resp.status_code == 200
    assert captured[0]["tenant_id"] == TENANT_ID
    assert captured[0]["status"] is None


def test_list_calls_query_tenant_mismatch_returns_403(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    resp = _client().get(
        f"/call?tenant_id={OTHER_TENANT_ID}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 403


def test_get_call_detail_without_token_returns_401():
    resp = _client().get(f"/call/{CALL_ID}")

    assert resp.status_code == 401


def test_get_call_detail_with_valid_token_returns_200(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[tuple[str, str]] = []

    async def fake_get_call_by_id_for_tenant(call_id: str, tenant_id: str):
        captured.append((call_id, tenant_id))
        return _call_payload(call_id=call_id, tenant_id=tenant_id)

    monkeypatch.setattr(call_history, "get_call_by_id_for_tenant", fake_get_call_by_id_for_tenant)

    resp = _client().get(f"/call/{CALL_ID}", headers=_auth_headers())

    assert resp.status_code == 200
    assert captured == [(CALL_ID, TENANT_ID)]
    assert resp.json()["data"]["id"] == CALL_ID


def test_get_call_detail_other_tenant_returns_404(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    async def fake_get_call_by_id_for_tenant(call_id: str, tenant_id: str):
        assert call_id == CALL_ID
        assert tenant_id == TENANT_ID
        return None

    monkeypatch.setattr(call_history, "get_call_by_id_for_tenant", fake_get_call_by_id_for_tenant)

    resp = _client().get(f"/call/{CALL_ID}", headers=_auth_headers())

    assert resp.status_code == 404


class FakeRecord(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


def _record() -> FakeRecord:
    now = datetime(2026, 5, 5, 1, 0, tzinfo=timezone.utc)
    return FakeRecord(
        {
            "id": CALL_ID,
            "tenant_id": TENANT_ID,
            "twilio_call_sid": "CA123",
            "caller_number": "+821012345678",
            "status": "completed",
            "started_at": now,
            "ended_at": now,
            "duration_sec": 180,
            "latency_log": {"stt": 120},
            "branch_stats": {"faq": 1},
            "created_at": now,
        }
    )


class FakeConn:
    def __init__(self):
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetchval_calls: list[tuple[str, tuple]] = []
        self.closed = False

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return [_record()]

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        return _record()

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return 1

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_list_calls_repository_uses_tenant_condition_and_filters(monkeypatch):
    conn = FakeConn()

    async def fake_connect(dsn):
        return conn

    monkeypatch.setattr(call_repo.asyncpg, "connect", fake_connect)

    result = await call_repo.list_calls_for_tenant(
        tenant_id=TENANT_ID,
        status="completed",
        started_from=datetime(2026, 5, 1, tzinfo=timezone.utc),
        started_to=datetime(2026, 5, 6, tzinfo=timezone.utc),
        offset=10,
        limit=200,
    )

    count_query, count_args = conn.fetchval_calls[0]
    list_query, list_args = conn.fetch_calls[0]
    assert "WHERE tenant_id = $1::uuid" in count_query
    assert "status = $2" in list_query
    assert "started_at >= $3" in list_query
    assert "started_at <= $4" in list_query
    assert "OFFSET $5" in list_query
    assert "LIMIT $6" in list_query
    assert count_args[0] == TENANT_ID
    assert list_args[-2:] == (10, 100)
    assert result["total"] == 1
    assert result["items"][0]["tenant_id"] == TENANT_ID


@pytest.mark.asyncio
async def test_get_call_detail_repository_uses_id_and_tenant_condition(monkeypatch):
    conn = FakeConn()

    async def fake_connect(dsn):
        return conn

    monkeypatch.setattr(call_repo.asyncpg, "connect", fake_connect)

    result = await call_repo.get_call_by_id_for_tenant(CALL_ID, TENANT_ID)

    query, args = conn.fetchrow_calls[0]
    assert "WHERE id = $1::uuid" in query
    assert "AND tenant_id = $2::uuid" in query
    assert args == (CALL_ID, TENANT_ID)
    assert result is not None
    assert result["id"] == CALL_ID


def test_call_detail_lookup_requires_tenant_id_argument():
    signature = inspect.signature(call_repo.get_call_by_id_for_tenant)

    assert "tenant_id" in signature.parameters
    assert signature.parameters["tenant_id"].default is inspect.Signature.empty
