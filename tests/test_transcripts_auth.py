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
from app.repositories import transcript_repo


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


def test_transcripts_without_token_returns_401():
    resp = _client().get(f"/call/{CALL_ID}/transcripts")

    assert resp.status_code == 401


def test_transcripts_with_valid_token_returns_200(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[tuple[str, str]] = []

    async def fake_get_transcripts_by_call_id(call_id: str, tenant_id: str):
        captured.append((call_id, tenant_id))
        return [
            {
                "id": "55555555-5555-5555-5555-555555555555",
                "call_id": call_id,
                "turn_index": 0,
                "speaker": "customer",
                "text": "hello",
                "response_path": None,
                "reviewer_applied": False,
                "reviewer_verdict": None,
                "is_barge_in": False,
                "spoken_at": "2026-05-05T00:00:00+00:00",
            }
        ]

    monkeypatch.setattr(call_history, "get_transcripts_by_call_id", fake_get_transcripts_by_call_id)

    resp = _client().get(f"/call/{CALL_ID}/transcripts", headers=_auth_headers())

    assert resp.status_code == 200
    assert captured == [(CALL_ID, TENANT_ID)]
    assert resp.json()["data"]["total"] == 1


def test_transcripts_other_tenant_call_returns_404(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    async def fake_get_transcripts_by_call_id(call_id: str, tenant_id: str):
        assert call_id == CALL_ID
        assert tenant_id == TENANT_ID
        return None

    monkeypatch.setattr(call_history, "get_transcripts_by_call_id", fake_get_transcripts_by_call_id)

    resp = _client().get(f"/call/{CALL_ID}/transcripts", headers=_auth_headers())

    assert resp.status_code == 404


class FakeRecord(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class FakeConn:
    def __init__(self, *, rows=None, exists=True):
        self.rows = rows or []
        self.exists = exists
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.fetchval_calls: list[tuple[str, tuple]] = []

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return self.rows

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return self.exists


class FakeAcquire:
    def __init__(self, conn: FakeConn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn: FakeConn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)


@pytest.mark.asyncio
async def test_repository_uses_calls_join_tenant_condition(monkeypatch):
    spoken_at = datetime(2026, 5, 5, tzinfo=timezone.utc)
    conn = FakeConn(
        rows=[
            FakeRecord(
                {
                    "id": "55555555-5555-5555-5555-555555555555",
                    "call_id": CALL_ID,
                    "turn_index": 0,
                    "speaker": "customer",
                    "text": "hello",
                    "response_path": None,
                    "reviewer_applied": False,
                    "reviewer_verdict": None,
                    "is_barge_in": False,
                    "spoken_at": spoken_at,
                }
            )
        ]
    )

    async def fake_get_pool():
        return FakePool(conn)

    monkeypatch.setattr(transcript_repo, "_get_pool", fake_get_pool)

    result = await transcript_repo.get_transcripts_by_call_id(CALL_ID, TENANT_ID)

    query, args = conn.fetch_calls[0]
    assert "JOIN calls c ON c.id = t.call_id" in query
    assert "c.tenant_id = $2::uuid" in query
    assert args == (CALL_ID, TENANT_ID)
    assert result is not None
    assert result[0]["spoken_at"] == spoken_at.isoformat()


def test_admin_transcript_lookup_requires_tenant_id_argument():
    signature = inspect.signature(transcript_repo.get_transcripts_by_call_id)

    assert "tenant_id" in signature.parameters
    assert signature.parameters["tenant_id"].default is inspect.Signature.empty
