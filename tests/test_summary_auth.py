from __future__ import annotations

import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.repositories.call_summary_repo as summary_repo
from app.api.v1 import admin_auth, summary
from app.core.security import create_access_token


USER_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "tenant-a"
OTHER_TENANT_ID = "tenant-b"
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
