from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import admin_auth, dashboard
from app.core.security import create_access_token


USER_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "tenant-a"
OTHER_TENANT_ID = "tenant-b"
EMAIL = "admin@example.test"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(dashboard.router, prefix="/dashboard")
    return TestClient(app)


def _token(tenant_id: str = TENANT_ID) -> str:
    return create_access_token(
        user_id=USER_ID,
        tenant_id=tenant_id,
        role="owner",
        email=EMAIL,
    )


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


def _auth_headers(token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token or _token()}"}


def _patch_admin_lookup(monkeypatch, tenant_id: str = TENANT_ID) -> None:
    async def fake_find_admin_user_by_id(user_id: str):
        assert user_id == USER_ID
        return _admin_context(tenant_id)

    monkeypatch.setattr(admin_auth, "find_admin_user_by_id", fake_find_admin_user_by_id)


def test_dashboard_without_token_returns_401():
    resp = _client().get("/dashboard/stats")

    assert resp.status_code == 401


def test_dashboard_with_valid_token_returns_200(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    async def fake_get_dashboard_overview(**kwargs):
        return {"tenant_id": kwargs["tenant_id"], "total_calls": 0}

    monkeypatch.setattr(dashboard, "get_dashboard_overview", fake_get_dashboard_overview)

    resp = _client().get("/dashboard/stats", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == TENANT_ID


def test_dashboard_query_tenant_equal_jwt_tenant_returns_200(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    async def fake_get_dashboard_overview(**kwargs):
        return {"tenant_id": kwargs["tenant_id"], "total_calls": 0}

    monkeypatch.setattr(dashboard, "get_dashboard_overview", fake_get_dashboard_overview)

    resp = _client().get(
        f"/dashboard/stats?tenant_id={TENANT_ID}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == TENANT_ID


def test_dashboard_query_tenant_mismatch_returns_403(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    resp = _client().get(
        f"/dashboard/stats?tenant_id={OTHER_TENANT_ID}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 403
    assert resp.json()["detail"] == "tenant 정보가 일치하지 않습니다."


def test_dashboard_uses_jwt_tenant_not_query_tenant(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[str] = []

    async def fake_get_dashboard_overview(**kwargs):
        captured.append(kwargs["tenant_id"])
        return {"tenant_id": kwargs["tenant_id"], "total_calls": 0}

    monkeypatch.setattr(dashboard, "get_dashboard_overview", fake_get_dashboard_overview)

    resp = _client().get(
        f"/dashboard/stats?tenant_id={TENANT_ID.upper()}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert captured == [TENANT_ID]
