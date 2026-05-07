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


def test_dashboard_recent_calls_without_token_returns_401():
    resp = _client().get("/dashboard/recent-calls")

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


def test_dashboard_recent_calls_uses_jwt_tenant(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_fetch_dashboard_recent_calls(**kwargs):
        captured.append(kwargs)
        return {
            "items": [
                {
                    "id": "call-1",
                    "caller_number": "+821012345678",
                    "status": "completed",
                    "started_at": "2026-05-05T01:00:00+00:00",
                    "duration_sec": 180,
                    "summary_short": "summary",
                    "customer_emotion": "negative",
                    "resolution_status": "escalated",
                    "priority": "high",
                }
            ],
            "total": 1,
            "offset": 2,
            "limit": 5,
        }

    monkeypatch.setattr(dashboard, "fetch_dashboard_recent_calls", fake_fetch_dashboard_recent_calls)

    resp = _client().get(
        "/dashboard/recent-calls?offset=2&limit=5",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert captured[0]["tenant_id"] == TENANT_ID
    assert captured[0]["offset"] == 2
    assert captured[0]["limit"] == 5
    assert resp.json()["data"]["items"][0]["priority"] == "high"


def test_dashboard_intent_distribution_uses_jwt_tenant(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_fetch_dashboard_intent_distribution(**kwargs):
        captured.append(kwargs)
        return [{"label": "예약/일정", "count": 12}]

    monkeypatch.setattr(
        dashboard,
        "fetch_dashboard_intent_distribution",
        fake_fetch_dashboard_intent_distribution,
    )

    resp = _client().get("/dashboard/intent-distribution?limit=7", headers=_auth_headers())

    assert resp.status_code == 200
    assert captured[0]["tenant_id"] == TENANT_ID
    assert captured[0]["limit"] == 7
    assert resp.json()["data"] == [{"label": "예약/일정", "count": 12}]


def test_dashboard_emotion_distribution_prefers_db_shape(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_fetch_dashboard_emotion_distribution_counts(**kwargs):
        captured.append(kwargs)
        return {"positive": 3, "neutral": 12, "negative": 4, "angry": 1}

    async def fake_get_emotion_distribution(**kwargs):
        raise AssertionError("legacy in-memory fallback should not be used")

    monkeypatch.setattr(
        dashboard,
        "fetch_dashboard_emotion_distribution_counts",
        fake_fetch_dashboard_emotion_distribution_counts,
    )
    monkeypatch.setattr(dashboard, "get_emotion_distribution", fake_get_emotion_distribution)

    resp = _client().get("/dashboard/emotion-distribution", headers=_auth_headers())

    assert resp.status_code == 200
    assert captured[0]["tenant_id"] == TENANT_ID
    assert resp.json() == {"positive": 3, "neutral": 12, "negative": 4, "angry": 1}
