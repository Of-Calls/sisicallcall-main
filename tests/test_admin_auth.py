from __future__ import annotations

from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import admin_auth
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


USER_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "22222222-2222-2222-2222-222222222222"
EMAIL = "admin@hanbat.test"
PASSWORD = "password1234"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(admin_auth.router, prefix="/auth")
    return TestClient(app)


def _admin_context(password_hash: str | None = None) -> dict:
    return {
        "user": {
            "id": USER_ID,
            "tenant_id": TENANT_ID,
            "email": EMAIL,
            "password_hash": password_hash or hash_password(PASSWORD),
            "name": "Hanbat Admin",
            "role": "owner",
            "is_active": True,
            "last_login_at": None,
        },
        "tenant": {
            "id": TENANT_ID,
            "name": "Hanbat",
            "industry": "restaurant",
            "plan": "basic",
            "twilio_number": "+821000000002",
            "is_active": True,
        },
    }


def test_password_hash_and_verify():
    password_hash = hash_password(PASSWORD)

    assert password_hash != PASSWORD
    assert verify_password(PASSWORD, password_hash) is True
    assert verify_password("wrong-password", password_hash) is False


def test_jwt_create_and_decode():
    token = create_access_token(
        user_id=USER_ID,
        tenant_id=TENANT_ID,
        role="owner",
        email=EMAIL,
    )

    payload = decode_access_token(token)
    assert payload["sub"] == USER_ID
    assert payload["tenant_id"] == TENANT_ID
    assert payload["role"] == "owner"
    assert payload["email"] == EMAIL
    assert "iat" in payload
    assert "exp" in payload


def test_jwt_invalid_token_fails():
    with pytest.raises(ValueError):
        decode_access_token("not-a-valid-token")


def test_login_success(monkeypatch):
    ctx = _admin_context()
    updated: list[str] = []

    async def fake_find_by_email(email: str):
        assert email == EMAIL
        return ctx

    async def fake_update_last_login(user_id: str):
        updated.append(user_id)

    monkeypatch.setattr(admin_auth, "find_admin_user_by_email", fake_find_by_email)
    monkeypatch.setattr(admin_auth, "update_last_login", fake_update_last_login)

    resp = _client().post(
        "/auth/login",
        json={"email": EMAIL, "password": PASSWORD},
    )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["token_type"] == "bearer"
    assert data["access_token"]
    assert data["user"]["id"] == str(UUID(USER_ID))
    assert data["user"]["email"] == EMAIL
    assert data["tenant"]["id"] == str(UUID(TENANT_ID))
    assert updated == [USER_ID]


def test_login_wrong_password_returns_401(monkeypatch):
    async def fake_find_by_email(email: str):
        return _admin_context()

    monkeypatch.setattr(admin_auth, "find_admin_user_by_email", fake_find_by_email)

    resp = _client().post(
        "/auth/login",
        json={"email": EMAIL, "password": "wrong-password"},
    )

    assert resp.status_code == 401


def test_me_success(monkeypatch):
    ctx = _admin_context()
    token = create_access_token(
        user_id=USER_ID,
        tenant_id=TENANT_ID,
        role="owner",
        email=EMAIL,
    )

    async def fake_find_by_id(user_id: str):
        assert user_id == USER_ID
        return ctx

    monkeypatch.setattr(admin_auth, "find_admin_user_by_id", fake_find_by_id)

    resp = _client().get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["user"]["email"] == EMAIL
    assert data["tenant"]["id"] == str(UUID(TENANT_ID))
