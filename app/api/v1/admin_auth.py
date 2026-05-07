from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import (
    create_access_token,
    decode_access_token,
    verify_password,
)
from app.repositories.admin_user_repo import (
    find_admin_user_by_email,
    find_admin_user_by_id,
    update_last_login,
)
from app.schemas.admin_auth import (
    AdminLoginData,
    AdminLoginRequest,
    AdminLoginResponse,
    AdminMeData,
    AdminMeResponse,
    AdminTenantResponse,
    AdminUserResponse,
)
from app.utils.logger import get_logger


logger = get_logger(__name__)
router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)


def _request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def _unauthorized(detail: str = "Invalid credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def _user_response(admin: dict[str, Any]) -> AdminUserResponse:
    user = admin["user"]
    return AdminUserResponse(
        id=user["id"],
        email=user["email"],
        name=user["name"],
        role=user["role"],
    )


def _tenant_response(admin: dict[str, Any]) -> AdminTenantResponse:
    tenant = admin["tenant"]
    return AdminTenantResponse(
        id=tenant["id"],
        name=tenant["name"],
        industry=tenant["industry"],
        plan=tenant["plan"],
        twilio_number=tenant["twilio_number"],
    )


async def get_current_admin_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Bearer token required")

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError:
        raise _unauthorized("Invalid token")

    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    if not user_id or not tenant_id:
        raise _unauthorized("Invalid token payload")

    try:
        admin = await find_admin_user_by_id(str(user_id))
    except Exception as exc:
        logger.warning("admin /me lookup failed user_id=%s err=%s", user_id, exc)
        raise HTTPException(status_code=500, detail="admin lookup failed")

    if admin is None:
        raise _unauthorized("Admin user not found")

    user = admin["user"]
    tenant = admin["tenant"]
    if str(user["tenant_id"]).lower() != str(tenant_id).lower():
        raise _forbidden("Token tenant mismatch")
    if not user["is_active"]:
        raise _forbidden("Admin user is inactive")
    if not tenant["is_active"]:
        raise _forbidden("Tenant is inactive")

    return admin


@router.post("/login", response_model=AdminLoginResponse)
async def login(body: AdminLoginRequest):
    req_id = _request_id()
    try:
        admin = await find_admin_user_by_email(str(body.email))
    except Exception as exc:
        logger.warning("admin login lookup failed email=%s err=%s", body.email, exc)
        raise HTTPException(status_code=500, detail="admin login unavailable")

    if admin is None:
        raise _unauthorized("Invalid email or password")

    user = admin["user"]
    tenant = admin["tenant"]
    if not verify_password(body.password, user["password_hash"]):
        raise _unauthorized("Invalid email or password")
    if not user["is_active"]:
        raise _forbidden("Admin user is inactive")
    if not tenant["is_active"]:
        raise _forbidden("Tenant is inactive")

    access_token = create_access_token(
        user_id=user["id"],
        tenant_id=user["tenant_id"],
        role=user["role"],
        email=user["email"],
    )

    try:
        await update_last_login(user["id"])
    except Exception as exc:
        logger.warning("last_login update failed user_id=%s err=%s", user["id"], exc)

    return AdminLoginResponse(
        request_id=req_id,
        data=AdminLoginData(
            access_token=access_token,
            user=_user_response(admin),
            tenant=_tenant_response(admin),
        ),
    )


@router.get("/me", response_model=AdminMeResponse)
async def me(admin: dict[str, Any] = Depends(get_current_admin_user)):
    return AdminMeResponse(
        request_id=_request_id(),
        data=AdminMeData(
            user=_user_response(admin),
            tenant=_tenant_response(admin),
        ),
    )
