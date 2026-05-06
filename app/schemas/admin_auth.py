from __future__ import annotations

from typing import Literal
from uuid import UUID

from email_validator import EmailNotValidError, validate_email
from pydantic import BaseModel, field_validator


AdminRole = Literal["owner", "admin", "staff", "manager", "agent"]


class AdminLoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_admin_email(cls, value: str) -> str:
        return _normalize_admin_email(value)


class AdminUserResponse(BaseModel):
    id: UUID
    email: str
    name: str
    role: AdminRole

    @field_validator("email")
    @classmethod
    def validate_admin_email(cls, value: str) -> str:
        return _normalize_admin_email(value)


class AdminTenantResponse(BaseModel):
    id: UUID
    name: str
    industry: str | None
    plan: str
    twilio_number: str


class AdminLoginData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AdminUserResponse
    tenant: AdminTenantResponse


class AdminLoginResponse(BaseModel):
    data: AdminLoginData
    request_id: str


class AdminMeData(BaseModel):
    user: AdminUserResponse
    tenant: AdminTenantResponse


class AdminMeResponse(BaseModel):
    data: AdminMeData
    request_id: str


def _normalize_admin_email(value: str) -> str:
    try:
        result = validate_email(
            value,
            check_deliverability=False,
            test_environment=True,
        )
    except EmailNotValidError as exc:
        raise ValueError(str(exc)) from exc
    return result.normalized
