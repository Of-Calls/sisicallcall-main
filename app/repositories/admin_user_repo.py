from __future__ import annotations

from typing import Any

import asyncpg

from app.utils.config import settings


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _row_to_admin_context(row: asyncpg.Record | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "user": {
            "id": str(row["user_id"]),
            "tenant_id": str(row["user_tenant_id"]),
            "email": row["email"],
            "password_hash": row["password_hash"],
            "name": row["user_name"],
            "role": row["role"],
            "is_active": bool(row["user_is_active"]),
            "last_login_at": row["last_login_at"],
        },
        "tenant": {
            "id": str(row["tenant_id"]),
            "name": row["tenant_name"],
            "industry": row["industry"],
            "plan": row["plan"],
            "twilio_number": row["twilio_number"],
            "is_active": bool(row["tenant_is_active"]),
        },
    }


async def find_admin_user_by_email(email: str) -> dict[str, Any] | None:
    normalized = _normalize_email(email)
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            """
            SELECT
                au.id AS user_id,
                au.tenant_id AS user_tenant_id,
                au.email,
                au.password_hash,
                au.name AS user_name,
                au.role,
                au.is_active AS user_is_active,
                au.last_login_at,
                t.id AS tenant_id,
                t.name AS tenant_name,
                t.industry,
                t.plan,
                t.twilio_number,
                t.is_active AS tenant_is_active
            FROM admin_users au
            JOIN tenants t ON t.id = au.tenant_id
            WHERE LOWER(au.email) = $1
            LIMIT 1
            """,
            normalized,
        )
        return _row_to_admin_context(row)
    finally:
        await conn.close()


async def find_admin_user_by_id(user_id: str) -> dict[str, Any] | None:
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            """
            SELECT
                au.id AS user_id,
                au.tenant_id AS user_tenant_id,
                au.email,
                au.password_hash,
                au.name AS user_name,
                au.role,
                au.is_active AS user_is_active,
                au.last_login_at,
                t.id AS tenant_id,
                t.name AS tenant_name,
                t.industry,
                t.plan,
                t.twilio_number,
                t.is_active AS tenant_is_active
            FROM admin_users au
            JOIN tenants t ON t.id = au.tenant_id
            WHERE au.id = $1::uuid
            LIMIT 1
            """,
            user_id,
        )
        return _row_to_admin_context(row)
    finally:
        await conn.close()


async def update_last_login(user_id: str) -> None:
    conn = await asyncpg.connect(_database_url())
    try:
        await conn.execute(
            """
            UPDATE admin_users
            SET last_login_at = now(), updated_at = now()
            WHERE id = $1::uuid
            """,
            user_id,
        )
    finally:
        await conn.close()


async def create_admin_user(
    *,
    tenant_id: str,
    email: str,
    password_hash: str,
    name: str,
    role: str = "admin",
    is_active: bool = True,
) -> str:
    normalized = _normalize_email(email)
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO admin_users (
                tenant_id, email, password_hash, name, role, is_active
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            ON CONFLICT (email) DO UPDATE
                SET tenant_id = EXCLUDED.tenant_id,
                    password_hash = EXCLUDED.password_hash,
                    name = EXCLUDED.name,
                    role = EXCLUDED.role,
                    is_active = EXCLUDED.is_active,
                    updated_at = now()
            RETURNING id
            """,
            tenant_id,
            normalized,
            password_hash,
            name,
            role,
            is_active,
        )
        return str(row["id"])
    finally:
        await conn.close()
