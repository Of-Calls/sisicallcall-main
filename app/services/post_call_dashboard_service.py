"""Post-call dashboard query service (KDT-94).

Builds the per-tenant aggregate payload that the front-end dashboard consumes.
The service is intentionally:

- ``tenant_id``-scoped — there is no "all tenants" call path.
- Read-only — it never touches the live call pipeline, OAuth state, or
  account-level structures.
- Auth-agnostic — login/tenant resolution is handled upstream when this
  branch merges into ``main``.

See ``docs/post_call_dashboard_contract.md`` for the response shape.
"""
from __future__ import annotations

import asyncpg

from app.repositories import post_call_dashboard_repo as repo
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _connect():
    return await asyncpg.connect(_database_url())


async def get_post_call_dashboard(
    tenant_id: str,
    *,
    limit_recent_calls: int = 10,
    limit_recent_actions: int = 20,
    date_from: str | None = None,
    date_to: str | None = None,
    conn=None,
) -> dict:
    """Return the dashboard aggregate dict for a single tenant.

    ``conn`` is exposed for testability and connection pooling (callers
    that already hold an asyncpg connection can pass it in). When omitted,
    the service opens and closes its own connection.

    The returned dict is JSON-serializable (all timestamps are ISO8601
    strings) and matches ``docs/post_call_dashboard_contract.md``.
    """
    if not tenant_id:
        raise ValueError("tenant_id is required")

    owns_conn = conn is None
    if owns_conn:
        conn = await _connect()

    try:
        summary = await repo.fetch_dashboard_summary(
            conn, tenant_id,
            date_from=date_from, date_to=date_to,
        )

        call_type = await repo.fetch_call_type_distribution(
            conn, tenant_id,
            date_from=date_from, date_to=date_to,
        )
        emotion = await repo.fetch_emotion_distribution(
            conn, tenant_id,
            date_from=date_from, date_to=date_to,
        )
        priority = await repo.fetch_priority_distribution(
            conn, tenant_id,
            date_from=date_from, date_to=date_to,
        )
        resolution = await repo.fetch_resolution_distribution(
            conn, tenant_id,
            date_from=date_from, date_to=date_to,
        )
        action_status = await repo.fetch_action_status_distribution(
            conn, tenant_id,
            date_from=date_from, date_to=date_to,
        )

        recent_calls = await repo.fetch_recent_calls(
            conn, tenant_id,
            limit=limit_recent_calls,
            date_from=date_from, date_to=date_to,
        )
        recent_actions = await repo.fetch_recent_actions(
            conn, tenant_id,
            limit=limit_recent_actions,
            date_from=date_from, date_to=date_to,
        )

        data_quality = await repo.fetch_data_quality(conn, tenant_id)
    finally:
        if owns_conn and conn is not None:
            await conn.close()

    return {
        "tenant_id": str(tenant_id),
        "range": {
            "from": date_from,
            "to":   date_to,
        },
        "summary": summary,
        "distributions": {
            "call_type":     call_type,
            "emotion":       emotion,
            "priority":      priority,
            "resolution":    resolution,
            "action_status": action_status,
        },
        "recent_calls":   recent_calls,
        "recent_actions": recent_actions,
        "data_quality":   data_quality,
    }
