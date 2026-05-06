"""Post-call dashboard query helpers (KDT-94).

Read-only queries that aggregate ``call_summaries`` / ``voc_analyses`` /
``mcp_action_logs`` for a single ``tenant_id``. Every function takes an
``asyncpg`` connection and a ``tenant_id`` — there is intentionally no
"all tenants" entry point because the dashboard is rendered per company.

Notes:
- ``calls.tenant_id`` and ``call_summaries.tenant_id`` / ``voc_analyses.tenant_id``
  are ``UUID``.
- ``mcp_action_logs.tenant_id`` is ``TEXT`` (legacy + multi-source compat),
  so action queries cast the parameter to ``$1::text`` and join with
  ``c.tenant_id::text`` when fall-through is needed.
- ``date_from`` / ``date_to`` accept ISO8601 strings and are applied against
  ``calls.started_at`` (or ``mcp_action_logs.created_at`` for action logs).
"""
from __future__ import annotations

from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────────────


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _row_to_dict(row: Any) -> dict:
    return dict(row) if row is not None else {}


def _append_date_clauses(
    sql: str,
    params: list,
    *,
    column: str,
    date_from: str | None,
    date_to: str | None,
) -> str:
    """Append optional ``$N::timestamptz`` range clauses and grow ``params``."""
    if date_from is not None:
        params.append(date_from)
        sql += f"\n  AND {column} >= ${len(params)}::timestamptz"
    if date_to is not None:
        params.append(date_to)
        sql += f"\n  AND {column} <= ${len(params)}::timestamptz"
    return sql


# ── Summary ───────────────────────────────────────────────────────────────────


async def fetch_dashboard_summary(
    conn,
    tenant_id: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Top-line counters for a tenant: total / completed / summarized / voc / actions."""

    calls_sql = """
        SELECT
          COUNT(*)::int                                              AS total_calls,
          COUNT(*) FILTER (WHERE status = 'completed')::int          AS completed_calls
        FROM calls
        WHERE tenant_id = $1::uuid
    """
    calls_params: list = [tenant_id]
    calls_sql = _append_date_clauses(
        calls_sql, calls_params,
        column="started_at", date_from=date_from, date_to=date_to,
    )

    summaries_sql = """
        SELECT COUNT(*)::int AS summarized_calls
        FROM call_summaries cs
        WHERE cs.tenant_id = $1::uuid
    """
    summaries_params: list = [tenant_id]
    if date_from is not None or date_to is not None:
        summaries_sql += "\n  AND EXISTS (SELECT 1 FROM calls c WHERE c.id = cs.call_id"
        if date_from is not None:
            summaries_params.append(date_from)
            summaries_sql += f"\n    AND c.started_at >= ${len(summaries_params)}::timestamptz"
        if date_to is not None:
            summaries_params.append(date_to)
            summaries_sql += f"\n    AND c.started_at <= ${len(summaries_params)}::timestamptz"
        summaries_sql += "\n  )"

    voc_sql = """
        SELECT COUNT(*)::int AS voc_analyzed_calls
        FROM voc_analyses va
        WHERE va.tenant_id = $1::uuid
    """
    voc_params: list = [tenant_id]
    if date_from is not None or date_to is not None:
        voc_sql += "\n  AND EXISTS (SELECT 1 FROM calls c WHERE c.id = va.call_id"
        if date_from is not None:
            voc_params.append(date_from)
            voc_sql += f"\n    AND c.started_at >= ${len(voc_params)}::timestamptz"
        if date_to is not None:
            voc_params.append(date_to)
            voc_sql += f"\n    AND c.started_at <= ${len(voc_params)}::timestamptz"
        voc_sql += "\n  )"

    actions_sql = """
        SELECT COUNT(*)::int AS action_logs
        FROM mcp_action_logs ml
        WHERE ml.tenant_id = $1::text
    """
    actions_params: list = [tenant_id]
    actions_sql = _append_date_clauses(
        actions_sql, actions_params,
        column="ml.created_at", date_from=date_from, date_to=date_to,
    )

    calls_row = await conn.fetchrow(calls_sql, *calls_params)
    summaries_row = await conn.fetchrow(summaries_sql, *summaries_params)
    voc_row = await conn.fetchrow(voc_sql, *voc_params)
    actions_row = await conn.fetchrow(actions_sql, *actions_params)

    return {
        "total_calls":        int((calls_row or {})["total_calls"] or 0) if calls_row else 0,
        "completed_calls":    int((calls_row or {})["completed_calls"] or 0) if calls_row else 0,
        "summarized_calls":   int((summaries_row or {})["summarized_calls"] or 0) if summaries_row else 0,
        "voc_analyzed_calls": int((voc_row or {})["voc_analyzed_calls"] or 0) if voc_row else 0,
        "action_logs":        int((actions_row or {})["action_logs"] or 0) if actions_row else 0,
    }


# ── Distributions ─────────────────────────────────────────────────────────────


async def fetch_call_type_distribution(
    conn,
    tenant_id: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Counts of ``voc_analyses.intent_result->>'primary_category'`` for the tenant."""

    sql = """
        SELECT
          COALESCE(NULLIF(va.intent_result->>'primary_category', ''), 'unknown') AS label,
          COUNT(*)::int AS count
        FROM voc_analyses va
        WHERE va.tenant_id = $1::uuid
    """
    params: list = [tenant_id]

    if date_from is not None or date_to is not None:
        sql += "\n  AND EXISTS (SELECT 1 FROM calls c WHERE c.id = va.call_id"
        if date_from is not None:
            params.append(date_from)
            sql += f"\n    AND c.started_at >= ${len(params)}::timestamptz"
        if date_to is not None:
            params.append(date_to)
            sql += f"\n    AND c.started_at <= ${len(params)}::timestamptz"
        sql += "\n  )"

    sql += "\nGROUP BY label\nORDER BY count DESC, label ASC"

    rows = await conn.fetch(sql, *params)
    return [{"label": r["label"], "count": int(r["count"])} for r in rows]


async def fetch_emotion_distribution(
    conn,
    tenant_id: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Counts of ``call_summaries.customer_emotion`` for the tenant."""

    sql = """
        SELECT
          COALESCE(NULLIF(cs.customer_emotion, ''), 'unknown') AS label,
          COUNT(*)::int AS count
        FROM call_summaries cs
        WHERE cs.tenant_id = $1::uuid
    """
    params: list = [tenant_id]

    if date_from is not None or date_to is not None:
        sql += "\n  AND EXISTS (SELECT 1 FROM calls c WHERE c.id = cs.call_id"
        if date_from is not None:
            params.append(date_from)
            sql += f"\n    AND c.started_at >= ${len(params)}::timestamptz"
        if date_to is not None:
            params.append(date_to)
            sql += f"\n    AND c.started_at <= ${len(params)}::timestamptz"
        sql += "\n  )"

    sql += "\nGROUP BY label\nORDER BY count DESC, label ASC"

    rows = await conn.fetch(sql, *params)
    return [{"label": r["label"], "count": int(r["count"])} for r in rows]


async def fetch_priority_distribution(
    conn,
    tenant_id: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Counts of ``voc_analyses.priority_result->>'priority'`` for the tenant."""

    sql = """
        SELECT
          COALESCE(NULLIF(va.priority_result->>'priority', ''), 'unknown') AS label,
          COUNT(*)::int AS count
        FROM voc_analyses va
        WHERE va.tenant_id = $1::uuid
    """
    params: list = [tenant_id]

    if date_from is not None or date_to is not None:
        sql += "\n  AND EXISTS (SELECT 1 FROM calls c WHERE c.id = va.call_id"
        if date_from is not None:
            params.append(date_from)
            sql += f"\n    AND c.started_at >= ${len(params)}::timestamptz"
        if date_to is not None:
            params.append(date_to)
            sql += f"\n    AND c.started_at <= ${len(params)}::timestamptz"
        sql += "\n  )"

    sql += "\nGROUP BY label\nORDER BY count DESC, label ASC"

    rows = await conn.fetch(sql, *params)
    return [{"label": r["label"], "count": int(r["count"])} for r in rows]


async def fetch_resolution_distribution(
    conn,
    tenant_id: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Counts of ``call_summaries.resolution_status`` for the tenant."""

    sql = """
        SELECT
          COALESCE(NULLIF(cs.resolution_status, ''), 'unknown') AS label,
          COUNT(*)::int AS count
        FROM call_summaries cs
        WHERE cs.tenant_id = $1::uuid
    """
    params: list = [tenant_id]

    if date_from is not None or date_to is not None:
        sql += "\n  AND EXISTS (SELECT 1 FROM calls c WHERE c.id = cs.call_id"
        if date_from is not None:
            params.append(date_from)
            sql += f"\n    AND c.started_at >= ${len(params)}::timestamptz"
        if date_to is not None:
            params.append(date_to)
            sql += f"\n    AND c.started_at <= ${len(params)}::timestamptz"
        sql += "\n  )"

    sql += "\nGROUP BY label\nORDER BY count DESC, label ASC"

    rows = await conn.fetch(sql, *params)
    return [{"label": r["label"], "count": int(r["count"])} for r in rows]


async def fetch_action_status_distribution(
    conn,
    tenant_id: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Counts of ``mcp_action_logs.status`` for the tenant.

    ``tenant_id`` is compared as TEXT because ``mcp_action_logs.tenant_id``
    is stored as ``TEXT``.
    """
    sql = """
        SELECT
          COALESCE(NULLIF(ml.status, ''), 'unknown') AS label,
          COUNT(*)::int AS count
        FROM mcp_action_logs ml
        WHERE ml.tenant_id = $1::text
    """
    params: list = [tenant_id]
    sql = _append_date_clauses(
        sql, params,
        column="ml.created_at", date_from=date_from, date_to=date_to,
    )
    sql += "\nGROUP BY label\nORDER BY count DESC, label ASC"

    rows = await conn.fetch(sql, *params)
    return [{"label": r["label"], "count": int(r["count"])} for r in rows]


# ── Recent rows ───────────────────────────────────────────────────────────────


async def fetch_recent_calls(
    conn,
    tenant_id: str,
    *,
    limit: int = 10,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Most recent calls for the tenant, joined with summary + voc fields."""

    sql = """
        SELECT
          c.id::text                                 AS call_id,
          c.tenant_id::text                          AS tenant_id,
          c.started_at,
          c.ended_at,
          c.duration_sec,
          c.caller_number,
          cs.summary_short,
          cs.customer_intent,
          cs.customer_emotion,
          cs.resolution_status,
          va.intent_result->>'primary_category'      AS primary_category,
          va.priority_result->>'priority'            AS priority,
          va.sentiment_result->>'sentiment'          AS sentiment
        FROM calls c
        LEFT JOIN call_summaries cs ON cs.call_id = c.id
        LEFT JOIN voc_analyses    va ON va.call_id = c.id
        WHERE c.tenant_id = $1::uuid
    """
    params: list = [tenant_id]
    sql = _append_date_clauses(
        sql, params,
        column="c.started_at", date_from=date_from, date_to=date_to,
    )

    params.append(int(limit))
    sql += (
        "\nORDER BY COALESCE(c.ended_at, c.started_at, c.created_at) DESC"
        f"\nLIMIT ${len(params)}"
    )

    rows = await conn.fetch(sql, *params)
    return [
        {
            "call_id":          r["call_id"],
            "tenant_id":        r["tenant_id"],
            "started_at":       _iso(r["started_at"]),
            "ended_at":         _iso(r["ended_at"]),
            "duration_sec":     r["duration_sec"],
            "caller_number":    r["caller_number"],
            "summary_short":    r["summary_short"],
            "customer_intent":  r["customer_intent"],
            "customer_emotion": r["customer_emotion"],
            "resolution_status": r["resolution_status"],
            "primary_category": r["primary_category"],
            "priority":         r["priority"],
            "sentiment":        r["sentiment"],
        }
        for r in rows
    ]


async def fetch_recent_actions(
    conn,
    tenant_id: str,
    *,
    limit: int = 20,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Most recent ``mcp_action_logs`` rows for the tenant.

    Rows where ``tenant_id`` was not stamped (legacy/migration) are recovered
    via a JOIN-on-call_id fallback so a tenant doesn't lose audit history.
    """
    sql = """
        SELECT
          ml.call_id,
          ml.tenant_id,
          ml.action_type,
          ml.tool_name,
          ml.status,
          ml.external_id,
          ml.error_message,
          ml.created_at
        FROM mcp_action_logs ml
        LEFT JOIN calls c ON c.id::text = ml.call_id
        WHERE (
              ml.tenant_id = $1::text
           OR (ml.tenant_id IS NULL AND c.tenant_id::text = $1::text)
        )
    """
    params: list = [tenant_id]
    sql = _append_date_clauses(
        sql, params,
        column="ml.created_at", date_from=date_from, date_to=date_to,
    )

    params.append(int(limit))
    sql += f"\nORDER BY ml.created_at DESC\nLIMIT ${len(params)}"

    rows = await conn.fetch(sql, *params)
    return [
        {
            "call_id":       r["call_id"],
            "tenant_id":     r["tenant_id"] or "",
            "action_type":   r["action_type"],
            "tool_name":     r["tool_name"],
            "status":        r["status"],
            "external_id":   r["external_id"],
            "error_message": r["error_message"],
            "created_at":    _iso(r["created_at"]),
        }
        for r in rows
    ]


# ── Data quality ──────────────────────────────────────────────────────────────


async def fetch_data_quality(conn, tenant_id: str) -> dict:
    """Tenant-scoped sanity checks used by the dashboard's "Data Quality" panel."""

    async def _count(sql: str, *params) -> int:
        row = await conn.fetchrow(sql, *params)
        if row is None:
            return 0
        # asyncpg Record supports both index- and key-based access
        try:
            value = row[0]
        except (KeyError, IndexError, TypeError):
            value = next(iter(row.values()), 0)
        return int(value or 0)

    missing_primary_category = await _count(
        """
        SELECT COUNT(*)::int
        FROM voc_analyses
        WHERE tenant_id = $1::uuid
          AND NULLIF(intent_result->>'primary_category', '') IS NULL
        """,
        tenant_id,
    )

    tenant_mismatch_summary = await _count(
        """
        SELECT COUNT(*)::int
        FROM call_summaries cs
        JOIN calls c ON c.id = cs.call_id
        WHERE cs.tenant_id <> c.tenant_id
          AND cs.tenant_id = $1::uuid
        """,
        tenant_id,
    )

    tenant_mismatch_voc = await _count(
        """
        SELECT COUNT(*)::int
        FROM voc_analyses va
        JOIN calls c ON c.id = va.call_id
        WHERE va.tenant_id <> c.tenant_id
          AND va.tenant_id = $1::uuid
        """,
        tenant_id,
    )

    tenant_mismatch_action_logs = await _count(
        """
        SELECT COUNT(*)::int
        FROM mcp_action_logs ml
        JOIN calls c ON c.id::text = ml.call_id
        WHERE ml.tenant_id IS DISTINCT FROM c.tenant_id::text
          AND c.tenant_id::text = $1::text
        """,
        tenant_id,
    )

    return {
        "missing_primary_category":    missing_primary_category,
        "tenant_mismatch_summary":     tenant_mismatch_summary,
        "tenant_mismatch_voc":         tenant_mismatch_voc,
        "tenant_mismatch_action_logs": tenant_mismatch_action_logs,
    }
