"""Tests for app.services.post_call_dashboard_service (KDT-94).

The dashboard service runs read-only SQL through asyncpg. These tests use a
fake connection that captures every (sql, params) pair so we can assert:

- every query is filtered by ``tenant_id``
- ``mcp_action_logs`` queries cast ``tenant_id`` to TEXT
- the JSONB extraction paths match the contract (``primary_category``,
  ``priority``, ``sentiment``)
- the assembled response contains the contract's required top-level keys
- there is no "all tenants" function in either the repo or the service
"""
from __future__ import annotations

import inspect
import os
import sys
from datetime import datetime, timezone
from uuid import UUID

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fake asyncpg connection ───────────────────────────────────────────────────


class _FakeRow(dict):
    """dict that also supports positional access — mirrors asyncpg.Record."""

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _FakeConn:
    """Captures SQL/params for both ``fetch`` and ``fetchrow``.

    ``fetch_handler`` lets a test return different rows per SQL pattern so a
    "recent_calls" payload doesn't clash with the columns expected by
    distribution queries.
    """

    def __init__(
        self,
        *,
        fetch_rows: list[dict] | None = None,
        fetchrow_value: dict | None = None,
        fetch_handler=None,
    ):
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self._fetch_rows = fetch_rows or []
        self._fetchrow_value = fetchrow_value or {}
        self._fetch_handler = fetch_handler
        self.closed = False

    async def fetch(self, sql: str, *params):
        self.fetch_calls.append((sql, params))
        if self._fetch_handler is not None:
            data = self._fetch_handler(sql, params) or []
            return [_FakeRow(r) for r in data]
        return [_FakeRow(r) for r in self._fetch_rows]

    async def fetchrow(self, sql: str, *params):
        self.fetchrow_calls.append((sql, params))
        return _FakeRow(self._fetchrow_value)

    async def close(self):
        self.closed = True

    # Combined view of every captured (sql, params) for cross-cutting assertions
    @property
    def all_calls(self) -> list[tuple[str, tuple]]:
        return self.fetch_calls + self.fetchrow_calls


def _empty_summary_row() -> dict:
    return {
        "total_calls": 0,
        "completed_calls": 0,
        "summarized_calls": 0,
        "voc_analyzed_calls": 0,
        "action_logs": 0,
    }


def _make_conn() -> _FakeConn:
    """Conn that returns sensible default empty payloads for every helper."""
    return _FakeConn(
        fetch_rows=[],
        fetchrow_value=_empty_summary_row(),
    )


TENANT_ID = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"


# ── Service-level tests ──────────────────────────────────────────────────────


class TestServiceResponseShape:
    @pytest.mark.asyncio
    async def test_response_has_required_top_level_keys(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        result = await get_post_call_dashboard(TENANT_ID, conn=conn)

        for key in (
            "tenant_id", "range",
            "summary", "distributions",
            "recent_calls", "recent_actions",
            "data_quality",
        ):
            assert key in result, f"missing top-level key: {key}"

        assert result["tenant_id"] == TENANT_ID
        assert result["range"] == {"from": None, "to": None}

    @pytest.mark.asyncio
    async def test_distributions_contains_all_five_keys(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        result = await get_post_call_dashboard(TENANT_ID, conn=conn)

        for key in ("call_type", "emotion", "priority", "resolution", "action_status"):
            assert key in result["distributions"], f"missing distribution: {key}"

    @pytest.mark.asyncio
    async def test_data_quality_has_four_required_keys(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        result = await get_post_call_dashboard(TENANT_ID, conn=conn)

        expected = {
            "missing_primary_category",
            "tenant_mismatch_summary",
            "tenant_mismatch_voc",
            "tenant_mismatch_action_logs",
        }
        assert set(result["data_quality"].keys()) == expected

    @pytest.mark.asyncio
    async def test_summary_has_required_keys(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        result = await get_post_call_dashboard(TENANT_ID, conn=conn)

        for key in (
            "total_calls", "completed_calls",
            "summarized_calls", "voc_analyzed_calls",
            "action_logs",
        ):
            assert key in result["summary"], f"missing summary key: {key}"

    @pytest.mark.asyncio
    async def test_empty_conn_yields_empty_lists(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        result = await get_post_call_dashboard(TENANT_ID, conn=conn)

        assert result["recent_calls"] == []
        assert result["recent_actions"] == []
        for dist in result["distributions"].values():
            assert dist == []

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        with pytest.raises(ValueError):
            await get_post_call_dashboard("", conn=conn)


# ── Tenant filtering ─────────────────────────────────────────────────────────


class TestTenantFiltering:
    @pytest.mark.asyncio
    async def test_every_query_has_tenant_id_in_params(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        assert conn.all_calls, "service must hit the connection at least once"
        for sql, params in conn.all_calls:
            assert TENANT_ID in params, (
                f"tenant_id missing from params for SQL:\n{sql}\nparams={params}"
            )

    @pytest.mark.asyncio
    async def test_uuid_queries_use_uuid_cast(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        # Queries on calls / call_summaries / voc_analyses must cast the
        # tenant param as ::uuid.
        uuid_targets = ("FROM calls", "FROM call_summaries", "FROM voc_analyses")
        uuid_sqls = [
            sql for sql, _ in conn.all_calls
            if any(t in sql for t in uuid_targets)
            and "FROM mcp_action_logs" not in sql
        ]
        assert uuid_sqls, "expected at least one UUID-keyed query"
        for sql in uuid_sqls:
            assert "$1::uuid" in sql, (
                "UUID-keyed tables must compare tenant_id as ::uuid:\n" + sql
            )

    @pytest.mark.asyncio
    async def test_mcp_action_logs_queries_use_text_cast(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        action_sqls = [
            sql for sql, _ in conn.all_calls
            if "mcp_action_logs" in sql
        ]
        assert action_sqls, "expected at least one mcp_action_logs query"
        for sql in action_sqls:
            assert "$1::text" in sql, (
                "mcp_action_logs queries must compare tenant_id as TEXT:\n" + sql
            )


# ── JSONB extraction paths ──────────────────────────────────────────────────


class TestJsonbExtractionPaths:
    @pytest.mark.asyncio
    async def test_call_type_distribution_uses_primary_category(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        matches = [
            sql for sql, _ in conn.all_calls
            if "intent_result->>'primary_category'" in sql
            and "GROUP BY label" in sql
            and "FROM voc_analyses" in sql
        ]
        assert matches, "call_type distribution must use intent_result->>'primary_category'"

    @pytest.mark.asyncio
    async def test_priority_distribution_uses_priority_result(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        matches = [
            sql for sql, _ in conn.all_calls
            if "priority_result->>'priority'" in sql
            and "GROUP BY label" in sql
        ]
        assert matches, "priority distribution must use priority_result->>'priority'"

    @pytest.mark.asyncio
    async def test_recent_calls_extracts_sentiment_from_voc(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        recent_calls_sqls = [
            sql for sql, _ in conn.all_calls
            if "FROM calls c" in sql
            and "LEFT JOIN voc_analyses" in sql
        ]
        assert recent_calls_sqls, "recent_calls SQL must JOIN voc_analyses"
        for sql in recent_calls_sqls:
            assert "sentiment_result->>'sentiment'" in sql, (
                "recent_calls must surface sentiment via sentiment_result->>'sentiment':\n" + sql
            )

    @pytest.mark.asyncio
    async def test_recent_calls_filters_on_calls_tenant_uuid(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(TENANT_ID, conn=conn)

        recent_calls_sqls = [
            sql for sql, _ in conn.all_calls
            if "FROM calls c" in sql
            and "LEFT JOIN voc_analyses" in sql
        ]
        for sql in recent_calls_sqls:
            assert "c.tenant_id = $1::uuid" in sql, (
                "recent_calls must filter by calls.tenant_id = $1::uuid:\n" + sql
            )


# ── Limit propagation ────────────────────────────────────────────────────────


class TestLimitPropagation:
    @pytest.mark.asyncio
    async def test_limit_recent_calls_passed_into_query(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(
            TENANT_ID,
            limit_recent_calls=7,
            limit_recent_actions=3,
            conn=conn,
        )

        recent_calls_calls = [
            (sql, params) for sql, params in conn.all_calls
            if "FROM calls c" in sql and "LEFT JOIN voc_analyses" in sql
        ]
        assert recent_calls_calls, "recent_calls query must fire"
        # limit is appended as the last positional parameter
        assert any(7 in params for _, params in recent_calls_calls), (
            f"limit_recent_calls=7 must be in params: {recent_calls_calls}"
        )

    @pytest.mark.asyncio
    async def test_limit_recent_actions_passed_into_query(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(
            TENANT_ID,
            limit_recent_calls=10,
            limit_recent_actions=3,
            conn=conn,
        )

        recent_actions_calls = [
            (sql, params) for sql, params in conn.all_calls
            if "FROM mcp_action_logs ml" in sql
            and "ORDER BY ml.created_at DESC" in sql
        ]
        assert recent_actions_calls, "recent_actions query must fire"
        assert any(3 in params for _, params in recent_actions_calls), (
            f"limit_recent_actions=3 must be in params: {recent_actions_calls}"
        )


# ── Date range propagation ───────────────────────────────────────────────────


class TestDateRangePropagation:
    @pytest.mark.asyncio
    async def test_date_from_to_appears_in_response_range(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        result = await get_post_call_dashboard(
            TENANT_ID,
            date_from="2026-01-01T00:00:00Z",
            date_to="2026-12-31T23:59:59Z",
            conn=conn,
        )

        assert result["range"] == {
            "from": "2026-01-01T00:00:00Z",
            "to":   "2026-12-31T23:59:59Z",
        }

    @pytest.mark.asyncio
    async def test_date_filters_appended_as_timestamptz_params(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        conn = _make_conn()
        await get_post_call_dashboard(
            TENANT_ID,
            date_from="2026-01-01T00:00:00Z",
            date_to="2026-02-01T00:00:00Z",
            conn=conn,
        )

        # at least one query should pass both date_from and date_to
        seen_from = any(
            "2026-01-01T00:00:00Z" in params for _, params in conn.all_calls
        )
        seen_to = any(
            "2026-02-01T00:00:00Z" in params for _, params in conn.all_calls
        )
        assert seen_from and seen_to, (
            "date_from and date_to must both reach at least one query"
        )

        # and at least one SQL must use ::timestamptz
        ts_sqls = [sql for sql, _ in conn.all_calls if "::timestamptz" in sql]
        assert ts_sqls, "expected ::timestamptz casts when date filters are set"


# ── Recent rows shape ────────────────────────────────────────────────────────


class TestRecentRowsShape:
    @pytest.mark.asyncio
    async def test_recent_calls_row_shape(self):
        """Service must reshape DB rows into the contract's recent_calls shape."""
        from app.services.post_call_dashboard_service import get_post_call_dashboard

        fake_row = {
            "call_id": "abc",
            "tenant_id": TENANT_ID,
            "started_at": datetime(2026, 5, 4, 10, 0, tzinfo=timezone.utc),
            "ended_at": datetime(2026, 5, 4, 10, 2, tzinfo=timezone.utc),
            "duration_sec": 120,
            "caller_number": "010-0000-0000",
            "summary_short": "예약 문의",
            "customer_intent": "예약",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "primary_category": "예약/일정",
            "priority": "medium",
            "sentiment": "neutral",
        }

        def _handler(sql: str, _params):
            # Only the recent_calls SQL gets the joined row; every other
            # fetch (the distribution queries) returns no rows.
            if "FROM calls c" in sql and "LEFT JOIN voc_analyses" in sql:
                return [fake_row]
            return []

        conn = _FakeConn(
            fetch_handler=_handler,
            fetchrow_value=_empty_summary_row(),
        )
        result = await get_post_call_dashboard(TENANT_ID, conn=conn)

        assert result["recent_calls"], "recent_calls must be populated"
        first = result["recent_calls"][0]
        for key in (
            "call_id", "started_at", "ended_at", "duration_sec",
            "caller_number", "summary_short", "customer_intent",
            "customer_emotion", "resolution_status",
            "primary_category", "priority", "sentiment",
        ):
            assert key in first, f"recent_calls row missing key: {key}"

        # timestamps must be ISO8601 strings, not datetime objects
        assert isinstance(first["started_at"], str)
        assert "2026-05-04" in first["started_at"]

    @pytest.mark.asyncio
    async def test_recent_actions_row_shape(self):
        from app.services.post_call_dashboard_service import get_post_call_dashboard
        from app.repositories import post_call_dashboard_repo as repo

        fake_row = {
            "call_id": "call-1",
            "tenant_id": TENANT_ID,
            "action_type": "send_slack_alert",
            "tool_name": "slack",
            "status": "success",
            "external_id": "C0B03TCMMSP:1777715377.023619",
            "error_message": None,
            "created_at": datetime(2026, 5, 4, 10, 5, tzinfo=timezone.utc),
        }
        conn = _FakeConn(fetch_rows=[fake_row], fetchrow_value=_empty_summary_row())
        rows = await repo.fetch_recent_actions(conn, TENANT_ID, limit=5)

        assert len(rows) == 1
        row = rows[0]
        for key in (
            "call_id", "tenant_id", "action_type", "tool_name",
            "status", "external_id", "error_message", "created_at",
        ):
            assert key in row, f"recent_actions row missing key: {key}"
        assert row["external_id"] == "C0B03TCMMSP:1777715377.023619"
        assert isinstance(row["created_at"], str)


# ── Module-level invariants ──────────────────────────────────────────────────


class TestModuleInvariants:
    def test_no_all_tenants_function_in_service(self):
        """No "all tenants" entry point may exist on the service module."""
        from app.services import post_call_dashboard_service as svc

        public_funcs = [
            name for name, obj in inspect.getmembers(svc, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == svc.__name__
        ]
        # The single public entry point is get_post_call_dashboard.
        # If a future refactor adds another, it must still require tenant_id.
        for name in public_funcs:
            forbidden = ("all_tenant", "all_tenants", "global", "everyone")
            assert not any(t in name for t in forbidden), (
                f"service exposes a non-tenant-scoped function: {name}"
            )

        sig = inspect.signature(svc.get_post_call_dashboard)
        assert "tenant_id" in sig.parameters
        assert sig.parameters["tenant_id"].default is inspect.Parameter.empty, (
            "tenant_id must be a required argument"
        )

    def test_no_all_tenants_function_in_repo(self):
        """The repo must not expose a tenant-less aggregate function either."""
        from app.repositories import post_call_dashboard_repo as repo

        public_funcs = [
            (name, obj) for name, obj in inspect.getmembers(repo, inspect.isfunction)
            if not name.startswith("_") and obj.__module__ == repo.__name__
        ]
        # Every non-helper public function must require tenant_id.
        for name, fn in public_funcs:
            sig = inspect.signature(fn)
            params = sig.parameters
            assert "tenant_id" in params, (
                f"repo function {name} must accept tenant_id"
            )
            assert params["tenant_id"].default is inspect.Parameter.empty, (
                f"repo function {name} must require tenant_id (no default)"
            )

    def test_tenant_id_string_round_trips_as_uuid(self):
        """Sanity: TENANT_ID is a real UUID — guards against typos in tests."""
        UUID(TENANT_ID)


# ── Repo SQL spot-checks ─────────────────────────────────────────────────────


class TestRepoSqlSpotChecks:
    @pytest.mark.asyncio
    async def test_action_status_distribution_uses_text_cast(self):
        from app.repositories import post_call_dashboard_repo as repo

        conn = _FakeConn()
        await repo.fetch_action_status_distribution(conn, TENANT_ID)

        assert conn.fetch_calls
        for sql, params in conn.fetch_calls:
            assert "ml.tenant_id = $1::text" in sql
            assert TENANT_ID in params

    @pytest.mark.asyncio
    async def test_data_quality_returns_int_zeros_for_empty_db(self):
        from app.repositories import post_call_dashboard_repo as repo

        # fetchrow returns an empty row → _count must coerce to 0
        conn = _FakeConn(fetchrow_value={"count": 0})
        result = await repo.fetch_data_quality(conn, TENANT_ID)

        assert set(result.keys()) == {
            "missing_primary_category",
            "tenant_mismatch_summary",
            "tenant_mismatch_voc",
            "tenant_mismatch_action_logs",
        }
        for value in result.values():
            assert isinstance(value, int)
            assert value == 0
