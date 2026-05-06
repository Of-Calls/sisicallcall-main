from __future__ import annotations

import json

import pytest

import app.repositories.call_summary_repo as summary_mod
import app.repositories.voc_analysis_repo as voc_mod
from app.repositories.call_summary_repo import save_summary
from app.repositories.voc_analysis_repo import save_voc_analysis


CALL_UUID = "1fc3c309-2c9d-4cd4-9378-d9d8ed4b1b3c"
TENANT_UUID = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"
OTHER_TENANT_UUID = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def reset_stores():
    summary_mod._reset()
    voc_mod._reset()
    yield
    summary_mod._reset()
    voc_mod._reset()


@pytest.mark.asyncio
async def test_uuid_summary_upserts_call_summaries(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class FakeConn:
        async def fetchrow(self, sql, *args):
            calls.append((sql, args))
            return {"tenant_id": TENANT_UUID}

        async def execute(self, sql, *args):
            calls.append((sql, args))
            return "INSERT 0 1"

        async def close(self):
            calls.append(("close", ()))

    async def fake_connect(url):
        calls.append(("connect", (url,)))
        return FakeConn()

    monkeypatch.setattr(summary_mod.asyncpg, "connect", fake_connect)

    summary = {
        "summary_short": "Refund delay repeat inquiry",
        "summary_detailed": "Customer complained about a delayed refund.",
        "customer_intent": "Check refund status",
        "customer_emotion": "angry",
        "resolution_status": "escalated",
        "keywords": ["refund", "delay", "callback"],
        "handoff_notes": "Payment team follow-up required",
    }

    await save_summary(CALL_UUID, TENANT_UUID, summary)
    await save_summary(CALL_UUID, TENANT_UUID, summary)

    inserts = [
        item for item in calls
        if "INSERT INTO call_summaries" in item[0]
    ]
    assert len(inserts) == 2
    sql, args = inserts[0]
    assert "ON CONFLICT (call_id)" in sql
    assert args[0] == CALL_UUID
    assert args[1] == TENANT_UUID
    assert args[2] == "Refund delay repeat inquiry"
    assert args[5] == "angry"
    assert args[6] == "escalated"
    assert json.loads(args[7]) == ["refund", "delay", "callback"]
    assert args[9] == "async"
    assert args[10] == "demo-mock-llm"


@pytest.mark.asyncio
async def test_uuid_voc_upserts_voc_analyses(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class FakeConn:
        async def fetchrow(self, sql, *args):
            calls.append((sql, args))
            return {"tenant_id": TENANT_UUID}

        async def execute(self, sql, *args):
            calls.append((sql, args))
            return "INSERT 0 1"

        async def close(self):
            calls.append(("close", ()))

    async def fake_connect(url):
        calls.append(("connect", (url,)))
        return FakeConn()

    monkeypatch.setattr(voc_mod.asyncpg, "connect", fake_connect)

    voc = {
        "sentiment_result": {"sentiment": "angry", "intensity": 0.93},
        "intent_result": {"primary_category": "refund"},
        "priority_result": {"priority": "critical", "action_required": True},
    }

    await save_voc_analysis(
        CALL_UUID,
        TENANT_UUID,
        voc,
        partial_success=True,
        failed_subagents=["priority_node"],
    )
    await save_voc_analysis(CALL_UUID, TENANT_UUID, voc)

    inserts = [
        item for item in calls
        if "INSERT INTO voc_analyses" in item[0]
    ]
    assert len(inserts) == 2
    sql, args = inserts[0]
    assert "ON CONFLICT (call_id)" in sql
    assert args[0] == CALL_UUID
    assert args[1] == TENANT_UUID
    assert json.loads(args[2])["sentiment"] == "angry"
    assert json.loads(args[3])["primary_category"] == "refund"
    assert json.loads(args[4])["priority"] == "critical"
    assert args[5] is True
    assert json.loads(args[6]) == ["priority_node"]


@pytest.mark.asyncio
async def test_uuid_summary_skips_db_upsert_when_tenant_mismatches(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class FakeConn:
        async def fetchrow(self, sql, *args):
            calls.append((sql, args))
            return {"tenant_id": OTHER_TENANT_UUID}

        async def execute(self, sql, *args):
            raise AssertionError("summary upsert should be skipped on tenant mismatch")

        async def close(self):
            calls.append(("close", ()))

    async def fake_connect(url):
        calls.append(("connect", (url,)))
        return FakeConn()

    monkeypatch.setattr(summary_mod.asyncpg, "connect", fake_connect)

    await save_summary(CALL_UUID, TENANT_UUID, {"summary_short": "tenant mismatch"})

    inserts = [
        item for item in calls
        if "INSERT INTO call_summaries" in item[0]
    ]
    assert inserts == []
    record = await summary_mod.get_summary_by_call_id(CALL_UUID)
    assert record is not None
    assert record["tenant_id"] == TENANT_UUID


@pytest.mark.asyncio
async def test_uuid_summary_skips_db_upsert_when_call_missing(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class FakeConn:
        async def fetchrow(self, sql, *args):
            calls.append((sql, args))
            return None

        async def execute(self, sql, *args):
            raise AssertionError("summary upsert should be skipped when call is missing")

        async def close(self):
            calls.append(("close", ()))

    async def fake_connect(url):
        calls.append(("connect", (url,)))
        return FakeConn()

    monkeypatch.setattr(summary_mod.asyncpg, "connect", fake_connect)

    await save_summary(CALL_UUID, TENANT_UUID, {"summary_short": "missing call"})

    inserts = [
        item for item in calls
        if "INSERT INTO call_summaries" in item[0]
    ]
    assert inserts == []


@pytest.mark.asyncio
async def test_uuid_voc_skips_db_upsert_when_tenant_mismatches(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class FakeConn:
        async def fetchrow(self, sql, *args):
            calls.append((sql, args))
            return {"tenant_id": OTHER_TENANT_UUID}

        async def execute(self, sql, *args):
            raise AssertionError("voc upsert should be skipped on tenant mismatch")

        async def close(self):
            calls.append(("close", ()))

    async def fake_connect(url):
        calls.append(("connect", (url,)))
        return FakeConn()

    monkeypatch.setattr(voc_mod.asyncpg, "connect", fake_connect)

    await save_voc_analysis(
        CALL_UUID,
        TENANT_UUID,
        {"priority_result": {"priority": "critical"}},
    )

    inserts = [
        item for item in calls
        if "INSERT INTO voc_analyses" in item[0]
    ]
    assert inserts == []
    record = await voc_mod.get_voc_by_call_id(CALL_UUID)
    assert record is not None
    assert record["tenant_id"] == TENANT_UUID


@pytest.mark.asyncio
async def test_uuid_voc_skips_db_upsert_when_call_missing(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    class FakeConn:
        async def fetchrow(self, sql, *args):
            calls.append((sql, args))
            return None

        async def execute(self, sql, *args):
            raise AssertionError("voc upsert should be skipped when call is missing")

        async def close(self):
            calls.append(("close", ()))

    async def fake_connect(url):
        calls.append(("connect", (url,)))
        return FakeConn()

    monkeypatch.setattr(voc_mod.asyncpg, "connect", fake_connect)

    await save_voc_analysis(
        CALL_UUID,
        TENANT_UUID,
        {"priority_result": {"priority": "critical"}},
    )

    inserts = [
        item for item in calls
        if "INSERT INTO voc_analyses" in item[0]
    ]
    assert inserts == []


@pytest.mark.asyncio
async def test_non_uuid_summary_skips_db_without_raising(monkeypatch):
    connect_calls: list[str] = []

    async def fake_connect(url):
        connect_calls.append(url)
        raise AssertionError("DB should not be called for non-UUID ids")

    monkeypatch.setattr(summary_mod.asyncpg, "connect", fake_connect)

    await save_summary(
        "demo-db-call-critical",
        "demo-tenant",
        {"summary_short": "demo", "customer_emotion": "neutral"},
    )

    assert connect_calls == []
    record = await summary_mod.get_summary_by_call_id("demo-db-call-critical")
    assert record is not None
    assert record["summary"]["summary_short"] == "demo"


@pytest.mark.asyncio
async def test_non_uuid_voc_skips_db_without_raising(monkeypatch):
    connect_calls: list[str] = []

    async def fake_connect(url):
        connect_calls.append(url)
        raise AssertionError("DB should not be called for non-UUID ids")

    monkeypatch.setattr(voc_mod.asyncpg, "connect", fake_connect)

    await save_voc_analysis(
        "demo-db-call-critical",
        "demo-tenant",
        {"priority_result": {"priority": "critical"}},
    )

    assert connect_calls == []
    record = await voc_mod.get_voc_by_call_id("demo-db-call-critical")
    assert record is not None
    assert record["voc_analysis"]["priority_result"]["priority"] == "critical"
