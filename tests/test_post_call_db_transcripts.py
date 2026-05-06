"""
tests/test_post_call_db_transcripts.py

app/services/db/transcripts.py 단위 테스트.
실제 PostgreSQL 서버 없이 FakeConn / FakeRecord 로 asyncpg 동작을 시뮬레이션한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.db.transcripts import (
    _is_uuid,
    _parse_jsonb,
    get_completed_call_context_from_db,
)


# ── Fake asyncpg 헬퍼 ─────────────────────────────────────────────────────────

class FakeRecord(dict):
    """asyncpg Record 대역. dict 서브클래싱으로 키 접근 지원."""
    def __getitem__(self, key):
        return super().__getitem__(key)


class FakeConn:
    """asyncpg Connection 대역."""

    def __init__(self, fetchrow_return=None, fetch_return=None, fetchrow_exc=None):
        self._fetchrow_return = fetchrow_return
        self._fetch_return = fetch_return or []
        self._fetchrow_exc = fetchrow_exc
        self.closed = False

    async def fetchrow(self, query, *args):
        if self._fetchrow_exc:
            raise self._fetchrow_exc
        return self._fetchrow_return

    async def fetch(self, query, *args):
        return self._fetch_return

    async def close(self):
        self.closed = True


# ── 공통 픽스처 데이터 ─────────────────────────────────────────────────────────

_CALL_UUID = "11111111-2222-3333-4444-555555555555"
_TENANT_ID = "tenant-abc"

_TS1 = datetime(2025, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
_TS2 = datetime(2025, 1, 1, 9, 1, 0, tzinfo=timezone.utc)
_TS3 = datetime(2025, 1, 1, 9, 2, 0, tzinfo=timezone.utc)


def _make_call_row(
    call_id=_CALL_UUID,
    tenant_id=_TENANT_ID,
    caller_number="+821012345678",
    status="completed",
    started_at=_TS1,
    ended_at=_TS2,
    branch_stats=None,
):
    return FakeRecord({
        "id": call_id,
        "tenant_id": tenant_id,
        "caller_number": caller_number,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "branch_stats": branch_stats,
    })


def _make_transcript_row(speaker, text, spoken_at=None):
    return FakeRecord({"speaker": speaker, "text": text, "spoken_at": spoken_at})


# ── _is_uuid 단위 테스트 ──────────────────────────────────────────────────────

def test_is_uuid_valid():
    assert _is_uuid("11111111-2222-3333-4444-555555555555") is True


def test_is_uuid_uppercase():
    assert _is_uuid("AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE") is True


def test_is_uuid_twilio_sid():
    assert _is_uuid("CA1234567890abcdef1234567890abcdef") is False


def test_is_uuid_empty():
    assert _is_uuid("") is False


# ── _parse_jsonb 단위 테스트 ──────────────────────────────────────────────────

def test_parse_jsonb_none():
    assert _parse_jsonb(None) == {}


def test_parse_jsonb_dict():
    d = {"faq": 2, "task": 1}
    assert _parse_jsonb(d) is d


def test_parse_jsonb_json_string():
    s = json.dumps({"escalation": 3})
    assert _parse_jsonb(s) == {"escalation": 3}


def test_parse_jsonb_invalid_string():
    assert _parse_jsonb("not-json{{{") == {}


# ── get_completed_call_context_from_db 통합 테스트 ────────────────────────────

@pytest.mark.asyncio
async def test_connect_failure_returns_none(monkeypatch):
    """asyncpg.connect 실패 시 None을 반환하고 예외를 전파하지 않는다."""
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(side_effect=OSError("DB 연결 거부됨")),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)
    assert result is None


@pytest.mark.asyncio
async def test_call_row_not_found_returns_none(monkeypatch):
    """calls 테이블에 해당 call_id 없으면 None을 반환한다."""
    conn = FakeConn(fetchrow_return=None)
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)
    assert result is None
    assert conn.closed is True


@pytest.mark.asyncio
async def test_tenant_id_mismatch_returns_none(monkeypatch):
    """tenant_id 불일치 시 None을 반환한다."""
    conn = FakeConn(fetchrow_return=_make_call_row(tenant_id="tenant-other"))
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID, tenant_id="tenant-mine")
    assert result is None
    assert conn.closed is True


@pytest.mark.asyncio
async def test_full_context_returned(monkeypatch):
    """call row + transcripts 가 있을 때 올바른 context dict를 반환한다."""
    transcript_rows = [
        _make_transcript_row("customer", "환불 요청합니다", _TS1),
        _make_transcript_row("agent",    "처리해드리겠습니다", _TS2),
        _make_transcript_row("customer", "감사합니다", _TS3),
    ]
    conn = FakeConn(
        fetchrow_return=_make_call_row(branch_stats={"faq": 1, "task": 2, "escalation": 0}),
        fetch_return=transcript_rows,
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID, tenant_id=_TENANT_ID)

    assert result is not None
    assert result["metadata"]["call_id"] == _CALL_UUID
    assert result["metadata"]["tenant_id"] == _TENANT_ID
    assert result["metadata"]["status"] == "completed"
    # caller_number(+82-prefixed E.164) → normalize_korean_phone 거쳐 로컬 형식으로
    assert result["metadata"]["customer_phone"] == "01012345678"
    assert result["metadata"]["start_time"] == _TS1.isoformat()
    assert result["metadata"]["end_time"] == _TS2.isoformat()

    assert len(result["transcripts"]) == 3
    assert result["transcripts"][0] == {
        "role": "customer",
        "text": "환불 요청합니다",
        "timestamp": _TS1.isoformat(),
    }
    assert result["transcripts"][1]["role"] == "agent"

    assert result["branch_stats"] == {"faq": 1, "task": 2, "escalation": 0}
    assert conn.closed is True


@pytest.mark.asyncio
async def test_no_transcripts_returns_empty_list(monkeypatch):
    """transcripts 없어도 transcripts=[] 로 정상 반환한다."""
    conn = FakeConn(fetchrow_return=_make_call_row(), fetch_return=[])
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)

    assert result is not None
    assert result["transcripts"] == []


@pytest.mark.asyncio
async def test_query_exception_returns_none(monkeypatch):
    """fetchrow가 예외를 던져도 None을 반환하고 예외를 전파하지 않는다."""
    conn = FakeConn(fetchrow_exc=RuntimeError("쿼리 실패"))
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)
    assert result is None


@pytest.mark.asyncio
async def test_twilio_sid_lookup(monkeypatch):
    """Twilio SID (비-UUID) 형식 call_id 도 정상 처리한다."""
    sid = "CA1234567890abcdef1234567890abcdef"
    conn = FakeConn(
        fetchrow_return=_make_call_row(call_id=_CALL_UUID),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(sid)
    assert result is not None
    assert result["metadata"]["call_id"] == sid


@pytest.mark.asyncio
async def test_branch_stats_jsonb_string_parsed(monkeypatch):
    """branch_stats가 JSON 문자열로 반환될 때 dict로 파싱된다."""
    branch_stats_json = json.dumps({"faq": 3, "task": 0, "escalation": 1})
    conn = FakeConn(
        fetchrow_return=_make_call_row(branch_stats=branch_stats_json),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)
    assert result is not None
    assert result["branch_stats"] == {"faq": 3, "task": 0, "escalation": 1}


@pytest.mark.asyncio
async def test_tenant_id_not_provided_skips_validation(monkeypatch):
    """tenant_id 미제공 시 tenant 검증을 생략하고 context를 반환한다."""
    conn = FakeConn(
        fetchrow_return=_make_call_row(tenant_id="any-tenant"),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID, tenant_id=None)
    assert result is not None
    assert result["metadata"]["tenant_id"] == "any-tenant"


@pytest.mark.asyncio
async def test_null_timestamps_handled(monkeypatch):
    """started_at / ended_at / spoken_at 가 None 이어도 정상 반환한다."""
    conn = FakeConn(
        fetchrow_return=_make_call_row(started_at=None, ended_at=None),
        fetch_return=[_make_transcript_row("customer", "테스트", spoken_at=None)],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)
    assert result is not None
    assert result["metadata"]["start_time"] is None
    assert result["metadata"]["end_time"] is None
    assert result["transcripts"][0]["timestamp"] is None


@pytest.mark.asyncio
async def test_status_none_defaults_to_completed(monkeypatch):
    """calls.status 가 NULL 이면 'completed' 를 기본값으로 반환한다."""
    conn = FakeConn(
        fetchrow_return=_make_call_row(status=None),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)
    assert result is not None
    assert result["metadata"]["status"] == "completed"


@pytest.mark.asyncio
async def test_conn_always_closed_on_success(monkeypatch):
    """정상 반환 경로에서도 conn.close() 가 반드시 호출된다."""
    conn = FakeConn(fetchrow_return=_make_call_row(), fetch_return=[])
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    await get_completed_call_context_from_db(_CALL_UUID)
    assert conn.closed is True


@pytest.mark.asyncio
async def test_conn_always_closed_on_row_not_found(monkeypatch):
    """row 없음(None 반환) 경로에서도 conn.close() 가 반드시 호출된다."""
    conn = FakeConn(fetchrow_return=None)
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    await get_completed_call_context_from_db(_CALL_UUID)
    assert conn.closed is True


# ── caller_number → customer_phone 매핑 ──────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_caller_number",
    [
        "+82-10-1234-5678",
        "+821012345678",
        "010-1234-5678",
        "01012345678",
    ],
)
async def test_caller_number_normalized_to_local_customer_phone(monkeypatch, raw_caller_number):
    """caller_number 4가지 표기가 모두 '01012345678' 로 정규화되어 metadata.customer_phone 에 들어간다."""
    conn = FakeConn(
        fetchrow_return=_make_call_row(caller_number=raw_caller_number),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)

    assert result is not None
    assert result["metadata"]["customer_phone"] == "01012345678"


@pytest.mark.asyncio
async def test_caller_number_null_drops_customer_phone_key(monkeypatch):
    """caller_number 가 NULL 이면 metadata 에 customer_phone 키 자체가 없다.

    (action_planner_node 가 metadata.get('customer_phone', '') 로 안전하게
    빈 문자열을 받기 위함.)
    """
    conn = FakeConn(
        fetchrow_return=_make_call_row(caller_number=None),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)

    assert result is not None
    assert "customer_phone" not in result["metadata"]


@pytest.mark.asyncio
async def test_caller_number_empty_string_drops_customer_phone_key(monkeypatch):
    """caller_number 가 '' 빈 문자열이어도 customer_phone 키가 비워진다."""
    conn = FakeConn(
        fetchrow_return=_make_call_row(caller_number=""),
        fetch_return=[],
    )
    monkeypatch.setattr(
        "app.services.db.transcripts.asyncpg.connect",
        AsyncMock(return_value=conn),
    )

    result = await get_completed_call_context_from_db(_CALL_UUID)

    assert result is not None
    assert "customer_phone" not in result["metadata"]
