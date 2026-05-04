"""
KDT-78: DASHBOARD API 테스트.

app/main.py 에 router 가 등록되지 않으므로 테스트 전용 미니 FastAPI app 을 만들어
include_router 로 검증한다.
"""
from __future__ import annotations

import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.repositories.call_summary_repo as summary_mod
import app.repositories.voc_analysis_repo as voc_mod
import app.repositories.mcp_action_log_repo as action_mod
import app.repositories.dashboard_repo as dashboard_mod

from app.api.v1.admin_auth import get_current_admin_user
from app.api.v1.dashboard import router as dashboard_router

from tests.fixtures.sample_transcripts import (
    PAYLOAD_RESOLVED_NEUTRAL,
    PAYLOAD_ANGRY_CRITICAL,
    PAYLOAD_NEGATIVE_REPEATED,
    ALL_SAMPLE_PAYLOADS,
)

# ── 테스트 전용 미니 앱 ───────────────────────────────────────────────────────

_app = FastAPI()
_app.include_router(dashboard_router, prefix="/dashboard")
_client = TestClient(_app)

DEFAULT_TENANT_ID = "tenant-a"


def _admin_context(tenant_id: str = DEFAULT_TENANT_ID) -> dict:
    return {
        "user": {
            "id": "11111111-1111-1111-1111-111111111111",
            "tenant_id": tenant_id,
            "email": "admin@example.test",
            "name": "Test Admin",
            "role": "owner",
            "is_active": True,
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


def _set_current_tenant(tenant_id: str = DEFAULT_TENANT_ID) -> None:
    async def fake_current_admin_user():
        return _admin_context(tenant_id)

    _app.dependency_overrides[get_current_admin_user] = fake_current_admin_user


# ── Store 격리 픽스처 ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_stores():
    _set_current_tenant()
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()
    yield
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()
    _set_current_tenant()


# ── 시딩 헬퍼 ────────────────────────────────────────────────────────────────

def _seed_dashboard(payload: dict) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    call_id = payload["call_id"]
    dashboard_mod._dashboard_store[call_id] = {
        **copy.deepcopy(payload),
        "call_id": call_id,
        "tenant_id": payload.get("tenant_id", ""),
        "created_at": now,
        "updated_at": now,
    }


def _seed_all_samples() -> None:
    for p in ALL_SAMPLE_PAYLOADS:
        _seed_dashboard(p)


def _seed_action_logs(call_id: str, tenant_id: str, actions: list[dict]) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    action_mod._action_store[call_id] = [
        {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "action_type": a.get("action_type", ""),
            "tool_name": a.get("tool", ""),
            "request_payload": a.get("params", {}),
            "response_payload": a.get("result", {}),
            "status": a.get("status", "success"),
            "external_id": a.get("external_id"),
            "error_message": a.get("error"),
            "created_at": now,
            "updated_at": now,
        }
        for a in actions
    ]


# ── 7. GET /dashboard/stats 성공 ──────────────────────────────────────────────

def test_get_dashboard_stats():
    _seed_all_samples()

    resp = _client.get("/dashboard/stats")
    assert resp.status_code == 200

    data = resp.json()
    required_keys = {
        "total_calls", "resolved_count", "escalated_count",
        "action_required_count", "mcp_success_count",
        "mcp_failed_count", "partial_success_count",
    }
    for key in required_keys:
        assert key in data, f"응답에 {key!r} 가 없음"

    assert data["total_calls"] == 2
    assert data["resolved_count"] == 1       # resolved_neutral
    assert data["escalated_count"] >= 1      # angry_critical (trigger=escalation_immediate)
    assert data["action_required_count"] == 1  # angry
    assert data["mcp_success_count"] == 2    # angry(2)
    assert data["mcp_failed_count"] == 0
    assert data["partial_success_count"] == 0


# ── 8. GET /dashboard/emotion-distribution 성공 ───────────────────────────────

def test_get_emotion_distribution():
    _seed_all_samples()

    resp = _client.get("/dashboard/emotion-distribution")
    assert resp.status_code == 200

    data = resp.json()
    assert data["neutral"] == 1
    assert data["angry"] == 1
    assert data["negative"] == 0
    assert data["positive"] == 0


# ── 9. GET /dashboard/priority-queue 성공 ────────────────────────────────────

def test_get_priority_queue():
    _seed_all_samples()

    resp = _client.get("/dashboard/priority-queue")
    assert resp.status_code == 200

    queue = resp.json()
    assert isinstance(queue, list)
    # resolved_neutral(low, action_required=False) 는 포함되지 않아야 함
    call_ids = [q["call_id"] for q in queue]
    assert PAYLOAD_RESOLVED_NEUTRAL["call_id"] not in call_ids

    # critical 이 high 보다 먼저
    priorities = [q["priority"] for q in queue]
    if "critical" in priorities and "high" in priorities:
        assert priorities.index("critical") < priorities.index("high")

    # 필수 필드 확인
    for item in queue:
        for key in ("call_id", "tenant_id", "priority", "summary_short",
                    "primary_category", "reason", "created_at"):
            assert key in item, f"priority_queue 항목에 {key!r} 가 없음"


# ── 10. GET /dashboard/action-logs 성공 ──────────────────────────────────────

def test_get_action_logs():
    _seed_action_logs(
        PAYLOAD_ANGRY_CRITICAL["call_id"],
        PAYLOAD_ANGRY_CRITICAL["tenant_id"],
        PAYLOAD_ANGRY_CRITICAL["executed_actions"],
    )
    _seed_action_logs(
        PAYLOAD_NEGATIVE_REPEATED["call_id"],
        PAYLOAD_NEGATIVE_REPEATED["tenant_id"],
        PAYLOAD_NEGATIVE_REPEATED["executed_actions"],
    )

    resp = _client.get("/dashboard/action-logs")
    assert resp.status_code == 200

    logs = resp.json()
    assert isinstance(logs, list)
    # angry(2) + negative(2) = 4 개
    assert len(logs) == 2


# ── 11. tenant_id 필터 동작 ───────────────────────────────────────────────────

def test_stats_tenant_filter():
    _seed_all_samples()

    resp_a = _client.get("/dashboard/stats?tenant_id=tenant-a")
    assert resp_a.status_code == 200
    # tenant-a: resolved_neutral + angry_critical = 2
    assert resp_a.json()["total_calls"] == 2

    resp_b = _client.get("/dashboard/stats?tenant_id=tenant-b")
    assert resp_b.status_code == 403
    assert resp_b.json()["detail"] == "tenant 정보가 일치하지 않습니다."


def test_priority_queue_tenant_filter():
    _seed_all_samples()

    resp_a = _client.get("/dashboard/priority-queue?tenant_id=tenant-a")
    data = resp_a.json()
    assert all(q["tenant_id"] == "tenant-a" for q in data)

    resp_b = _client.get("/dashboard/priority-queue?tenant_id=tenant-b")
    assert resp_b.status_code == 403


def test_action_logs_tenant_filter():
    _seed_action_logs("call-alpha", "tenant-a",
                      [{"action_type": "a", "tool": "gmail",
                        "status": "success", "external_id": None,
                        "error": None, "result": {}, "params": {}}])
    _seed_action_logs("call-beta", "tenant-b",
                      [{"action_type": "b", "tool": "company_db",
                        "status": "success", "external_id": None,
                        "error": None, "result": {}, "params": {}},
                       {"action_type": "c", "tool": "calendar",
                        "status": "failed", "external_id": None,
                        "error": "timeout", "result": {}, "params": {}}])

    resp = _client.get("/dashboard/action-logs?tenant_id=tenant-a")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = _client.get("/dashboard/action-logs?tenant_id=tenant-b")
    assert resp.status_code == 403


# ── 12. 빈 저장소에서 기본값 안전 반환 ────────────────────────────────────────

def test_empty_store_stats_safe():
    resp = _client.get("/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {
        "total_calls": 0,
        "resolved_count": 0,
        "escalated_count": 0,
        "action_required_count": 0,
        "mcp_success_count": 0,
        "mcp_failed_count": 0,
        "partial_success_count": 0,
    }


def test_empty_store_emotion_distribution_safe():
    resp = _client.get("/dashboard/emotion-distribution")
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"positive": 0, "neutral": 0, "negative": 0, "angry": 0}


def test_empty_store_priority_queue_safe():
    resp = _client.get("/dashboard/priority-queue")
    assert resp.status_code == 200
    assert resp.json() == []


def test_empty_store_action_logs_safe():
    resp = _client.get("/dashboard/action-logs")
    assert resp.status_code == 200
    assert resp.json() == []


# ── emotion-distribution 필드 검증 ───────────────────────────────────────────

def test_emotion_distribution_has_all_four_keys():
    resp = _client.get("/dashboard/emotion-distribution")
    data = resp.json()
    for key in ("positive", "neutral", "negative", "angry"):
        assert key in data, f"감정 분포에 {key!r} 키가 없음"


# ── started_from / started_to 필터 ────────────────────────────────────────────

def test_stats_started_from_filter():
    """started_from 이 미래 날짜면 모든 레코드가 필터링된다."""
    _seed_all_samples()

    resp = _client.get("/dashboard/stats?started_from=9999-01-01T00:00:00Z")
    assert resp.status_code == 200
    assert resp.json()["total_calls"] == 0


def test_action_logs_started_to_filter():
    """started_to 가 과거 날짜면 모든 로그가 필터링된다."""
    _seed_action_logs(
        PAYLOAD_ANGRY_CRITICAL["call_id"],
        PAYLOAD_ANGRY_CRITICAL["tenant_id"],
        PAYLOAD_ANGRY_CRITICAL["executed_actions"],
    )

    resp = _client.get("/dashboard/action-logs?started_to=2000-01-01T00:00:00Z")
    assert resp.status_code == 200
    assert resp.json() == []
