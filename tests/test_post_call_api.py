"""
KDT-78: POST-CALL / SUMMARY API 테스트.

app/main.py 에 router 가 등록되지 않으므로 테스트 전용 미니 FastAPI app 을 만들어
include_router 로 검증한다.
"""
from __future__ import annotations

import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.v1.post_call as post_call_api
import app.repositories.call_summary_repo as summary_mod
import app.repositories.voc_analysis_repo as voc_mod
import app.repositories.mcp_action_log_repo as action_mod
import app.repositories.dashboard_repo as dashboard_mod
from app.repositories.call_summary_repo import _context_store as _ctx_store

from app.api.v1.admin_auth import get_current_admin_user
from app.api.v1.post_call import router as post_call_router
from app.api.v1.summary import router as summary_router

from tests.fixtures.sample_transcripts import (
    PAYLOAD_RESOLVED_NEUTRAL,
    PAYLOAD_ANGRY_CRITICAL,
    PAYLOAD_NEGATIVE_REPEATED,
)

# ── 테스트 전용 미니 앱 ───────────────────────────────────────────────────────

_app = FastAPI()
_app.include_router(post_call_router, prefix="/post-call")
_app.include_router(summary_router, prefix="/summary")
_client = TestClient(_app)


def _summary_admin_context(tenant_id: str = "tenant-a") -> dict:
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


async def _fake_current_admin_user():
    return _summary_admin_context()


_app.dependency_overrides[get_current_admin_user] = _fake_current_admin_user


# ── Store 격리 픽스처 ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_stores():
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()
    yield
    summary_mod._reset()
    voc_mod._reset()
    action_mod._reset()
    dashboard_mod._reset()
    _ctx_store.clear()


# ── 시딩 헬퍼 (동기 — 내부 store 직접 접근) ─────────────────────────────────

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


def _seed_summary(call_id: str, tenant_id: str, summary: dict) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    summary_mod._summary_store[call_id] = {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "summary": copy.deepcopy(summary),
        "created_at": now,
        "updated_at": now,
    }


def _seed_call_context(call_id: str, tenant_id: str = "default") -> None:
    """run 테스트용 — completed_call_runner가 context를 찾을 수 있도록 주입한다."""
    import copy
    _ctx_store[call_id] = copy.deepcopy({
        "metadata": {"call_id": call_id, "tenant_id": tenant_id},
        "transcripts": [{"role": "customer", "text": "테스트 문의입니다"}],
        "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
    })


def _seed_action_logs(call_id: str, tenant_id: str, actions: list[dict]) -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    action_mod._action_store[call_id] = [
        {
            "id": str(uuid4()),
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


def _patch_call_lookup(monkeypatch, *, exists: bool = True, tenant_id: str = "tenant-a") -> None:
    async def fake_get_call_by_id_for_tenant(call_id: str, jwt_tenant_id: str):
        assert jwt_tenant_id == tenant_id
        if not exists:
            return None
        return {"id": call_id, "tenant_id": jwt_tenant_id}

    monkeypatch.setattr(
        post_call_api,
        "get_call_by_id_for_tenant",
        fake_get_call_by_id_for_tenant,
    )


# ── 1. GET /post-call/{call_id} 성공 ─────────────────────────────────────────

def test_get_post_call_success():
    _seed_dashboard(PAYLOAD_RESOLVED_NEUTRAL)

    resp = _client.get(f"/post-call/{PAYLOAD_RESOLVED_NEUTRAL['call_id']}")
    assert resp.status_code == 200

    data = resp.json()
    assert data["call_id"] == PAYLOAD_RESOLVED_NEUTRAL["call_id"]
    assert "summary" in data
    assert "voc_analysis" in data
    assert "priority_result" in data
    assert "action_plan" in data
    assert "executed_actions" in data
    assert "errors" in data
    assert "partial_success" in data
    assert data["summary"]["customer_emotion"] == "neutral"


# ── 2. GET /post-call/{call_id} not found ────────────────────────────────────

def test_get_post_call_not_found():
    resp = _client.get("/post-call/nonexistent-call-999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── 3. GET /post-call/{call_id}/actions 성공 ──────────────────────────────────

def test_get_call_actions_success(monkeypatch):
    _patch_call_lookup(monkeypatch)
    call_id = PAYLOAD_ANGRY_CRITICAL["call_id"]
    _seed_action_logs(call_id, PAYLOAD_ANGRY_CRITICAL["tenant_id"], [
        {
            "action_type": "send_manager_email",
            "tool": "gmail",
            "params": {"to": "manager@example.com"},
            "result": {"sent": True},
            "status": "success",
        },
        {
            "action_type": "schedule_callback",
            "tool": "calendar",
            "params": {"title": "callback"},
            "result": {},
            "status": "failed",
            "error": "calendar unavailable",
        },
    ])

    resp = _client.get(f"/post-call/{call_id}/actions")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total"] == 2
    assert data["items"][0]["call_id"] == call_id
    assert data["items"][0]["tenant_id"] == PAYLOAD_ANGRY_CRITICAL["tenant_id"]
    assert data["items"][0]["action_type"] == "send_manager_email"
    assert data["items"][0]["action_detail"] == "gmail"
    assert data["items"][0]["status"] == "success"
    assert data["items"][1]["status"] == "fail"
    assert data["items"][1]["error_message"] == "calendar unavailable"


def test_get_call_actions_empty_for_call_with_no_actions(monkeypatch):
    _patch_call_lookup(monkeypatch)

    resp = _client.get("/post-call/no-actions-call/actions")
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}


def test_get_call_actions_other_tenant_call_returns_404(monkeypatch):
    _patch_call_lookup(monkeypatch, exists=False)

    resp = _client.get("/post-call/call-other-tenant/actions")

    assert resp.status_code == 404


# ── 4. POST /post-call/{call_id}/run manual 실행 성공 ─────────────────────────

def test_run_post_call_manual():
    _seed_call_context("run-test-001", "test-tenant")
    resp = _client.post("/post-call/run-test-001/run?trigger=manual&tenant_id=test-tenant")
    assert resp.status_code == 200

    data = resp.json()
    assert data["ok"] is True
    assert "result" in data

    result = data["result"]
    assert result["call_id"] == "run-test-001"
    assert result["trigger"] == "manual"
    assert isinstance(result["partial_success"], bool)
    assert isinstance(result["errors"], list)
    assert "summary" in result


def test_run_post_call_invalid_trigger():
    # trigger 검증은 context 조회 전에 수행되므로 seed 불필요
    resp = _client.post("/post-call/run-test-002/run?trigger=invalid_trigger")
    assert resp.status_code == 400
    assert "trigger" in resp.json()["detail"].lower() or "unknown" in resp.json()["detail"].lower()


def test_run_post_call_call_ended_trigger():
    _seed_call_context("run-test-003")
    resp = _client.post("/post-call/run-test-003/run?trigger=call_ended")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_run_post_call_escalation_trigger():
    _seed_call_context("run-test-004")
    resp = _client.post("/post-call/run-test-004/run?trigger=escalation_immediate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    # escalation_immediate 는 summary 만 실행하므로 voc/priority/action_plan 이 없을 수 있음
    result = data["result"]
    assert result["trigger"] == "escalation_immediate"


# ── 5. GET /summary/{call_id} 성공 ───────────────────────────────────────────

def test_get_summary_success():
    summary = {
        "summary_short": "요금 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
    }
    _seed_summary("sum-001", "tenant-a", summary)

    resp = _client.get("/summary/sum-001")
    assert resp.status_code == 200

    data = resp.json()
    assert data["call_id"] == "sum-001"
    assert data["tenant_id"] == "tenant-a"
    assert data["summary"]["summary_short"] == "요금 문의"
    assert "created_at" in data
    assert "updated_at" in data


# ── 6. GET /summary/{call_id} not found ──────────────────────────────────────

def test_get_summary_not_found():
    resp = _client.get("/summary/nonexistent-sum-999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── post-call detail 구조 검증 ────────────────────────────────────────────────

def test_post_call_detail_has_all_standard_keys():
    """GET /post-call/{call_id} 응답이 프론트용 6개 필드를 모두 갖는지 확인한다."""
    _seed_dashboard(PAYLOAD_ANGRY_CRITICAL)

    resp = _client.get(f"/post-call/{PAYLOAD_ANGRY_CRITICAL['call_id']}")
    assert resp.status_code == 200
    data = resp.json()

    required_keys = {
        "call_id", "summary", "voc_analysis", "priority_result",
        "action_plan", "executed_actions", "errors", "partial_success",
    }
    for key in required_keys:
        assert key in data, f"응답에 {key!r} 가 없음"


def test_post_call_detail_executed_actions_list():
    """executed_actions 가 리스트로 반환된다."""
    _seed_dashboard(PAYLOAD_ANGRY_CRITICAL)

    resp = _client.get(f"/post-call/{PAYLOAD_ANGRY_CRITICAL['call_id']}")
    data = resp.json()
    assert isinstance(data["executed_actions"], list)
    assert len(data["executed_actions"]) == len(PAYLOAD_ANGRY_CRITICAL["executed_actions"])
