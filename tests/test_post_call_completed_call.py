"""
종료된 통화 데이터 후처리 통합 테스트.

검증 범위:
  1. DB context 있으면 PostCallAgent까지 실행 → ok=True
  2. context_provider가 DB context를 in-memory seed보다 우선 사용
  3. DB context 없고 seed context 있으면 seed context 사용
  4. context 없으면 run_post_call_for_completed_call → ok=False
  5. transcripts None → [] 정규화
  6. metadata call_id / tenant_id 보강
  7. branch_stats None → {} 정규화
  8. PostCallAgent partial_success=True → ok=True
  9. API POST /post-call/{call_id}/run → completed call runner 호출
  10. NeMo / Chroma / STT / TTS / app.main / app.api.v1.call import 없음

주의:
  app.main, app.api.v1.call, conversational graph, NeMo, Chroma를 import하지 않는다.
  API 테스트는 라우터만 등록한 미니 앱을 사용한다.
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
    summary_mod._context_store.clear()


@pytest.fixture(autouse=True)
def force_post_call_tests_mock_mode(monkeypatch, tmp_path):
    for key in (
        "GMAIL_MCP_REAL",
        "CALENDAR_MCP_REAL",
        "JIRA_MCP_REAL",
        "SLACK_MCP_REAL",
        "SMS_MCP_REAL",
        "NOTION_MCP_REAL",
        "COMPANY_DB_MCP_REAL",
        "MCP_COMPANY_DB_REAL",
        "MCP_USE_TENANT_OAUTH",
        "POST_CALL_ENABLE_NOTION_RECORD",
    ):
        monkeypatch.setenv(key, "false")
    monkeypatch.setenv("MCP_ACTION_LOG_STORE", "file")
    monkeypatch.setenv("MCP_ACTION_LOG_FILE", str(tmp_path / "mcp_action_logs.json"))


# ── API 테스트용 미니 앱 ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """post_call 라우터만 등록한 미니 앱 (NeMo/Chroma/call.py import 없음)."""
    from app.api.v1.post_call import router as post_call_router
    _app = FastAPI()
    _app.include_router(post_call_router, prefix="/post-call")
    return TestClient(_app)


# ── 샘플 데이터 ──────────────────────────────────────────────────────────────

_SAMPLE_DB_CTX = {
    "metadata": {
        "call_id":   "db-call-001",
        "tenant_id": "tenant-db",
        "start_time": "2026-04-28T10:00:00Z",
        "end_time":   "2026-04-28T10:05:00Z",
        "status":    "completed",
    },
    "transcripts": [
        {"role": "customer", "text": "요금제 변경 문의드려요", "timestamp": "2026-04-28T10:00:10Z"},
        {"role": "agent",    "text": "확인해드리겠습니다",     "timestamp": "2026-04-28T10:00:20Z"},
    ],
    "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
}


# ── 1. DB context 있으면 PostCallAgent까지 실행 → ok=True ────────────────────

@pytest.mark.asyncio
async def test_completed_runner_with_db_context(monkeypatch):
    """DB에서 context를 가져오면 PostCallAgent가 실행되어 ok=True가 반환된다."""
    from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call

    async def fake_db(call_id, tenant_id=None):
        return copy.deepcopy(_SAMPLE_DB_CTX)

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    result = await run_post_call_for_completed_call(
        call_id="db-call-001",
        tenant_id="tenant-db",
        trigger="call_ended",
    )

    assert result["ok"] is True
    assert result["error"] is None
    assert isinstance(result["result"], dict)
    assert result["result"]["call_id"] == "db-call-001"


# ── 2. context_provider: DB context를 in-memory seed보다 우선 사용 ──────────

@pytest.mark.asyncio
async def test_context_provider_db_takes_priority_over_seed(monkeypatch):
    """DB context와 seed context가 모두 있을 때 DB context를 반환한다."""
    from app.agents.post_call.context_provider import (
        get_call_context_for_post_call,
        seed_test_context,
    )

    # seed context 주입
    await seed_test_context(
        call_id="prio-test-001",
        tenant_id="tenant-seed",
        transcripts=[{"role": "customer", "text": "seed 발화"}],
    )

    # DB context는 다른 내용
    db_ctx = copy.deepcopy(_SAMPLE_DB_CTX)
    db_ctx["metadata"]["call_id"] = "prio-test-001"
    db_ctx["transcripts"] = [{"role": "customer", "text": "DB 발화"}]

    async def fake_db(call_id, tenant_id=None):
        return copy.deepcopy(db_ctx)

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("prio-test-001")

    assert ctx is not None
    assert ctx["transcripts"][0]["text"] == "DB 발화"


# ── 3. DB context 없고 seed context 있으면 seed context 사용 ─────────────────

@pytest.mark.asyncio
async def test_context_provider_falls_back_to_seed(monkeypatch):
    """DB가 None을 반환하면 seed context를 반환한다."""
    from app.agents.post_call.context_provider import (
        get_call_context_for_post_call,
        seed_test_context,
    )

    async def fake_db(call_id, tenant_id=None):
        return None

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    await seed_test_context(
        call_id="seed-fallback-001",
        tenant_id="tenant-seed",
        transcripts=[{"role": "customer", "text": "seed 발화"}],
    )

    ctx = await get_call_context_for_post_call("seed-fallback-001")

    assert ctx is not None
    assert ctx["transcripts"][0]["text"] == "seed 발화"


# ── 4. context 없으면 ok=False ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_completed_runner_no_context_returns_ok_false(monkeypatch):
    """DB도 None, seed도 없으면 ok=False, error='call_context_not_found'."""
    from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call

    async def fake_db(call_id, tenant_id=None):
        return None

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    result = await run_post_call_for_completed_call("no-ctx-call-001")

    assert result["ok"] is False
    assert result["result"] is None
    assert result["error"] == "call_context_not_found"


# ── 5. transcripts None → [] 정규화 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_normalizes_transcripts_none(monkeypatch):
    """DB context의 transcripts가 None이면 []로 정규화된다."""
    from app.agents.post_call.context_provider import get_call_context_for_post_call

    async def fake_db(call_id, tenant_id=None):
        return {
            "metadata":    {"call_id": call_id, "tenant_id": "t"},
            "transcripts": None,
            "branch_stats": {"faq": 0, "task": 0, "escalation": 0},
        }

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("norm-test-001")

    assert ctx is not None
    assert ctx["transcripts"] == []


# ── 6. metadata call_id / tenant_id 보강 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_normalizes_metadata_fields(monkeypatch):
    """DB context의 metadata에 call_id / tenant_id가 없으면 인자로 보강된다."""
    from app.agents.post_call.context_provider import get_call_context_for_post_call

    async def fake_db(call_id, tenant_id=None):
        return {
            "metadata":    {},  # call_id / tenant_id 없음
            "transcripts": [{"role": "customer", "text": "문의"}],
            "branch_stats": {},
        }

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("meta-norm-001", tenant_id="tenant-x")

    assert ctx is not None
    assert ctx["metadata"]["call_id"] == "meta-norm-001"
    assert ctx["metadata"]["tenant_id"] == "tenant-x"


# ── 7. branch_stats None → {} 정규화 ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_normalizes_branch_stats_none(monkeypatch):
    """DB context의 branch_stats가 None이면 {}로 정규화된다."""
    from app.agents.post_call.context_provider import get_call_context_for_post_call

    async def fake_db(call_id, tenant_id=None):
        return {
            "metadata":    {"call_id": call_id},
            "transcripts": [{"role": "customer", "text": "문의"}],
            "branch_stats": None,
        }

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("branch-norm-001")

    assert ctx is not None
    assert ctx["branch_stats"] == {}


# ── 7-b. customer_phone 정규화 / 보존 / NULL 매핑 ────────────────────────────

@pytest.mark.asyncio
async def test_context_provider_normalizes_customer_phone(monkeypatch):
    """metadata.customer_phone 가 다양한 한국 형식이어도 로컬 형식으로 통일된다."""
    from app.agents.post_call.context_provider import get_call_context_for_post_call

    async def fake_db(call_id, tenant_id=None):
        return {
            "metadata":    {"call_id": call_id, "tenant_id": "t", "customer_phone": "+82-10-1234-5678"},
            "transcripts": [],
            "branch_stats": {},
        }

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("phone-norm-001")
    assert ctx is not None
    assert ctx["metadata"]["customer_phone"] == "01012345678"


@pytest.mark.asyncio
async def test_context_provider_drops_empty_customer_phone(monkeypatch):
    """caller_number 가 NULL/empty 일 때 customer_phone 키 자체가 비워진다."""
    from app.agents.post_call.context_provider import get_call_context_for_post_call

    async def fake_db(call_id, tenant_id=None):
        return {
            "metadata":    {"call_id": call_id, "tenant_id": "t", "customer_phone": None},
            "transcripts": [],
            "branch_stats": {},
        }

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("phone-null-001")
    assert ctx is not None
    # action_planner 가 metadata.get("customer_phone", "") 로 안전하게 빈 문자열을
    # 받을 수 있도록 키 자체를 비워 둔다.
    assert "customer_phone" not in ctx["metadata"]
    assert ctx["metadata"].get("customer_phone", "") == ""


@pytest.mark.asyncio
async def test_context_provider_preserves_existing_normalized_phone(monkeypatch):
    """이미 정규화된 customer_phone 은 그대로 보존된다 (idempotent)."""
    from app.agents.post_call.context_provider import get_call_context_for_post_call

    async def fake_db(call_id, tenant_id=None):
        return {
            "metadata":    {"call_id": call_id, "tenant_id": "t", "customer_phone": "01098765432"},
            "transcripts": [],
            "branch_stats": {},
        }

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    ctx = await get_call_context_for_post_call("phone-keep-001")
    assert ctx is not None
    assert ctx["metadata"]["customer_phone"] == "01098765432"


# ── 8. PostCallAgent partial_success=True → ok=True ─────────────────────────

@pytest.mark.asyncio
async def test_completed_runner_partial_success_is_ok_true(monkeypatch):
    """PostCallAgent가 partial_success=True로 끝나도 ok=True를 반환한다."""
    from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call
    from app.agents.post_call import agent as agent_mod

    async def fake_db(call_id, tenant_id=None):
        return copy.deepcopy(_SAMPLE_DB_CTX)

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    async def fake_run(self, call_id, trigger, tenant_id):
        return {
            "call_id": call_id, "trigger": trigger,
            "partial_success": True,
            "errors": [{"node": "action_router", "error": "mock fail"}],
            "summary": {}, "voc_analysis": {}, "priority_result": {},
            "action_plan": [], "executed_actions": [],
        }

    monkeypatch.setattr(agent_mod.PostCallAgent, "run", fake_run)

    result = await run_post_call_for_completed_call(
        call_id="db-call-001",
        tenant_id="tenant-db",
        trigger="call_ended",
    )

    assert result["ok"] is True
    assert result["result"]["partial_success"] is True


# ── 9. API POST /post-call/{call_id}/run → completed call runner 호출 ────────

def test_api_run_uses_completed_call_runner(api_client, monkeypatch):
    """POST /post-call/{call_id}/run 이 completed_call_runner를 사용한다."""
    # post_call.py가 run_post_call_for_completed_call을 직접 import하므로
    # post_call 모듈 네임스페이스에서 패치해야 한다.
    import app.api.v1.post_call as post_call_mod

    called_with: list[dict] = []

    async def fake_runner(call_id, tenant_id="default", trigger="call_ended"):
        called_with.append({"call_id": call_id, "tenant_id": tenant_id, "trigger": trigger})
        return {
            "ok": True,
            "result": {
                "call_id": call_id, "trigger": trigger,
                "partial_success": False, "errors": [],
                "summary": {}, "voc_analysis": {}, "priority_result": {},
                "action_plan": [], "executed_actions": [],
            },
            "error": None,
        }

    monkeypatch.setattr(post_call_mod, "run_post_call_for_completed_call", fake_runner)

    resp = api_client.post("/post-call/api-test-001/run?trigger=call_ended&tenant_id=t-api")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    assert len(called_with) == 1
    assert called_with[0]["call_id"] == "api-test-001"
    assert called_with[0]["tenant_id"] == "t-api"
    assert called_with[0]["trigger"] == "call_ended"


def test_api_run_no_context_returns_404(api_client, monkeypatch):
    """context가 없으면 API는 404를 반환한다."""
    import app.agents.post_call.completed_call_runner as runner_mod

    async def fake_runner(call_id, tenant_id="default", trigger="call_ended"):
        return {"ok": False, "result": None, "error": "call_context_not_found"}

    monkeypatch.setattr(runner_mod, "run_post_call_for_completed_call", fake_runner)

    resp = api_client.post("/post-call/no-ctx-call/run")
    assert resp.status_code == 404


def test_api_run_invalid_trigger_returns_400(api_client):
    """잘못된 trigger는 context 조회 전에 400을 반환한다."""
    resp = api_client.post("/post-call/any-call/run?trigger=bad_trigger")
    assert resp.status_code == 400
    assert "unknown" in resp.json()["detail"].lower()


# ── 11. angry/critical 시나리오 → 필수 액션 생성·실행 ────────────────────────

@pytest.mark.asyncio
async def test_angry_critical_generates_and_executes_required_actions(monkeypatch):
    """통화 context가 angry+escalated+critical이면
    send_manager_email / send_slack_alert / add_priority_queue 액션이
    생성되고 실행된다.
    """
    from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call

    async def fake_db(call_id, tenant_id=None):
        return copy.deepcopy(_SAMPLE_DB_CTX)

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    # LLM caller를 angry + critical 응답으로 교체
    _ANGRY_SUMMARY = {
        "summary_short": "[TEST] 긴급 에스컬레이션",
        "summary_detailed": "[TEST] 고객이 매우 화남",
        "customer_intent": "불만 처리",
        "customer_emotion": "angry",
        "resolution_status": "escalated",
        "keywords": ["불만", "에스컬레이션"],
        "handoff_notes": None,
    }
    _ANGRY_VOC = {
        "sentiment_result": {"sentiment": "angry", "intensity": 0.9, "reason": "매우 화남"},
        "intent_result": {
            "primary_category": "불만",
            "sub_categories": [],
            "is_repeat_topic": False,
            "faq_candidate": False,
        },
        "priority_result": {
            "priority": "critical",
            "action_required": True,
            "suggested_action": None,
            "reason": "즉시 대응 필요",
        },
    }
    _ANGRY_PRIORITY = {
        "priority": "critical",
        "tier": "critical",
        "action_required": True,
        "suggested_action": None,
        "reason": "즉시 대응 필요",
    }

    class AngryMockLLM:
        async def call_json(self, system_prompt, user_message, max_tokens=1024):
            # 새 통합 분석 노드 (우선 검사)
            if "ANALYSIS_COMBINED" in system_prompt:
                return {
                    "summary": _ANGRY_SUMMARY,
                    "voc_analysis": _ANGRY_VOC,
                    "priority_result": _ANGRY_PRIORITY,
                }
            # 리뷰 게이트 — pass 반환
            if "REVIEW_VERDICT" in system_prompt:
                return {"verdict": "pass", "confidence": 0.95, "issues": [], "corrections": {}, "blocked_actions": [], "reason": "pass"}
            # 하위 호환
            if "summary_short" in system_prompt:
                return _ANGRY_SUMMARY
            if "sentiment_result" in system_prompt:
                return _ANGRY_VOC
            return _ANGRY_PRIORITY

    angry_mock = AngryMockLLM()

    import app.agents.post_call.nodes.post_call_analysis_node as analysis_node_mod
    import app.agents.post_call.nodes.review_node as review_node_mod
    import app.agents.post_call.nodes.summary_node as summary_node_mod
    import app.agents.post_call.nodes.voc_analysis_node as voc_node_mod
    import app.agents.post_call.nodes.priority_node as priority_node_mod

    monkeypatch.setattr(analysis_node_mod, "_caller", angry_mock)
    monkeypatch.setattr(review_node_mod, "_caller", angry_mock)
    monkeypatch.setattr(summary_node_mod, "_caller", angry_mock)
    monkeypatch.setattr(voc_node_mod, "_caller", angry_mock)
    monkeypatch.setattr(priority_node_mod, "_caller", angry_mock)

    result = await run_post_call_for_completed_call(
        call_id="db-call-001",
        tenant_id="tenant-db",
        trigger="call_ended",
    )

    assert result["ok"] is True
    agent_result = result["result"]

    executed = agent_result.get("executed_actions", [])
    assert isinstance(executed, list)
    assert len(executed) > 0, "executed_actions가 비어 있음"

    action_types = {a["action_type"] for a in executed}

    assert "send_manager_email" in action_types, \
        f"send_manager_email 없음: {action_types}"
    assert "send_slack_alert" in action_types, \
        f"send_slack_alert 없음: {action_types}"
    assert "add_priority_queue" in action_types, \
        f"add_priority_queue 없음: {action_types}"

    # 모든 action이 표준 6-key 포맷인지 확인
    for a in executed:
        for key in ("action_type", "tool", "status", "external_id", "error", "result"):
            assert key in a, f"action {a.get('action_type')} 에 {key!r} 키 없음"


# ── 시연용 demo context 통합 테스트 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_demo_context_generates_required_action_plan(monkeypatch):
    """demo context가 action_planner에서 Calendar/Slack/SMS/Notion 액션을 생성한다."""
    from tests.fixtures.demo_post_call_context import (
        DEMO_POST_CALL_CONTEXT, DEMO_LLM_SUMMARY, DEMO_LLM_VOC, DEMO_LLM_PRIORITY,
    )
    from app.agents.post_call.nodes.action_planner_node import _build_plan

    monkeypatch.setenv("POST_CALL_ENABLE_NOTION_RECORD", "true")

    plan = _build_plan(
        call_id=DEMO_POST_CALL_CONTEXT["metadata"]["call_id"],
        tenant_id=DEMO_POST_CALL_CONTEXT["metadata"]["tenant_id"],
        summary=DEMO_LLM_SUMMARY,
        voc=DEMO_LLM_VOC,
        priority=DEMO_LLM_PRIORITY,
        customer_phone=DEMO_POST_CALL_CONTEXT["metadata"]["customer_phone"],
    )

    assert plan["action_required"] is True
    action_types = {a["action_type"] for a in plan["actions"]}

    # Calendar: 콜백 필요 → schedule_callback
    assert "schedule_callback" in action_types, f"schedule_callback 없음: {action_types}"
    # Slack: critical + angry+escalated
    assert "send_slack_alert" in action_types, f"send_slack_alert 없음: {action_types}"
    # SMS: 콜백 → send_callback_sms 또는 VOC → send_voc_receipt_sms
    sms_types = {"send_callback_sms", "send_voc_receipt_sms"}
    assert action_types & sms_types, f"SMS 액션 없음: {action_types}"
    # Notion: 통화 1건 = row 1개 — create_notion_call_record 하나만 생성
    assert "create_notion_call_record" in action_types, f"create_notion_call_record 없음: {action_types}"
    assert "create_notion_voc_record" not in action_types, f"create_notion_voc_record가 자동 생성되면 안 됨: {action_types}"
    notion_action_count = sum(1 for a in plan["actions"] if a["tool"] == "notion")
    assert notion_action_count == 1, f"Notion action은 1개여야 함, 실제: {notion_action_count}"

    monkeypatch.delenv("POST_CALL_ENABLE_NOTION_RECORD", raising=False)


@pytest.mark.asyncio
async def test_demo_context_customer_phone_in_sms_params(monkeypatch):
    """SMS 액션 params에 customer_phone이 포함된다."""
    from tests.fixtures.demo_post_call_context import (
        DEMO_POST_CALL_CONTEXT, DEMO_LLM_SUMMARY, DEMO_LLM_VOC, DEMO_LLM_PRIORITY,
    )
    from app.agents.post_call.nodes.action_planner_node import _build_plan

    plan = _build_plan(
        call_id="demo-call-critical",
        tenant_id="demo-tenant",
        summary=DEMO_LLM_SUMMARY,
        voc=DEMO_LLM_VOC,
        priority=DEMO_LLM_PRIORITY,
        customer_phone="01099990000",
    )

    sms_actions = [a for a in plan["actions"] if a["tool"] == "sms"]
    assert len(sms_actions) > 0, "SMS 액션이 없음"
    for a in sms_actions:
        assert a["params"].get("customer_phone") == "01099990000", \
            f"customer_phone 없음: {a['params']}"


@pytest.mark.asyncio
async def test_demo_notion_record_env_adds_call_record(monkeypatch):
    """POST_CALL_ENABLE_NOTION_RECORD=true이면 create_notion_call_record가 추가된다."""
    from tests.fixtures.demo_post_call_context import (
        DEMO_LLM_SUMMARY, DEMO_LLM_VOC, DEMO_LLM_PRIORITY,
    )
    from app.agents.post_call.nodes.action_planner_node import _build_plan

    monkeypatch.setenv("POST_CALL_ENABLE_NOTION_RECORD", "true")

    plan = _build_plan(
        call_id="demo-call-critical",
        tenant_id="demo-tenant",
        summary=DEMO_LLM_SUMMARY,
        voc=DEMO_LLM_VOC,
        priority=DEMO_LLM_PRIORITY,
        customer_phone="01099990000",
    )

    action_types = {a["action_type"] for a in plan["actions"]}
    assert "create_notion_call_record" in action_types

    monkeypatch.delenv("POST_CALL_ENABLE_NOTION_RECORD", raising=False)


@pytest.mark.asyncio
async def test_demo_all_actions_standard_format(monkeypatch):
    """demo plan의 모든 action이 표준 포맷을 유지한다."""
    from tests.fixtures.demo_post_call_context import (
        DEMO_LLM_SUMMARY, DEMO_LLM_VOC, DEMO_LLM_PRIORITY,
    )
    from app.agents.post_call.nodes.action_planner_node import _build_plan
    from app.agents.post_call.actions.executor import execute_actions

    plan = _build_plan(
        call_id="demo-fmt-001",
        tenant_id="demo-tenant",
        summary=DEMO_LLM_SUMMARY,
        voc=DEMO_LLM_VOC,
        priority=DEMO_LLM_PRIORITY,
        customer_phone="01099990000",
    )

    results = await execute_actions(
        call_id="demo-fmt-001",
        tenant_id="demo-tenant",
        actions=plan["actions"],
    )

    for r in results:
        for key in ("action_type", "tool", "status", "external_id", "error", "result"):
            assert key in r, f"action {r.get('action_type')} 에 {key!r} 키 없음"


@pytest.mark.asyncio
async def test_demo_slack_failure_does_not_stop_others(monkeypatch):
    """Slack 실패가 Calendar/SMS/Notion 실행을 막지 않는다."""
    from app.agents.post_call.actions.executor import execute_actions
    from app.agents.post_call.actions.registry import register, unregister

    class ExplodingSlack:
        async def execute(self, action, *, call_id, tenant_id=""):
            raise RuntimeError("Slack 의도적 폭발")

    register("slack_demo_fail", ExplodingSlack())
    try:
        actions = [
            {"action_type": "send_slack_alert",          "tool": "slack_demo_fail", "params": {}, "status": "pending"},
            {"action_type": "schedule_callback",         "tool": "calendar",        "params": {}, "status": "pending"},
            {"action_type": "send_callback_sms",         "tool": "sms",             "params": {"customer_phone": "01099990000"}, "status": "pending"},
            {"action_type": "create_notion_call_record", "tool": "notion",          "params": {}, "status": "pending"},
        ]
        results = await execute_actions(call_id="slack-fail-demo", tenant_id="demo-tenant", actions=actions)
        assert len(results) == 4
        assert results[0]["status"] == "failed"   # slack 폭발
        assert results[1]["status"] == "success"  # calendar ok
        assert results[2]["status"] == "success"  # sms ok
        assert results[3]["status"] == "success"  # notion ok
    finally:
        unregister("slack_demo_fail")


@pytest.mark.asyncio
async def test_demo_full_flow_with_mock_llm(monkeypatch):
    """demo context + mock LLM으로 전체 Post-call 플로우 실행 → executed_actions 검증."""
    from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call
    from tests.fixtures.demo_post_call_context import (
        DEMO_POST_CALL_CONTEXT, DEMO_LLM_SUMMARY, DEMO_LLM_VOC, DEMO_LLM_PRIORITY,
        DEMO_LLM_ANALYSIS, DEMO_LLM_REVIEW_PASS,
    )
    import copy

    ctx = copy.deepcopy(DEMO_POST_CALL_CONTEXT)

    async def fake_db(call_id, tenant_id=None):
        return copy.deepcopy(ctx)

    monkeypatch.setattr(
        "app.agents.post_call.context_provider.get_completed_call_context_from_db",
        fake_db,
    )

    class DemoMockLLM:
        async def call_json(self, system_prompt, user_message, max_tokens=1024):
            # 새 통합 분석 노드 (우선 검사)
            if "ANALYSIS_COMBINED" in system_prompt:
                return DEMO_LLM_ANALYSIS
            # 리뷰 게이트
            if "REVIEW_VERDICT" in system_prompt:
                return DEMO_LLM_REVIEW_PASS
            # 하위 호환
            if "summary_short" in system_prompt:
                return DEMO_LLM_SUMMARY
            if "sentiment_result" in system_prompt:
                return DEMO_LLM_VOC
            return DEMO_LLM_PRIORITY

    import app.agents.post_call.nodes.post_call_analysis_node as analysis_mod
    import app.agents.post_call.nodes.review_node as review_node_mod
    import app.agents.post_call.nodes.summary_node as summary_mod
    import app.agents.post_call.nodes.voc_analysis_node as voc_mod
    import app.agents.post_call.nodes.priority_node as priority_mod

    mock_llm = DemoMockLLM()
    monkeypatch.setattr(analysis_mod, "_caller", mock_llm)
    monkeypatch.setattr(review_node_mod, "_caller", mock_llm)
    monkeypatch.setattr(summary_mod, "_caller", mock_llm)
    monkeypatch.setattr(voc_mod, "_caller", mock_llm)
    monkeypatch.setattr(priority_mod, "_caller", mock_llm)

    result = await run_post_call_for_completed_call(
        call_id="demo-call-critical",
        tenant_id="demo-tenant",
        trigger="call_ended",
    )

    assert result["ok"] is True
    agent_result = result["result"]
    executed = agent_result.get("executed_actions", [])
    assert isinstance(executed, list)
    assert len(executed) > 0, "executed_actions가 비어 있음"

    action_types = {a["action_type"] for a in executed}
    assert "schedule_callback" in action_types, f"schedule_callback 없음: {action_types}"
    assert "send_slack_alert" in action_types, f"send_slack_alert 없음: {action_types}"

    # 모든 action이 표준 6-key 포맷
    for a in executed:
        for key in ("action_type", "tool", "status", "external_id", "error", "result"):
            assert key in a, f"action {a.get('action_type')} 에 {key!r} 키 없음"
