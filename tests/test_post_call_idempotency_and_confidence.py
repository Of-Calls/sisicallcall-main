"""KDT-73 — Slack/SMS 중복 발송 차단 + Reviewer R1~R4 + confidence 강등 단위 테스트.

작업 A (idempotency 좁힘):
  send_slack_alert / send_voc_receipt_sms / send_manager_email 의 idempotency_token 이
  (call_id, action_type) 단위로 좁혀져, message/subject 가 달라도 token 동일.

작업 B (R1~R4 + confidence):
  reviewer_agent_node 가 finalize_review 의 confidence 값으로 verdict 자동 강등.

LLM 은 monkeypatch 로 결정론적 fake 로 교체.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.agents.post_call.nodes.analysis_planner_agent_node as planner_mod
import app.agents.post_call.nodes.reviewer_agent_node as reviewer_mod
import app.agents.post_call.tools.action_catalog as catalog_mod


# ── 픽스처 ────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_llm_singletons(monkeypatch):
    monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)
    planner_mod._llm = None
    reviewer_mod._llm = None
    yield
    planner_mod._llm = None
    reviewer_mod._llm = None


def _patch_full_catalog(monkeypatch):
    from app.models.tenant_integration import IntegrationStatus
    fake = [
        SimpleNamespace(provider=p, status=IntegrationStatus.connected)
        for p in ("slack", "google_calendar", "jira", "gmail")
    ]
    monkeypatch.setattr(catalog_mod, "list_integrations", lambda tenant_id: fake)


def _record_args() -> dict:
    return {
        "summary_short": "x", "customer_intent": "x",
        "customer_emotion": "angry", "resolution_status": "escalated",
        "priority": "high", "primary_category": "민원/불만",
    }


def _planner_state(transcripts: list[dict] | None = None) -> dict:
    return {
        "call_id": "test-call-001",
        "tenant_id": "test-tenant",
        "trigger": "call_ended",
        "call_metadata": {},
        "transcripts": transcripts if transcripts is not None else [
            {"role": "customer", "text": "환불 요청 정말 화가 납니다"},
        ],
        "branch_stats": {},
        "summary": None, "voc_analysis": None, "priority_result": None,
        "action_plan": None, "executed_actions": [],
        "dashboard_payload": None, "errors": [], "partial_success": False,
        "analysis_result": None, "proposed_actions": [],
        "analysis_planner_rationale": "", "analysis_llm_usage": None,
        "review_result": None, "review_verdict": None,
        "approved_actions": [], "corrections_to_analysis": {},
        "escalate_reason": None, "reviewer_steps": 0,
        "review_llm_usage": None, "human_review_required": False,
        "analysis_retry_count": 0, "review_feedback": [],
        "blocked_actions": [], "review_retry_count": 0,
    }


def _reviewer_state(*, proposed: list[dict]) -> dict:
    base = _planner_state()
    base["analysis_result"] = {
        "summary": {
            "summary_short": "환불 요청",
            "summary_detailed": "고객 환불 요청",
            "customer_intent": "환불 요청",
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "keywords": ["환불"],
            "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "angry", "intensity": 0.7, "reason": ""},
            "intent_result": {"primary_category": "민원/불만", "sub_categories": [],
                              "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "high", "action_required": True,
                                "suggested_action": None, "reason": ""},
        },
        "priority_result": {"priority": "high", "tier": "high",
                            "action_required": True, "suggested_action": None,
                            "reason": ""},
    }
    base["proposed_actions"] = proposed
    return base


# ────────────────────────────────────────────────────────────────────────────
# 작업 A — Slack / SMS / Email 중복 발송 차단
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_alert_no_duplicate_same_intent(monkeypatch):
    """LLM 이 같은 의도로 send_slack_alert 를 5번 propose (message 표현만 다름) →
    옵션 1 정책에 따라 모든 token 이 동일해야 한다.

    executor 가 첫 시도 success 후 나머지 4건은 idempotency skip 됨.
    """
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
            {"id": "s1", "name": "propose_send_slack_alert",
             "arguments": {"message": "강한 불만 통화 검토 필요"}},
            {"id": "s2", "name": "propose_send_slack_alert",
             "arguments": {"message": "고객이 환불을 요청하며 강하게 항의"}},
            {"id": "s3", "name": "propose_send_slack_alert",
             "arguments": {"message": "민원 제기 의사 표명 — 즉시 보고 필요"}},
            {"id": "s4", "name": "propose_send_slack_alert",
             "arguments": {"message": "고객 분노 표출 — 후속 조치 요청"}},
            {"id": "s5", "name": "propose_send_slack_alert",
             "arguments": {"message": "강한 escalation 발생"}},
        ],
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    result = await planner_mod.analysis_planner_agent_node(_planner_state())

    slack_actions = [a for a in result["proposed_actions"]
                     if a["action_type"] == "send_slack_alert"]
    assert len(slack_actions) == 5, "planner 가 5건 모두 propose 했어야 함"
    tokens = {a["idempotency_token"] for a in slack_actions}
    assert len(tokens) == 1, (
        f"같은 의도의 send_slack_alert 5건은 token 1개여야 함. 실제: {tokens}"
    )


@pytest.mark.asyncio
async def test_sms_no_duplicate_same_intent(monkeypatch):
    """e2e-001 재현: LLM 이 propose_send_sms_followup 를 7번 (각 message 약간씩 다름)
    호출 → 모든 token 동일 → 1건만 발송됨."""
    _patch_full_catalog(monkeypatch)

    messages = [
        "환불 요청이 상부에 보고되었습니다.",
        "불만 사항이 접수되었습니다.",
        "환불 요청 검토 후 연락드리겠습니다.",
        "민원 관련 추가 안내가 필요하시면 연락주세요.",
        "환불 요청이 상부에 보고되었습니다. 검토 중입니다.",
        "불만 사항이 접수되었으며 빠르게 처리하겠습니다.",
        "환불 요청이 상부에 보고되었습니다. 빠른 시일 내 연락드립니다.",
    ]
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
            *[
                {"id": f"sm{i}", "name": "propose_send_sms_followup",
                 "arguments": {"phone": "01000000000", "message": m}}
                for i, m in enumerate(messages)
            ],
        ],
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    result = await planner_mod.analysis_planner_agent_node(_planner_state())

    sms_actions = [a for a in result["proposed_actions"]
                   if a["action_type"] == "send_voc_receipt_sms"]
    # _MAX_TOOL_CALLS=8 → record_analysis 1 + sms 7 = 8 정확히 한도
    assert len(sms_actions) >= 1
    tokens = {a["idempotency_token"] for a in sms_actions}
    assert len(tokens) == 1, (
        f"같은 의도의 send_voc_receipt_sms 다중 호출은 token 1개여야 함. 실제: {tokens}"
    )


@pytest.mark.asyncio
async def test_manager_email_no_duplicate_same_intent(monkeypatch):
    """옵션 4: send_manager_email 의 idempotency 도 (call_id, action_type) 단일.
    subject/body 가 달라도 token 동일 → 한 통화당 supervisor 알림 1건."""
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
            {"id": "e1", "name": "propose_send_email_supervisor",
             "arguments": {"subject": "[ALERT] 강한 불만 통화 — 환불 요청",
                           "body": "강한 불만 표출"}},
            {"id": "e2", "name": "propose_send_email_supervisor",
             "arguments": {"subject": "[ALERT] 민원 가능성 통화",
                           "body": "민원 신고 의사 표명"}},
            {"id": "e3", "name": "propose_send_email_supervisor",
             "arguments": {"subject": "[ALERT] 환불 escalation",
                           "body": "검토 요청"}},
        ],
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    result = await planner_mod.analysis_planner_agent_node(_planner_state())

    email_actions = [a for a in result["proposed_actions"]
                     if a["action_type"] == "send_manager_email"]
    assert len(email_actions) == 3
    tokens = {a["idempotency_token"] for a in email_actions}
    assert len(tokens) == 1, (
        f"send_manager_email 다중 호출은 token 1개여야 함. 실제: {tokens}"
    )


def test_idempotency_field_table_options_1_and_4():
    """Regression guard: 안내성 액션 3종의 _IDEMPOTENCY_FIELDS 가 빈 리스트인지 확인.

    이 값이 다시 message/subject 를 포함하면 e2e-001 의 중복 발송 회귀 발생.
    """
    fields = planner_mod._IDEMPOTENCY_FIELDS
    assert fields["send_slack_alert"] == []
    assert fields["send_voc_receipt_sms"] == []
    assert fields["send_manager_email"] == []
    # 의도 분리가 가능한 action_type 은 fields 유지
    assert fields["create_jira_issue"] == ["summary"]
    assert fields["schedule_callback"] == ["preferred_time"]
    assert fields["create_voc_issue"] == ["voc_content"]


# ────────────────────────────────────────────────────────────────────────────
# 작업 B — Reviewer R1~R4 fail 판정 + confidence 강등
# ────────────────────────────────────────────────────────────────────────────


def _reviewer_fail_response(reason: str, *, confidence: float | None = None) -> dict:
    args: dict = {"verdict": "fail", "summary_reason": reason}
    if confidence is not None:
        args["confidence"] = confidence
    return {
        "tool_calls": [
            {"id": "e", "name": "escalate_to_human", "arguments": {"reason": reason}},
            {"id": "f", "name": "finalize_review", "arguments": args},
        ],
        "text": "", "raw_message": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "mock"},
    }


def _reviewer_pass_response(*, confidence: float | None = None,
                            approve_action_ids: list[str] | None = None) -> dict:
    calls: list[dict] = []
    for aid in (approve_action_ids or []):
        calls.append({"id": f"a_{aid}", "name": "approve_action",
                      "arguments": {"action_id": aid}})
    final_args: dict = {"verdict": "pass", "summary_reason": "ok"}
    if confidence is not None:
        final_args["confidence"] = confidence
    calls.append({"id": "f", "name": "finalize_review", "arguments": final_args})
    return {
        "tool_calls": calls,
        "text": "", "raw_message": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "mock"},
    }


@pytest.mark.asyncio
async def test_reviewer_fail_on_grounding_violation_R1(monkeypatch):
    """R1 — 분석이 transcript 와 모순. reviewer 가 fail finalize → verdict=fail."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_fail_response(
            "R1 grounding 위반 — transcript 에 '환불' 단어 없는데 intent='환불 요청'"
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "high",
         "params": {"message": "x"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True
    assert "R1" in (result["escalate_reason"] or "")
    assert result["approved_actions"] == []


@pytest.mark.asyncio
async def test_reviewer_fail_on_action_mismatch_R2(monkeypatch):
    """R2 — 액션 params 가 transcript 와 불일치. fail."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_fail_response(
            "R2 action mismatch — 단순 운영시간 문의에 jira_issue 부적절"
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "create_jira_issue", "tool": "jira", "priority": "critical",
         "params": {"summary": "x", "description": "y"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "fail"
    assert "R2" in (result["escalate_reason"] or "")


@pytest.mark.asyncio
async def test_reviewer_fail_on_risk_escalation_R3(monkeypatch):
    """R3 — 위험 액션 (자동 환불 처리 등). fail + escalate."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_fail_response(
            "R3 risk escalation — 본인인증 우회 자동 처리 위험"
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "critical",
         "params": {"message": "본인인증 우회 처리 안내"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "fail"
    assert "R3" in (result["escalate_reason"] or "")


@pytest.mark.asyncio
async def test_reviewer_confidence_downgrade_pass_to_correctable(monkeypatch):
    """R4 — verdict=pass + confidence=0.55 → correctable 강등."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_pass_response(
            confidence=0.55,
            approve_action_ids=["a0_send_slack_alert_slack"],
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "high",
         "params": {"message": "x"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "correctable"
    rr = result["review_result"]
    assert rr["original_verdict"] == "pass"
    assert rr["confidence"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_reviewer_confidence_downgrade_pass_to_fail(monkeypatch):
    """R4 — verdict=pass + confidence=0.35 → fail 강등."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_pass_response(
            confidence=0.35,
            approve_action_ids=["a0_send_slack_alert_slack"],
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "high",
         "params": {"message": "x"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True
    rr = result["review_result"]
    assert rr["original_verdict"] == "pass"
    assert rr["confidence"] == pytest.approx(0.35)
    assert "low_confidence" in (result["escalate_reason"] or "")
    # fail 강등 시 외부 액션은 차단
    assert result["approved_actions"] == []


@pytest.mark.asyncio
async def test_reviewer_no_downgrade_when_confidence_high(monkeypatch):
    """R4 — confidence=0.85 → verdict 그대로 pass."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_pass_response(
            confidence=0.85,
            approve_action_ids=["a0_send_slack_alert_slack"],
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "high",
         "params": {"message": "x"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "pass"
    rr = result["review_result"]
    assert rr["original_verdict"] == "pass"
    assert rr["confidence"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_reviewer_confidence_optional_backward_compat(monkeypatch):
    """R4 — finalize_review 에 confidence 없이 호출 → 강등 없음, 기존 호환."""
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(
        return_value=_reviewer_pass_response(
            confidence=None,
            approve_action_ids=["a0_send_slack_alert_slack"],
        )
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _reviewer_state(proposed=[
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "high",
         "params": {"message": "x"}, "status": "pending"},
    ])
    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "pass"
    rr = result["review_result"]
    assert rr["confidence"] is None
    assert rr["original_verdict"] == "pass"


def test_apply_confidence_downgrade_boundaries():
    """강등 함수 단위 검증: 경계값 + None + fail 보존."""
    dg = reviewer_mod._apply_confidence_downgrade
    # confidence=None → 무변경
    assert dg("pass", None) == "pass"
    assert dg("correctable", None) == "correctable"
    assert dg("fail", None) == "fail"
    # 0.4 미만 → fail
    assert dg("pass", 0.39) == "fail"
    assert dg("correctable", 0.39) == "fail"
    # 0.4 이상 0.6 미만 + pass → correctable
    assert dg("pass", 0.4) == "correctable"
    assert dg("pass", 0.55) == "correctable"
    # 0.6 이상 → 무변경
    assert dg("pass", 0.6) == "pass"
    assert dg("pass", 0.85) == "pass"
    # fail 은 그대로 (재상승 금지)
    assert dg("fail", 1.0) == "fail"
    # correctable + 0.4~0.6 → 그대로 correctable
    assert dg("correctable", 0.55) == "correctable"


def test_coerce_confidence_handles_bad_input():
    """LLM 이 잘못된 값을 보내도 안전 처리."""
    cc = reviewer_mod._coerce_confidence
    assert cc(None) is None
    assert cc("not a number") is None
    assert cc(float("nan")) is None
    assert cc(-1.0) == 0.0
    assert cc(2.5) == 1.0
    assert cc(0.5) == 0.5
    assert cc("0.7") == 0.7


# ────────────────────────────────────────────────────────────────────────────
# find_existing_action — status 무관 idempotency 매칭
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_idempotency_blocks_skipped_replay(monkeypatch):
    """첫 호출이 sms_config_missing 으로 skipped INSERT → 두 번째 호출 (같은 token) 차단.

    e2e-complex-001 에서 관찰된 패턴 — SMS 4 row 모두 skipped 으로 쌓이는 것을 방지.
    executor 가 status='skipped' 인 row 도 idempotency match 로 인식해야 한다.
    """
    import app.agents.post_call.actions.executor as executor_mod

    action = {
        "action_type": "send_voc_receipt_sms",
        "tool": "sms",
        "priority": "high",
        "params": {"customer_phone": "01000000000", "message": "x"},
        "idempotency_token": "44136fa355b3678a",  # 옵션 1 의 fields=[] hash
    }

    # 첫 호출에서 skipped row 가 DB 에 들어간 상태를 가정
    db_rows: list[dict] = [{
        "call_id": "c-skip-001",
        "action_type": "send_voc_receipt_sms",
        "tool_name": "sms",
        "status": "skipped",
        "error_message": "sms_config_missing",
        "external_id": None,
        "request_payload": {"idempotency_token": "44136fa355b3678a"},
    }]

    async def fake_find_existing(call_id, action_type, tool, idempotency_token=None):
        for r in db_rows:
            if (r["call_id"] == call_id and r["action_type"] == action_type
                    and r["tool_name"] == tool
                    and r["request_payload"].get("idempotency_token") == idempotency_token):
                return r
        return None

    monkeypatch.setattr(executor_mod, "find_existing_action", fake_find_existing)

    # gateway 가 호출되면 안 됨 (idempotency skip)
    gateway_calls = []
    from app.services.mcp.connectors import mcp_gateway_connector as mgc

    async def fake_gateway_execute(self, action, *, call_id, tenant_id):
        gateway_calls.append(action)
        return {"status": "success", "external_id": "x", "result": {}}

    monkeypatch.setattr(mgc.MCPGatewayConnector, "execute", fake_gateway_execute)

    executor = executor_mod.ActionExecutor()
    results = await executor.execute_actions(
        call_id="c-skip-001", tenant_id="t", actions=[action]
    )

    assert len(results) == 1
    assert results[0]["status"] == "skipped"
    assert results[0]["error"].startswith("already_attempted"), (
        f"skipped row 매칭 시 reason='already_attempted(skipped)' 여야 함. 실제: {results[0]['error']}"
    )
    assert gateway_calls == [], "idempotency 차단 시 gateway 호출 없어야 함"
    # result 에 previous_status 포함
    assert results[0]["result"]["previous_status"] == "skipped"
    assert results[0]["result"]["idempotency"] == "already_attempted(skipped)"


@pytest.mark.asyncio
async def test_idempotency_existing_success_row_preserved(monkeypatch):
    """기존 success row 매칭 시 reason='already_succeeded' 유지 (기존 동작 보존)."""
    import app.agents.post_call.actions.executor as executor_mod

    action = {
        "action_type": "send_slack_alert",
        "tool": "slack",
        "priority": "high",
        "params": {"message": "x"},
        "idempotency_token": "44136fa355b3678a",
    }

    db_rows = [{
        "call_id": "c-succ-001",
        "action_type": "send_slack_alert",
        "tool_name": "slack",
        "status": "success",
        "error_message": None,
        "external_id": "C:ts",
        "request_payload": {"idempotency_token": "44136fa355b3678a"},
    }]

    async def fake_find_existing(call_id, action_type, tool, idempotency_token=None):
        for r in db_rows:
            if (r["call_id"] == call_id and r["action_type"] == action_type
                    and r["tool_name"] == tool
                    and r["request_payload"].get("idempotency_token") == idempotency_token):
                return r
        return None

    monkeypatch.setattr(executor_mod, "find_existing_action", fake_find_existing)

    from app.services.mcp.connectors import mcp_gateway_connector as mgc

    async def fake_gateway_execute(self, action, *, call_id, tenant_id):
        raise AssertionError("gateway should not be called for idempotent action")

    monkeypatch.setattr(mgc.MCPGatewayConnector, "execute", fake_gateway_execute)

    executor = executor_mod.ActionExecutor()
    results = await executor.execute_actions(
        call_id="c-succ-001", tenant_id="t", actions=[action]
    )

    assert results[0]["status"] == "skipped"
    assert results[0]["error"] == "already_succeeded"
    assert results[0]["result"]["idempotency"] == "already_succeeded"
    assert results[0]["result"]["previous_status"] == "success"
    assert results[0]["result"]["previous_external_id"] == "C:ts"


@pytest.mark.asyncio
async def test_idempotency_no_existing_row_proceeds(monkeypatch):
    """매칭되는 row 없으면 정상 gateway 호출 (회귀: 첫 호출은 차단되면 안 됨)."""
    import app.agents.post_call.actions.executor as executor_mod

    action = {
        "action_type": "send_slack_alert",
        "tool": "slack",
        "priority": "high",
        "params": {"message": "first"},
        "idempotency_token": "abc",
    }

    async def fake_find_existing(call_id, action_type, tool, idempotency_token=None):
        return None  # 첫 호출 — 매칭 없음

    monkeypatch.setattr(executor_mod, "find_existing_action", fake_find_existing)

    sent = []
    from app.services.mcp.connectors import mcp_gateway_connector as mgc

    async def fake_gateway_execute(self, action, *, call_id, tenant_id):
        sent.append(action)
        return {"status": "success", "external_id": "ok", "result": {}}

    monkeypatch.setattr(mgc.MCPGatewayConnector, "execute", fake_gateway_execute)

    executor = executor_mod.ActionExecutor()
    results = await executor.execute_actions(
        call_id="c-new-001", tenant_id="t", actions=[action]
    )

    assert len(sent) == 1, "첫 호출은 차단되면 안 됨"
    assert results[0]["status"] == "success"


@pytest.mark.asyncio
async def test_idempotency_failed_row_also_blocks(monkeypatch):
    """status='failed' row 도 차단 (재시도 의미 없음)."""
    import app.agents.post_call.actions.executor as executor_mod

    action = {
        "action_type": "send_slack_alert",
        "tool": "slack",
        "params": {"message": "x"},
        "idempotency_token": "abc",
    }

    db_rows = [{
        "call_id": "c-fail-001",
        "action_type": "send_slack_alert",
        "tool_name": "slack",
        "status": "failed",
        "error_message": "channel_not_found",
        "external_id": None,
        "request_payload": {"idempotency_token": "abc"},
    }]

    async def fake_find_existing(call_id, action_type, tool, idempotency_token=None):
        for r in db_rows:
            if (r["call_id"] == call_id and r["action_type"] == action_type
                    and r["tool_name"] == tool
                    and r["request_payload"].get("idempotency_token") == idempotency_token):
                return r
        return None

    monkeypatch.setattr(executor_mod, "find_existing_action", fake_find_existing)

    from app.services.mcp.connectors import mcp_gateway_connector as mgc

    async def fake_gateway_execute(self, action, *, call_id, tenant_id):
        raise AssertionError("gateway should not be called")

    monkeypatch.setattr(mgc.MCPGatewayConnector, "execute", fake_gateway_execute)

    executor = executor_mod.ActionExecutor()
    results = await executor.execute_actions(
        call_id="c-fail-001", tenant_id="t", actions=[action]
    )

    assert results[0]["status"] == "skipped"
    assert results[0]["error"] == "already_attempted(failed)"
    assert results[0]["result"]["previous_status"] == "failed"


def test_find_existing_action_module_exports():
    """app.repositories 의 find_existing_action export 보장."""
    from app.repositories import find_existing_action, find_successful_action
    # 두 함수 모두 import 가능 — backward compat 보장
    assert callable(find_existing_action)
    assert callable(find_successful_action)
    # 클래스 메서드도 노출
    from app.repositories import MCPActionLogRepository
    assert hasattr(MCPActionLogRepository, "find_existing_action")
    assert hasattr(MCPActionLogRepository, "find_successful_action")
