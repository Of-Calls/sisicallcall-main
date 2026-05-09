"""KDT-73 — analysis_planner_agent + reviewer_agent 단위/통합 테스트.

LLM 은 모두 monkeypatch 로 결정론적 fake 로 교체. tenant_integration_repo 도
fake 로 교체하여 OAuth 카탈로그 필터를 명시 테스트한다.
"""
from __future__ import annotations

import copy
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
    planner_mod._llm = None
    reviewer_mod._llm = None
    yield
    planner_mod._llm = None
    reviewer_mod._llm = None


def _state(
    *,
    call_id: str = "t-001",
    tenant_id: str = "t-tenant",
    transcripts: list[dict] | None = None,
    analysis: dict | None = None,
    proposed: list[dict] | None = None,
) -> dict:
    return {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "trigger": "call_ended",
        "call_metadata": {},
        "transcripts": transcripts if transcripts is not None else [],
        "branch_stats": {},
        "summary": None,
        "voc_analysis": None,
        "priority_result": None,
        "action_plan": None,
        "executed_actions": [],
        "dashboard_payload": None,
        "errors": [],
        "partial_success": False,
        "analysis_result": analysis,
        "proposed_actions": proposed or [],
        "analysis_planner_rationale": "",
        "analysis_llm_usage": None,
        "review_result": None,
        "review_verdict": None,
        "approved_actions": [],
        "corrections_to_analysis": {},
        "escalate_reason": None,
        "reviewer_steps": 0,
        "review_llm_usage": None,
        "human_review_required": False,
        "blocked_actions": [],
        "review_retry_count": 0,
    }


def _patch_full_catalog(monkeypatch):
    """list_integrations 를 모든 OAuth provider connected 로 fake."""
    fake = [
        SimpleNamespace(provider=p, status=SimpleNamespace(value="connected"))
        for p in ("slack", "google_calendar", "jira", "gmail")
    ]
    # IntegrationStatus.connected 비교: action_catalog 에서 i.status == IntegrationStatus.connected 비교
    from app.models.tenant_integration import IntegrationStatus
    fake = [
        SimpleNamespace(provider=p, status=IntegrationStatus.connected)
        for p in ("slack", "google_calendar", "jira", "gmail")
    ]
    monkeypatch.setattr(catalog_mod, "list_integrations", lambda tenant_id: fake)


def _patch_no_oauth(monkeypatch, *, exclude: tuple[str, ...] = ()):
    """slack 만 미연결로 fake (예시) — exclude 지정한 provider 만 connected 에서 빼고 나머지는 connected."""
    from app.models.tenant_integration import IntegrationStatus

    all_providers = ("slack", "google_calendar", "jira", "gmail")
    fake = [
        SimpleNamespace(provider=p, status=IntegrationStatus.connected)
        for p in all_providers
        if p not in exclude
    ]
    monkeypatch.setattr(catalog_mod, "list_integrations", lambda tenant_id: fake)


# ────────────────────────────────────────────────────────────────────────────
# analysis_planner_agent
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_planner_happy_path_angry_customer(monkeypatch):
    """화난 고객 transcript → emotion=angry, priority=high(critical+)
    + propose_send_slack_alert + (옵션) propose_send_email_supervisor."""
    _patch_full_catalog(monkeypatch)

    state = _state(transcripts=[
        {"role": "customer", "text": "이거 진짜 화나네요. 환불 안 해주면 민원 넣을 거예요"},
        {"role": "agent", "text": "죄송합니다, 처리해드릴게요."},
    ])

    result = await planner_mod.analysis_planner_agent_node(state)

    analysis = result["analysis_result"]
    assert analysis["summary"]["customer_emotion"] in ("angry", "negative")
    assert analysis["priority_result"]["priority"] in ("high", "critical")

    proposed = result["proposed_actions"]
    proposed_types = {a["action_type"] for a in proposed}
    # mock 은 angry 면 slack 을 반드시 propose
    assert "send_slack_alert" in proposed_types


@pytest.mark.asyncio
async def test_planner_simple_inquiry_no_external_action(monkeypatch):
    """단순 정보 문의 → propose_no_action + (Notion env 있으면) Notion call_record.

    Notion 기본 기록은 모든 통화에 propose 되므로 'no external action' = Notion 외
    연락 / 발송 액션 없음 의미. Notion 미연결 환경에서 propose_actions==[] 검증
    은 별도 테스트 (test_planner_notion_skipped_when_not_connected).
    """
    _patch_full_catalog(monkeypatch)
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

    state = _state(transcripts=[
        {"role": "customer", "text": "운영 시간이 어떻게 되나요"},
        {"role": "agent", "text": "오전 9시부터 오후 6시입니다"},
    ])

    result = await planner_mod.analysis_planner_agent_node(state)

    # Notion 미연결 + 단순 inquiry → 외부 액션 0
    assert result["proposed_actions"] == []
    assert "propose_no_action" in result["analysis_planner_rationale"]


@pytest.mark.asyncio
async def test_planner_tenant_without_slack_drops_slack_tool(monkeypatch):
    """tenant 가 Slack 미연결 → 카탈로그에 propose_send_slack_alert 빠짐 →
    angry 시나리오에서도 slack 액션 propose 안 됨."""
    _patch_no_oauth(monkeypatch, exclude=("slack",))

    state = _state(transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요. 환불 처리 안 해주면 민원 넣을 거예요"},
    ])

    result = await planner_mod.analysis_planner_agent_node(state)
    proposed_types = {a["action_type"] for a in result["proposed_actions"]}
    assert "send_slack_alert" not in proposed_types


@pytest.mark.asyncio
async def test_planner_max_tool_calls_enforced(monkeypatch):
    """LLM 이 N+ tool_call 을 시도해도 노드는 _MAX_TOOL_CALLS 이하만 처리."""
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    overflow = [
        {"id": f"x{i}", "name": "propose_send_slack_alert", "arguments": {"urgency": "warning", "message": "x"}}
        for i in range(10)
    ]
    overflow.insert(0, {
        "id": "rec",
        "name": "record_analysis",
        "arguments": {
            "summary_short": "x", "customer_intent": "x",
            "customer_emotion": "neutral", "resolution_status": "resolved",
            "priority": "low", "primary_category": "기타",
        },
    })
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": overflow, "text": "", "raw_message": None,
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)

    state = _state(transcripts=[{"role": "customer", "text": "test"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    # record_analysis(1) + propose(<=MAX-1) — _MAX_TOOL_CALLS 6 일 땐 propose 5 까지
    assert len(result["proposed_actions"]) <= planner_mod._MAX_TOOL_CALLS - 1


@pytest.mark.asyncio
async def test_planner_drops_unknown_tool_call(monkeypatch):
    """카탈로그에 없는 도구 호출은 drop + rationale 에 기록."""
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {
                "id": "rec",
                "name": "record_analysis",
                "arguments": {
                    "summary_short": "x", "customer_intent": "x",
                    "customer_emotion": "neutral", "resolution_status": "resolved",
                    "priority": "low", "primary_category": "기타",
                },
            },
            {"id": "evil", "name": "delete_database", "arguments": {}},
        ],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)

    state = _state(transcripts=[{"role": "customer", "text": "hi"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    assert result["proposed_actions"] == []
    assert "delete_database" in result["analysis_planner_rationale"]


@pytest.mark.asyncio
async def test_planner_empty_transcript_safe(monkeypatch):
    _patch_full_catalog(monkeypatch)

    state = _state(transcripts=[])
    result = await planner_mod.analysis_planner_agent_node(state)

    assert result["analysis_result"] is not None
    assert result["proposed_actions"] == []
    assert result["partial_success"] is True
    assert any(e.get("warning") == "empty_transcript" for e in result["errors"])


@pytest.mark.asyncio
async def test_planner_llm_failure_returns_fallback(monkeypatch):
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(side_effect=RuntimeError("LLM down"))
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)

    state = _state(transcripts=[{"role": "customer", "text": "test"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    assert result["analysis_result"] is not None
    assert result["proposed_actions"] == []
    assert result["human_review_required"] is True
    assert result["partial_success"] is True


# ────────────────────────────────────────────────────────────────────────────
# reviewer_agent
# ────────────────────────────────────────────────────────────────────────────


def _baseline_analysis() -> dict:
    return {
        "summary": {
            "summary_short": "환불 요청 통화",
            "summary_detailed": "고객이 환불을 요청함",
            "customer_intent": "환불 요청",
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "keywords": ["환불"],
            "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "angry", "intensity": 0.8, "reason": ""},
            "intent_result": {
                "primary_category": "환불/결제",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "high",
                "action_required": True,
                "suggested_action": None,
                "reason": "",
            },
        },
        "priority_result": {
            "priority": "high", "tier": "high",
            "action_required": True, "suggested_action": None, "reason": "",
        },
    }


def _angry_transcript() -> list[dict]:
    return [
        {"role": "customer", "text": "환불 처리해주세요. 정말 화가 납니다"},
        {"role": "agent", "text": "처리해드릴게요"},
    ]


@pytest.mark.asyncio
async def test_reviewer_pass_approves_all(monkeypatch):
    """transcript 와 분석/액션이 일치 → verdict=pass, 모두 approve."""
    proposed = [
        {
            "action_type": "send_slack_alert", "tool": "slack", "priority": "high",
            "params": {"call_id": "c1", "tenant_id": "t1", "channel_type": "warning",
                       "message": "[review] 환불 불만"},
            "status": "pending",
        }
    ]
    state = _state(
        transcripts=_angry_transcript(),
        analysis=_baseline_analysis(),
        proposed=proposed,
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "pass"
    assert len(result["approved_actions"]) == 1
    assert result["human_review_required"] is False


@pytest.mark.asyncio
async def test_reviewer_rejects_inappropriate_jira_for_simple_inquiry(monkeypatch):
    """단순 문의 transcript + critical jira 후보 → mock reviewer 가 jira 만 reject."""
    proposed = [
        {
            "action_type": "create_jira_issue", "tool": "jira", "priority": "critical",
            "params": {"call_id": "c1", "summary": "x", "description": "y"},
            "status": "pending",
        }
    ]
    analysis = {
        "summary": {
            "summary_short": "단순 문의",
            "summary_detailed": "운영시간 문의",
            "customer_intent": "운영시간 확인",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "keywords": ["운영시간"],
            "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.0, "reason": ""},
            "intent_result": {"primary_category": "운영시간/위치", "sub_categories": [],
                              "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "low", "action_required": False,
                                "suggested_action": None, "reason": ""},
        },
        "priority_result": {"priority": "low", "tier": "low", "action_required": False,
                            "suggested_action": None, "reason": ""},
    }
    state = _state(
        transcripts=[{"role": "customer", "text": "단순 문의 — 운영시간 알려주세요"}],
        analysis=analysis,
        proposed=proposed,
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    # jira 액션은 reject → approved 0, rejected 1
    assert result["review_verdict"] in ("correctable", "fail")
    assert len(result["approved_actions"]) == 0


@pytest.mark.asyncio
async def test_reviewer_max_steps_force_close_rejects_pending(monkeypatch):
    """max_steps 도달 → finalize_review 호출 안 된 채 종료, 결정 안 된 액션은 reject."""
    fake_llm = MagicMock()
    # 매 step 마다 re_read_transcript 만 호출하고 finalize 안 함
    async def loop_response(*args, **kwargs):
        return {
            "tool_calls": [{
                "id": "x",
                "name": "re_read_transcript",
                "arguments": {"query": "환불"},
            }],
            "text": "",
            "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=loop_response)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    proposed = [
        {"action_type": "send_slack_alert", "tool": "slack", "priority": "high",
         "params": {}, "status": "pending"},
    ]
    state = _state(
        transcripts=_angry_transcript(),
        analysis=_baseline_analysis(),
        proposed=proposed,
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["reviewer_steps"] == 5
    assert result["approved_actions"] == []
    assert result["review_verdict"] in ("correctable", "fail")


@pytest.mark.asyncio
async def test_reviewer_correct_analysis_records_corrections(monkeypatch):
    """correct_analysis 도구가 호출되면 corrections_to_analysis 와 analysis_result 양쪽에 반영."""
    fake_llm = MagicMock()

    state = _state(
        transcripts=[{"role": "customer", "text": "환불 요청합니다"}],
        analysis={
            "summary": {
                "summary_short": "환불 요청",
                "summary_detailed": "x",
                "customer_intent": "환불",
                "customer_emotion": "angry",
                "resolution_status": "escalated",
                "keywords": [],
                "handoff_notes": "고객이 폭언 함 — transcript 에 없음",
            },
            "voc_analysis": {
                "sentiment_result": {"sentiment": "angry", "intensity": 0.7, "reason": ""},
                "intent_result": {"primary_category": "환불/결제", "sub_categories": [],
                                  "is_repeat_topic": False, "faq_candidate": False},
                "priority_result": {"priority": "high", "action_required": True,
                                    "suggested_action": None, "reason": ""},
            },
            "priority_result": {"priority": "high", "tier": "high", "action_required": True,
                                "suggested_action": None, "reason": ""},
        },
        proposed=[],
    )

    async def correct_then_finalize(*args, **kwargs):
        return {
            "tool_calls": [
                {
                    "id": "c1", "name": "correct_analysis",
                    "arguments": {
                        "field": "summary.handoff_notes",
                        "new_value": "",
                        "reason": "transcript 에 폭언 없음",
                        "transcript_evidence": "환불 요청합니다",
                    },
                },
                {
                    "id": "f1", "name": "finalize_review",
                    "arguments": {"verdict": "correctable", "summary_reason": "handoff_notes 교정"},
                },
            ],
            "text": "",
            "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=correct_then_finalize)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "correctable"
    assert "summary.handoff_notes" in result["corrections_to_analysis"]
    # 빈 문자열은 _coerce_sentinel 에 의해 None 으로 변환됨
    assert result["analysis_result"]["summary"]["handoff_notes"] is None


@pytest.mark.asyncio
async def test_reviewer_escalate_to_human_returns_fail(monkeypatch):
    fake_llm = MagicMock()

    async def escalate_then_finalize(*args, **kwargs):
        return {
            "tool_calls": [
                {"id": "e1", "name": "escalate_to_human",
                 "arguments": {"reason": "위험한 액션 — 자동 판단 불가"}},
                {"id": "f1", "name": "finalize_review",
                 "arguments": {"summary_reason": "fail"}},
            ],
            "text": "", "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=escalate_then_finalize)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _state(
        transcripts=_angry_transcript(),
        analysis=_baseline_analysis(),
        proposed=[{"action_type": "send_slack_alert", "tool": "slack",
                   "priority": "high", "params": {}, "status": "pending"}],
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    assert result["review_verdict"] == "fail"
    assert result["approved_actions"] == []
    assert result["human_review_required"] is True
    assert result["escalate_reason"] is not None


# ────────────────────────────────────────────────────────────────────────────
# 통합: 그래프 전체 + save_intermediate 보장
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_save_intermediate_runs_before_reviewer(monkeypatch):
    """save_intermediate 가 reviewer 전에 호출되어 분석이 저장되는지 확인.

    reviewer 가 fail 을 반환하더라도 call_summaries 에 분석이 남아 있어야 한다.
    """
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.load_context_node as lcn

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "g-001", "tenant_id": "t-graph"},
        "transcripts": [{"role": "customer", "text": "환불 안 해주면 민원 넣을게요"}],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    save_calls: list[str] = []

    async def fake_save(state, mode="final"):
        save_calls.append(mode)
        return {"dashboard_payload": {"save_mode": mode}, "errors": list(state.get("errors", []))}

    import app.agents.post_call.nodes.save_result_node as srm
    monkeypatch.setattr(srm, "save_result_node", fake_save)

    # reviewer 강제 fail
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [{"id": "x", "name": "escalate_to_human",
                        "arguments": {"reason": "test fail"}},
                       {"id": "f", "name": "finalize_review",
                        "arguments": {"summary_reason": "fail"}}],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # 그래프 재컴파일 — fake save_result_node 가 사용되도록
    from app.agents.post_call.graph import build_post_call_graph
    agent = PostCallAgent()
    agent._graph = build_post_call_graph()

    result = await agent.run("g-001", trigger="call_ended", tenant_id="t-graph")

    # 두 번 저장 — intermediate(1) + final(1)
    assert save_calls == ["intermediate", "final"]
    assert result["review_verdict"] == "fail"


@pytest.mark.asyncio
async def test_graph_pass_executes_actions(monkeypatch):
    """reviewer pass → action_executor 가 호출되어 executed_actions 에 결과 누적."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.load_context_node as lcn
    import app.agents.post_call.nodes.action_executor_node as exec_node

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "g-002"},
        "transcripts": [{"role": "customer", "text": "환불 안 해주면 정말 화나요"}],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    # ActionExecutor.execute_actions 를 fake — 외부 호출 X
    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[{
        "action_type": "send_slack_alert", "tool": "slack",
        "status": "success", "external_id": "fake-1",
        "error": None, "result": {"via_mcp": False},
    }])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    # save_result_node 는 그대로 (in-memory store) — 실제 동작 확인용
    agent = PostCallAgent()
    result = await agent.run("g-002", trigger="call_ended", tenant_id="t-graph2")

    assert result["review_verdict"] in ("pass", "correctable")
    # mock planner 가 angry 시나리오로 propose → reviewer mock approve →
    # fake executor 호출
    if result["approved_actions"]:
        assert fake_executor.execute_actions.await_count >= 1


# ════════════════════════════════════════════════════════════════════════════
# v2 회귀 — D-1 ~ D-6 fix 검증
# ════════════════════════════════════════════════════════════════════════════


def _record_args(*, priority="low", emotion="neutral", resolution="resolved", category="기타"):
    return {
        "summary_short": "x",
        "customer_intent": "x",
        "customer_emotion": emotion,
        "resolution_status": resolution,
        "priority": priority,
        "primary_category": category,
        "action_required": False,
        "is_repeat_topic": False,
        "faq_candidate": False,
        "keywords": [],
        "handoff_notes": "",
    }


# ── D-1: ISO 강제 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_d1_planner_natural_language_time_dropped(monkeypatch):
    """LLM 이 자연어 preferred_time 보내면 빈 문자열로 강제 + human_review_required."""
    _patch_full_catalog(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args(category="예약/일정")},
            {"id": "cb", "name": "propose_schedule_callback", "arguments": {
                "preferred_time": "내일 오후 3시",
                "phone": "01012345678",
                "reason": "콜백",
            }},
        ],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)

    state = _state(transcripts=[{"role": "customer", "text": "콜백 부탁"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    assert proposed[0]["params"]["preferred_time"] == ""
    assert result.get("human_review_required") is True
    assert "schedule_callback_invalid_time_format" in result["analysis_planner_rationale"]


@pytest.mark.asyncio
async def test_d1_planner_iso_time_passes_through(monkeypatch):
    """이미 ISO 포맷이면 그대로 통과 + human_review_required 미설정."""
    _patch_full_catalog(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args(category="예약/일정")},
            {"id": "cb", "name": "propose_schedule_callback", "arguments": {
                "preferred_time": "2026-05-10 15:00",
                "phone": "01012345678",
                "reason": "콜백",
            }},
        ],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)

    state = _state(transcripts=[{"role": "customer", "text": "콜백 부탁"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    assert proposed[0]["params"]["preferred_time"] == "2026-05-10 15:00"
    assert result.get("human_review_required") in (None, False)


# ── D-2: evidence guard ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_d2_reviewer_drops_default_correction_without_evidence(monkeypatch):
    """neutral 로의 변경 + transcript_evidence 없음 → 보정 drop, corrections_dropped 기록."""
    fake_llm = MagicMock()

    async def correct_then_finalize(*args, **kwargs):
        return {
            "tool_calls": [
                {"id": "c1", "name": "correct_analysis", "arguments": {
                    "field": "summary.customer_emotion",
                    "new_value": "neutral",
                    "reason": "transcript에 명시 없음",
                    "transcript_evidence": "",
                }},
                {"id": "f", "name": "finalize_review", "arguments": {"summary_reason": "done"}},
            ],
            "text": "", "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=correct_then_finalize)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    analysis = {
        "summary": {
            "summary_short": "환불 요청",
            "summary_detailed": "환불",
            "customer_intent": "환불",
            "customer_emotion": "angry",
            "resolution_status": "escalated",
            "keywords": [],
            "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "angry", "intensity": 0.9, "reason": ""},
            "intent_result": {"primary_category": "환불/결제", "sub_categories": [],
                              "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "high", "action_required": True,
                                "suggested_action": None, "reason": ""},
        },
        "priority_result": {"priority": "high", "tier": "high", "action_required": True,
                            "suggested_action": None, "reason": ""},
    }
    state = _state(
        transcripts=[{"role": "customer", "text": "정말 화가 납니다. 환불해주세요."}],
        analysis=analysis,
        proposed=[],
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    assert "summary.customer_emotion" not in result["corrections_to_analysis"]
    dropped = result["review_result"]["corrections_dropped"]
    assert len(dropped) == 1
    assert dropped[0]["field"] == "summary.customer_emotion"
    # 분석 결과는 angry 그대로
    assert result["analysis_result"]["summary"]["customer_emotion"] == "angry"


@pytest.mark.asyncio
async def test_d2_reviewer_applies_correction_with_valid_evidence(monkeypatch):
    """transcript 에 substring 으로 존재하는 evidence 가 있으면 보정 적용."""
    fake_llm = MagicMock()

    async def correct_then_finalize(*args, **kwargs):
        return {
            "tool_calls": [
                {"id": "c1", "name": "correct_analysis", "arguments": {
                    "field": "summary.customer_emotion",
                    "new_value": "angry",
                    "reason": "고객이 화남 표현",
                    "transcript_evidence": "정말 화가 납니다",
                }},
                {"id": "f", "name": "finalize_review", "arguments": {"summary_reason": "done"}},
            ],
            "text": "", "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=correct_then_finalize)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    analysis = {
        "summary": {
            "summary_short": "x", "summary_detailed": "x",
            "customer_intent": "x", "customer_emotion": "neutral",
            "resolution_status": "resolved", "keywords": [], "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.0, "reason": ""},
            "intent_result": {"primary_category": "기타", "sub_categories": [],
                              "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "low", "action_required": False,
                                "suggested_action": None, "reason": ""},
        },
        "priority_result": {"priority": "low", "tier": "low", "action_required": False,
                            "suggested_action": None, "reason": ""},
    }
    state = _state(
        transcripts=[{"role": "customer", "text": "정말 화가 납니다. 환불 요청합니다."}],
        analysis=analysis,
        proposed=[],
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    assert "summary.customer_emotion" in result["corrections_to_analysis"]
    assert result["analysis_result"]["summary"]["customer_emotion"] == "angry"


# ── D-3: null sentinel coerce ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_d3_correct_analysis_string_null_coerced_to_none(monkeypatch):
    """new_value='null' (string) → None 으로 coerce. evidence 가 transcript 에 있어야 적용."""
    fake_llm = MagicMock()

    async def correct_then_finalize(*args, **kwargs):
        return {
            "tool_calls": [
                {"id": "c1", "name": "correct_analysis", "arguments": {
                    "field": "summary.handoff_notes",
                    "new_value": "null",   # 문자열 "null"
                    "reason": "transcript 에 폭언 없음",
                    "transcript_evidence": "환불 요청합니다",
                }},
                {"id": "f", "name": "finalize_review", "arguments": {"summary_reason": "done"}},
            ],
            "text": "", "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=correct_then_finalize)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    analysis = {
        "summary": {
            "summary_short": "x", "summary_detailed": "x",
            "customer_intent": "x", "customer_emotion": "angry",
            "resolution_status": "escalated", "keywords": [],
            "handoff_notes": "고객이 폭언함 — 환각",
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "angry", "intensity": 0.7, "reason": ""},
            "intent_result": {"primary_category": "환불/결제", "sub_categories": [],
                              "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "high", "action_required": True,
                                "suggested_action": None, "reason": ""},
        },
        "priority_result": {"priority": "high", "tier": "high", "action_required": True,
                            "suggested_action": None, "reason": ""},
    }
    state = _state(
        transcripts=[{"role": "customer", "text": "환불 요청합니다"}],
        analysis=analysis,
        proposed=[],
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    # 문자열 "null" 이 None 으로 coerce 되어 handoff_notes = None
    assert result["analysis_result"]["summary"]["handoff_notes"] is None


# ── D-4: priority single source of truth ──────────────────────────────────────


@pytest.mark.asyncio
async def test_d4_planner_does_not_inject_priority_into_jira_args(monkeypatch):
    """planner 가 propose_create_jira_ticket args 에서 priority 를 받지 않으므로
    state priority_result.priority 만 source.
    """
    _patch_full_catalog(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args(priority="medium")},
            # LLM 이 priority 를 args 에 넣어도 (스키마에 없는 키) 무시되어야 함
            {"id": "j", "name": "propose_create_jira_ticket", "arguments": {
                "summary": "issue", "description": "desc",
            }},
        ],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)

    state = _state(transcripts=[{"role": "customer", "text": "x"}])
    result = await planner_mod.analysis_planner_agent_node(state)
    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    # ActionItem.priority == analysis priority
    assert proposed[0]["priority"] == "medium"
    # params 에는 priority 자동 주입 안 됨 (executor 가 후속 주입)
    assert "priority" not in proposed[0]["params"]
    # labels 는 priority 동기화
    assert "medium" in (proposed[0]["params"].get("labels") or [])


@pytest.mark.asyncio
async def test_d4_reviewer_priority_correction_syncs_approved_actions(monkeypatch):
    """reviewer 가 priority_result.priority 를 high → medium 로 보정하면
    approved_actions[].priority 와 params.priority / labels 도 medium 으로 sync.
    """
    fake_llm = MagicMock()

    async def correct_then_approve(*args, **kwargs):
        return {
            "tool_calls": [
                {"id": "c", "name": "correct_analysis", "arguments": {
                    "field": "priority_result.priority",
                    "new_value": "medium",
                    "reason": "transcript 에 critical 신호 부족",
                    "transcript_evidence": "환불해주세요",
                }},
                {"id": "a", "name": "approve_action", "arguments": {
                    "action_id": "a0_send_slack_alert_slack",
                }},
                {"id": "f", "name": "finalize_review", "arguments": {"summary_reason": "done"}},
            ],
            "text": "", "raw_message": None,
        }

    fake_llm.generate_with_tools = AsyncMock(side_effect=correct_then_approve)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    analysis = {
        "summary": {
            "summary_short": "x", "summary_detailed": "x",
            "customer_intent": "x", "customer_emotion": "angry",
            "resolution_status": "escalated", "keywords": [], "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "angry", "intensity": 0.7, "reason": ""},
            "intent_result": {"primary_category": "환불/결제", "sub_categories": [],
                              "is_repeat_topic": False, "faq_candidate": False},
            "priority_result": {"priority": "high", "action_required": True,
                                "suggested_action": None, "reason": ""},
        },
        "priority_result": {"priority": "high", "tier": "high", "action_required": True,
                            "suggested_action": None, "reason": ""},
    }
    proposed = [
        {
            "action_type": "send_slack_alert", "tool": "slack", "priority": "high",
            "params": {"call_id": "c1", "tenant_id": "t1", "channel_type": "warning",
                       "message": "x", "labels": ["sisicallcall", "high"]},
            "status": "pending",
        }
    ]
    state = _state(
        transcripts=[{"role": "customer", "text": "환불해주세요. 화나네요."}],
        analysis=analysis,
        proposed=proposed,
    )

    result = await reviewer_mod.reviewer_agent_node(state)

    # 보정 적용
    assert result["analysis_result"]["priority_result"]["priority"] == "medium"
    # approved actions sync
    approved = result["approved_actions"]
    assert len(approved) == 1
    assert approved[0]["priority"] == "medium"
    assert approved[0]["params"].get("priority") in (None, "medium")  # 원래 없거나 sync
    labels = approved[0]["params"].get("labels") or []
    assert "medium" in labels and "high" not in labels


# ── D-6: model_used 동적 ──────────────────────────────────────────────────────


def test_d6_model_used_real_mode(monkeypatch):
    """real 모드 + 명시 모델 이면 model_used 에 그 모델명."""
    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setenv("POST_CALL_LLM_MODEL", "gpt-4o")

    from app.agents.post_call.nodes.save_result_node import _resolve_model_used
    assert _resolve_model_used() == "gpt-4o"


def test_d6_model_used_real_mode_default(monkeypatch):
    """real 모드 + 모델 미지정 이면 default 'gpt-4o'."""
    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.delenv("POST_CALL_LLM_MODEL", raising=False)

    from app.agents.post_call.nodes.save_result_node import _resolve_model_used
    assert _resolve_model_used() == "gpt-4o"


def test_d6_model_used_mock_mode(monkeypatch):
    """mock 모드면 model_used == 'mock'."""
    monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")
    monkeypatch.delenv("POST_CALL_USE_REAL_LLM", raising=False)

    from app.agents.post_call.nodes.save_result_node import _resolve_model_used
    assert _resolve_model_used() == "mock"


@pytest.mark.asyncio
async def test_d6_model_used_propagates_to_summary(agent_factory, monkeypatch):
    """end-to-end: real 모드에서 agent.run() 후 summary.model_used != 'demo-mock-llm'."""
    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setenv("POST_CALL_LLM_MODEL", "gpt-4o-mini")

    # 그러나 LLM 은 mock 으로 강제 (네트워크 회피) — _llm 직접 주입
    fake_planner = MagicMock()
    fake_planner.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
            {"id": "no", "name": "propose_no_action", "arguments": {"reason": "ok"}},
        ],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_planner)
    fake_reviewer = MagicMock()
    fake_reviewer.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "f", "name": "finalize_review", "arguments": {"summary_reason": "ok"}},
        ],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer)
    _patch_full_catalog(monkeypatch)

    import app.agents.post_call.nodes.load_context_node as lcn
    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "d6-001"},
        "transcripts": [{"role": "customer", "text": "안녕하세요"}],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    agent = agent_factory()
    result = await agent.run("d6-001", trigger="call_ended", tenant_id="t-d6")
    assert result["summary"]["model_used"] == "gpt-4o-mini"


@pytest.fixture
def agent_factory():
    def _make():
        from app.agents.post_call.agent import PostCallAgent
        return PostCallAgent()
    return _make


# ════════════════════════════════════════════════════════════════════════════
# v3 — V3-1~V3-4 + 텔레메트리 인프라
# ════════════════════════════════════════════════════════════════════════════

# ── V3-2: KST 시각 + 과거/미래 범위 검증 ─────────────────────────────────────


def test_v3_2_validate_callback_time_iso_future_passes():
    from app.agents.post_call.nodes.analysis_planner_agent_node import _validate_callback_time, _kst_now
    from datetime import timedelta
    future = (_kst_now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    value, violation = _validate_callback_time(future)
    assert value == future
    assert violation is None


def test_v3_2_validate_callback_time_past_blocked():
    from app.agents.post_call.nodes.analysis_planner_agent_node import _validate_callback_time
    value, violation = _validate_callback_time("2020-01-01 12:00")
    assert value == ""
    assert violation is not None
    assert "past_time" in violation


def test_v3_2_validate_callback_time_too_far_future_blocked():
    from app.agents.post_call.nodes.analysis_planner_agent_node import _validate_callback_time, _kst_now
    from datetime import timedelta
    far = (_kst_now() + timedelta(days=200)).strftime("%Y-%m-%d %H:%M")
    value, violation = _validate_callback_time(far)
    assert value == ""
    assert violation is not None
    assert "too_far_future" in violation


def test_v3_2_validate_callback_time_natural_language_blocked():
    from app.agents.post_call.nodes.analysis_planner_agent_node import _validate_callback_time
    value, violation = _validate_callback_time("내일 오후 3시")
    assert value == ""
    assert violation is not None
    assert "invalid_time_format" in violation


@pytest.mark.asyncio
async def test_v3_2_planner_past_time_triggers_human_review(monkeypatch):
    """planner 가 과거 ISO 받으면 빈 문자열 + violations + human_review_required."""
    _patch_full_catalog(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args(category="예약/일정")},
            {"id": "cb", "name": "propose_schedule_callback", "arguments": {
                "preferred_time": "2020-05-10 15:00",
                "phone": "01012345678",
                "reason": "콜백",
            }},
        ],
        "text": "", "raw_message": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "mock"},
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)
    state = _state(transcripts=[{"role": "customer", "text": "콜백 부탁"}])
    result = await planner_mod.analysis_planner_agent_node(state)
    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    assert proposed[0]["params"]["preferred_time"] == ""
    assert result.get("human_review_required") is True
    assert "past_time" in result["analysis_planner_rationale"]


def test_v3_2_today_label_uses_kst():
    from app.agents.post_call.nodes.analysis_planner_agent_node import _today_label
    label = _today_label()
    assert "KST" in label or "Asia/Seoul" in label


# ── V3-1/3: actions 0 fast finalize ────────────────────────────────────────


@pytest.mark.asyncio
async def test_v3_1_3_empty_actions_fast_finalize(monkeypatch):
    """proposed_actions=[] 인 경우 reviewer 첫 step 에서 finalize 가능 (mock LLM)."""
    fake_llm = MagicMock()
    # mock LLM: 첫 step 에 바로 finalize_review 만 호출
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "f", "name": "finalize_review",
             "arguments": {"summary_reason": "no actions, no obvious errors"}},
        ],
        "text": "", "raw_message": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "model": "mock"},
    })
    monkeypatch.setattr(reviewer_mod, "_llm", fake_llm)

    state = _state(
        transcripts=[{"role": "customer", "text": "안녕하세요"}],
        analysis={
            "summary": {
                "summary_short": "x", "summary_detailed": "x",
                "customer_intent": "x", "customer_emotion": "neutral",
                "resolution_status": "resolved", "keywords": [], "handoff_notes": None,
            },
            "voc_analysis": {
                "sentiment_result": {"sentiment": "neutral", "intensity": 0.0, "reason": ""},
                "intent_result": {"primary_category": "기타", "sub_categories": [],
                                  "is_repeat_topic": False, "faq_candidate": False},
                "priority_result": {"priority": "low", "action_required": False,
                                    "suggested_action": None, "reason": ""},
            },
            "priority_result": {"priority": "low", "tier": "low", "action_required": False,
                                "suggested_action": None, "reason": ""},
        },
        proposed=[],
    )
    result = await reviewer_mod.reviewer_agent_node(state)
    assert result["reviewer_steps"] <= 2
    assert result["review_verdict"] == "pass"


# ── V3-4: slack urgency derived from priority ─────────────────────────────


@pytest.mark.asyncio
async def test_v3_4_slack_urgency_derived_from_priority(monkeypatch):
    """propose_send_slack_alert args 에서 urgency 제거. priority='high' 시
    derived urgency='critical' 자동 주입."""
    _patch_full_catalog(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis",
             "arguments": _record_args(priority="high", emotion="angry", resolution="escalated")},
            {"id": "s", "name": "propose_send_slack_alert",
             "arguments": {"message": "긴급"}},  # urgency 인자 없음
        ],
        "text": "", "raw_message": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "mock"},
    })
    monkeypatch.setattr(planner_mod, "_llm", fake_llm)
    state = _state(transcripts=[{"role": "customer", "text": "x"}])
    result = await planner_mod.analysis_planner_agent_node(state)
    proposed = result["proposed_actions"]
    assert len(proposed) == 1
    p = proposed[0]
    assert p["action_type"] == "send_slack_alert"
    # high → critical
    assert p["params"]["urgency"] == "critical"
    assert p["params"]["channel_type"] == "critical"


def test_v3_4_priority_to_urgency_mapping():
    from app.agents.post_call.nodes.analysis_planner_agent_node import _PRIORITY_TO_URGENCY
    assert _PRIORITY_TO_URGENCY["low"] == "info"
    assert _PRIORITY_TO_URGENCY["medium"] == "warning"
    assert _PRIORITY_TO_URGENCY["high"] == "critical"
    assert _PRIORITY_TO_URGENCY["critical"] == "critical"


# ── 텔레메트리 인프라 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_telemetry_planner_state_keys_present(monkeypatch):
    """planner 노드 호출 후 state.analysis_planner_telemetry 키 채워짐."""
    _patch_full_catalog(monkeypatch)
    state = _state(transcripts=[{"role": "customer", "text": "안녕하세요"}])
    result = await planner_mod.analysis_planner_agent_node(state)
    t = result.get("analysis_planner_telemetry")
    assert t is not None
    assert t["calls"] == 1
    assert "tokens" in t
    for k in ("prompt", "completion", "total", "model"):
        assert k in t["tokens"]
    assert isinstance(t["tool_counts"], dict)
    assert "latency_ms" in t and t["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_telemetry_reviewer_state_keys_present(monkeypatch):
    """reviewer 노드 호출 후 state.reviewer_telemetry 키 채워짐."""
    state = _state(
        transcripts=[{"role": "customer", "text": "x"}],
        analysis={
            "summary": {"summary_short": "x", "summary_detailed": "x",
                        "customer_intent": "x", "customer_emotion": "neutral",
                        "resolution_status": "resolved", "keywords": [], "handoff_notes": None},
            "voc_analysis": {
                "sentiment_result": {"sentiment": "neutral", "intensity": 0.0, "reason": ""},
                "intent_result": {"primary_category": "기타", "sub_categories": [],
                                  "is_repeat_topic": False, "faq_candidate": False},
                "priority_result": {"priority": "low", "action_required": False,
                                    "suggested_action": None, "reason": ""},
            },
            "priority_result": {"priority": "low", "tier": "low", "action_required": False,
                                "suggested_action": None, "reason": ""},
        },
        proposed=[],
    )
    result = await reviewer_mod.reviewer_agent_node(state)
    t = result.get("reviewer_telemetry")
    assert t is not None
    for k in ("calls", "tokens", "tool_counts", "steps", "max_steps_reached", "latency_ms"):
        assert k in t
    for k in ("prompt", "completion", "total", "model"):
        assert k in t["tokens"]
    assert isinstance(t["tool_counts"], dict)


@pytest.mark.asyncio
async def test_telemetry_dashboard_payload_includes_telemetry(monkeypatch):
    """save_final 단계 → dashboard_payload.telemetry 에 두 에이전트 telemetry 포함."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.load_context_node as lcn
    import app.agents.post_call.nodes.action_executor_node as exec_node

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "tel-001"},
        "transcripts": [{"role": "customer", "text": "안녕"}],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    _patch_full_catalog(monkeypatch)

    agent = PostCallAgent()
    result = await agent.run("tel-001", trigger="call_ended", tenant_id="t-tel")

    payload = result["dashboard_payload"]
    tel = payload.get("telemetry") or {}
    assert "analysis_planner" in tel
    assert "reviewer" in tel
    # 둘 중 하나는 None 일 수 있음 (escalation_immediate 분기) — 여기는 call_ended 라 둘 다 있음
    assert tel["analysis_planner"] is not None
    assert tel["reviewer"] is not None


@pytest.mark.asyncio
async def test_telemetry_mock_mode_zero_tokens(monkeypatch):
    """mock 모드에서도 telemetry 키 존재, tokens=0."""
    _patch_full_catalog(monkeypatch)
    state = _state(transcripts=[{"role": "customer", "text": "안녕"}])
    result = await planner_mod.analysis_planner_agent_node(state)
    t = result["analysis_planner_telemetry"]
    assert t["tokens"]["total"] == 0
    assert t["tokens"]["model"] == "mock"


@pytest.mark.asyncio
async def test_graph_fail_skips_action_executor(monkeypatch):
    """reviewer verdict=fail → human_queue 분기 → executor 미실행."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.load_context_node as lcn
    import app.agents.post_call.nodes.action_executor_node as exec_node

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "g-003"},
        "transcripts": [{"role": "customer", "text": "환불 처리 정말 화나네요"}],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    # reviewer 강제 fail
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [{"id": "e", "name": "escalate_to_human",
                        "arguments": {"reason": "test"}},
                       {"id": "f", "name": "finalize_review",
                        "arguments": {"summary_reason": "fail"}}],
        "text": "", "raw_message": None,
    })
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # executor 가 호출되면 안 됨
    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock()
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    result = await agent.run("g-003", trigger="call_ended", tenant_id="t-graph3")

    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True
    assert fake_executor.execute_actions.await_count == 0
    assert result["executed_actions"] == []


# ════════════════════════════════════════════════════════════════════════════
# E-5 — Notion 카탈로그 + supervisor email 가이드 강화
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_planner_notion_always_proposed_when_connected(monkeypatch):
    """Notion env 가 채워져 있으면 단순 inquiry 통화도 propose_create_notion_call_record 호출."""
    _patch_full_catalog(monkeypatch)
    monkeypatch.setenv("NOTION_API_TOKEN", "secret_test_token_xxx")
    monkeypatch.setenv("NOTION_DATABASE_ID", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    state = _state(transcripts=[
        {"role": "customer", "text": "운영 시간이 어떻게 되나요"},
        {"role": "agent", "text": "오전 9시부터 오후 6시입니다"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)

    proposed_types = {a["action_type"] for a in result["proposed_actions"]}
    assert "create_notion_call_record" in proposed_types, \
        f"단순 inquiry 도 notion call_record 필요. 실제: {proposed_types}"


@pytest.mark.asyncio
async def test_planner_notion_voc_for_angry(monkeypatch):
    """angry transcript → propose_create_notion_voc_record 도 함께 호출."""
    _patch_full_catalog(monkeypatch)
    monkeypatch.setenv("NOTION_API_TOKEN", "secret_test_token_xxx")
    monkeypatch.setenv("NOTION_DATABASE_ID", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    state = _state(transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요. 환불 처리 안 해주면 민원 넣을 거예요"},
        {"role": "agent", "text": "죄송합니다, 처리해드릴게요"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)
    proposed_types = {a["action_type"] for a in result["proposed_actions"]}
    assert "create_notion_call_record" in proposed_types
    assert "create_notion_voc_record" in proposed_types, \
        f"angry 시 notion voc_record 필요. 실제: {proposed_types}"


@pytest.mark.asyncio
async def test_planner_notion_skipped_when_not_connected(monkeypatch):
    """Notion env 가 비어 있으면 (token 또는 db id 미설정) 카탈로그에서 제외 → propose 안 됨."""
    _patch_full_catalog(monkeypatch)
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

    state = _state(transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요. 환불 처리 안 해주면 민원 넣을 거예요"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)
    proposed_types = {a["action_type"] for a in result["proposed_actions"]}
    assert "create_notion_call_record" not in proposed_types
    assert "create_notion_voc_record" not in proposed_types

    # 카탈로그에서도 제외 확인
    catalog = catalog_mod.get_action_catalog("test-tenant")
    catalog_names = {e["name"] for e in catalog}
    assert "propose_create_notion_call_record" not in catalog_names
    assert "propose_create_notion_voc_record" not in catalog_names


@pytest.mark.asyncio
async def test_planner_supervisor_email_for_high_priority(monkeypatch):
    """priority=high (angry) → propose_send_email_supervisor 호출
    (이전엔 critical 만 호출했지만 이제 high+ 또는 angry 도 호출)."""
    _patch_full_catalog(monkeypatch)

    state = _state(transcripts=[
        {"role": "customer", "text": "환불 처리 정말 화가 나네요. 민원 제기할 거예요"},
        {"role": "agent", "text": "죄송합니다"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)

    analysis = result["analysis_result"]
    assert analysis["priority_result"]["priority"] in ("high", "critical")
    proposed_types = {a["action_type"] for a in result["proposed_actions"]}
    assert "send_manager_email" in proposed_types, \
        f"high priority 또는 angry 시 supervisor email 필요. 실제: {proposed_types}"
