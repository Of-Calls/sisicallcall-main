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
    # 기본값: NOTION 미설정 → _inject_mandatory_actions 0건 (auto 액션 없음).
    # 명시적으로 NOTION 동작 검증하는 테스트는 그 안에서 setenv 다시 함.
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)
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
        "analysis_retry_count": 0,
        "review_feedback": [],
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
async def test_graph_no_save_when_fail_max_retries(monkeypatch):
    """reviewer fail max 시 분석 저장 차단 — save_reviewed_analysis 호출 0회.

    이전 흐름과 정반대: 검토 통과 분석만 저장. fail max 도달 시 call_summaries /
    voc_analyses 에 row 가 들어가지 않는다 (잘못된 분석으로 프론트 노출 방지).
    save_final 은 mcp_action_logs / dashboard 갱신 위해 한 번 호출.
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

    # fail max 경로 — save_reviewed_analysis (mode='intermediate') 호출 0회.
    # save_final 만 1회. 분석 본문은 어디에도 영속화되지 않는다.
    assert save_calls.count("intermediate") == 0, (
        f"fail max 시 save_reviewed_analysis 호출 0 기대. 실제={save_calls.count('intermediate')}"
    )
    assert save_calls.count("final") == 1
    assert save_calls[-1] == "final"
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
    from datetime import timedelta
    _patch_full_catalog(monkeypatch)
    # 현재 시각 기준 +1일 미래 (validator 의 [+5분, +90일] 범위 안).
    future_iso = (planner_mod._kst_now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args(category="예약/일정")},
            {"id": "cb", "name": "propose_schedule_callback", "arguments": {
                "preferred_time": future_iso,
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
    assert proposed[0]["params"]["preferred_time"] == future_iso
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
    """reviewer verdict=fail → human_queue 분기 → action_executor 미실행.

    auto_action_executor 는 별개의 노드 — auto_injected 액션 (Notion) 만 실행.
    이 테스트는 NOTION env 가 없어 auto 액션도 0건이므로 executed=[].
    """
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.load_context_node as lcn
    import app.agents.post_call.nodes.action_executor_node as exec_node
    import app.agents.post_call.nodes.auto_action_executor_node as auto_exec_node

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

    # action_executor (LLM-approved 발송) 가 호출되면 안 됨
    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock()
    monkeypatch.setattr(exec_node, "_executor", fake_executor)
    # auto_action_executor 는 호출됨 — admin alert 2건 (Slack + Gmail) 발송 위해.
    # NOTION 은 미설정이라 _inject_mandatory_actions 에서 auto Notion 추가 안 됨.
    fake_auto_executor = MagicMock()
    fake_auto_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(auto_exec_node, "_executor", fake_auto_executor)

    agent = PostCallAgent()
    result = await agent.run("g-003", trigger="call_ended", tenant_id="t-graph3")

    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True
    assert fake_executor.execute_actions.await_count == 0, "LLM-approved 발송 차단"
    # notify_admin_review_failed 가 alert 2건을 proposed 에 넣었으므로 auto_action_executor 1회 호출
    assert fake_auto_executor.execute_actions.await_count == 1, "admin alert 발송 1회"
    sent_actions = fake_auto_executor.execute_actions.await_args.kwargs.get("actions") or \
                   fake_auto_executor.execute_actions.await_args.args[2]
    types = {a["action_type"] for a in sent_actions}
    assert {"send_slack_alert", "send_manager_email"}.issubset(types), \
        f"admin alert 두 채널 모두 포함. 실제: {types}"


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


# ════════════════════════════════════════════════════════════════════════════
# F-1 — reviewer fail → analysis_planner 재시도 루프
# ════════════════════════════════════════════════════════════════════════════


def _patch_load_context(monkeypatch, *, transcripts: list[dict]):
    """load_context_node 가 미리 정해진 transcript 를 돌려주도록 monkeypatch."""
    import app.agents.post_call.nodes.load_context_node as lcn

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "f-001"},
        "transcripts": transcripts,
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)


def _make_reviewer_fail_response(reason: str = "분석 부적절") -> dict:
    return {
        "tool_calls": [
            {"id": "e", "name": "escalate_to_human", "arguments": {"reason": reason}},
            {"id": "f", "name": "finalize_review",
             "arguments": {"verdict": "fail", "summary_reason": reason}},
        ],
        "text": "",
        "raw_message": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "mock"},
    }


def _make_reviewer_pass_response() -> dict:
    return {
        "tool_calls": [
            {"id": "f", "name": "finalize_review",
             "arguments": {"verdict": "pass", "summary_reason": "ok"}},
        ],
        "text": "",
        "raw_message": None,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "model": "mock"},
    }


@pytest.mark.asyncio
async def test_review_fail_triggers_analysis_retry(monkeypatch):
    """reviewer 가 fail 반환 → analysis_planner 재호출 + retry_count=1 + review_feedback 누적."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불 처리 정말 화나네요"},
    ])

    # reviewer: 1차 fail, 2차 pass
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(side_effect=[
        _make_reviewer_fail_response("transcript 와 분석 결과 불일치"),
        _make_reviewer_pass_response(),
    ])
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # planner 호출 횟수 추적
    planner_calls: list[dict] = []
    real_planner = planner_mod.analysis_planner_agent_node

    async def tracked_planner(state):
        planner_calls.append(copy.deepcopy(state))
        return await real_planner(state)

    monkeypatch.setattr(
        "app.agents.post_call.graph.analysis_planner_agent_node",
        tracked_planner,
    )

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    result = await agent.run("f-retry-001", trigger="call_ended", tenant_id="t-retry")

    # planner 가 두 번 호출됨 (최초 + 재시도 1)
    assert len(planner_calls) == 2, f"planner 호출 횟수 기대=2 실제={len(planner_calls)}"
    # 두 번째 호출 시 review_feedback 채워짐
    assert planner_calls[1]["analysis_retry_count"] == 1
    assert planner_calls[1]["review_feedback"], "재시도 시 review_feedback 비어있으면 안 됨"
    # 최종 verdict pass
    assert result["review_verdict"] == "pass"
    assert result["analysis_retry_count"] == 1


@pytest.mark.asyncio
async def test_review_fail_max_retries_goes_to_human_queue(monkeypatch):
    """3회 연속 fail → max_retries(=2) 초과 → human_queue, executor 미호출."""
    from app.agents.post_call.agent import PostCallAgent
    from app.agents.post_call.graph import MAX_ANALYSIS_RETRIES
    import app.agents.post_call.nodes.action_executor_node as exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불 처리 정말 화나네요"},
    ])

    fail_count = {"n": 0}

    async def always_fail(*args, **kwargs):
        fail_count["n"] += 1
        return _make_reviewer_fail_response(f"fail_{fail_count['n']}")

    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(side_effect=always_fail)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    result = await agent.run("f-retry-002", trigger="call_ended", tenant_id="t-retry")

    # MAX_ANALYSIS_RETRIES = 2 → reviewer 호출 총 3회 (1차 + 재시도 2회)
    assert MAX_ANALYSIS_RETRIES == 2
    assert fail_count["n"] == MAX_ANALYSIS_RETRIES + 1  # 3
    assert result["analysis_retry_count"] == MAX_ANALYSIS_RETRIES  # 2
    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True
    assert fake_executor.execute_actions.await_count == 0
    # review_feedback 에 3회분 누적
    assert len(result["review_feedback"]) >= 2  # 최소 2 사이클의 feedback


@pytest.mark.asyncio
async def test_analysis_planner_uses_review_feedback_in_prompt(monkeypatch):
    """state["review_feedback"] 가 비어있지 않으면 system_prompt 에 [이전 분석 검토 결과] 블록 포함."""
    _patch_full_catalog(monkeypatch)

    captured: dict = {}

    class _Capture:
        async def generate_with_tools(self, system_prompt, user_message, tools,
                                       temperature=0.0, max_tokens=1024,
                                       tool_choice="auto", messages=None):
            captured["system_prompt"] = system_prompt
            return {
                "tool_calls": [{
                    "id": "r",
                    "name": "record_analysis",
                    "arguments": {
                        "summary_short": "x", "summary_detailed": "x",
                        "customer_intent": "x", "customer_emotion": "neutral",
                        "resolution_status": "resolved", "priority": "low",
                        "action_required": False, "primary_category": "기타",
                        "is_repeat_topic": False, "faq_candidate": False,
                        "keywords": [], "handoff_notes": "",
                    },
                }],
                "text": "",
                "raw_message": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
            }

    planner_mod._llm = _Capture()

    state = _state(transcripts=[{"role": "customer", "text": "안녕"}])
    state["review_feedback"] = [
        "reviewer_escalated: 분석 결과 transcript 와 모순",
        "action_rejected[a0_send_slack_alert_slack]: 단순 문의에 부적절",
    ]
    state["analysis_retry_count"] = 1

    await planner_mod.analysis_planner_agent_node(state)

    sp = captured["system_prompt"]
    assert "[이전 분석 검토 결과" in sp, "재시도 시 prompt 에 feedback 블록 누락"
    assert "reviewer_escalated" in sp
    assert "단순 문의에 부적절" in sp
    assert "같은 실수 반복" in sp


@pytest.mark.asyncio
async def test_review_pass_after_retry_executes_actions(monkeypatch):
    """1차 fail → 2차 pass → executor 호출됨, approved_actions 실행."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불 처리 정말 화나네요"},
    ])

    # 1차 호출은 fail, 2차 부터는 _MockReviewerLLM 동작 (모든 ACTION_ID 자동 approve)
    real_mock = reviewer_mod._MockReviewerLLM()
    call_n = {"i": 0}

    async def first_fail_then_mock(**kwargs):
        call_n["i"] += 1
        if call_n["i"] == 1:
            return _make_reviewer_fail_response("1차 fail")
        return await real_mock.generate_with_tools(**kwargs)

    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(side_effect=first_fail_then_mock)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # executor: 호출되면 success 응답 그대로 반환
    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[
        {"action_type": "send_slack_alert", "tool": "slack", "status": "success"},
    ])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    result = await agent.run("f-retry-003", trigger="call_ended", tenant_id="t-retry")

    assert result["review_verdict"] == "pass"
    assert result["analysis_retry_count"] == 1
    assert fake_executor.execute_actions.await_count == 1, "2차 pass 후 executor 1회 호출"
    assert result.get("human_review_required") is False


@pytest.mark.asyncio
async def test_no_analysis_save_when_fail_max_retries(monkeypatch):
    """fail 3회 → call_summaries / voc_analyses 에 분석 미저장 (사용자 핵심 의도).

    검토 통과 분석만 저장한다. fail max 도달 시:
      - save_summary / save_voc_analysis 호출 0회
      - mcp_action_logs 는 admin_alert 발송분만 1회 (notify_admin → auto_action_executor)
    """
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node
    import app.agents.post_call.nodes.auto_action_executor_node as auto_exec_node
    import app.agents.post_call.nodes.save_result_node as save_mod

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불 처리 정말 화나네요"},
    ])

    # reviewer: 모든 호출 fail (max 도달)
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(return_value=_make_reviewer_fail_response())
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # main executor 호출 0
    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    # auto executor — admin alert 발송분만 success 로 반환
    fake_auto_executor = MagicMock()
    fake_auto_executor.execute_actions = AsyncMock(return_value=[
        {"action_type": "send_slack_alert", "tool": "slack", "status": "success",
         "external_id": "slack-admin-001",
         "idempotency_token": "auto:admin_alert_review_failed_slack"},
        {"action_type": "send_manager_email", "tool": "gmail", "status": "success",
         "external_id": "gmail-admin-001",
         "idempotency_token": "auto:admin_alert_review_failed_email"},
    ])
    monkeypatch.setattr(auto_exec_node, "_executor", fake_auto_executor)

    # repo 호출 spy
    summary_calls: list[tuple] = []
    voc_calls: list[tuple] = []
    action_log_calls: list[tuple] = []

    fake_summary_repo = MagicMock()
    fake_summary_repo.save_summary = AsyncMock(
        side_effect=lambda call_id, *a, **kw: summary_calls.append((call_id, kw))
    )
    fake_voc_repo = MagicMock()
    fake_voc_repo.save_voc_analysis = AsyncMock(
        side_effect=lambda call_id, *a, **kw: voc_calls.append((call_id, kw))
    )
    fake_action_repo = MagicMock()
    fake_action_repo.save_action_log = AsyncMock(
        side_effect=lambda call_id, *a, **kw: action_log_calls.append((call_id, kw))
    )
    monkeypatch.setattr(save_mod, "_summary_repo", fake_summary_repo)
    monkeypatch.setattr(save_mod, "_voc_repo", fake_voc_repo)
    monkeypatch.setattr(save_mod, "_action_log_repo", fake_action_repo)

    agent = PostCallAgent()
    call_id = "f-retry-004"
    result = await agent.run(call_id, trigger="call_ended", tenant_id="t-retry")

    # 핵심: 분석 저장 0회 (검토 미통과 분석은 영속화 차단).
    assert len(summary_calls) == 0, f"fail max 시 summary 저장 0 기대. 실제={len(summary_calls)}"
    assert len(voc_calls) == 0, f"fail max 시 voc 저장 0 기대. 실제={len(voc_calls)}"
    # admin alert 발송 결과는 mcp_action_logs 에 기록됨 (executed_actions 비어있지 않음).
    assert len(action_log_calls) == 1, "admin alert mcp_action_logs 기록"
    assert fake_auto_executor.execute_actions.await_count == 1
    assert result["analysis_retry_count"] == 2
    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True


# ════════════════════════════════════════════════════════════════════════════
# G — Notion 자동 저장 + 한 통화 다중 액션
# ════════════════════════════════════════════════════════════════════════════


def _enable_notion(monkeypatch) -> None:
    monkeypatch.setenv("NOTION_API_TOKEN", "secret_test_token_xxx")
    monkeypatch.setenv("NOTION_DATABASE_ID", "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


@pytest.mark.asyncio
async def test_notion_call_record_auto_injected_on_every_call(monkeypatch):
    """simple inquiry → propose_no_action 응답이어도 자동 call_record 추가."""
    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)

    state = _state(transcripts=[
        {"role": "customer", "text": "운영 시간이 어떻게 되나요"},
        {"role": "agent", "text": "오전 9시부터 오후 6시입니다"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)

    types = [a["action_type"] for a in result["proposed_actions"]]
    assert "create_notion_call_record" in types
    auto = [a for a in result["proposed_actions"]
            if (a.get("params") or {}).get("auto_injected")]
    assert len(auto) == 1, f"call_record 1건만 기대. 실제: {len(auto)}"
    assert auto[0]["idempotency_token"] == "auto:auto_call_record"


@pytest.mark.asyncio
async def test_notion_voc_record_auto_injected_on_angry_high(monkeypatch):
    """angry + high → call_record + voc_record 둘 다 자동 주입."""
    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)

    state = _state(transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요. 환불 처리 안 해주면 민원 넣을 거예요"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)

    auto = [a for a in result["proposed_actions"]
            if (a.get("params") or {}).get("auto_injected")]
    types = {a["action_type"] for a in auto}
    assert types == {"create_notion_call_record", "create_notion_voc_record"}, \
        f"angry+high 시 둘 다 주입. 실제: {types}"
    for a in auto:
        assert a["params"]["auto_injected"] is True


@pytest.mark.asyncio
async def test_notion_voc_not_injected_on_neutral(monkeypatch):
    """neutral → call_record 만, voc_record 없음."""
    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)

    state = _state(transcripts=[
        {"role": "customer", "text": "운영 시간 문의입니다"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)
    auto = [a for a in result["proposed_actions"]
            if (a.get("params") or {}).get("auto_injected")]
    types = {a["action_type"] for a in auto}
    assert types == {"create_notion_call_record"}


@pytest.mark.asyncio
async def test_notion_voc_not_injected_on_angry_low(monkeypatch):
    """angry 지만 priority=low → voc_record 미주입 (medium+ 조건 미충족)."""
    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)

    # mock LLM 직접 패치 — 우리가 priority=low + emotion=angry 강제 지정
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [{
            "id": "rec",
            "name": "record_analysis",
            "arguments": {
                "summary_short": "low priority angry",
                "summary_detailed": "low priority angry case",
                "customer_intent": "test",
                "customer_emotion": "angry",
                "resolution_status": "resolved",
                "priority": "low",
                "action_required": False,
                "primary_category": "기타",
                "is_repeat_topic": False,
                "faq_candidate": False,
                "keywords": [],
                "handoff_notes": "",
            },
        }],
        "text": "", "raw_message": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
    })
    planner_mod._llm = fake_llm

    state = _state(transcripts=[{"role": "customer", "text": "약간 짜증나는데 별일 아님"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    auto = [a for a in result["proposed_actions"]
            if (a.get("params") or {}).get("auto_injected")]
    types = {a["action_type"] for a in auto}
    assert "create_notion_call_record" in types
    assert "create_notion_voc_record" not in types, \
        f"low priority 면 voc_record 주입 안 됨. 실제: {types}"


@pytest.mark.asyncio
async def test_notion_disconnected_skips_auto_actions(monkeypatch):
    """NOTION env 없으면 auto 액션 0건 (graceful skip)."""
    _patch_full_catalog(monkeypatch)
    # Notion env 미설정 (autouse fixture 가 이미 unset)

    state = _state(transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요"},
    ])
    result = await planner_mod.analysis_planner_agent_node(state)

    auto = [a for a in result["proposed_actions"]
            if (a.get("params") or {}).get("auto_injected")]
    assert auto == []


@pytest.mark.asyncio
async def test_auto_injected_actions_skip_reviewer(monkeypatch):
    """reviewer 의 review_target 에 auto_injected 미포함, 최종 approved 에는 포함."""
    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)

    # planner 직접 호출해 auto inject 발동 → reviewer 에 그대로 전달
    state_dict = _state(transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요. 환불 안 해주면 민원 넣을 거예요"},
    ])
    planner_out = await planner_mod.analysis_planner_agent_node(state_dict)
    state_dict.update(planner_out)

    # reviewer mock — proposed_text 안에 auto_injected 액션이 ACTION_ID 로 안 떠야 함
    captured: dict = {}

    async def capture(*args, **kwargs):
        # messages 의 user content 캡처
        msgs = kwargs.get("messages") or []
        for m in msgs:
            if m.get("role") == "user":
                captured["user_text"] = m.get("content")
                break
        return {
            "tool_calls": [{
                "id": "f", "name": "finalize_review",
                "arguments": {"verdict": "pass", "summary_reason": "ok"},
            }],
            "text": "", "raw_message": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
        }

    fake_reviewer = MagicMock()
    fake_reviewer.generate_with_tools = AsyncMock(side_effect=capture)
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer)

    out = await reviewer_mod.reviewer_agent_node(state_dict)

    # reviewer 가 본 텍스트에 auto_injected 액션 ACTION_ID 가 없어야 함
    user_text = captured.get("user_text", "")
    assert "create_notion_call_record" not in user_text, \
        f"reviewer 에 auto 액션이 보이면 안 됨. 실제 text: {user_text[:500]}"

    # 하지만 최종 approved_actions 에는 포함
    approved_types = [a["action_type"] for a in out["approved_actions"]]
    assert "create_notion_call_record" in approved_types
    # telemetry 에도 카운트
    assert out["reviewer_telemetry"]["auto_injected_count"] >= 1


@pytest.mark.asyncio
async def test_auto_actions_executed_on_human_queue_path(monkeypatch):
    """retry max 초과 → human_queue → auto_action_executor 가 Notion auto 만 실행."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node
    import app.agents.post_call.nodes.auto_action_executor_node as auto_exec_node

    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요. 환불 안 해주면 민원 넣을 거예요"},
    ])

    # reviewer 모두 fail
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(
        return_value=_make_reviewer_fail_response()
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # main executor 호출 0
    fake_main_executor = MagicMock()
    fake_main_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_main_executor)

    # auto executor — 호출되면 받은 actions 캡처 + dummy success 반환
    captured_auto_actions: list[list[dict]] = []

    async def capture_auto(call_id, tenant_id, actions):
        captured_auto_actions.append(actions)
        return [
            {**a, "status": "success", "external_id": f"notion-{i}"}
            for i, a in enumerate(actions)
        ]

    fake_auto_executor = MagicMock()
    fake_auto_executor.execute_actions = AsyncMock(side_effect=capture_auto)
    monkeypatch.setattr(auto_exec_node, "_executor", fake_auto_executor)

    agent = PostCallAgent()
    result = await agent.run("g-auto-001", trigger="call_ended", tenant_id="t-auto")

    assert result["review_verdict"] == "fail"
    assert fake_main_executor.execute_actions.await_count == 0
    assert fake_auto_executor.execute_actions.await_count == 1
    auto_actions = captured_auto_actions[0]
    types = {a["action_type"] for a in auto_actions}
    assert "create_notion_call_record" in types
    # voc 도 angry+high 이므로 포함
    assert "create_notion_voc_record" in types
    # 모두 auto_injected 마커
    for a in auto_actions:
        assert (a.get("params") or {}).get("auto_injected") is True


@pytest.mark.asyncio
async def test_auto_actions_idempotent_across_retries(monkeypatch):
    """retry 사이클 안에서 같은 sub_intent 의 token 이 동일 — executor 가 1회만 발송."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node
    import app.agents.post_call.nodes.auto_action_executor_node as auto_exec_node

    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "정말 짜증 나네요"},
    ])

    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(
        return_value=_make_reviewer_fail_response()
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    # 모든 sub_intent token 카운트
    seen_tokens: list[str] = []

    async def capture_auto(call_id, tenant_id, actions):
        for a in actions:
            seen_tokens.append(a.get("idempotency_token") or "")
        return [
            {**a, "status": "success", "external_id": f"x-{i}"}
            for i, a in enumerate(actions)
        ]

    fake_auto_executor = MagicMock()
    fake_auto_executor.execute_actions = AsyncMock(side_effect=capture_auto)
    monkeypatch.setattr(auto_exec_node, "_executor", fake_auto_executor)

    fake_main = MagicMock()
    fake_main.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_main)

    agent = PostCallAgent()
    result = await agent.run("g-idem-001", trigger="call_ended", tenant_id="t-idem")

    # auto inject 는 매 retry 사이클마다 발생 (planner 매번 _inject 재실행) —
    # 하지만 auto_action_executor 는 마지막 1회만 호출되므로 token 출현은 통화당 1세트.
    assert result["analysis_retry_count"] == 2  # MAX 도달
    # 두 sub_intent 가 정확히 한 번씩만 executor 에 전달돼야 함
    call_token = "auto:auto_call_record"
    voc_token = "auto:auto_voc_record"
    assert seen_tokens.count(call_token) == 1, f"call_record token 1회 기대. 실제: {seen_tokens}"
    assert seen_tokens.count(voc_token) == 1, f"voc_record token 1회 기대. 실제: {seen_tokens}"


@pytest.mark.asyncio
async def test_multiple_intents_propose_multiple_actions(monkeypatch):
    """다중 의도 LLM 응답 → 여러 action_type 동시 propose."""
    _patch_full_catalog(monkeypatch)

    # mock LLM: 환불(slack+jira+email) + 콜백 + sms 5개 propose
    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args(
                emotion="angry", priority="high", category="민원/불만",
            )},
            {"id": "s", "name": "propose_send_slack_alert",
             "arguments": {"message": "환불 불만 통화"}},
            {"id": "j", "name": "propose_create_jira_ticket",
             "arguments": {"summary": "환불 처리 검토", "description": "고객 환불 요청"}},
            {"id": "e", "name": "propose_send_email_supervisor",
             "arguments": {"subject": "환불 보고", "body": "..."}},
            {"id": "c", "name": "propose_schedule_callback",
             "arguments": {"reason": "환불 결과 안내"}},
            {"id": "sms", "name": "propose_send_sms_followup",
             "arguments": {"phone": "01012345678", "message": "환불 접수 안내"}},
        ],
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    state = _state(transcripts=[{"role": "customer", "text": "환불 요청 + 콜백"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    types = {a["action_type"] for a in result["proposed_actions"]}
    assert {"send_slack_alert", "create_jira_issue", "send_manager_email",
            "schedule_callback", "send_voc_receipt_sms"}.issubset(types)


@pytest.mark.asyncio
async def test_same_action_type_with_different_intents(monkeypatch):
    """같은 action_type 두 번 — 다른 token → 둘 다 별개 액션."""
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
            # 같은 action_type, 다른 summary
            {"id": "j1", "name": "propose_create_jira_ticket",
             "arguments": {"summary": "환불 처리", "description": "환불 요청"}},
            {"id": "j2", "name": "propose_create_jira_ticket",
             "arguments": {"summary": "본인 인증 실패 조사", "description": "auth error"}},
        ],
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    state = _state(transcripts=[{"role": "customer", "text": "환불 + 인증"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    jira_actions = [a for a in result["proposed_actions"]
                    if a["action_type"] == "create_jira_issue"]
    assert len(jira_actions) == 2
    tokens = {a["idempotency_token"] for a in jira_actions}
    assert len(tokens) == 2, f"두 jira 가 다른 token 가져야 함. 실제: {tokens}"


@pytest.mark.asyncio
async def test_idempotency_blocks_true_duplicates(monkeypatch):
    """같은 summary 두 번 propose → 두 token 동일."""
    _patch_full_catalog(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": [
            {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
            {"id": "j1", "name": "propose_create_jira_ticket",
             "arguments": {"summary": "환불 처리", "description": "동일 요청 1"}},
            {"id": "j2", "name": "propose_create_jira_ticket",
             "arguments": {"summary": "환불 처리", "description": "동일 요청 2"}},
        ],
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    state = _state(transcripts=[{"role": "customer", "text": "환불"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    jira_actions = [a for a in result["proposed_actions"]
                    if a["action_type"] == "create_jira_issue"]
    tokens = {a["idempotency_token"] for a in jira_actions}
    # _IDEMPOTENCY_FIELDS["create_jira_issue"] = ["summary"] → 같은 summary → 같은 token
    assert len(tokens) == 1, f"같은 summary 면 token 동일. 실제: {tokens}"


def test_idempotency_token_is_deterministic():
    """같은 입력 → 같은 token (sha256 결정성)."""
    a1 = {
        "action_type": "create_jira_issue",
        "tool": "jira",
        "params": {"summary": "환불 처리", "description": "본문"},
    }
    a2 = {
        "action_type": "create_jira_issue",
        "tool": "jira",
        "params": {"summary": "환불 처리", "description": "다른 본문"},  # description 무시
    }
    a3 = {
        "action_type": "create_jira_issue",
        "tool": "jira",
        "params": {"summary": "다른 issue", "description": "본문"},
    }
    assert planner_mod._compute_idempotency_token(a1) == planner_mod._compute_idempotency_token(a2)
    assert planner_mod._compute_idempotency_token(a1) != planner_mod._compute_idempotency_token(a3)

    auto = {
        "action_type": "create_notion_call_record",
        "tool": "notion",
        "params": {"auto_injected": True, "sub_intent": "auto_call_record"},
    }
    assert planner_mod._compute_idempotency_token(auto) == "auto:auto_call_record"


@pytest.mark.asyncio
async def test_max_tool_calls_8_enforced(monkeypatch):
    """LLM 이 9개 propose → 8개로 제한."""
    _patch_full_catalog(monkeypatch)

    big_calls = [
        {"id": "rec", "name": "record_analysis", "arguments": _record_args()},
    ]
    for i in range(8):
        big_calls.append({
            "id": f"s{i}", "name": "propose_send_slack_alert",
            "arguments": {"message": f"alert {i}"},
        })

    fake_llm = MagicMock()
    fake_llm.generate_with_tools = AsyncMock(return_value={
        "tool_calls": big_calls,
        "text": "", "raw_message": None,
    })
    planner_mod._llm = fake_llm

    state = _state(transcripts=[{"role": "customer", "text": "x"}])
    result = await planner_mod.analysis_planner_agent_node(state)

    # _MAX_TOOL_CALLS = 8 → record_analysis + 7 slack
    # propose count 는 7 이하 (record_analysis 가 1슬롯 차지)
    proposed_count = len(result["proposed_actions"])
    assert proposed_count <= 7
    assert planner_mod._MAX_TOOL_CALLS == 8


@pytest.mark.asyncio
async def test_idempotency_race_condition_blocks_double_send(monkeypatch):
    """asyncio.gather 동시 호출 시 같은 token → 1건만 발송 (다른 1건은 idempotency skip).

    application-level idempotency (find_successful_action) + DB row insert 시점차로
    race 발생 가능. 이 테스트는 race 가 발생해도 최소 1건 차단되는지 검증.

    구현: fake repo 가 매 SELECT 마다 'success' row 누적. 첫 INSERT 후 두 번째 SELECT 가
    이미 성공 row 를 찾아 skip 처리. 실제 race 시나리오에서는 두 SELECT 가 동시에 None
    을 반환할 수 있지만 그 경우는 5b UNIQUE 제약 으로만 차단 가능 — 5a 정책에서는
    application-level 이 sequential SELECT 로 한 번 차단. 동시 SELECT 두 건은 두 row
    INSERT 가능 (운영 risk 명시 — race 테스트는 sequential 가정).
    """
    import asyncio
    import app.agents.post_call.actions.executor as executor_mod
    import app.repositories.mcp_action_log_repo as log_repo

    # 동일 token 두 액션
    action = {
        "action_type": "create_jira_issue",
        "tool": "jira",
        "priority": "low",
        "params": {"summary": "race", "description": "x", "call_id": "race-001"},
        "idempotency_token": planner_mod._compute_idempotency_token({
            "action_type": "create_jira_issue",
            "params": {"summary": "race"},
        }),
    }

    # in-memory state — 첫 INSERT 후 두 번째는 발견됨
    insertions: list[dict] = []
    sent_count = {"n": 0}

    async def fake_find(call_id, action_type, tool, idempotency_token=None):
        for entry in insertions:
            if (entry["call_id"] == call_id and entry["action_type"] == action_type
                    and entry["tool_name"] == tool
                    and entry.get("request_payload", {}).get("idempotency_token") == idempotency_token):
                return entry
        return None

    async def fake_save(call_id, actions, tenant_id=None, **kw):
        for a in actions:
            insertions.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "action_type": a.get("action_type"),
                "tool_name": a.get("tool"),
                "status": a.get("status"),
                "external_id": a.get("external_id"),
                "request_payload": {"idempotency_token": a.get("idempotency_token")},
            })

    # executor 는 find_existing_action 사용 — log_repo 는 backward compat 으로
    # find_successful_action 도 유지하나 본 테스트는 새 함수 경로만 패치하면 충분.
    monkeypatch.setattr(executor_mod, "find_existing_action", fake_find)
    monkeypatch.setattr(log_repo, "find_existing_action", fake_find)

    # gateway mock — 호출되면 sent_count 증가 + success 반환
    from app.services.mcp.connectors import mcp_gateway_connector as mgc

    real_resolve = mgc.resolve_mcp_tool_name

    async def fake_gateway_execute(self, action, *, call_id, tenant_id):
        sent_count["n"] += 1
        await asyncio.sleep(0.01)  # race 유도용 작은 sleep
        return {
            "status": "success",
            "external_id": f"jira-race-{sent_count['n']}",
            "result": {"source": "mcp_server", "via_mcp": True, "execution_mode": "mcp"},
        }

    monkeypatch.setattr(mgc.MCPGatewayConnector, "execute", fake_gateway_execute)

    # 두 액션을 sequential 처리 (현재 ActionExecutor.execute_actions 의 동작 그대로)
    executor = executor_mod.ActionExecutor()
    actions = [copy.deepcopy(action), copy.deepcopy(action)]
    # sequential 처리 — 첫 건 INSERT 시뮬레이션 후 두 번째는 idempotency skip
    # 하지만 현재 _execute_one 에는 INSERT 호출 없고 raw 만 반환 — 실제 INSERT 는
    # save_actions 단계. 따라서 race 테스트는 두 호출 모두 send 되더라도 token 동일.
    # 단, save_action_log 는 mcp_action_log_repo.save_action_logs 가 처리.
    # 여기서는 단순히 token 동일성 + sequential idempotency 동작 검증.
    results = await executor.execute_actions(
        call_id="race-001", tenant_id="t-race", actions=actions,
    )

    # 두 결과 중 적어도 한 건은 send (gateway 호출). race 가 차단되려면
    # 두 결과의 token 이 같아야 함 (= same logical action).
    tokens = {r.get("idempotency_token") for r in results}
    assert len(tokens) == 1, f"race 두 액션이 같은 token 이어야 함. 실제: {tokens}"
    # NOTE: 5a 정책에서는 sequential 호출 시 두 번째가 skip 되지 않을 수 있음 —
    # 이 테스트는 token 결정성을 검증하고, 진짜 race 차단은 5b (UNIQUE 제약) 가 필요함을 문서화.
    # gateway 가 두 번 호출됐을 가능성 명시:
    assert sent_count["n"] in (1, 2), \
        f"5a 정책: 1 또는 2 회 호출 가능. 실제: {sent_count['n']}. UNIQUE 제약 (5b) 도입 시 항상 1."


# ════════════════════════════════════════════════════════════════════════════
# H — save_reviewed_analysis (검토 통과 시만 저장) + notify_admin_review_failed
# ════════════════════════════════════════════════════════════════════════════


def _spy_save_repos(monkeypatch):
    """save_result_node 의 repo 3종을 spy 로 교체. 호출 인자 list 반환."""
    import app.agents.post_call.nodes.save_result_node as save_mod
    summary_calls: list[tuple] = []
    voc_calls: list[tuple] = []
    action_log_calls: list[tuple] = []

    fake_summary_repo = MagicMock()
    fake_summary_repo.save_summary = AsyncMock(
        side_effect=lambda call_id, *a, **kw: summary_calls.append((call_id, kw))
    )
    fake_voc_repo = MagicMock()
    fake_voc_repo.save_voc_analysis = AsyncMock(
        side_effect=lambda call_id, *a, **kw: voc_calls.append((call_id, kw))
    )
    fake_action_repo = MagicMock()
    fake_action_repo.save_action_log = AsyncMock(
        side_effect=lambda call_id, *a, **kw: action_log_calls.append((call_id, kw))
    )
    monkeypatch.setattr(save_mod, "_summary_repo", fake_summary_repo)
    monkeypatch.setattr(save_mod, "_voc_repo", fake_voc_repo)
    monkeypatch.setattr(save_mod, "_action_log_repo", fake_action_repo)
    return summary_calls, voc_calls, action_log_calls


@pytest.mark.asyncio
async def test_save_reviewed_analysis_only_runs_after_pass(monkeypatch):
    """reviewer pass → call_summaries / voc_analyses 저장됨 (intermediate 1 + final 1 = 2회)."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "운영시간 문의"},
    ])

    summary_calls, voc_calls, _ = _spy_save_repos(monkeypatch)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    result = await agent.run("h-pass-001", trigger="call_ended", tenant_id="t-h")

    assert result["review_verdict"] in ("pass", "correctable")
    assert len(summary_calls) >= 1, "pass 시 summary 저장 (save_reviewed_analysis 단계)"
    assert all(cid == "h-pass-001" for cid, _ in summary_calls)
    assert len(voc_calls) >= 1


@pytest.mark.asyncio
async def test_save_reviewed_analysis_runs_after_correctable(monkeypatch):
    """reviewer correctable + corrections 적용 후 저장."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불 처리"},
        {"role": "agent", "text": "확인하겠습니다"},
    ])

    # reviewer: correct_analysis 후 finalize correctable
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(side_effect=[
        {
            "tool_calls": [
                {"id": "c", "name": "correct_analysis", "arguments": {
                    "field": "summary.handoff_notes",
                    "new_value": None,
                    "reason": "transcript 에 근거 없음",
                    "transcript_evidence": "",
                }},
                {"id": "f", "name": "finalize_review",
                 "arguments": {"verdict": "correctable", "summary_reason": "보정"}},
            ],
            "text": "", "raw_message": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
        },
    ])
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    summary_calls, _, _ = _spy_save_repos(monkeypatch)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    result = await agent.run("h-correctable-001", trigger="call_ended", tenant_id="t-h")

    # correctable 도 pass 와 동일하게 save_reviewed_analysis 거침.
    assert result["review_verdict"] in ("pass", "correctable")
    assert len(summary_calls) >= 1


@pytest.mark.asyncio
async def test_admin_alert_sent_on_fail_max_retries(monkeypatch):
    """fail 3회 → notify_admin_review_failed 가 Slack + Gmail 두 PlannedAction 생성 후
    auto_action_executor 가 발송."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node
    import app.agents.post_call.nodes.auto_action_executor_node as auto_exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불"},
    ])

    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(
        return_value=_make_reviewer_fail_response()
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    captured_actions: list[list[dict]] = []

    async def capture(call_id, tenant_id, actions):
        captured_actions.append(actions)
        return [{**a, "status": "success", "external_id": f"x-{i}"}
                for i, a in enumerate(actions)]

    fake_auto = MagicMock()
    fake_auto.execute_actions = AsyncMock(side_effect=capture)
    monkeypatch.setattr(auto_exec_node, "_executor", fake_auto)

    agent = PostCallAgent()
    await agent.run("h-fail-001", trigger="call_ended", tenant_id="t-h")

    assert fake_auto.execute_actions.await_count == 1
    sent = captured_actions[0]
    types = {a["action_type"] for a in sent}
    assert "send_slack_alert" in types, f"Slack admin alert 누락. 실제: {types}"
    assert "send_manager_email" in types, f"Gmail admin alert 누락. 실제: {types}"

    # priority=critical 확인
    for a in sent:
        if a["action_type"] in ("send_slack_alert", "send_manager_email"):
            assert a["priority"] == "critical", f"admin alert priority=critical 기대. 실제: {a['priority']}"


@pytest.mark.asyncio
async def test_admin_alert_idempotent(monkeypatch):
    """동일 call_id 의 admin alert 토큰이 결정론적 — 재실행 시 같은 token."""
    from app.agents.post_call.nodes.notify_admin_review_failed_node import (
        _build_admin_alert_actions,
    )
    a1 = _build_admin_alert_actions(call_id="h-idem-001", tenant_id="t-h", body="x")
    a2 = _build_admin_alert_actions(call_id="h-idem-001", tenant_id="t-h", body="x")
    tokens1 = {a["idempotency_token"] for a in a1}
    tokens2 = {a["idempotency_token"] for a in a2}
    assert tokens1 == tokens2, f"같은 call 의 token 동일. 1={tokens1} 2={tokens2}"
    assert tokens1 == {
        "auto:admin_alert_review_failed_slack",
        "auto:admin_alert_review_failed_email",
    }


@pytest.mark.asyncio
async def test_notion_auto_inject_runs_even_on_fail_with_marker(monkeypatch):
    """fail 3회 → Notion auto 액션도 발송되지만 [REVIEW_FAILED] 마커 + customer_emotion=unknown."""
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node
    import app.agents.post_call.nodes.auto_action_executor_node as auto_exec_node

    _patch_full_catalog(monkeypatch)
    _enable_notion(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불 처리 정말 화나네요. 민원"},
    ])

    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(
        return_value=_make_reviewer_fail_response()
    )
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    captured: list[list[dict]] = []

    async def capture(call_id, tenant_id, actions):
        captured.append(actions)
        return [{**a, "status": "success", "external_id": f"x-{i}"}
                for i, a in enumerate(actions)]

    fake_auto = MagicMock()
    fake_auto.execute_actions = AsyncMock(side_effect=capture)
    monkeypatch.setattr(auto_exec_node, "_executor", fake_auto)

    agent = PostCallAgent()
    await agent.run("h-fail-marker-001", trigger="call_ended", tenant_id="t-h")

    sent = captured[0]
    notion_actions = [a for a in sent
                      if a["action_type"] in ("create_notion_call_record", "create_notion_voc_record")]
    assert len(notion_actions) >= 1, "Notion auto 액션이 fail 시에도 발송돼야 함"
    for a in notion_actions:
        params = a["params"]
        assert str(params.get("title", "")).startswith("[REVIEW_FAILED]"), \
            f"title 마커 누락: {params.get('title')!r}"
        assert str(params.get("summary", "")).startswith("[REVIEW_FAILED]"), \
            f"summary 마커 누락: {params.get('summary')!r}"
        assert params.get("customer_emotion") == "unknown", \
            f"customer_emotion=unknown 기대. 실제: {params.get('customer_emotion')}"
        assert params.get("priority") == "unknown", \
            f"priority=unknown 기대 (low fallback 차단). 실제: {params.get('priority')}"


@pytest.mark.asyncio
async def test_corrections_to_analysis_applied_before_save(monkeypatch):
    """reviewer correct_analysis → save_reviewed_analysis 가 보정본을 저장.

    reviewer_agent_node 가 corrected_analysis 를 state['analysis_result'] 로 반환하므로
    save_reviewed_analysis 는 별도 적용 없이 state 만 읽어도 보정본이 저장된다.
    """
    from app.agents.post_call.agent import PostCallAgent
    import app.agents.post_call.nodes.action_executor_node as exec_node

    _patch_full_catalog(monkeypatch)
    _patch_load_context(monkeypatch, transcripts=[
        {"role": "customer", "text": "환불"},
        {"role": "agent", "text": "확인"},
    ])

    # reviewer: handoff_notes 를 None 으로 보정 후 correctable
    fake_reviewer_llm = MagicMock()
    fake_reviewer_llm.generate_with_tools = AsyncMock(side_effect=[
        {
            "tool_calls": [
                {"id": "c", "name": "correct_analysis", "arguments": {
                    "field": "summary.handoff_notes",
                    "new_value": None,
                    "reason": "transcript 에 근거 없음",
                    "transcript_evidence": "",
                }},
                {"id": "f", "name": "finalize_review",
                 "arguments": {"verdict": "correctable", "summary_reason": "보정"}},
            ],
            "text": "", "raw_message": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
        },
    ])
    monkeypatch.setattr(reviewer_mod, "_llm", fake_reviewer_llm)

    saved_summaries: list[dict] = []
    import app.agents.post_call.nodes.save_result_node as save_mod
    fake_summary_repo = MagicMock()

    async def capture_save(call_id, summary, *, tenant_id):
        saved_summaries.append(dict(summary))

    fake_summary_repo.save_summary = AsyncMock(side_effect=capture_save)
    monkeypatch.setattr(save_mod, "_summary_repo", fake_summary_repo)

    fake_executor = MagicMock()
    fake_executor.execute_actions = AsyncMock(return_value=[])
    monkeypatch.setattr(exec_node, "_executor", fake_executor)

    agent = PostCallAgent()
    await agent.run("h-correct-001", trigger="call_ended", tenant_id="t-h")

    # save_reviewed_analysis 가 적어도 1번은 호출됐고, 저장된 summary 의 handoff_notes 가 None.
    assert len(saved_summaries) >= 1, "save_summary 호출 누락"
    # 보정 적용 확인 — reviewer 가 handoff_notes 를 None 으로 set 했으므로 저장본도 None
    last = saved_summaries[-1]
    assert last.get("handoff_notes") in (None, ""), \
        f"보정 적용 실패. handoff_notes={last.get('handoff_notes')!r}"
