"""KDT-73 — Notion call_record / voc_record 데이터 구조 재구성 단위 테스트.

call_record: 통화 보관소 — 원본 transcript_full + 메타 (LLM 가공 최소화)
voc_record:  분석 인사이트 — LLM 요약/감정/우선순위 (reviewer 검증 통과한 것)
record_type 컬럼으로 Notion DB 안에서 구분.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.agents.post_call.nodes.analysis_planner_agent_node as planner_mod
import app.agents.post_call.tools.action_catalog as catalog_mod
from app.services.mcp.server.providers import notion_tools


@pytest.fixture(autouse=True)
def reset_llm_singletons(monkeypatch):
    monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")
    # Notion 자동 액션이 작동하려면 NOTION_API_TOKEN + NOTION_DATABASE_ID 필요
    monkeypatch.setenv("NOTION_API_TOKEN", "test-token")
    monkeypatch.setenv("NOTION_DATABASE_ID", "test-db")
    planner_mod._llm = None
    yield
    planner_mod._llm = None


def _patch_full_catalog(monkeypatch):
    from app.models.tenant_integration import IntegrationStatus
    fake = [
        SimpleNamespace(provider=p, status=IntegrationStatus.connected)
        for p in ("slack", "google_calendar", "jira", "gmail")
    ]
    monkeypatch.setattr(catalog_mod, "list_integrations", lambda tenant_id: fake)


def _planner_state(*, transcripts, call_metadata, branch_stats) -> dict:
    return {
        "call_id": "call-split-001",
        "tenant_id": "t-split",
        "trigger": "call_ended",
        "call_metadata": call_metadata,
        "transcripts": transcripts,
        "branch_stats": branch_stats,
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


def _angry_transcripts() -> list[dict]:
    return [
        {"role": "customer", "text": "환불 안 해주면 민원 넣을 거예요. 정말 화가 납니다."},
        {"role": "agent", "text": "죄송합니다."},
    ]


def _angry_metadata() -> dict:
    return {
        "call_id": "call-split-001",
        "tenant_id": "t-split",
        "customer_phone": "010-1234-5678",
        "start_time": "2026-05-11T14:00:00+00:00",
        "end_time": "2026-05-11T14:05:30+00:00",
    }


# ────────────────────────────────────────────────────────────────────────────
# call_record params 재구성 검증
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_record_contains_full_transcript(monkeypatch):
    """call_record params 에 transcript_full 필드 + turn 개수 일치."""
    _patch_full_catalog(monkeypatch)

    transcripts = [
        {"role": "customer", "text": "안녕하세요"},
        {"role": "agent", "text": "네 안녕하세요"},
        {"role": "customer", "text": "환불 요청 드려요"},
    ]
    state = _planner_state(
        transcripts=transcripts,
        call_metadata=_angry_metadata(),
        branch_stats={"faq": 0, "task": 0, "escalation": 1},
    )
    result = await planner_mod.analysis_planner_agent_node(state)

    call_records = [a for a in result["proposed_actions"]
                    if a["action_type"] == "create_notion_call_record"]
    assert len(call_records) == 1
    params = call_records[0]["params"]
    assert "transcript_full" in params
    assert isinstance(params["transcript_full"], list)
    assert len(params["transcript_full"]) == len(transcripts)
    # turn-by-turn 구조 확인
    assert params["transcript_full"][0] == {"turn": 0, "speaker": "customer", "text": "안녕하세요"}
    assert params["transcript_full"][2] == {"turn": 2, "speaker": "customer", "text": "환불 요청 드려요"}


@pytest.mark.asyncio
async def test_call_record_no_llm_summary(monkeypatch):
    """call_record params 에서 LLM 가공 필드 (summary / customer_emotion / priority) 제거."""
    _patch_full_catalog(monkeypatch)

    state = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={},
    )
    result = await planner_mod.analysis_planner_agent_node(state)

    call_records = [a for a in result["proposed_actions"]
                    if a["action_type"] == "create_notion_call_record"]
    params = call_records[0]["params"]

    # LLM 가공 필드 제거 검증
    assert "summary" not in params, f"call_record 에 summary 잔존: {params.get('summary')!r}"
    assert "summary_short" not in params, f"call_record 에 summary_short 잔존: {params.get('summary_short')!r}"
    assert "customer_emotion" not in params, f"call_record 에 customer_emotion 잔존"
    # priority 는 ActionItem 의 top-level 필드라 별도 — params 에서만 제거
    # (단, notify_admin_review_failed_node 가 fail 시 unknown 으로 set 할 수는 있음)

    # 원본 메타 필드 존재 검증
    assert params["caller_number"] == "010-1234-5678"
    assert params["started_at"] == "2026-05-11T14:00:00+00:00"
    assert params["ended_at"] == "2026-05-11T14:05:30+00:00"
    assert params["duration_sec"] == 330  # 5분 30초
    assert params["record_type"] == "call_record"


@pytest.mark.asyncio
async def test_call_record_has_record_type_field(monkeypatch):
    """call_record params 에 record_type='call_record' 존재."""
    _patch_full_catalog(monkeypatch)
    state = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={},
    )
    result = await planner_mod.analysis_planner_agent_node(state)

    call_record = next(a for a in result["proposed_actions"]
                       if a["action_type"] == "create_notion_call_record")
    assert call_record["params"]["record_type"] == "call_record"


# ────────────────────────────────────────────────────────────────────────────
# voc_record params 검증 (기존 분석 인사이트 + record_type)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_voc_record_contains_analysis(monkeypatch):
    """voc_record params 에 LLM 분석 필드 (sentiment / priority / suggested_action) 포함."""
    _patch_full_catalog(monkeypatch)
    state = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={"faq": 0, "task": 0, "escalation": 1},
    )
    result = await planner_mod.analysis_planner_agent_node(state)

    voc_records = [a for a in result["proposed_actions"]
                   if a["action_type"] == "create_notion_voc_record"]
    assert len(voc_records) == 1, "angry+high 시나리오에서 voc_record 1건 주입 기대"
    params = voc_records[0]["params"]

    # LLM 분석 필드 존재
    assert "customer_emotion" in params and params["customer_emotion"] in ("angry", "negative")
    assert "priority" in params and params["priority"] in ("high", "critical", "medium")
    assert "voc_content" in params
    assert "summary" in params
    assert params["record_type"] == "voc_record"


@pytest.mark.asyncio
async def test_voc_record_has_record_type_field(monkeypatch):
    """voc_record params 에 record_type='voc_record' 존재."""
    _patch_full_catalog(monkeypatch)
    state = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={},
    )
    result = await planner_mod.analysis_planner_agent_node(state)
    voc_record = next(a for a in result["proposed_actions"]
                      if a["action_type"] == "create_notion_voc_record")
    assert voc_record["params"]["record_type"] == "voc_record"


@pytest.mark.asyncio
async def test_voc_record_only_on_angry_high(monkeypatch):
    """neutral 통화 → voc_record 미주입. angry+high → 주입."""
    _patch_full_catalog(monkeypatch)

    # 1. neutral 통화
    state_neutral = _planner_state(
        transcripts=[
            {"role": "customer", "text": "운영시간이 어떻게 되나요"},
            {"role": "agent", "text": "오전 9시부터 6시"},
        ],
        call_metadata={"call_id": "x", "tenant_id": "t"},
        branch_stats={"faq": 1},
    )
    result = await planner_mod.analysis_planner_agent_node(state_neutral)
    types = {a["action_type"] for a in result["proposed_actions"]}
    assert "create_notion_call_record" in types, "모든 통화 call_record 주입"
    assert "create_notion_voc_record" not in types, "neutral 은 voc_record 미주입"

    # 2. angry+high 통화
    state_angry = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={"escalation": 1},
    )
    result = await planner_mod.analysis_planner_agent_node(state_angry)
    types = {a["action_type"] for a in result["proposed_actions"]}
    assert "create_notion_call_record" in types
    assert "create_notion_voc_record" in types, "angry+high 는 voc_record 주입"


# ────────────────────────────────────────────────────────────────────────────
# idempotency 회귀
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_record_idempotency_preserved(monkeypatch):
    """두 번 propose → 같은 idempotency_token (auto:auto_call_record)."""
    _patch_full_catalog(monkeypatch)

    # 두 번 동일 state 로 호출
    state1 = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={"escalation": 1},
    )
    state2 = _planner_state(
        transcripts=_angry_transcripts(),
        call_metadata=_angry_metadata(),
        branch_stats={"escalation": 1},
    )
    r1 = await planner_mod.analysis_planner_agent_node(state1)
    planner_mod._llm = None  # reset between calls (mock LLM)
    r2 = await planner_mod.analysis_planner_agent_node(state2)

    cr1 = next(a for a in r1["proposed_actions"]
               if a["action_type"] == "create_notion_call_record")
    cr2 = next(a for a in r2["proposed_actions"]
               if a["action_type"] == "create_notion_call_record")
    assert cr1["idempotency_token"] == "auto:auto_call_record"
    assert cr2["idempotency_token"] == "auto:auto_call_record"
    # 두 token 동일 → DB idempotency 차단 작동
    assert cr1["idempotency_token"] == cr2["idempotency_token"]


# ────────────────────────────────────────────────────────────────────────────
# notion_tools — page properties / body 구조
# ────────────────────────────────────────────────────────────────────────────


def test_notion_call_properties_no_llm_fields():
    """_build_call_properties 결과에 Customer Emotion / Priority / Summary 없음."""
    params = {
        "call_id": "abc-1234-def",
        "tenant_id": "t-1",
        "record_type": "call_record",
        "caller_number": "010-9999-8888",
        "started_at": "2026-05-11T10:00:00+00:00",
        "duration_sec": 120,
        "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
        "transcript_full": [
            {"turn": 0, "speaker": "customer", "text": "안녕하세요"},
        ],
    }
    props = notion_tools._build_call_properties(params, call_id="abc-1234-def")
    assert props["Record Type"] == {"select": {"name": "call"}}
    assert "Caller Number" in props
    assert "Branch Stats" in props
    assert "Duration Sec" in props
    assert "Started At" in props
    # LLM 가공 필드 부재
    assert "Customer Emotion" not in props
    assert "Priority" not in props
    assert "Summary" not in props
    assert "VOC Category" not in props


def test_notion_voc_properties_has_llm_fields():
    """_build_voc_properties 결과에 분석 필드 모두 존재."""
    params = {
        "call_id": "abc",
        "tenant_id": "t",
        "record_type": "voc_record",
        "summary_short": "환불 요청 통화",
        "customer_emotion": "angry",
        "priority": "high",
        "voc_category": "민원/불만",
        "suggested_action": "팀장 보고",
        "action_required": True,
    }
    props = notion_tools._build_voc_properties(params, call_id="abc")
    assert props["Record Type"] == {"select": {"name": "voc"}}
    assert props["Customer Emotion"] == {"select": {"name": "angry"}}
    assert props["Priority"] == {"select": {"name": "high"}}
    assert props["Summary"]["rich_text"][0]["text"]["content"] == "환불 요청 통화"
    assert props["Suggested Action"]["rich_text"][0]["text"]["content"] == "팀장 보고"
    assert props["Action Required"] == {"checkbox": True}


def test_notion_page_body_contains_transcript_turns():
    """_build_call_children: heading_1 + 각 turn paragraph block."""
    transcripts = [
        {"turn": 0, "speaker": "customer", "text": "환불 처리해주세요"},
        {"turn": 1, "speaker": "agent", "text": "확인해드리겠습니다"},
        {"turn": 2, "speaker": "customer", "text": "감사합니다"},
    ]
    children = notion_tools._build_call_children({"transcript_full": transcripts})

    # 첫 block 은 heading_1 "통화 내역"
    assert children[0]["type"] == "heading_1"
    assert children[0]["heading_1"]["rich_text"][0]["text"]["content"] == "통화 내역"

    # 이후 turn 개수만큼 paragraph
    paragraphs = [c for c in children if c["type"] == "paragraph"]
    assert len(paragraphs) == 3

    # 첫 paragraph: [고객] prefix + bold
    first = paragraphs[0]["paragraph"]["rich_text"]
    assert first[0]["text"]["content"] == "[고객] "
    assert first[0]["annotations"]["bold"] is True
    assert first[1]["text"]["content"] == "환불 처리해주세요"

    # 두 번째 paragraph: [상담원] prefix
    second = paragraphs[1]["paragraph"]["rich_text"]
    assert second[0]["text"]["content"] == "[상담원] "


def test_notion_page_body_empty_when_no_transcripts():
    """transcript_full 없으면 heading_1 만 들어가는지 (paragraph 0)."""
    children = notion_tools._build_call_children({})
    assert len(children) == 1
    assert children[0]["type"] == "heading_1"


def test_notion_call_name_review_failed_marker():
    """notify_admin_review_failed_node 가 title='[REVIEW_FAILED] ...' 로 mutate 한 경우
    Name 에도 prefix 가 표시되어야 한다 — 운영자가 Notion 에서 즉시 식별."""
    params = {
        "call_id": "abc-1234",
        "caller_number": "010-0000-0000",
        "title": "[REVIEW_FAILED] 통화 기록",  # notify_admin_review_failed_node 가 mutate
    }
    props = notion_tools._build_call_properties(params, call_id="abc-1234")
    name = props["Name"]["title"][0]["text"]["content"]
    assert name.startswith("[REVIEW_FAILED] "), f"Name 에 marker 누락: {name!r}"


def test_notion_voc_name_review_failed_marker():
    params = {
        "call_id": "abc",
        "summary_short": "환불",
        "title": "[REVIEW_FAILED] VOC",
    }
    props = notion_tools._build_voc_properties(params, call_id="abc")
    name = props["Name"]["title"][0]["text"]["content"]
    assert name.startswith("[REVIEW_FAILED] ")


# ────────────────────────────────────────────────────────────────────────────
# MCP gateway 매핑
# ────────────────────────────────────────────────────────────────────────────


def test_voc_record_routes_to_separate_mcp_tool():
    """create_notion_voc_record 가 별도 MCP tool 로 라우팅."""
    from app.services.mcp.connectors.mcp_gateway_connector import resolve_mcp_tool_name
    assert resolve_mcp_tool_name("notion", "create_notion_call_record") == "notion.create_notion_call_record"
    assert resolve_mcp_tool_name("notion", "create_notion_voc_record") == "notion.create_notion_voc_record"


def test_notion_voc_record_tool_function_exists():
    """notion_tools 에 create_notion_voc_record 함수가 별도로 존재."""
    assert hasattr(notion_tools, "create_notion_voc_record")
    assert callable(notion_tools.create_notion_voc_record)
    assert hasattr(notion_tools, "create_notion_call_record")


# ────────────────────────────────────────────────────────────────────────────
# helpers — 단위
# ────────────────────────────────────────────────────────────────────────────


def test_serialize_transcripts():
    out = planner_mod._serialize_transcripts([
        {"role": "customer", "text": "hi"},
        {"role": "agent", "text": "hello"},
        {"speaker": "customer", "text": "bye"},  # speaker alias
    ])
    assert out == [
        {"turn": 0, "speaker": "customer", "text": "hi"},
        {"turn": 1, "speaker": "agent", "text": "hello"},
        {"turn": 2, "speaker": "customer", "text": "bye"},
    ]


def test_compute_duration_sec():
    assert planner_mod._compute_duration_sec({
        "start_time": "2026-05-11T10:00:00+00:00",
        "end_time":   "2026-05-11T10:05:30+00:00",
    }) == 330
    assert planner_mod._compute_duration_sec({}) is None
    assert planner_mod._compute_duration_sec({"start_time": "invalid"}) is None
