"""
PostCallAgent 통합 테스트 (2-에이전트 그래프).

LLM 호출은 POST_CALL_LLM_MODE=mock 으로 결정론적 mock 으로 강제.
"""
from __future__ import annotations

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.post_call.agent import PostCallAgent


# ── 픽스처 ────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def force_mock_llm(monkeypatch):
    """모든 테스트에서 mock LLM 강제."""
    monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")
    import app.agents.post_call.nodes.analysis_planner_agent_node as planner_mod
    import app.agents.post_call.nodes.reviewer_agent_node as reviewer_mod
    planner_mod._llm = None
    reviewer_mod._llm = None
    yield
    planner_mod._llm = None
    reviewer_mod._llm = None


@pytest.fixture
def agent():
    return PostCallAgent()


# ── 기본 통합 플로우 ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_call_ended_full_pipeline(agent):
    result = await agent.run("call-001", trigger="call_ended", tenant_id="test")

    assert result["call_id"] == "call-001"
    assert result["trigger"] == "call_ended"
    assert result["summary"] is not None
    assert result["voc_analysis"] is not None
    assert result["priority_result"] is not None
    assert result["analysis_result"] is not None
    # reviewer 가 실행되어 verdict 가 채워졌어야 한다
    assert result["review_verdict"] in ("pass", "correctable", "fail")
    assert result["dashboard_payload"] is not None
    assert isinstance(result["executed_actions"], list)


@pytest.mark.asyncio
async def test_run_escalation_immediate_skips_reviewer_and_actions(agent):
    """escalation_immediate → 분석 + save_intermediate 만 실행, reviewer/액션 스킵."""
    result = await agent.run("call-002", trigger="escalation_immediate", tenant_id="test")

    assert result["call_id"] == "call-002"
    assert result["trigger"] == "escalation_immediate"
    assert result["summary"] is not None
    # reviewer 미실행 → verdict 기본값 None
    assert result["review_verdict"] is None
    assert result["executed_actions"] == []
    assert result["dashboard_payload"] is not None


@pytest.mark.asyncio
async def test_run_manual_full_pipeline(agent):
    result = await agent.run("call-003", trigger="manual", tenant_id="test")

    assert result["trigger"] == "manual"
    assert result["summary"] is not None
    assert result["voc_analysis"] is not None
    assert result["priority_result"] is not None


@pytest.mark.asyncio
async def test_invalid_trigger_raises(agent):
    with pytest.raises(ValueError, match="Unknown trigger"):
        await agent.run("call-004", trigger="bad_trigger")


# ── 스키마 검증 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_schema(agent):
    result = await agent.run("call-010", trigger="call_ended")
    s = result["summary"]
    assert s is not None
    for key in ("summary_short", "summary_detailed", "customer_intent",
                "customer_emotion", "resolution_status", "keywords"):
        assert key in s, f"summary missing key: {key}"
    assert s["customer_emotion"] in ("positive", "neutral", "negative", "angry")
    assert s["resolution_status"] in ("resolved", "escalated", "abandoned")
    assert isinstance(s["keywords"], list)


@pytest.mark.asyncio
async def test_voc_schema(agent):
    result = await agent.run("call-011", trigger="call_ended")
    voc = result["voc_analysis"]
    assert voc is not None
    assert "sentiment_result" in voc
    assert "intent_result" in voc
    assert "priority_result" in voc

    sr = voc["sentiment_result"]
    assert sr["sentiment"] in ("positive", "neutral", "negative", "angry")
    assert isinstance(sr["intensity"], (int, float))
    assert 0.0 <= sr["intensity"] <= 1.0

    pr = voc["priority_result"]
    assert pr["priority"] in ("low", "medium", "high", "critical")
    assert isinstance(pr["action_required"], bool)


@pytest.mark.asyncio
async def test_priority_schema(agent):
    result = await agent.run("call-012", trigger="call_ended")
    p = result["priority_result"]
    assert p is not None
    assert p["priority"] in ("low", "medium", "high", "critical")
    assert "tier" in p
    assert p["tier"] == p["priority"]
    assert isinstance(p["action_required"], bool)


# ── empty transcript 처리 ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_transcript_completes(agent, monkeypatch):
    """transcripts 가 비어 있어도 agent.run() 이 끝까지 완료된다 (human_queue 분기)."""
    import app.agents.post_call.nodes.load_context_node as lcn

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "empty-001"},
        "transcripts": [],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    result = await agent.run("empty-001", trigger="call_ended")

    assert result is not None
    assert result["partial_success"] is True
    assert any(
        e.get("warning") == "empty_transcript"
        or "transcript" in str(e.get("error", "")).lower()
        for e in result["errors"]
    ), f"empty_transcript 에러 없음: {result['errors']}"


@pytest.mark.asyncio
async def test_empty_transcript_escalation_completes(agent, monkeypatch):
    """escalation_immediate + 빈 녹취도 끝까지 완료된다."""
    import app.agents.post_call.nodes.load_context_node as lcn

    mock_repo = MagicMock()
    mock_repo.get_call_context = AsyncMock(return_value={
        "metadata": {"call_id": "empty-002"},
        "transcripts": [],
        "branch_stats": {},
    })
    monkeypatch.setattr(lcn, "_repo", mock_repo)

    result = await agent.run("empty-002", trigger="escalation_immediate")

    assert result is not None
    assert result["partial_success"] is True
    # escalation_immediate 는 reviewer 진입 안 함
    assert result["review_verdict"] is None


# ── LLM 실패 처리 ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_llm_failure_partial_success(agent, monkeypatch):
    """analysis_planner LLM 실패 → fallback 분석 + human_review_required."""
    import app.agents.post_call.nodes.analysis_planner_agent_node as planner_mod

    failing = MagicMock()
    failing.generate_with_tools = AsyncMock(side_effect=RuntimeError("LLM 오류"))
    monkeypatch.setattr(planner_mod, "_llm", failing)

    result = await agent.run("call-022", trigger="call_ended")

    assert result is not None
    assert result["partial_success"] is True
    assert result["human_review_required"] is True
    # save 는 끝까지 도달
    assert result["dashboard_payload"] is not None


# ── dashboard payload ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dashboard_payload_has_required_keys(agent):
    result = await agent.run("call-031", trigger="call_ended", tenant_id="demo")
    payload = result["dashboard_payload"]
    assert payload is not None
    for key in (
        "call_id", "tenant_id", "trigger", "summary", "voc_analysis",
        "priority_result", "executed_actions", "errors", "partial_success",
        "analysis_result", "proposed_actions", "review_result",
        "review_verdict", "approved_actions",
    ):
        assert key in payload, f"dashboard_payload missing key: {key}"


@pytest.mark.asyncio
async def test_dashboard_partial_success_consistent_clean(agent):
    result = await agent.run("consist-001", trigger="call_ended")
    assert result["dashboard_payload"]["partial_success"] == result["partial_success"]


# ── scripts/run_post_call_agent.py 회귀 ────────────────────────────────────────

def test_run_post_call_script_mock():
    """scripts/run_post_call_agent.py 가 Mock LLM 으로 정상 종료된다."""
    import subprocess
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POST_CALL_LLM_MODE": "mock"}
    result = subprocess.run(
        [sys.executable, "scripts/run_post_call_agent.py", "--trigger", "call_ended"],
        capture_output=True,
        timeout=30,
        cwd=str(project_root),
        env=env,
    )
    assert result.returncode == 0, (
        f"스크립트 실행 실패 (returncode={result.returncode})\n"
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )


def test_run_post_call_script_escalation_mock():
    """--trigger escalation_immediate 도 정상 종료된다."""
    import subprocess
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "POST_CALL_LLM_MODE": "mock"}
    result = subprocess.run(
        [sys.executable, "scripts/run_post_call_agent.py", "--trigger", "escalation_immediate"],
        capture_output=True,
        timeout=30,
        cwd=str(project_root),
        env=env,
    )
    assert result.returncode == 0, (
        f"escalation_immediate 스크립트 실패\n"
        f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
    )
