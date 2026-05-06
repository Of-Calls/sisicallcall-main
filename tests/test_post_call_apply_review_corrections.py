from __future__ import annotations

import pytest

from app.agents.post_call.nodes.apply_review_corrections_node import (
    apply_review_corrections_node,
)


def _state() -> dict:
    return {
        "call_id": "correction-test-001",
        "tenant_id": "tenant-a",
        "trigger": "call_ended",
        "call_metadata": {},
        "transcripts": [],
        "branch_stats": {},
        "summary": {
            "summary_short": "기존 요약",
            "customer_intent": "기존 의도",
            "customer_emotion": "negative",
            "resolution_status": "resolved",
        },
        "voc_analysis": {
            "intent_result": {
                "primary_category": "제품/서비스 문의",
                "reason": "기존 근거",
                "is_repeat_topic": True,
            },
            "sentiment_result": {
                "sentiment": "neutral",
                "reason": "기존 감정 근거",
            },
            "priority_result": {
                "priority": "high",
                "action_required": True,
                "reason": "기존 우선순위 근거",
            },
        },
        "priority_result": {
            "priority": "high",
            "tier": "high",
            "action_required": True,
            "reason": "기존 top-level 우선순위",
        },
        "analysis_result": {
            "summary": {},
            "voc_analysis": {},
            "priority_result": {},
        },
        "review_result": {
            "verdict": "correctable",
            "confidence": 0.35,
            "corrections": {},
            "reason": "보정 필요",
        },
        "review_verdict": "correctable",
        "review_retry_count": 0,
        "human_review_required": False,
        "blocked_actions": [],
        "action_plan": None,
        "executed_actions": [],
        "dashboard_payload": None,
        "errors": [],
        "partial_success": False,
    }


@pytest.mark.asyncio
async def test_nested_voc_correction_preserves_existing_nested_fields():
    state = _state()
    state["review_result"]["corrections"] = {
        "voc_analysis": {
            "intent_result": {"primary_category": "의료/시설 문의"},
        }
    }

    result = await apply_review_corrections_node(state)
    voc = result["voc_analysis"]

    assert voc["intent_result"]["primary_category"] == "의료/시설 문의"
    assert voc["intent_result"]["reason"] == "기존 근거"
    assert voc["intent_result"]["is_repeat_topic"] is True
    assert voc["sentiment_result"]["sentiment"] == "neutral"
    assert voc["priority_result"]["priority"] == "high"


@pytest.mark.asyncio
async def test_correction_diff_contains_core_changed_fields():
    state = _state()
    state["review_result"]["corrections"] = {
        "summary": {
            "summary_short": "수정된 요약",
            "customer_emotion": "neutral",
        },
        "voc_analysis": {
            "intent_result": {"primary_category": "의료/시설 문의"},
            "sentiment_result": {"sentiment": "neutral"},
        },
        "priority_result": {
            "priority": "medium",
            "action_required": False,
        },
    }

    result = await apply_review_corrections_node(state)
    diff_by_path = {item["path"]: item for item in result["review_correction_diff"]}

    assert diff_by_path["summary.summary_short"]["before"] == "기존 요약"
    assert diff_by_path["summary.summary_short"]["after"] == "수정된 요약"
    assert diff_by_path["summary.customer_emotion"]["before"] == "negative"
    assert diff_by_path["summary.customer_emotion"]["after"] == "neutral"
    assert diff_by_path["voc_analysis.intent_result.primary_category"]["before"] == "제품/서비스 문의"
    assert diff_by_path["voc_analysis.intent_result.primary_category"]["after"] == "의료/시설 문의"
    assert diff_by_path["priority_result.priority"]["before"] == "high"
    assert diff_by_path["priority_result.priority"]["after"] == "medium"
    assert diff_by_path["priority_result.action_required"]["before"] is True
    assert diff_by_path["priority_result.action_required"]["after"] is False


@pytest.mark.asyncio
async def test_analysis_result_is_updated_with_merged_corrections():
    state = _state()
    state["review_result"]["corrections"] = {
        "priority_result": {"priority": "medium", "action_required": False}
    }

    result = await apply_review_corrections_node(state)

    assert result["priority_result"]["priority"] == "medium"
    assert result["analysis_result"]["priority_result"]["priority"] == "medium"
    assert result["analysis_result"]["voc_analysis"]["intent_result"]["reason"] == "기존 근거"


@pytest.mark.asyncio
async def test_empty_corrections_keep_state_and_return_empty_diff():
    state = _state()

    result = await apply_review_corrections_node(state)

    assert result["summary"] == state["summary"]
    assert result["voc_analysis"] == state["voc_analysis"]
    assert result["priority_result"] == state["priority_result"]
    assert result["review_correction_diff"] == []
