from __future__ import annotations

import pytest

import app.agents.post_call.nodes.review_node as review_mod


def _state() -> dict:
    return {
        "call_id": "review-test-001",
        "tenant_id": "tenant-a",
        "trigger": "call_ended",
        "call_metadata": {},
        "transcripts": [{"role": "customer", "text": "예약 변경 가능한가요?"}],
        "branch_stats": {},
        "summary": {},
        "voc_analysis": {},
        "priority_result": {},
        "action_plan": None,
        "executed_actions": [],
        "dashboard_payload": None,
        "errors": [],
        "partial_success": False,
        "analysis_result": {
            "summary": {"summary_short": "예약 변경 문의"},
            "voc_analysis": {"intent_result": {"primary_category": "예약/일정"}},
            "priority_result": {"priority": "medium"},
        },
        "review_result": None,
        "review_verdict": None,
        "review_retry_count": 0,
        "human_review_required": False,
        "blocked_actions": [],
    }


class _FakeReviewCaller:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response or {}
        self.error = error

    async def call_json(self, **kwargs):
        if self.error is not None:
            raise self.error
        return dict(self.response)


def test_confidence_numeric_string_is_converted_to_float():
    confidence, meta = review_mod._normalize_confidence({"confidence": "0.87"})

    assert confidence == 0.87
    assert meta["confidence_source"] == "llm"
    assert meta["confidence_missing"] is False
    assert meta["confidence_parse_error"] is False


def test_confidence_missing_is_marked_as_default():
    confidence, meta = review_mod._normalize_confidence({})

    assert confidence == 0.0
    assert meta["confidence_source"] == "default"
    assert meta["confidence_missing"] is True
    assert meta["confidence_parse_error"] is False


def test_confidence_parse_error_is_distinguishable():
    confidence, meta = review_mod._normalize_confidence({"confidence": "not-a-number"})

    assert confidence == 0.0
    assert meta["confidence_source"] == "parse_error"
    assert meta["confidence_missing"] is False
    assert meta["confidence_parse_error"] is True


def test_confidence_out_of_range_is_clamped():
    high, high_meta = review_mod._normalize_confidence({"confidence": 1.7})
    low, low_meta = review_mod._normalize_confidence({"confidence": -0.5})

    assert high == 1.0
    assert low == 0.0
    assert high_meta["confidence_source"] == "clamped"
    assert low_meta["confidence_source"] == "clamped"


def test_confidence_from_llm_fallback_is_marked_as_fallback():
    confidence, meta = review_mod._normalize_confidence({
        "confidence": 0.95,
        "_llm_fallback": True,
    })

    assert confidence == 0.95
    assert meta["confidence_source"] == "fallback"


@pytest.mark.asyncio
async def test_review_node_preserves_reason_and_corrected_keys(monkeypatch):
    monkeypatch.setattr(
        review_mod,
        "_caller",
        _FakeReviewCaller({
            "verdict": "correctable",
            "confidence": "0.42",
            "issues": [],
            "corrections": {
                "summary": {"customer_emotion": "neutral"},
                "voc_analysis": {},
                "priority_result": {"priority": "medium"},
            },
            "blocked_actions": [],
            "reason": "priority_result 보정 필요",
        }),
    )

    result = await review_mod.review_node(_state())
    review_result = result["review_result"]

    assert result["review_verdict"] == "correctable"
    assert review_result["confidence"] == 0.42
    assert review_result["confidence_source"] == "llm"
    assert review_result["reason"] == "priority_result 보정 필요"
    assert review_result["corrected_keys"] == ["summary", "priority_result"]


@pytest.mark.asyncio
async def test_review_node_preserves_llm_fallback_metadata(monkeypatch):
    monkeypatch.setattr(
        review_mod,
        "_caller",
        _FakeReviewCaller({
            "verdict": "pass",
            "confidence": 0.95,
            "issues": [],
            "corrections": {},
            "blocked_actions": [],
            "reason": "mock fallback review",
            "_llm_fallback": True,
            "_llm_fallback_reason": "LLM JSON parse failed twice",
        }),
    )

    result = await review_mod.review_node(_state())
    review_result = result["review_result"]

    assert review_result["confidence_source"] == "fallback"
    assert review_result["llm_fallback"] is True
    assert "JSON parse failed" in review_result["llm_fallback_reason"]


@pytest.mark.asyncio
async def test_review_node_warns_on_correctable_zero_confidence(monkeypatch):
    warnings: list[str] = []

    def fake_warning(message, *args, **kwargs):
        warnings.append(message % args if args else message)

    monkeypatch.setattr(review_mod.logger, "warning", fake_warning)
    monkeypatch.setattr(
        review_mod,
        "_caller",
        _FakeReviewCaller({
            "verdict": "correctable",
            "confidence": 0.0,
            "issues": [],
            "corrections": {"summary": {"customer_emotion": "neutral"}},
            "blocked_actions": [],
            "reason": "summary 보정 필요",
        }),
    )

    result = await review_mod.review_node(_state())

    assert result["review_result"]["confidence"] == 0.0
    assert any("review confidence suspicious" in item for item in warnings)


@pytest.mark.asyncio
async def test_review_node_missing_confidence_does_not_break_pipeline(monkeypatch):
    warnings: list[str] = []

    def fake_warning(message, *args, **kwargs):
        warnings.append(message % args if args else message)

    monkeypatch.setattr(review_mod.logger, "warning", fake_warning)
    monkeypatch.setattr(
        review_mod,
        "_caller",
        _FakeReviewCaller({
            "verdict": "correctable",
            "issues": [],
            "corrections": {"voc_analysis": {"intent_result": {"primary_category": "예약/일정"}}},
            "blocked_actions": [],
            "reason": "intent_result 보정 필요",
        }),
    )

    result = await review_mod.review_node(_state())
    review_result = result["review_result"]

    assert result["review_verdict"] == "correctable"
    assert review_result["confidence"] == 0.0
    assert review_result["confidence_missing"] is True
    assert review_result["confidence_source"] == "default"
    assert any("review confidence suspicious" in item for item in warnings)


@pytest.mark.asyncio
async def test_review_node_exception_uses_fallback_fail_result(monkeypatch):
    monkeypatch.setattr(
        review_mod,
        "_caller",
        _FakeReviewCaller(error=RuntimeError("review boom")),
    )

    result = await review_mod.review_node(_state())

    assert result["review_verdict"] == "fail"
    assert result["human_review_required"] is True
    assert result["review_result"]["confidence_source"] == "fallback"
