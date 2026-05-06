from __future__ import annotations

import pytest

import app.agents.post_call.llm_caller as llm_mod


@pytest.fixture(autouse=True)
def clear_llm_env(monkeypatch):
    monkeypatch.delenv("POST_CALL_LLM_MODE", raising=False)
    monkeypatch.delenv("POST_CALL_USE_REAL_LLM", raising=False)
    monkeypatch.delenv("POST_CALL_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_mod.settings, "openai_api_key", "")


def test_default_post_call_llm_mode_is_mock():
    assert llm_mod.get_post_call_llm_mode() == "mock"
    assert llm_mod.describe_post_call_llm() == "Demo Mock LLM"


def test_post_call_llm_mode_real_selects_real_description(monkeypatch):
    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setenv("POST_CALL_LLM_MODEL", "gpt-4o-mini")

    assert llm_mod.get_post_call_llm_mode() == "real"
    assert llm_mod.describe_post_call_llm() == "OpenAI Real LLM (gpt-4o-mini)"


def test_legacy_post_call_use_real_llm_still_selects_real(monkeypatch):
    monkeypatch.setenv("POST_CALL_USE_REAL_LLM", "true")

    assert llm_mod.get_post_call_llm_mode() == "real"


def test_cli_llm_mode_overrides_environment(monkeypatch):
    import scripts.run_post_call_from_db as db_runner

    monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")

    assert db_runner._apply_llm_mode("real") == "real"
    assert llm_mod.get_post_call_llm_mode() == "real"
    assert db_runner._apply_llm_mode("mock") == "mock"
    assert llm_mod.get_post_call_llm_mode() == "mock"


@pytest.mark.asyncio
async def test_real_llm_json_parsing_success(monkeypatch):
    class FakeOpenAIService:
        def __init__(self, model=None):
            self.model = model

        async def generate(self, **kwargs):
            return """
            {
              "summary": {
                "summary_short": "Reservation change request",
                "summary_detailed": "Customer asked to change a reservation.",
                "customer_intent": "Change reservation",
                "customer_emotion": "neutral",
                "resolution_status": "resolved",
                "keywords": ["reservation", "change"],
                "handoff_notes": null
              },
              "voc_analysis": {
                "sentiment_result": {"sentiment": "neutral", "intensity": 0.3, "reason": "informational"},
                "intent_result": {"primary_category": "예약/일정", "sub_categories": ["예약 변경"], "is_repeat_topic": false, "faq_candidate": false},
                "priority_result": {"priority": "medium", "action_required": false, "suggested_action": null, "reason": "ordinary request"}
              },
              "priority_result": {"priority": "medium", "tier": "medium", "action_required": false, "suggested_action": null, "reason": "ordinary request"}
            }
            """

    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setattr(llm_mod, "PostCallOpenAIService", FakeOpenAIService)

    caller = llm_mod.make_analysis_caller()
    result = await caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert result["summary"]["summary_short"] == "Reservation change request"
    assert result["voc_analysis"]["intent_result"]["primary_category"] == "예약/일정"


@pytest.mark.asyncio
async def test_real_llm_parse_failure_falls_back_to_mock(monkeypatch):
    class BadOpenAIService:
        def __init__(self, model=None):
            self.model = model

        async def generate(self, **kwargs):
            return "not-json"

    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setattr(llm_mod, "PostCallOpenAIService", BadOpenAIService)

    caller = llm_mod.make_analysis_caller()
    result = await caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert "summary" in result
    assert "voc_analysis" in result
    assert result["priority_result"]["priority"] == "low"
    assert result["_llm_fallback"] is True
    assert "LLM JSON parse failed" in result["_llm_fallback_reason"]


@pytest.mark.asyncio
async def test_real_llm_missing_openai_api_key_is_clear():
    service = llm_mod.PostCallOpenAIService(model="gpt-4o-mini")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await service.generate(
            system_prompt="system",
            user_message="user",
            max_tokens=10,
        )


# ── Token usage extraction ────────────────────────────────────────────────────

class _FakeUsageObj:
    def __init__(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _FakeResponse:
    def __init__(self, usage):
        self.usage = usage


def test_extract_openai_usage_from_attribute_object():
    response = _FakeResponse(_FakeUsageObj(120, 30, 150))
    usage = llm_mod.extract_openai_usage(response)

    assert usage == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }


def test_extract_openai_usage_from_dict_response():
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    usage = llm_mod.extract_openai_usage(response)

    assert usage["total_tokens"] == 15


def test_extract_openai_usage_missing_returns_none():
    class NoUsage:
        pass

    usage = llm_mod.extract_openai_usage(NoUsage())
    assert usage == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_extract_openai_usage_partial_fields():
    response = _FakeResponse(_FakeUsageObj(100, 0, 100))
    response.usage.completion_tokens = None  # type: ignore[assignment]
    usage = llm_mod.extract_openai_usage(response)

    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] is None


# ── Estimated cost ────────────────────────────────────────────────────────────

def test_compute_estimated_cost_known_model():
    cost = llm_mod.compute_estimated_cost_usd(1_000_000, 1_000_000, "gpt-4o-mini")
    # input 0.15 + output 0.60 per 1M
    assert cost == round(0.15 + 0.60, 6)


def test_compute_estimated_cost_unknown_model_returns_none():
    assert llm_mod.compute_estimated_cost_usd(100, 50, "unknown-model") is None


def test_compute_estimated_cost_missing_model_returns_none():
    assert llm_mod.compute_estimated_cost_usd(100, 50, None) is None


def test_compute_estimated_cost_zero_tokens_returns_none():
    assert llm_mod.compute_estimated_cost_usd(0, 0, "gpt-4o-mini") is None
    assert llm_mod.compute_estimated_cost_usd(None, None, "gpt-4o-mini") is None


def test_compute_estimated_cost_handles_only_prompt_tokens():
    cost = llm_mod.compute_estimated_cost_usd(2_000_000, 0, "gpt-4o-mini")
    assert cost == round(2 * 0.15, 6)


# ── Usage attached to LLM caller result ───────────────────────────────────────

@pytest.mark.asyncio
async def test_real_llm_attaches_usage_metadata(monkeypatch):
    """When the OpenAI provider exposes _last_usage, the caller attaches it."""
    class FakeOpenAIService:
        def __init__(self, model=None):
            self.model = model
            self._last_usage = None

        async def generate(self, **kwargs):
            self._last_usage = {
                "prompt_tokens": 500,
                "completion_tokens": 120,
                "total_tokens": 620,
            }
            return '{"summary": {"summary_short": "ok"}}'

    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setattr(llm_mod, "PostCallOpenAIService", FakeOpenAIService)

    caller = llm_mod.make_analysis_caller()
    result = await caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert "_llm_usage" in result
    usage = result["_llm_usage"]
    assert usage["prompt_tokens"] == 500
    assert usage["completion_tokens"] == 120
    assert usage["total_tokens"] == 620
    assert usage["purpose"] == "analysis"
    assert usage["source"] == "openai"
    assert usage["fallback"] is False


@pytest.mark.asyncio
async def test_real_llm_fallback_marks_fallback_in_usage(monkeypatch):
    """When the real LLM fails and falls back to mock, _llm_usage marks fallback."""
    class BadOpenAIService:
        def __init__(self, model=None):
            self.model = model
            self._last_usage = None

        async def generate(self, **kwargs):
            return "not-json"

    monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
    monkeypatch.setattr(llm_mod, "PostCallOpenAIService", BadOpenAIService)

    caller = llm_mod.make_analysis_caller()
    result = await caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert result["_llm_fallback"] is True
    assert "_llm_usage" in result
    assert result["_llm_usage"]["source"] == "fallback"
    assert result["_llm_usage"]["fallback"] is True
    assert result["_llm_usage"]["total_tokens"] is None


@pytest.mark.asyncio
async def test_mock_caller_does_not_attach_usage():
    """MockLLMCaller never attaches _llm_usage — pipelines treat it as None."""
    mock_caller = llm_mod.MockLLMCaller()
    result = await mock_caller.call_json("ANALYSIS_COMBINED", "transcript")

    assert "_llm_usage" not in result
