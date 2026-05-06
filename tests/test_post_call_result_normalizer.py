from __future__ import annotations

import pytest

from app.agents.post_call.normalizer import (
    infer_primary_category,
    normalize_post_call_result,
)
from app.agents.post_call.nodes import save_result_node as save_node


def _base_state() -> dict:
    return {
        "call_id": "normalizer-call-001",
        "tenant_id": "tenant-a",
        "trigger": "manual",
        "call_metadata": {},
        "transcripts": [],
        "branch_stats": {},
        "summary": {
            "summary_short": "고객이 예약 변경 가능 여부를 문의했습니다.",
            "customer_intent": "예약 변경 문의",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "keywords": ["예약", "변경"],
        },
        "voc_analysis": {
            "intent_result": {},
            "priority_result": {},
            "sentiment_result": {},
        },
        "priority_result": {},
        "action_plan": None,
        "executed_actions": [],
        "dashboard_payload": None,
        "errors": [],
        "partial_success": False,
        "analysis_result": None,
        "review_result": None,
        "review_verdict": "pass",
        "review_retry_count": 0,
        "human_review_required": False,
        "blocked_actions": [],
    }


def test_primary_category_missing_is_filled_from_intent_and_keywords():
    state = _base_state()

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "예약/일정"


def test_existing_primary_category_is_preserved():
    state = _base_state()
    state["voc_analysis"]["intent_result"] = {
        "primary_category": "LLM 분류",
        "reason": "already classified",
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "LLM 분류"


def test_existing_specific_primary_category_is_not_overwritten_by_tenant_rule():
    state = _base_state()
    state["tenant_industry"] = "hospital"
    state["summary"] = {
        "summary_short": "고객이 응급실 위치와 주차장을 문의했습니다.",
        "customer_intent": "응급실 위치 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["응급실", "주차장"],
    }
    state["voc_analysis"]["intent_result"] = {
        "primary_category": "제품/서비스 문의",
        "reason": "already specific",
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "제품/서비스 문의"


def test_problem_row_like_hospital_input_gets_non_empty_category():
    state = {
        **_base_state(),
        "tenant_id": "05665d1a-e25b-44f7-be6c-a262603dbfd5",
        "tenant_industry": "hospital",
        "summary": {
            "summary_short": "고객은 응급실 이용 방법과 주차 요금, 교통편 및 음식 판매 여부에 대해 문의했습니다.",
            "customer_intent": "응급실 이용 방법 및 주차 요금 문의",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "keywords": ["응급실", "주차장", "주차요금", "대중교통", "음식"],
        },
        "voc_analysis": {
            "intent_result": {
                "reason": "고객이 음식 판매 여부에 대해 반복적으로 질문함",
                "is_repeat_topic": True,
            },
            "priority_result": {
                "priority": "medium",
                "action_required": False,
            },
            "sentiment_result": {
                "sentiment": "neutral",
                "intensity": 0.0,
            },
        },
    }

    result = normalize_post_call_result(state)
    category = result["voc_analysis"]["intent_result"]["primary_category"]

    assert category == "의료/시설 문의"


def test_generic_category_is_refined_for_problem_row_like_hospital_input():
    state = _base_state()
    state["tenant_industry"] = "hospital"
    state["summary"] = {
        "summary_short": "고객은 응급실 이용 방법과 주차 요금, 교통편 및 음식 판매 여부에 대해 문의했습니다.",
        "customer_intent": "응급실 이용 방법 및 주차 요금 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["응급실", "주차장", "주차요금", "대중교통", "음식"],
    }
    state["voc_analysis"]["intent_result"] = {
        "primary_category": "기타",
        "reason": "고객이 음식 판매 여부에 대해 반복적으로 질문함",
        "is_repeat_topic": True,
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "의료/시설 문의"


def test_hospital_clinical_keywords_use_medical_care_category():
    state = _base_state()
    state["tenant_industry"] = "hospital"
    state["summary"] = {
        "summary_short": "고객이 진료 검사 결과와 처방 가능 여부를 문의했습니다.",
        "customer_intent": "진료 및 처방 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["진료", "검사", "처방"],
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "의료/진료 문의"


def test_hospital_like_keywords_without_tenant_industry_do_not_fall_to_other():
    category = infer_primary_category(
        "고객이 응급실 주차장과 교통편을 문의했습니다.",
        ["응급실", "주차장", "교통편"],
        tenant_industry=None,
    )

    assert category == "의료/시설 문의"


def test_restaurant_tenant_menu_reservation_breaktime_uses_restaurant_category():
    state = _base_state()
    state["tenant_industry"] = "restaurant"
    state["summary"] = {
        "summary_short": "고객이 메뉴와 예약 가능 여부, 브레이크타임을 문의했습니다.",
        "customer_intent": "메뉴 및 예약 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["메뉴", "예약", "브레이크타임"],
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "메뉴/예약 문의"


def test_government_tenant_welfare_application_uses_admin_category():
    state = _base_state()
    state["context"] = {"tenant": {"industry": "government"}}
    state["summary"] = {
        "summary_short": "고객이 구청 청년복지 지원금 신청 방법을 문의했습니다.",
        "customer_intent": "청년복지 신청 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["구청", "청년복지", "신청", "지원금"],
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "복지/행정 문의"


def test_finance_tenant_contract_keywords_use_finance_category():
    state = _base_state()
    state["tenant"] = {"industry": "finance"}
    state["summary"] = {
        "summary_short": "고객이 보험 계약과 대출 상품을 문의했습니다.",
        "customer_intent": "보험 계약 및 대출 상품 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["보험", "계약", "대출", "상품"],
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "금융/계약 문의"


def test_generic_category_stays_other_when_no_clear_rule_matches():
    state = _base_state()
    state["summary"] = {
        "summary_short": "고객이 기타 내용을 문의했습니다.",
        "customer_intent": "기타 문의",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": [],
    }
    state["voc_analysis"]["intent_result"] = {
        "primary_category": "기타",
        "reason": "명확한 분류 기준 없음",
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "기타"


def test_priority_missing_uses_safe_fallback():
    state = _base_state()

    result = normalize_post_call_result(state)

    assert result["priority_result"]["priority"] in {"low", "medium", "high", "critical"}
    assert result["priority_result"]["priority"] == "medium"
    assert result["voc_analysis"]["priority_result"]["priority"] == "medium"


def test_priority_invalid_angry_uses_high_fallback():
    state = _base_state()
    state["summary"]["customer_emotion"] = "angry"
    state["priority_result"] = {"priority": "urgent"}

    result = normalize_post_call_result(state)

    assert result["priority_result"]["priority"] == "high"


def test_sentiment_missing_uses_customer_emotion():
    state = _base_state()
    state["summary"]["customer_emotion"] = "negative"

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["sentiment_result"]["sentiment"] == "negative"


def test_summary_short_missing_gets_fallback():
    state = _base_state()
    state["summary"] = {
        "customer_intent": "주차 요금 문의",
        "customer_emotion": "neutral",
        "keywords": None,
    }

    result = normalize_post_call_result(state)

    assert result["summary"]["summary_short"] == "주차 요금 문의 내용을 후처리했습니다."
    assert result["summary"]["keywords"] == []


def test_bad_shapes_do_not_raise_and_return_safe_fields():
    state = _base_state()
    state["summary"] = None
    state["voc_analysis"] = "bad-shape"
    state["priority_result"] = None

    result = normalize_post_call_result(state)

    assert result["summary"]["summary_short"]
    assert result["voc_analysis"]["intent_result"]["primary_category"]
    assert result["voc_analysis"]["sentiment_result"]["sentiment"] == "neutral"
    assert result["priority_result"]["priority"] == "medium"


def test_tenant_industry_fallback_can_be_used_from_metadata():
    state = _base_state()
    state["call_metadata"] = {"industry": "finance"}
    state["summary"] = {
        "summary_short": "고객이 보험 상품과 납부 방법을 문의했습니다.",
        "customer_intent": "",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "keywords": ["보험", "납부"],
    }

    result = normalize_post_call_result(state)

    assert result["voc_analysis"]["intent_result"]["primary_category"] == "금융/계약 문의"


def test_infer_primary_category_without_tenant_industry_uses_common_rules():
    category = infer_primary_category(
        "고객이 음식 판매 여부와 이용 방법을 문의했습니다.",
        ["음식", "이용 방법"],
        tenant_industry=None,
    )

    assert category == "제품/서비스 문의"


@pytest.mark.asyncio
async def test_save_result_node_persists_normalized_summary_and_voc(monkeypatch):
    captured: dict = {}

    class FakeSummaryRepo:
        async def save_summary(self, call_id, summary, tenant_id=""):
            captured["summary"] = summary
            captured["summary_tenant_id"] = tenant_id

    class FakeVOCRepo:
        async def save_voc_analysis(
            self,
            call_id,
            voc,
            tenant_id="",
            partial_success=False,
            failed_subagents=None,
        ):
            captured["voc"] = voc
            captured["voc_tenant_id"] = tenant_id

    class FakeActionLogRepo:
        async def save_action_log(self, call_id, actions, tenant_id=None):
            captured["action_tenant_id"] = tenant_id

    class FakeDashboardRepo:
        async def upsert_dashboard(self, call_id, payload):
            captured["dashboard_payload"] = payload

    monkeypatch.setattr(save_node, "_summary_repo", FakeSummaryRepo())
    monkeypatch.setattr(save_node, "_voc_repo", FakeVOCRepo())
    monkeypatch.setattr(save_node, "_action_log_repo", FakeActionLogRepo())
    monkeypatch.setattr(save_node, "_dashboard_repo", FakeDashboardRepo())

    state = _base_state()
    state["voc_analysis"]["intent_result"] = {"reason": "고객이 예약 변경을 문의함"}

    result = await save_node.save_result_node(state)

    assert captured["summary_tenant_id"] == "tenant-a"
    assert captured["voc_tenant_id"] == "tenant-a"
    assert captured["voc"]["intent_result"]["primary_category"] == "예약/일정"
    assert captured["dashboard_payload"]["voc_analysis"]["intent_result"]["primary_category"] == "예약/일정"
    assert result["voc_analysis"]["intent_result"]["primary_category"] == "예약/일정"
