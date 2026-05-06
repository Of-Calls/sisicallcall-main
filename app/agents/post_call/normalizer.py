from __future__ import annotations

import copy
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_SENTIMENTS = frozenset({"positive", "neutral", "negative", "angry"})
_VALID_PRIORITIES = frozenset({"low", "medium", "high", "critical"})
_VALID_RESOLUTION_STATUSES = frozenset({"resolved", "escalated", "abandoned"})


def normalize_post_call_result(state: dict) -> dict:
    """Normalize final post-call result fields before persistence.

    This is a deterministic safety net after analysis/review. Existing valid LLM
    output is preserved; only missing, blank, or invalid schema fields are filled.
    """
    try:
        normalized = copy.deepcopy(state) if isinstance(state, dict) else {}
        call_id = _clean_str(normalized.get("call_id"))
        context = _build_context(normalized)

        raw_summary = _coalesce_dict(
            normalized.get("summary"),
            _nested_dict(normalized.get("analysis_result"), "summary"),
        )
        raw_voc = _coalesce_dict(
            normalized.get("voc_analysis"),
            _nested_dict(normalized.get("analysis_result"), "voc_analysis"),
        )
        raw_priority = _coalesce_dict(
            normalized.get("priority_result"),
            raw_voc.get("priority_result"),
            _nested_dict(normalized.get("analysis_result"), "priority_result"),
        )

        summary_hint = normalize_summary(raw_summary, context, emit_logs=False)
        intent_result = normalize_intent_result(
            _as_dict(raw_voc.get("intent_result")),
            {**context, "summary": summary_hint},
            call_id=call_id,
        )
        priority_result = normalize_priority_result(
            raw_priority,
            {**context, "summary": summary_hint, "intent_result": intent_result},
            call_id=call_id,
        )
        sentiment_result = normalize_sentiment_result(
            _coalesce_dict(raw_voc.get("sentiment_result"), {"sentiment": raw_voc.get("sentiment")}),
            {**context, "summary": summary_hint},
            call_id=call_id,
        )
        summary = normalize_summary(
            raw_summary,
            {**context, "intent_result": intent_result},
            call_id=call_id,
        )

        voc_analysis = dict(raw_voc)
        voc_analysis["sentiment_result"] = sentiment_result
        voc_analysis["intent_result"] = intent_result
        voc_analysis["priority_result"] = priority_result

        normalized["summary"] = summary
        normalized["voc_analysis"] = voc_analysis
        normalized["priority_result"] = priority_result

        analysis_result = _as_dict(normalized.get("analysis_result"))
        if analysis_result:
            analysis_result["summary"] = summary
            analysis_result["voc_analysis"] = voc_analysis
            analysis_result["priority_result"] = priority_result
            normalized["analysis_result"] = analysis_result

        return normalized
    except Exception as exc:
        logger.warning("post_call_normalizer: failed err=%s", exc)
        return copy.deepcopy(state) if isinstance(state, dict) else {}


def normalize_summary(
    summary: dict,
    fallback_context: dict,
    call_id: str = "",
    emit_logs: bool = True,
) -> dict:
    result = _as_dict(summary)

    intent_result = _as_dict(fallback_context.get("intent_result"))
    category = _clean_str(intent_result.get("primary_category"))

    summary_short = _clean_str(result.get("summary_short") or result.get("short"))
    customer_intent = _clean_str(result.get("customer_intent") or result.get("intent"))

    if not summary_short:
        summary_short = (
            f"{customer_intent} 내용을 후처리했습니다."
            if customer_intent
            else "고객 문의 내용을 후처리했습니다."
        )
        if emit_logs:
            logger.warning(
                "post_call_normalizer: summary_short fallback call_id=%s value=%s",
                call_id,
                summary_short,
            )

    if not customer_intent:
        customer_intent = category or summary_short
        if emit_logs:
            logger.warning(
                "post_call_normalizer: customer_intent fallback call_id=%s value=%s",
                call_id,
                customer_intent,
            )

    emotion = _clean_str(result.get("customer_emotion") or result.get("emotion"))
    if emotion not in _VALID_SENTIMENTS:
        emotion = "neutral"

    resolution_status = _clean_str(result.get("resolution_status") or result.get("status"))
    if resolution_status not in _VALID_RESOLUTION_STATUSES:
        resolution_status = "resolved"

    keywords = _as_list(result.get("keywords"))

    result["summary_short"] = summary_short
    result["summary_detailed"] = result.get("summary_detailed") or result.get("detailed")
    result["customer_intent"] = customer_intent
    result["customer_emotion"] = emotion
    result["resolution_status"] = resolution_status
    result["keywords"] = keywords
    result.setdefault("handoff_notes", None)
    result.setdefault("generation_mode", "async")
    result.setdefault("model_used", "demo-mock-llm")
    return result


def normalize_voc_analysis(voc_analysis: dict, fallback_context: dict) -> dict:
    result = _as_dict(voc_analysis)
    call_id = _clean_str(fallback_context.get("call_id"))
    result["intent_result"] = normalize_intent_result(
        _as_dict(result.get("intent_result")),
        fallback_context,
        call_id=call_id,
    )
    result["priority_result"] = normalize_priority_result(
        _as_dict(result.get("priority_result")),
        {**fallback_context, "intent_result": result["intent_result"]},
        call_id=call_id,
    )
    result["sentiment_result"] = normalize_sentiment_result(
        _coalesce_dict(result.get("sentiment_result"), {"sentiment": result.get("sentiment")}),
        fallback_context,
        call_id=call_id,
    )
    return result


def normalize_sentiment_result(
    sentiment_result: dict,
    fallback_context: dict,
    call_id: str = "",
) -> dict:
    result = _as_dict(sentiment_result)
    sentiment = _clean_str(result.get("sentiment"))
    if sentiment not in _VALID_SENTIMENTS:
        summary = _as_dict(fallback_context.get("summary"))
        fallback = _clean_str(summary.get("customer_emotion"))
        sentiment = fallback if fallback in _VALID_SENTIMENTS else "neutral"
        logger.warning(
            "post_call_normalizer: sentiment fallback call_id=%s value=%s reason=missing_or_invalid_sentiment",
            call_id,
            sentiment,
        )
    result["sentiment"] = sentiment
    result.setdefault("intensity", 0.0)
    result.setdefault("reason", "")
    return result


def normalize_intent_result(
    intent_result: dict,
    fallback_context: dict,
    call_id: str = "",
) -> dict:
    result = _as_dict(intent_result)
    primary_category = _clean_str(result.get("primary_category"))
    if not primary_category or primary_category == "기타":
        summary = _as_dict(fallback_context.get("summary"))
        inferred_category = infer_primary_category(
            text=" ".join(
                part
                for part in (
                    _clean_str(summary.get("summary_short")),
                    _clean_str(summary.get("customer_intent")),
                    _clean_str(result.get("reason")),
                )
                if part
            ),
            keywords=_as_list(summary.get("keywords")),
            tenant_industry=_clean_str(fallback_context.get("tenant_industry")) or None,
        )
        if not primary_category or inferred_category != "기타":
            reason = "missing_primary_category" if not primary_category else "generic_primary_category"
            primary_category = inferred_category
            logger.warning(
                "post_call_normalizer: primary_category filled call_id=%s value=%s reason=%s",
                call_id,
                primary_category,
                reason,
            )
    result["primary_category"] = primary_category
    if not isinstance(result.get("sub_categories"), list):
        result["sub_categories"] = []
    result["is_repeat_topic"] = bool(result.get("is_repeat_topic", False))
    result["faq_candidate"] = bool(result.get("faq_candidate", False))
    return result


def normalize_priority_result(
    priority_result: dict,
    fallback_context: dict,
    call_id: str = "",
) -> dict:
    result = _as_dict(priority_result)
    priority = _clean_str(result.get("priority") or result.get("tier"))
    if priority not in _VALID_PRIORITIES:
        priority = _infer_priority(result, fallback_context)
        logger.warning(
            "post_call_normalizer: priority fallback call_id=%s value=%s reason=missing_or_invalid_priority",
            call_id,
            priority,
        )
    result["priority"] = priority
    result["tier"] = priority
    result["action_required"] = bool(result.get("action_required", False))
    result.setdefault("suggested_action", None)
    result.setdefault("reason", "")
    return result


def infer_primary_category(
    text: str,
    keywords: list[str],
    tenant_industry: str | None = None,
) -> str:
    haystack = _normalize_text(" ".join([text, *[str(k) for k in keywords]]))
    industry = _normalize_text(tenant_industry or "")

    industry_category = _infer_industry_category(haystack, industry)
    if industry_category:
        return industry_category

    hospital_like = _infer_hospital_like_category(haystack, industry_known=False)
    if hospital_like:
        return hospital_like
    restaurant_like = _infer_restaurant_like_category(haystack, industry_known=False)
    if restaurant_like:
        return restaurant_like
    government_like = _infer_government_like_category(haystack, industry_known=False)
    if government_like:
        return government_like
    finance_like = _infer_finance_like_category(haystack, industry_known=False)
    if finance_like:
        return finance_like

    if _has_any(haystack, ("예약", "일정", "변경", "취소", "예약방법", "예약 방법")):
        return "예약/일정"
    if _has_any(haystack, ("환불", "결제", "금액", "요금 환불")):
        return "환불/결제"
    if _has_any(haystack, ("불만", "민원", "화남", "짜증", "항의", "지연", "반복 문의")):
        return "민원/불만"
    if _has_any(haystack, ("운영시간", "영업시간", "위치", "주소", "주차", "주차장", "대중교통", "교통편", "브레이크타임")):
        return "운영시간/위치"
    if _has_any(haystack, ("메뉴", "음식", "제품", "서비스", "판매 여부", "이용 방법", "신청 방법", "준비물")):
        return "제품/서비스 문의"
    if _has_any(haystack, ("상담원", "담당자", "연결", "콜백", "전화 주세요")):
        return "상담원 연결"
    return "기타"


def _infer_industry_category(haystack: str, industry: str) -> str | None:
    if industry == "hospital":
        return _infer_hospital_like_category(haystack, industry_known=True)
    if industry == "restaurant":
        return _infer_restaurant_like_category(haystack, industry_known=True)
    if industry == "government":
        return _infer_government_like_category(haystack, industry_known=True)
    if industry == "finance":
        return _infer_finance_like_category(haystack, industry_known=True)
    return None


def _infer_hospital_like_category(haystack: str, *, industry_known: bool) -> str | None:
    clinical_terms = (
        "응급실",
        "진료",
        "병원",
        "의료",
        "외래",
        "입원",
        "퇴원",
        "간호",
        "의사",
        "처방",
        "검사",
    )
    facility_terms = (
        "주차",
        "주차장",
        "주차요금",
        "교통편",
        "대중교통",
        "면회",
        "보호자",
        "음식 판매",
        "음식",
        "편의시설",
        "매점",
        "식당",
    )
    has_clinical = _has_any(haystack, clinical_terms)
    has_facility = _has_any(haystack, facility_terms)
    if has_facility and (industry_known or has_clinical or _has_any(haystack, ("병원", "의료", "응급실"))):
        return "의료/시설 문의"
    if has_clinical:
        return "의료/진료 문의"
    return None


def _infer_restaurant_like_category(haystack: str, *, industry_known: bool) -> str | None:
    menu_terms = ("메뉴", "음식", "제철메뉴", "식당")
    reservation_terms = ("예약", "좌석", "단체")
    location_terms = ("브레이크타임", "영업시간", "위치", "주차")
    has_restaurant_context = industry_known or _has_any(
        haystack,
        ("식당", "제철메뉴", "포장", "배달", "브레이크타임", "좌석", "단체"),
    )
    if not has_restaurant_context:
        return None
    if _has_any(haystack, reservation_terms) and _has_any(haystack, menu_terms):
        return "메뉴/예약 문의"
    if _has_any(haystack, menu_terms):
        return "메뉴/예약 문의"
    if _has_any(haystack, reservation_terms):
        return "예약/일정"
    if _has_any(haystack, location_terms):
        return "운영시간/위치"
    return None


def _infer_government_like_category(haystack: str, *, industry_known: bool) -> str | None:
    admin_terms = ("구청", "청년복지", "복지", "신청", "서류", "지원금", "행정", "접수", "증명서", "주민센터")
    complaint_terms = ("민원", "불만", "항의")
    if _has_any(haystack, complaint_terms):
        return "민원/불만"
    if industry_known or _has_any(haystack, ("구청", "청년복지", "복지", "지원금", "주민센터")):
        if _has_any(haystack, admin_terms):
            return "복지/행정 문의"
    return None


def _infer_finance_like_category(haystack: str, *, industry_known: bool) -> str | None:
    contract_terms = ("보험", "대출", "금융", "상품", "계약", "청구", "계좌")
    payment_terms = ("납부", "결제", "환불")
    identity_terms = ("본인확인", "본인 확인", "인증")
    if _has_any(haystack, identity_terms):
        return "본인확인"
    if industry_known or _has_any(haystack, contract_terms):
        if _has_any(haystack, contract_terms):
            return "금융/계약 문의"
    if _has_any(haystack, payment_terms):
        return "환불/결제"
    return None


def _infer_priority(priority_result: dict, fallback_context: dict) -> str:
    summary = _as_dict(fallback_context.get("summary"))
    intent_result = _as_dict(fallback_context.get("intent_result"))
    text = _normalize_text(
        " ".join(
            str(part)
            for part in (
                summary.get("summary_short"),
                summary.get("customer_intent"),
                summary.get("keywords"),
                priority_result.get("reason"),
                intent_result.get("primary_category"),
                intent_result.get("reason"),
            )
            if part
        )
    )
    emotion = _clean_str(summary.get("customer_emotion"))
    status = _clean_str(summary.get("resolution_status"))
    action_required = bool(priority_result.get("action_required"))

    if _has_any(text, ("위험", "응급", "장애", "긴급")):
        return "critical"
    if emotion == "angry":
        return "high"
    if status == "escalated" and _has_any(text, ("민원", "불만", "환불 지연", "지연")):
        return "high"
    if action_required or status == "escalated":
        return "medium"
    return "medium"


def _build_context(state: dict) -> dict:
    metadata = _as_dict(state.get("call_metadata"))
    state_tenant = _as_dict(state.get("tenant"))
    call_context = _as_dict(state.get("call_context"))
    call_context_tenant = _as_dict(call_context.get("tenant"))
    state_metadata = _as_dict(state.get("metadata"))
    context = _as_dict(state.get("context"))
    context_tenant = _as_dict(context.get("tenant"))
    tenant = _as_dict(metadata.get("tenant"))
    return {
        "call_id": _clean_str(state.get("call_id")),
        "tenant_id": _clean_str(state.get("tenant_id") or metadata.get("tenant_id")),
        "tenant_name": _clean_str(
            state.get("tenant_name")
            or metadata.get("tenant_name")
            or state_metadata.get("tenant_name")
            or state_tenant.get("name")
            or call_context_tenant.get("name")
            or context_tenant.get("name")
            or tenant.get("name")
        ),
        "tenant_industry": _clean_str(
            state.get("tenant_industry")
            or metadata.get("tenant_industry")
            or metadata.get("industry")
            or state_metadata.get("tenant_industry")
            or state_metadata.get("industry")
            or call_context.get("tenant_industry")
            or call_context.get("industry")
            or context.get("tenant_industry")
            or context.get("industry")
            or state_tenant.get("industry")
            or call_context_tenant.get("industry")
            or context_tenant.get("industry")
            or tenant.get("industry")
        ),
    }


def _coalesce_dict(*values: Any) -> dict:
    for value in values:
        if isinstance(value, dict) and value:
            return copy.deepcopy(value)
    return {}


def _nested_dict(value: Any, key: str) -> dict:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return copy.deepcopy(value[key])
    return {}


def _as_dict(value: Any) -> dict:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_text(value: str) -> str:
    return _clean_str(value).lower()


def _has_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in haystack for needle in needles)
