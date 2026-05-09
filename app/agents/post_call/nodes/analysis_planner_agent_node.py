"""Agent 1 — analysis_planner_agent.

통화 transcript 를 분석하여 다음을 생성한다:
  1. analysis_result (summary + voc_analysis + priority_result, 기존 키 호환)
  2. proposed_actions: tenant 카탈로그 안에서 LLM 이 자율 선택한 액션 후보

LLM 호출은 단 1회 — record_analysis + propose_* 들을 한 번의 tool_calls 로 받는다.
실제 외부 액션 실행은 reviewer 통과 후 action_executor 가 담당.

이 노드는 *propose 만* 한다. 절대 외부 호출을 하지 않는다.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo
    _KST = ZoneInfo("Asia/Seoul")
except Exception:  # noqa: BLE001 — Windows 일부 환경에서 tzdata 미설치
    # KST 는 UTC+9 (DST 없음) — 고정 offset 으로 대체. 의미상 동일.
    _KST = timezone(timedelta(hours=9), name="Asia/Seoul")  # type: ignore[assignment]

from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.tools.action_catalog import (
    find_entry,
    get_action_catalog,
    to_openai_tools,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 테스트에서 monkeypatch 로 교체. POST_CALL_LLM_MODE=mock 일 때는 _MockPlannerLLM 사용.
_llm: Any = None

_MAX_TOOL_CALLS = 6
# V3-2: callback 허용 미래 범위 — task_branch_node connector 도 동일 정책
_CALLBACK_MIN_OFFSET = timedelta(minutes=5)
_CALLBACK_MAX_OFFSET = timedelta(days=90)

# ISO 8601 'YYYY-MM-DD HH:MM' — task_branch_node 와 동일 패턴
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}$")

# priority → slack urgency 매핑 (V3-4: SoT 단일화)
_PRIORITY_TO_URGENCY = {
    "low": "info",
    "medium": "warning",
    "high": "critical",
    "critical": "critical",
}


def _kst_now() -> datetime:
    return datetime.now(_KST)


def _today_label() -> str:
    """planner system prompt 에 주입할 KST 현재 시각 라벨."""
    return _kst_now().strftime("%Y-%m-%d %H:%M (Asia/Seoul KST)")


_SYSTEM_PROMPT_TEMPLATE = """당신은 콜센터 통화 분석 + 후속 액션 계획 전문가입니다. [POST_CALL_PLANNER]

[현재 시각] {today}

[작업]
1. 제공된 통화 녹취만을 근거로 분석을 수행하세요. 녹취에 없는 내용을 추측 금지.
2. 분석 결과는 반드시 record_analysis 도구를 한 번 호출하여 기록하세요.
3. 분석 후 필요한 외부 액션이 있으면 propose_* 도구로 후보를 등록하세요.
   액션이 필요 없으면 propose_no_action 을 호출하세요.
4. 도구 호출은 record_analysis + propose_* 합쳐서 최대 6개까지만.

[시각 처리 — 매우 중요]
- propose_schedule_callback.preferred_time 은 반드시 위 [현재 시각] 기준으로 계산한
  *미래* 절대 시각을 'YYYY-MM-DD HH:MM' (KST) 형식으로 채우세요.
- 학습 cutoff 의 과거 날짜를 채우면 안 됩니다 — [현재 시각] 의 연도/월/일을 정확히 사용.
- "내일 오후 3시" → [현재 시각] 의 다음 날짜 + "15:00" 으로 계산.
- transcript 에 시각 표현이 없거나 모호하면 빈 문자열로.

[액션 선택 가이드 — 각 항목은 독립적으로 평가하고 해당하면 모두 호출]

항상 평가해야 할 도구들 (조건 충족 시 한 번씩 호출, 누락 금지):

A. **모든 통화 (잡음/무음 제외)** → propose_create_notion_call_record 호출.
   Notion DB 에 기본 row 1건. 카탈로그에 없으면 Notion 미연결이라 생략.

B. 단순 콜백 요청 → propose_schedule_callback

C. 강한 불만 (angry / negative) + 에스컬레이션 (escalated / abandoned) →
   - propose_send_slack_alert (필수)
   - propose_create_jira_ticket (필수)

D. **angry / negative emotion 또는 priority 가 high / critical** →
   - propose_create_notion_voc_record 추가 (call_record 와 별도)
   - 단순 inquiry / resolved 통화에는 VOC record 호출 금지

E. **angry emotion 또는 priority 가 high / critical** →
   - propose_send_email_supervisor 호출 (supervisor 알림)
   - critical 만이 아닌 high / angry 도 포함

F. 단순 정보 문의 / 해결된 통화 (action_required=false) → propose_no_action

[조합 예시]
- angry + escalated + high : A(notion call) + C(slack+jira) + D(notion voc) + E(email) → 5개 호출
- neutral + resolved + low : A(notion call) + F(no action) → 2개 호출
- neutral + 콜백 요청 : A(notion call) + B(callback) → 2개 호출

카탈로그에 없는 도구는 호출 금지. (없으면 그 액션은 propose 하지 않는다.)

반드시 record_analysis 를 포함하여 도구 호출을 시작하세요. 텍스트 응답만 내면 안 됩니다."""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(today=_today_label())


def _validate_callback_time(raw: str) -> tuple[str, str | None]:
    """preferred_time 을 정규식 + 미래 범위 검증.

    반환: (저장할 값, violation_note | None)
        - 빈 입력 → ("", None) 통과
        - 정규식 불일치 → ("", "schedule_callback_invalid_time_format=...")
        - 파싱 실패 → ("", "schedule_callback_unparseable_time=...")
        - 과거 (현재 -inf ~ 현재 +5분 미만) → ("", "schedule_callback_past_time=...")
        - 90일 초과 미래 → ("", "schedule_callback_too_far_future=...")
        - 정상 미래 → (정규화 값, None)
    """
    raw = (raw or "").strip()
    if not raw:
        return "", None
    if not _ISO_DATETIME_RE.match(raw):
        return "", f"schedule_callback_invalid_time_format={raw!r}"
    try:
        dt_naive = datetime.strptime(raw, "%Y-%m-%d %H:%M")
        dt_kst = dt_naive.replace(tzinfo=_KST)
    except ValueError:
        return "", f"schedule_callback_unparseable_time={raw!r}"
    now = _kst_now()
    if dt_kst < now + _CALLBACK_MIN_OFFSET:
        return "", f"schedule_callback_past_time={raw!r}"
    if dt_kst > now + _CALLBACK_MAX_OFFSET:
        return "", f"schedule_callback_too_far_future={raw!r}"
    return raw, None


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    return "\n".join(f"[{t.get('role','?')}] {t.get('text','')}" for t in transcripts)


def _empty_analysis(reason: str) -> dict:
    return {
        "summary": {
            "summary_short": "통화 내용 없음" if "transcript" in reason else "분석 실패",
            "summary_detailed": reason,
            "customer_intent": "알 수 없음",
            "customer_emotion": "neutral",
            "resolution_status": "resolved",
            "keywords": [],
            "handoff_notes": None,
        },
        "voc_analysis": {
            "sentiment_result": {"sentiment": "neutral", "intensity": 0.0, "reason": reason},
            "intent_result": {
                "primary_category": "알 수 없음",
                "sub_categories": [],
                "is_repeat_topic": False,
                "faq_candidate": False,
            },
            "priority_result": {
                "priority": "low",
                "action_required": False,
                "suggested_action": None,
                "reason": reason,
            },
        },
        "priority_result": {
            "priority": "low",
            "tier": "low",
            "action_required": False,
            "suggested_action": None,
            "reason": reason,
        },
    }


def _record_to_analysis(args: dict) -> dict:
    """record_analysis 인자 → analysis_result dict (기존 스키마)."""
    summary = {
        "summary_short": str(args.get("summary_short") or "")[:200],
        "summary_detailed": str(args.get("summary_detailed") or args.get("summary_short") or "")[:1000],
        "customer_intent": str(args.get("customer_intent") or ""),
        "customer_emotion": args.get("customer_emotion") or "neutral",
        "resolution_status": args.get("resolution_status") or "resolved",
        "keywords": list(args.get("keywords") or [])[:10],
        "handoff_notes": (args.get("handoff_notes") or None) or None,
    }
    if summary["handoff_notes"] == "":
        summary["handoff_notes"] = None

    priority = args.get("priority") or "low"
    action_required = bool(args.get("action_required", False))
    primary_category = args.get("primary_category") or "기타"

    voc = {
        "sentiment_result": {
            "sentiment": summary["customer_emotion"],
            "intensity": 0.5 if summary["customer_emotion"] in ("negative", "angry") else 0.2,
            "reason": "",
        },
        "intent_result": {
            "primary_category": primary_category,
            "sub_categories": [],
            "is_repeat_topic": bool(args.get("is_repeat_topic", False)),
            "faq_candidate": bool(args.get("faq_candidate", False)),
        },
        "priority_result": {
            "priority": priority,
            "action_required": action_required,
            "suggested_action": None,
            "reason": "",
        },
    }
    priority_result = {
        "priority": priority,
        "tier": priority,
        "action_required": action_required,
        "suggested_action": None,
        "reason": "",
    }
    return {
        "summary": summary,
        "voc_analysis": voc,
        "priority_result": priority_result,
    }


def _propose_to_planned_action(
    *,
    catalog_entry: dict,
    args: dict,
    call_id: str,
    tenant_id: str,
    priority: str,
) -> tuple[dict | None, str | None]:
    """propose_* tool_call → PlannedAction-호환 dict.

    반환: (action_dict | None, violation_note | None)
        propose_no_action → (None, None)
        ISO 위반 등 가드 위반 → action 은 만들되 params 에 빈값, violation_note 반환.
        priority 는 단일 source — analysis priority_result.priority 만 사용.
        params 에 priority 자동 주입 안 함 (executor 가 ActionItem.priority 로 조회).
    """
    action_type = catalog_entry.get("action_type")
    tool_name = catalog_entry.get("tool")
    if action_type is None or action_type == "_no_action":
        return None, None

    base_params: dict = {
        "call_id": call_id,
        "tenant_id": tenant_id,
    }

    violation: str | None = None
    name = catalog_entry["name"]
    if name == "propose_send_slack_alert":
        # V3-4: urgency 는 priority 에서 자동 derive — LLM 인자 받지 않음
        derived_urgency = _PRIORITY_TO_URGENCY.get(priority, "warning")
        base_params.update({
            "channel_type": derived_urgency,
            "urgency": derived_urgency,
            "message": args.get("message", ""),
        })
    elif name == "propose_schedule_callback":
        # V3-2: ISO 정규식 + 과거/먼 미래 범위 검증
        preferred, violation = _validate_callback_time(args.get("preferred_time", ""))
        base_params.update({
            "preferred_time": preferred,
            "customer_phone": args.get("phone", ""),
            "callback_reason": args.get("reason", ""),
        })
    elif name == "propose_create_jira_ticket":
        base_params.update({
            "summary": args.get("summary", ""),
            "description": args.get("description", ""),
            "labels": ["sisicallcall", "post-call", priority],
        })
    elif name == "propose_send_sms_followup":
        base_params.update({
            "customer_phone": args.get("phone", ""),
            "message": args.get("message", ""),
        })
    elif name == "propose_send_email_supervisor":
        base_params.update({
            "subject": args.get("subject", ""),
            "body": args.get("body", ""),
        })
    elif name == "propose_create_notion_call_record":
        base_params.update({
            "title": args.get("title", ""),
            "summary_short": args.get("title", ""),  # Notion connector 호환
            "summary": args.get("summary", ""),
            "customer_emotion": args.get("sentiment", "neutral"),
            "priority": args.get("priority", "low"),
        })
    elif name == "propose_create_notion_voc_record":
        base_params.update({
            "title": args.get("title", ""),
            "summary_short": args.get("title", ""),
            "voc_content": args.get("voc_content", ""),
            "summary": args.get("voc_content", ""),
            "customer_emotion": args.get("sentiment", "neutral"),
            "priority": args.get("priority", "low"),
            "suggested_action": args.get("suggested_action", ""),
        })

    return (
        {
            "action_type": action_type,
            "tool": tool_name,
            "priority": priority,  # 단일 source — analysis 의 priority_result.priority
            "params": base_params,
            "status": "pending",
            "proposed_by": "analysis_planner_agent",
        },
        violation,
    )


def _post_call_llm_mode() -> str:
    raw = (os.environ.get("POST_CALL_LLM_MODE") or "").strip().lower()
    if raw in {"mock", "real"}:
        return raw
    legacy = (os.environ.get("POST_CALL_USE_REAL_LLM") or "").strip().lower()
    return "real" if legacy in {"1", "true", "yes", "on"} else "mock"


def _get_llm():
    """real 모드면 GPT4OService, 아니면 _MockPlannerLLM. 테스트는 monkeypatch 로 _llm 직접 교체."""
    global _llm
    if _llm is not None:
        return _llm
    if _post_call_llm_mode() == "real":
        from app.services.llm.gpt4o import GPT4OService
        _llm = GPT4OService()
    else:
        _llm = _MockPlannerLLM()
    return _llm


class _MockPlannerLLM:
    """POST_CALL_LLM_MODE=mock 또는 키 부재 시 사용되는 결정론적 LLM.

    transcript 의 키워드를 보고 angry/critical 시나리오 vs 단순 문의를 구분한다.
    OAuth 카탈로그가 부족하면 부족한대로만 propose 한다.
    """

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tool_choice: str = "required",
        messages: list[dict] | None = None,
    ) -> dict:
        # transcript 라인만 추출 ([role] text 줄). prefix 의 "통화 녹취" 같은 문구가
        # 'Lower 화' 매칭되지 않도록.
        transcript_lines = []
        for line in (user_message or "").splitlines():
            if line.startswith("[customer]") or line.startswith("[agent]"):
                transcript_lines.append(line)
        text = "\n".join(transcript_lines).lower() if transcript_lines else (user_message or "").lower()
        is_angry = any(kw in text for kw in ("짜증", "불만", "최악", "환불", "민원", "어이없", "화나", "화남", "화가"))
        is_critical = any(kw in text for kw in ("위험", "응급", "긴급", "장애", "법적"))

        available_names = {t["function"]["name"] for t in tools}

        emotion = "angry" if is_angry else "neutral"
        priority = "critical" if is_critical else ("high" if is_angry else "low")
        action_required = is_angry or is_critical
        resolution = "escalated" if is_angry or is_critical else "resolved"

        calls: list[dict] = []

        # record_analysis 무조건 포함
        calls.append({
            "id": "call_record",
            "name": "record_analysis",
            "arguments": {
                "summary_short": "[MOCK] 강한 불만 통화" if is_angry else "[MOCK] 일반 문의",
                "summary_detailed": "[MOCK] 결정론적 mock 응답",
                "customer_intent": "불만 처리" if is_angry else "정보 문의",
                "customer_emotion": emotion,
                "resolution_status": resolution,
                "priority": priority,
                "action_required": action_required,
                "primary_category": "민원/불만" if is_angry else "단순 문의",
                "is_repeat_topic": False,
                "faq_candidate": False,
                "keywords": ["불만"] if is_angry else ["문의"],
                "handoff_notes": "팀장 보고 필요" if is_critical else "",
            },
        })

        # Notion call_record: 모든 통화 기본 (카탈로그에 있을 때만)
        if "propose_create_notion_call_record" in available_names:
            calls.append({
                "id": "call_notion_call",
                "name": "propose_create_notion_call_record",
                "arguments": {
                    "title": "[MOCK] 통화 기본 기록",
                    "summary": "[MOCK] 결정론적 mock 응답",
                    "sentiment": emotion,
                    "priority": priority,
                },
            })

        if action_required:
            if "propose_send_slack_alert" in available_names:
                # V3-4: urgency 인자 제거 — planner 가 priority 에서 자동 derive
                calls.append({
                    "id": "call_slack",
                    "name": "propose_send_slack_alert",
                    "arguments": {
                        "message": "[MOCK] 강한 불만 통화 검토 필요",
                    },
                })
            # supervisor email: high / critical priority 또는 angry emotion
            if (is_critical or priority in ("high", "critical") or is_angry) and \
                    "propose_send_email_supervisor" in available_names:
                calls.append({
                    "id": "call_email",
                    "name": "propose_send_email_supervisor",
                    "arguments": {
                        "subject": "[ALERT] 통화 보고",
                        "body": "[MOCK] supervisor 보고 필요",
                    },
                })
            if "propose_create_jira_ticket" in available_names:
                calls.append({
                    "id": "call_jira",
                    "name": "propose_create_jira_ticket",
                    "arguments": {
                        "summary": "[MOCK] 후속 처리 필요",
                        "description": "[MOCK] 자동 생성된 mock 이슈",
                    },
                })
            # Notion voc_record: angry/negative emotion 또는 high+ priority
            if (is_angry or emotion in ("angry", "negative") or priority in ("high", "critical")) and \
                    "propose_create_notion_voc_record" in available_names:
                calls.append({
                    "id": "call_notion_voc",
                    "name": "propose_create_notion_voc_record",
                    "arguments": {
                        "title": "[MOCK] VOC 기록",
                        "voc_content": "[MOCK] 강한 불만 / 후속 처리 필요",
                        "sentiment": emotion,
                        "priority": priority,
                        "suggested_action": "[MOCK] 환불 처리 + 24h 콜백",
                    },
                })
        else:
            calls.append({
                "id": "call_no",
                "name": "propose_no_action",
                "arguments": {"reason": "[MOCK] 단순 문의 — 외부 액션 불필요"},
            })

        return {
            "tool_calls": calls[:_MAX_TOOL_CALLS],
            "text": "",
            "raw_message": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
        }


def _empty_telemetry(latency_ms: int = 0) -> dict:
    """텔레메트리 키 일관성 — usage/tool_counts 가 항상 채워진 dict."""
    return {
        "calls": 0,
        "tokens": {"prompt": 0, "completion": 0, "total": 0, "model": ""},
        "tool_counts": {},
        "latency_ms": latency_ms,
    }


async def analysis_planner_agent_node(state: PostCallAgentState) -> dict:
    call_id: str = state["call_id"]
    tenant_id: str = state["tenant_id"]
    transcripts: list = state.get("transcripts") or []  # type: ignore[call-overload]
    errors: list = list(state.get("errors", []))  # type: ignore[call-overload]

    t0 = time.perf_counter()

    # ── 빈 transcript 안전 처리 ──────────────────────────────────────────────
    if not transcripts:
        logger.warning("analysis_planner: 녹취 없음 call_id=%s — empty fallback", call_id)
        analysis = _empty_analysis("transcripts 없음 — fallback 분석")
        errors.append({
            "node": "analysis_planner_agent",
            "warning": "empty_transcript",
            "error": "transcripts 없음 — fallback 사용",
        })
        return {
            "analysis_result": analysis,
            "summary": analysis["summary"],
            "voc_analysis": analysis["voc_analysis"],
            "priority_result": analysis["priority_result"],
            "proposed_actions": [],
            "analysis_planner_rationale": "empty_transcript",
            "analysis_planner_telemetry": _empty_telemetry(int((time.perf_counter() - t0) * 1000)),
            "errors": errors,
            "partial_success": True,
        }

    catalog = get_action_catalog(tenant_id)
    tools = to_openai_tools(catalog)
    transcripts_text = _format_transcripts(transcripts)
    user_message = f"통화 녹취:\n{transcripts_text}"

    llm = _get_llm()
    last_error: str | None = None
    response: dict | None = None

    # LLM 호출 — 1회 retry
    for attempt in range(2):
        try:
            response = await llm.generate_with_tools(
                system_prompt=_build_system_prompt(),
                user_message=user_message,
                tools=tools,
                temperature=0.0,
                max_tokens=1500,
                tool_choice="required",
            )
            break
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "analysis_planner LLM 실패 attempt=%d call_id=%s err=%s",
                attempt, call_id, exc,
            )

    if response is None:
        logger.error("analysis_planner LLM 두 번째 시도도 실패 call_id=%s err=%s", call_id, last_error)
        errors.append({"node": "analysis_planner_agent", "error": f"llm_failed: {last_error}"})
        analysis = _empty_analysis("LLM 호출 실패 — fallback 분석")
        return {
            "analysis_result": analysis,
            "summary": analysis["summary"],
            "voc_analysis": analysis["voc_analysis"],
            "priority_result": analysis["priority_result"],
            "proposed_actions": [],
            "analysis_planner_rationale": "llm_call_failed",
            "analysis_planner_telemetry": _empty_telemetry(int((time.perf_counter() - t0) * 1000)),
            "errors": errors,
            "partial_success": True,
            "human_review_required": True,
        }

    tool_calls: list[dict] = list(response.get("tool_calls") or [])[:_MAX_TOOL_CALLS]
    usage = response.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": ""}
    if not tool_calls:
        logger.warning("analysis_planner: tool_calls 없음 call_id=%s — fallback", call_id)
        analysis = _empty_analysis("LLM tool_calls 없음 — fallback 분석")
        errors.append({
            "node": "analysis_planner_agent",
            "warning": "no_tool_calls",
            "error": "LLM 이 tool 을 호출하지 않음",
        })
        return {
            "analysis_result": analysis,
            "summary": analysis["summary"],
            "voc_analysis": analysis["voc_analysis"],
            "priority_result": analysis["priority_result"],
            "proposed_actions": [],
            "analysis_planner_rationale": "no_tool_calls",
            "analysis_planner_telemetry": {
                "calls": 1,
                "tokens": {
                    "prompt": int(usage.get("prompt_tokens", 0)),
                    "completion": int(usage.get("completion_tokens", 0)),
                    "total": int(usage.get("total_tokens", 0)),
                    "model": str(usage.get("model", "")),
                },
                "tool_counts": {},
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            },
            "errors": errors,
            "partial_success": True,
            "human_review_required": True,
        }

    # ── tool_calls 처리: record_analysis 추출 + propose_* 매핑 ─────────────────
    # 1차 패스: record_analysis 부터 추출 (priority 가 propose 시 필요)
    analysis: dict | None = None
    propose_calls: list[tuple[str, dict, dict]] = []  # (name, args, catalog_entry)
    rationale_parts: list[str] = []
    no_action_seen = False
    dropped_unknown: list[str] = []
    violations: list[str] = []
    tool_counts: dict[str, int] = {}

    for call in tool_calls:
        name = call.get("name") or ""
        args = call.get("arguments") or {}
        entry = find_entry(catalog, name)
        tool_counts[name] = tool_counts.get(name, 0) + 1

        if name == "record_analysis":
            try:
                analysis = _record_to_analysis(args)
            except Exception as exc:
                logger.error("analysis_planner: record_analysis 파싱 실패 call_id=%s err=%s", call_id, exc)
                errors.append({"node": "analysis_planner_agent", "error": f"record_analysis_parse: {exc}"})
            continue

        if entry is None:
            dropped_unknown.append(name)
            logger.warning(
                "analysis_planner: 카탈로그 외 도구 호출 drop call_id=%s tool=%s",
                call_id, name,
            )
            continue

        if name == "propose_no_action":
            no_action_seen = True
            rationale_parts.append(f"propose_no_action: {args.get('reason', '')}")
            continue

        propose_calls.append((name, args, entry))

    # 2차 패스: 단일 priority source 로 propose 매핑
    proposed: list[dict] = []
    priority = (
        (analysis or {}).get("priority_result", {}).get("priority", "low")
        if analysis else "low"
    )
    for name, args, entry in propose_calls:
        planned, violation = _propose_to_planned_action(
            catalog_entry=entry,
            args=args,
            call_id=call_id,
            tenant_id=tenant_id,
            priority=priority,
        )
        if violation:
            violations.append(violation)
            logger.warning(
                "analysis_planner: violation call_id=%s tool=%s violation=%s",
                call_id, name, violation,
            )
        if planned is not None:
            proposed.append(planned)
            rationale_parts.append(f"{name} → {planned['action_type']}")

    # record_analysis 누락 — fallback 분석으로 채워서 저장은 가능하게
    if analysis is None:
        logger.warning(
            "analysis_planner: record_analysis 누락 call_id=%s — fallback 분석",
            call_id,
        )
        errors.append({
            "node": "analysis_planner_agent",
            "warning": "missing_record_analysis",
            "error": "LLM 이 record_analysis 를 호출하지 않음",
        })
        analysis = _empty_analysis("record_analysis 누락 — fallback")
        # priority 가 low fallback 이므로 propose 했던 액션들의 우선순위 재정렬 안 함

    # propose_no_action + 다른 propose 가 동시에 오면 propose_* 우선
    if no_action_seen and proposed:
        rationale_parts.append("propose_no_action 무시 — 다른 propose 함께 호출됨")

    rationale = "; ".join(rationale_parts) if rationale_parts else "no_actions_proposed"
    if dropped_unknown:
        rationale = f"{rationale}; dropped_unknown={dropped_unknown}"
    if violations:
        rationale = f"{rationale}; violations={violations}"

    latency_ms = int((time.perf_counter() - t0) * 1000)
    telemetry = {
        "calls": 1,
        "tokens": {
            "prompt": int(usage.get("prompt_tokens", 0)),
            "completion": int(usage.get("completion_tokens", 0)),
            "total": int(usage.get("total_tokens", 0)),
            "model": str(usage.get("model", "")),
        },
        "tool_counts": tool_counts,
        "latency_ms": latency_ms,
    }

    logger.info(
        "post_call telemetry node=analysis_planner call_id=%s tenant=%s "
        "tokens=%d latency_ms=%d emotion=%s priority=%s proposed=%d dropped=%d violations=%d",
        call_id, tenant_id,
        telemetry["tokens"]["total"], latency_ms,
        analysis["summary"].get("customer_emotion"),
        analysis["priority_result"].get("priority"),
        len(proposed), len(dropped_unknown), len(violations),
    )

    out: dict = {
        "analysis_result": analysis,
        "summary": analysis["summary"],
        "voc_analysis": analysis["voc_analysis"],
        "priority_result": analysis["priority_result"],
        "proposed_actions": proposed,
        "analysis_planner_rationale": rationale,
        "analysis_planner_telemetry": telemetry,
        "errors": errors,
    }
    # ISO 위반 등 가드 위반은 사람 검토 대기 — reviewer 가 별도 검증 가능하나
    # silent fallback 보다 명시 escalation 이 안전.
    if violations:
        out["human_review_required"] = True
    return out
