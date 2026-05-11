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
    is_notion_ready,
    to_openai_tools,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 테스트에서 monkeypatch 로 교체. POST_CALL_LLM_MODE=mock 일 때는 _MockPlannerLLM 사용.
_llm: Any = None

_MAX_TOOL_CALLS = 8  # 다중 의도 통화 + retry feedback 케이스 대응 (이전 6 → 8)
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
4. 도구 호출은 record_analysis + propose_* 합쳐서 최대 8개까지만.

(참고: Notion 통화 기록 / VOC 기록은 별도 자동 액션으로 시스템이 항상 처리한다 —
 propose 대상이 아니다. 카탈로그에서 빠져 있어도 정상.)

[시각 처리 — 매우 중요]
- propose_schedule_callback.preferred_time 은 반드시 위 [현재 시각] 기준으로 계산한
  *미래* 절대 시각을 'YYYY-MM-DD HH:MM' (KST) 형식으로 채우세요.
- 학습 cutoff 의 과거 날짜를 채우면 안 됩니다 — [현재 시각] 의 연도/월/일을 정확히 사용.
- "내일 오후 3시" → [현재 시각] 의 다음 날짜 + "15:00" 으로 계산.
- transcript 에 시각 표현이 없거나 모호하면 빈 문자열로.

[다중 의도 — 매우 중요]
한 통화에 여러 의도가 동시에 있을 수 있다 — 단, 의도가 **명확히 다른 action_type 으로
분리되는 경우**만 다중 호출.
예: "환불 요청(불만) + 내일 오후 3시 콜백 + 본인인증 필요" →
    propose_send_email_supervisor + propose_schedule_callback + propose_create_jira_ticket
    (서로 다른 action_type — 세 개 모두 호출 OK)

[같은 action_type 다중 호출 — 강한 제한]
- 안내성 액션 (propose_send_sms_followup / propose_send_slack_alert /
  propose_send_email_supervisor) 은 **한 통화당 1회로 통합**하라.
- 같은 의도를 다른 표현으로 여러 번 propose 하는 것은 명백한 중복. 금지.
- 같은 action_type 을 2번 이상 호출하려면, **의도가 본질적으로 다르고**
  (예: 별개 VOC 사안 — 배송 누락 + 가격 오안내) 핵심 params 의 식별 필드가
  다른 경우에만. 안내 메시지 본문이 다르다는 이유만으로는 부족하다.
- 시스템이 안내성 액션을 (call_id, action_type) 단위로 중복 차단하므로
  본 가이드를 어기면 두 번째 호출은 어차피 skip 된다 — 토큰 낭비 금지.

[BAD example — 절대 하지 말 것]
환불 불만 통화 → propose_send_sms_followup × 7
  message="환불 요청이 접수되었습니다 …"
  message="불만 사항이 접수되었습니다 …"
  message="환불 요청이 상부에 보고되었습니다 …"
  ... (의도는 모두 "VOC 접수 안내" 1개. 표현만 다름 → 1건으로 통합해야 함)

[GOOD example]
배송 누락 + 가격 오안내 2건 VOC → propose_create_jira_ticket × 2
  summary="배송 누락 — 품목 2건" / description="..."
  summary="결제 금액 오안내 — 광고 5만원 vs 청구 7만원" / description="..."
  (summary 가 분명히 다른 사안이고 한 티켓에 묶지 말라고 고객이 명시 요청)

[액션 선택 가이드 — 각 항목은 독립적으로 평가하고 해당하면 모두 호출]

A. 단순 콜백 요청 → propose_schedule_callback

B. 강한 불만 (angry / negative) + 에스컬레이션 (escalated / abandoned) →
   - propose_send_slack_alert (필수)
   - propose_create_jira_ticket (필수)

C. **angry emotion 또는 priority 가 high / critical** →
   - propose_send_email_supervisor 호출 (supervisor 알림)
   - critical 만이 아닌 high / angry 도 포함

D. 단순 정보 문의 / 해결된 통화 (action_required=false) → propose_no_action

[조합 예시]
- angry + escalated + high : B(slack+jira) + C(email) → 3개 호출
- neutral + 콜백 요청 : A(callback) → 1개 호출
- 다중 의도 (환불+콜백+인증) : A + B + C 모두 + 추가 jira → 5개+ 호출
  (Notion 기록은 자동 처리됨 — 호출 불필요)

카탈로그에 없는 도구는 호출 금지. (없으면 그 액션은 propose 하지 않는다.)

반드시 record_analysis 를 포함하여 도구 호출을 시작하세요. 텍스트 응답만 내면 안 됩니다."""


def _build_system_prompt(review_feedback: list[str] | None = None) -> str:
    base = _SYSTEM_PROMPT_TEMPLATE.format(today=_today_label())
    feedback = [s for s in (review_feedback or []) if s and str(s).strip()]
    if not feedback:
        return base
    bullets = "\n".join(f"- {s}" for s in feedback)
    retry_block = (
        "\n\n[이전 분석 검토 결과 — 다음 문제가 지적되었다]\n"
        f"{bullets}\n"
        "위 지적사항을 반영해서 분석을 다시 작성하라. "
        "같은 실수 반복 시 인간 검토 큐로 빠진다."
    )
    return base + retry_block


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
    # NOTE: Notion (call_record / voc_record) 는 자동 주입 액션이라 카탈로그에서 제거됨.
    # _inject_mandatory_actions() 가 별도 처리.

    action = {
        "action_type": action_type,
        "tool": tool_name,
        "priority": priority,  # 단일 source — analysis 의 priority_result.priority
        "params": base_params,
        "status": "pending",
        "proposed_by": "analysis_planner_agent",
    }
    action["idempotency_token"] = _compute_idempotency_token(action)
    return action, violation


# ── Idempotency token — 동일 의도 두 번째 호출 차단 / 다른 의도는 별개 ─────
import hashlib  # noqa: E402
import json as _json  # noqa: E402

_IDEMPOTENCY_FIELDS = {
    # 안내성 통합 — 한 통화당 1건만 발송. LLM 이 message 표현만 바꿔 N번 propose 해도
    # 모두 같은 token 으로 산출되어 executor 가 첫 시도 후 skip.
    # 같은 통화에서 "의도가 정말 다른 SMS/Slack 2건" 은 1건으로 차단됨 (안전 우선 trade-off).
    "send_voc_receipt_sms": [],
    "send_slack_alert": [],
    "send_manager_email": [],
    # 의도가 다르면 자연스럽게 분리되는 필드만 포함
    "create_jira_issue": ["summary"],
    "schedule_callback": ["preferred_time"],
    "send_callback_sms": ["customer_phone"],
    "create_voc_issue": ["voc_content"],
    # auto_injected 는 sub_intent 마커로 처리 (아래 분기)
    "create_notion_call_record": [],
    "create_notion_voc_record": ["voc_content"],
}


def _compute_idempotency_token(action: dict) -> str:
    """결정론적 idempotency token. 같은 의도 → 같은 token, 다른 의도 → 다른 token.

    auto_injected 액션은 (sub_intent 마커 + 통화당 1건) 로 단순화.
    LLM-proposed 는 action_type 별 핵심 field hash.
    """
    params = action.get("params") or {}
    if params.get("auto_injected"):
        sub = str(params.get("sub_intent") or "auto").strip() or "auto"
        return f"auto:{sub}"
    action_type = action.get("action_type") or ""
    fields = _IDEMPOTENCY_FIELDS.get(action_type, [])
    payload = {k: params.get(k) for k in fields}
    raw = _json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ── 자동 주입 액션 (Notion 회사 DB 기록) ──────────────────────────────────────
# Notion DB 를 회사 DB 로 가정. 모든 통화 → call_record. angry+medium+ → voc_record.
# LLM 자율 판단 대상 아님. retry 사이클에서 매번 재실행되지만 idempotency_token 으로
# executor 가 첫 시도 후 skip. fail (max retry 초과) 시에도 auto_action_executor 가 발송.

def _compute_duration_sec(metadata: dict) -> int | None:
    """metadata.start_time / end_time (ISO) → duration_sec. 실패 시 None."""
    s = str(metadata.get("start_time") or "").strip()
    e = str(metadata.get("end_time") or "").strip()
    if not s or not e:
        return None
    try:
        s_dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        e_dt = datetime.fromisoformat(e.replace("Z", "+00:00"))
        delta = (e_dt - s_dt).total_seconds()
        return int(delta) if delta >= 0 else None
    except Exception:
        return None


def _serialize_transcripts(transcripts: list[dict]) -> list[dict]:
    """state.transcripts → call_record params.transcript_full 형식.

    [{"turn": 0, "speaker": "customer", "text": "..."},
     {"turn": 1, "speaker": "agent",    "text": "..."}, ...]
    """
    out: list[dict] = []
    for i, t in enumerate(transcripts or []):
        speaker = t.get("role") or t.get("speaker") or "unknown"
        out.append({
            "turn": i,
            "speaker": str(speaker),
            "text": str(t.get("text") or ""),
        })
    return out


def _inject_mandatory_actions(
    *,
    tenant_id: str,
    call_id: str,
    analysis: dict,
    priority: str,
    proposed: list[dict],
    transcripts: list[dict] | None = None,
    call_metadata: dict | None = None,
    branch_stats: dict | None = None,
) -> list[dict]:
    """카탈로그 외부의 자동 액션을 proposed_actions 끝에 append.

    Notion 미연결 (token/db id 없음) → 자동 액션 0건. graceful skip.

    call_record: 통화 보관소 — 원본 transcript_full + 메타 (LLM 가공 최소화).
    voc_record:  분석 인사이트 — LLM 요약/분석 (reviewer 검증 통과한 것).
    두 record 는 Notion DB 의 'Record Type' 컬럼으로 구분 (call/voc).
    """
    if not is_notion_ready():
        return proposed

    summary_dict = (analysis or {}).get("summary") or {}
    summary_short = str(summary_dict.get("summary_short") or "")[:80]
    summary_detailed = str(
        summary_dict.get("summary_detailed") or summary_short or ""
    )
    sentiment = str(summary_dict.get("customer_emotion") or "neutral")
    handoff_notes = str(summary_dict.get("handoff_notes") or "")

    metadata = call_metadata or {}
    out = list(proposed)

    # 1. call_record — 모든 통화 무조건 (idempotency: auto:auto_call_record)
    # LLM 가공 필드 (summary / customer_emotion / priority) 제거.
    # 원본 transcript_full + 메타데이터만 보관 — Notion page body 에 turn-by-turn.
    call_record = {
        "action_type": "create_notion_call_record",
        "tool": "notion",
        "priority": priority,
        "params": {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "record_type": "call_record",
            "caller_number": str(metadata.get("customer_phone") or ""),
            "started_at": str(metadata.get("start_time") or ""),
            "ended_at": str(metadata.get("end_time") or ""),
            "duration_sec": _compute_duration_sec(metadata),
            "transcript_full": _serialize_transcripts(transcripts or []),
            "branch_stats": dict(branch_stats or {}),
            "auto_injected": True,
            "sub_intent": "auto_call_record",
        },
        "status": "pending",
        "proposed_by": "auto_inject",
    }
    call_record["idempotency_token"] = _compute_idempotency_token(call_record)
    out.append(call_record)

    # 2. voc_record — 조건부 (angry/negative + medium+ priority).
    # LLM 분석 인사이트 — summary / sentiment / priority / suggested_action.
    if sentiment in ("angry", "negative") and priority in ("medium", "high", "critical"):
        voc_record = {
            "action_type": "create_notion_voc_record",
            "tool": "notion",
            "priority": priority,
            "params": {
                "call_id": call_id,
                "tenant_id": tenant_id,
                "record_type": "voc_record",
                "title": summary_short or "VOC",
                "summary_short": summary_short or "VOC",
                "voc_content": summary_detailed,
                "summary": summary_detailed,
                "customer_emotion": sentiment,
                "priority": priority,
                "suggested_action": handoff_notes,
                "auto_injected": True,
                "sub_intent": "auto_voc_record",
            },
            "status": "pending",
            "proposed_by": "auto_inject",
        }
        voc_record["idempotency_token"] = _compute_idempotency_token(voc_record)
        out.append(voc_record)

    return out


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

        # NOTE: Notion (call_record / voc_record) 는 자동 주입 액션이라 mock 에서도 호출 안 함.
        #       _inject_mandatory_actions() 가 catalog 외부에서 추가.

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

    review_feedback: list[str] = list(state.get("review_feedback") or [])  # type: ignore[call-overload]
    retry_count = int(state.get("analysis_retry_count") or 0)  # type: ignore[call-overload]
    system_prompt = _build_system_prompt(review_feedback)
    if retry_count > 0:
        logger.info(
            "analysis_planner: 재시도 호출 call_id=%s retry_count=%d feedback_items=%d",
            call_id, retry_count, len(review_feedback),
        )

    llm = _get_llm()
    last_error: str | None = None
    response: dict | None = None

    # LLM 호출 — 1회 retry
    for attempt in range(2):
        try:
            response = await llm.generate_with_tools(
                system_prompt=system_prompt,
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

    # ── 자동 액션 주입 (Notion 회사 DB 기록 — LLM 자율 판단 대상 아님) ────────
    proposed = _inject_mandatory_actions(
        tenant_id=tenant_id,
        call_id=call_id,
        analysis=analysis,
        priority=priority,
        proposed=proposed,
        transcripts=transcripts,
        call_metadata=state.get("call_metadata") or {},  # type: ignore[call-overload]
        branch_stats=state.get("branch_stats") or {},  # type: ignore[call-overload]
    )
    auto_injected_count = sum(
        1 for a in proposed if (a.get("params") or {}).get("auto_injected")
    )
    if auto_injected_count:
        rationale_parts.append(f"auto_injected={auto_injected_count}")

    # propose_no_action + 다른 propose 가 동시에 오면 propose_* 우선.
    # auto_injected 도 propose 로 카운트한다 (Notion 자동 액션 = '액션 있음').
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
        "retry_count": retry_count,
        "auto_injected_count": auto_injected_count,
    }

    logger.info(
        "post_call telemetry node=analysis_planner call_id=%s tenant=%s "
        "tokens=%d latency_ms=%d emotion=%s priority=%s proposed=%d dropped=%d violations=%d retry_count=%d",
        call_id, tenant_id,
        telemetry["tokens"]["total"], latency_ms,
        analysis["summary"].get("customer_emotion"),
        analysis["priority_result"].get("priority"),
        len(proposed), len(dropped_unknown), len(violations),
        retry_count,
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
