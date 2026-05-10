"""reviewer 가 max_retries 까지 fail 시 관리자에게 긴급 알림 발송 + Notion 마킹.

이 노드의 책임:

1. 관리자 Slack 채널에 긴급 알림 (urgency=critical)
2. 관리자 Gmail 에 긴급 알림 메일
3. state["proposed_actions"] 의 Notion auto_injected 액션 params 를 mutate —
   분석 본문이 신뢰 불가임을 Notion row 에 명시:
     params["title"]    : "[REVIEW_FAILED] " prefix
     params["summary"]  : "[REVIEW_FAILED] " prefix
     params["customer_emotion"] = "unknown"
     params["priority"]         = "unknown"

생성된 알림 PlannedAction 2건은 auto_injected=true + 결정론적 idempotency_token
("auto:admin_alert_review_failed_slack" / "_email") 으로 ActionExecutor 에 위임.
같은 call_id 재실행 시 mcp_action_logs 의 success row 를 보고 skip.

이 노드는 save_reviewed_analysis 를 거치지 않으므로 (분석 본문은 비신뢰),
call_summaries / voc_analyses 에 row 가 들어가지 않는다 — 사용자 요구사항 그대로.
mcp_action_logs 에는 admin_alert + Notion call_record 가 들어감.

다음 노드: auto_action_executor → save_final → END.
"""
from __future__ import annotations

import copy

from app.agents.post_call.nodes.analysis_planner_agent_node import (
    _compute_idempotency_token,
)
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)

_REVIEW_FAILED_MARKER = "[REVIEW_FAILED] "
_NOTION_AUTO_TYPES = frozenset({
    "create_notion_call_record",
    "create_notion_voc_record",
})


def _mask_phone(phone: str) -> str:
    s = (phone or "").strip()
    if len(s) < 4:
        return "***"
    return f"{s[:3]}-****-{s[-4:]}"


def _format_admin_message_body(
    *,
    call_id: str,
    tenant_id: str,
    review_feedback: list[str],
    retry_count: int,
    customer_phone: str | None,
    transcripts: list[dict],
) -> str:
    """[긴급 검토 필요] 본문 — Slack/Gmail 공용 텍스트."""
    lines: list[str] = [
        "[긴급 검토 필요] 통화 분석 검토 실패",
        "",
        f"call_id: {call_id}",
        f"tenant_id: {tenant_id}",
        f"고객 번호: {_mask_phone(customer_phone or '')}",
        f"retry 시도 횟수: {retry_count}",
        "",
        "검토 실패 사유:",
    ]
    for fb in (review_feedback or [])[:6]:
        lines.append(f"  - {fb}")
    if not review_feedback:
        lines.append("  - (사유 미기록)")
    lines.append("")
    lines.append(f"녹취 길이: {len(transcripts)} turns")
    lines.append("분석 결과는 신뢰할 수 없습니다 — transcript 를 직접 검토해주세요.")
    return "\n".join(lines)


def _mutate_notion_actions_in_place(
    proposed_actions: list[dict],
) -> int:
    """state["proposed_actions"] 의 Notion auto_injected 액션 params 를 in-place mutate.

    반환: 실제로 수정된 액션 개수.
    """
    n_mutated = 0
    for action in proposed_actions:
        params = action.get("params") or {}
        if not params.get("auto_injected"):
            continue
        if action.get("action_type") not in _NOTION_AUTO_TYPES:
            continue

        title = str(params.get("title") or "")
        summary = str(params.get("summary") or "")
        if not title.startswith(_REVIEW_FAILED_MARKER):
            params["title"] = f"{_REVIEW_FAILED_MARKER}{title}".strip()
        if not summary.startswith(_REVIEW_FAILED_MARKER):
            params["summary"] = f"{_REVIEW_FAILED_MARKER}{summary}".strip()
        # voc_record 는 voc_content 도 별도 필드 — 함께 마킹
        if "voc_content" in params:
            voc_content = str(params.get("voc_content") or "")
            if not voc_content.startswith(_REVIEW_FAILED_MARKER):
                params["voc_content"] = f"{_REVIEW_FAILED_MARKER}{voc_content}".strip()
        params["customer_emotion"] = "unknown"
        params["priority"] = "unknown"  # low fallback 차단 (상위 fan-out 에서 unknown 처리 가능)
        action["params"] = params
        n_mutated += 1
    return n_mutated


def _build_admin_alert_actions(
    *,
    call_id: str,
    tenant_id: str,
    body: str,
) -> list[dict]:
    """Slack + Gmail 두 PlannedAction 을 결정론적 token 과 함께 생성.

    auto_injected=True 마커로 reviewer 우회는 무관 (이미 reviewer 후 단계).
    같은 sub_intent 면 같은 token → mcp_action_logs 에서 skip 가능.
    """
    slack_action = {
        "action_type": "send_slack_alert",
        "tool": "slack",
        "priority": "critical",
        "params": {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "channel_type": "critical",
            "urgency": "critical",
            "message": body,
            "auto_injected": True,
            "sub_intent": "admin_alert_review_failed_slack",
        },
        "status": "pending",
        "proposed_by": "notify_admin_review_failed",
    }
    slack_action["idempotency_token"] = _compute_idempotency_token(slack_action)

    email_action = {
        "action_type": "send_manager_email",
        "tool": "gmail",
        "priority": "critical",
        "params": {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "subject": f"[긴급 검토 필요] 통화 {call_id} 분석 검토 실패",
            "body": body,
            "auto_injected": True,
            "sub_intent": "admin_alert_review_failed_email",
        },
        "status": "pending",
        "proposed_by": "notify_admin_review_failed",
    }
    email_action["idempotency_token"] = _compute_idempotency_token(email_action)

    return [slack_action, email_action]


async def notify_admin_review_failed_node(state: PostCallAgentState) -> dict:
    call_id: str = state["call_id"]
    tenant_id: str = state.get("tenant_id") or ""  # type: ignore[call-overload]
    review_feedback: list[str] = list(state.get("review_feedback") or [])  # type: ignore[call-overload]
    retry_count = int(state.get("analysis_retry_count") or 0)  # type: ignore[call-overload]
    transcripts = list(state.get("transcripts") or [])  # type: ignore[call-overload]
    call_metadata = state.get("call_metadata") or {}  # type: ignore[call-overload]

    customer_phone = (
        call_metadata.get("caller_number")
        or call_metadata.get("customer_phone")
        or ""
    )

    body = _format_admin_message_body(
        call_id=call_id,
        tenant_id=tenant_id,
        review_feedback=review_feedback,
        retry_count=retry_count,
        customer_phone=customer_phone,
        transcripts=transcripts,
    )

    alert_actions = _build_admin_alert_actions(
        call_id=call_id,
        tenant_id=tenant_id,
        body=body,
    )

    # state["proposed_actions"] 의 Notion auto 액션을 [REVIEW_FAILED] 로 mutate
    proposed = list(state.get("proposed_actions") or [])  # type: ignore[call-overload]
    proposed_mutated = [copy.deepcopy(a) for a in proposed]
    notion_marked = _mutate_notion_actions_in_place(proposed_mutated)

    # Notion auto 들 + alert 두 개를 합쳐 다음 노드 (auto_action_executor) 가 한 번에 발송
    combined = proposed_mutated + alert_actions

    logger.info(
        "notify_admin_review_failed call_id=%s tenant=%s retry=%d "
        "alerts=2 notion_marked=%d feedback_items=%d",
        call_id, tenant_id, retry_count,
        notion_marked, len(review_feedback),
    )

    return {
        "proposed_actions": combined,
        "human_review_required": True,
        # human_queue_node 와의 호환 — 사람 검토 큐 등록 사유 명시
        "escalate_reason": (
            state.get("escalate_reason")
            or "max_retries_review_fail"
        ),
        "action_plan": {
            "action_required": True,
            "actions": alert_actions,
            "rationale": "max_retries_review_fail — admin alert + Notion marker",
        },
    }
