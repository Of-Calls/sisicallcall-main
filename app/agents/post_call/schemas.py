from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Action 관련 ──────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    create_voc_issue = "create_voc_issue"
    send_manager_email = "send_manager_email"
    schedule_callback = "schedule_callback"
    add_priority_queue = "add_priority_queue"
    mark_faq_candidate = "mark_faq_candidate"
    create_jira_issue = "create_jira_issue"
    send_slack_alert = "send_slack_alert"
    send_callback_sms = "send_callback_sms"
    send_voc_receipt_sms = "send_voc_receipt_sms"
    send_reservation_confirmation = "send_reservation_confirmation"
    create_notion_call_record = "create_notion_call_record"
    create_notion_voc_record = "create_notion_voc_record"


class Tool(str, Enum):
    company_db = "company_db"
    gmail = "gmail"
    calendar = "calendar"
    internal_dashboard = "internal_dashboard"
    jira = "jira"
    slack = "slack"
    sms = "sms"
    notion = "notion"


class ActionStatus(str, Enum):
    pending = "pending"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class ActionItem(BaseModel):
    action_type: ActionType
    tool: Tool
    params: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.pending
    result: Optional[dict] = None
    error: Optional[str] = None


class ActionPlan(BaseModel):
    actions: list[ActionItem] = Field(default_factory=list)
    rationale: str = ""


# ── LLM 출력 열거형 ───────────────────────────────────────────────────────────

class CustomerEmotion(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    angry = "angry"


class ResolutionStatus(str, Enum):
    resolved = "resolved"
    escalated = "escalated"
    abandoned = "abandoned"


class PriorityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ── Summary 출력 스키마 ───────────────────────────────────────────────────────

class SummaryResult(BaseModel):
    summary_short: str
    summary_detailed: str
    customer_intent: str
    customer_emotion: CustomerEmotion = CustomerEmotion.neutral
    resolution_status: ResolutionStatus = ResolutionStatus.resolved
    keywords: list[str] = Field(default_factory=list)
    handoff_notes: Optional[str] = None


# ── VOC 출력 스키마 ───────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    sentiment: CustomerEmotion = CustomerEmotion.neutral
    intensity: float = 0.0       # 0.0 ~ 1.0
    reason: str = ""


class IntentResult(BaseModel):
    primary_category: str
    sub_categories: list[str] = Field(default_factory=list)
    is_repeat_topic: bool = False
    faq_candidate: bool = False


class VOCPriorityResult(BaseModel):
    priority: PriorityLevel = PriorityLevel.low
    action_required: bool = False
    suggested_action: Optional[str] = None
    reason: str = ""


class VOCResult(BaseModel):
    sentiment_result: SentimentResult
    intent_result: IntentResult
    priority_result: VOCPriorityResult


# ── Action Planner 노드 전용 출력 스키마 (priority 필드 + action_required 추가) ──

class PlannedAction(BaseModel):
    """Action Planner 노드가 생성하는 액션.
    ActionItem 과 executor 호환을 유지하면서 priority 필드를 추가한다."""
    action_type: ActionType
    tool: Tool
    priority: PriorityLevel = PriorityLevel.low
    params: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.pending
    result: Optional[dict] = None
    error: Optional[str] = None


class ActionPlanResult(BaseModel):
    """Action Planner 노드 출력. ActionPlan 에 action_required 필드를 추가한다."""
    action_required: bool = False
    actions: list[PlannedAction] = Field(default_factory=list)
    rationale: str = ""


# ── Priority Node 출력 스키마 ─────────────────────────────────────────────────

class PriorityNodeResult(BaseModel):
    priority: PriorityLevel = PriorityLevel.low
    # action_planner_node 가 priority.get("tier") 를 참조하므로 tier 를 유지
    tier: str = PriorityLevel.low.value
    action_required: bool = False
    suggested_action: Optional[str] = None
    reason: str = ""


# ── 하위 호환 alias (기존 코드가 VOCAnalysis / PriorityResult 를 import 하는 경우 대비) ──

class VOCAnalysis(BaseModel):
    """Deprecated — VOCResult 로 교체. 하위 호환용."""
    sentiment: str
    issues: list[str]
    keywords: list[str]
    escalation_reason: Optional[str] = None
    faq_candidates: list[str] = Field(default_factory=list)


class PriorityResult(BaseModel):
    """Deprecated — PriorityNodeResult 로 교체. 하위 호환용."""
    score: int
    tier: str
    reason: str


# ── Review Gate 스키마 ────────────────────────────────────────────────────────

class ReviewVerdictValues:
    """review_node verdict 허용값. 'pass'는 Python 예약어이므로 클래스 상수로 관리."""
    PASS = "pass"
    CORRECTABLE = "correctable"
    RETRY = "retry"
    FAIL = "fail"
    _VALID = frozenset({"pass", "correctable", "retry", "fail"})

    @classmethod
    def is_valid(cls, v: str) -> bool:
        return v in cls._VALID


class ReviewIssue(BaseModel):
    type: str
    message: str
    evidence: Optional[str] = None


class ReviewResult(BaseModel):
    verdict: str = "fail"
    confidence: float = 0.0
    confidence_missing: bool = False
    confidence_parse_error: bool = False
    confidence_source: str = "llm"
    llm_fallback: bool = False
    llm_fallback_reason: str = ""
    issues: list[ReviewIssue] = Field(default_factory=list)
    corrections: dict[str, Any] = Field(default_factory=dict)
    corrected_keys: list[str] = Field(default_factory=list)
    blocked_actions: list[str] = Field(default_factory=list)
    reason: str = ""
