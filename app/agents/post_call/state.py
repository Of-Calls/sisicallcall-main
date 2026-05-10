from __future__ import annotations
from typing import Optional
from typing_extensions import TypedDict


class PostCallAgentState(TypedDict):
    call_id: str
    tenant_id: str
    trigger: str                    # "call_ended" | "escalation_immediate" | "manual"
    call_metadata: dict
    transcripts: list[dict]
    branch_stats: dict
    summary: Optional[dict]
    voc_analysis: Optional[dict]
    priority_result: Optional[dict]
    action_plan: Optional[dict]
    executed_actions: list[dict]
    dashboard_payload: Optional[dict]
    errors: list[dict]
    partial_success: bool

    # ── Agent 1 (analysis_planner_agent) 출력 ─────────────────────────────────
    analysis_result: Optional[dict]      # 통합 분석 (summary + voc_analysis + priority_result)
    proposed_actions: list[dict]         # PlannedAction-호환 dict 리스트 (LLM 자율 선택)
    analysis_planner_rationale: str      # 도구 선택/스킵 사유
    analysis_planner_telemetry: Optional[dict]  # tokens / tool_counts / latency_ms / model
    analysis_llm_usage: Optional[dict]   # legacy alias (호환)

    # ── Agent 2 (reviewer_agent) 출력 ─────────────────────────────────────────
    review_result: Optional[dict]        # {verdict, approved_actions, corrections_to_analysis, escalate_reason, steps}
    review_verdict: Optional[str]        # "pass" | "correctable" | "fail"
    approved_actions: list[dict]         # reviewer 가 승인/보정한 액션
    corrections_to_analysis: dict        # reviewer 가 분석 결과에 적용한 패치
    escalate_reason: Optional[str]
    reviewer_steps: int                  # ReAct 루프 step 수 (legacy alias)
    reviewer_telemetry: Optional[dict]   # tokens / tool_counts / steps / max_steps_reached / latency_ms
    review_llm_usage: Optional[dict]     # legacy alias

    # ── Routing flags ─────────────────────────────────────────────────────────
    human_review_required: bool          # True 이면 외부 action 실행 금지

    # ── Reviewer fail → analysis 재시도 루프 ──────────────────────────────────
    analysis_retry_count: int            # planner 재실행 횟수 (0 = 최초 시도)
    review_feedback: list[str]           # 이전 reviewer 가 지적한 fail 사유 누적

    # ── Legacy fields (호환 유지 — 곧 비울 예정) ───────────────────────────────
    blocked_actions: list[str]
    review_retry_count: int
