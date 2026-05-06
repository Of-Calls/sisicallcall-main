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
    # ── Review Gate 필드 (통합 분석 + 검토 게이트) ─────────────────────────────
    analysis_result: Optional[dict]     # post_call_analysis_node 통합 출력
    review_result: Optional[dict]       # review_node 출력
    review_verdict: Optional[str]       # "pass" | "correctable" | "retry" | "fail"
    review_retry_count: int             # 재분석 횟수 (최대 1회)
    human_review_required: bool         # True이면 외부 action 금지
    blocked_actions: list[str]          # review가 차단한 action_type 또는 tool 이름
    # ── LLM token usage 메타데이터 (real LLM 호출 시 채워짐) ──────────────────
    analysis_llm_usage: Optional[dict]  # {purpose, model, prompt/completion/total_tokens, source, fallback}
    review_llm_usage: Optional[dict]
