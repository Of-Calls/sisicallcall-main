"""Post-call 2-에이전트 그래프.

  load_context
    → analysis_planner_agent          (Agent 1: 분석 + 액션 후보 propose + Notion auto inject)
    → reviewer_agent                   (Agent 2: ReAct 루프 검증, auto_injected 우회)
        ├ pass / correctable          → save_reviewed_analysis (검토 통과 분석만 영속화)
        │                               → action_executor (auto + LLM-approved 발송)
        │                               → save_final → END
        ├ fail (retry < MAX)          → increment_analysis_retry → analysis_planner_agent
        │                               (review_feedback 누적 + 분석 재실행)
        └ fail (retry ≥ MAX)          → notify_admin_review_failed
                                        (Slack + Gmail 긴급 알림 PlannedAction 생성 +
                                         Notion auto 액션 [REVIEW_FAILED] 마킹)
                                        → human_queue (사람 큐 등록)
                                        → auto_action_executor (auto + alert 발송)
                                        → save_final → END

핵심 변경 (이전 흐름과 차이):
  - save_intermediate (reviewer 전 raw 분석 저장) 단계 삭제 — 검토 통과 분석만 저장
  - escalation_immediate trigger 는 reviewer/저장 모두 우회하고 바로 END
  - fail+max retry 시 분석 본문은 call_summaries / voc_analyses 에 들어가지 않음
  - 대신 관리자 알림 + Notion 마킹으로 사람 검토 유도

자동 주입 액션 (Notion 회사 DB 기록) 은 reviewer 우회. retry 사이클 동안 매번
재주입되지만 idempotency_token 으로 첫 시도 후 executor 가 skip — 통화당 1건 보장.

retry 가드:
  - MAX_ANALYSIS_RETRIES = 2 (총 3회 = 최초 + 재시도 2회)
  - retry 한도 초과 시 notify_admin_review_failed → human_queue → auto_action_executor
  - retry 시 LLM 비용 증가 — telemetry 에 retry_count 기록
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.nodes.action_executor_node import action_executor_node
from app.agents.post_call.nodes.analysis_planner_agent_node import analysis_planner_agent_node
from app.agents.post_call.nodes.auto_action_executor_node import auto_action_executor_node
from app.agents.post_call.nodes.human_queue_node import human_queue_node
from app.agents.post_call.nodes.increment_analysis_retry_node import (
    increment_analysis_retry_node,
)
from app.agents.post_call.nodes.load_context_node import load_context_node
from app.agents.post_call.nodes.notify_admin_review_failed_node import (
    notify_admin_review_failed_node,
)
from app.agents.post_call.nodes.reviewer_agent_node import reviewer_agent_node
from app.agents.post_call.nodes.save_result_node import (
    save_final_node,
    save_reviewed_analysis_node,
)


# 분석 재시도 한도 — env 로 노출 안 함 (안전 마진).
MAX_ANALYSIS_RETRIES = 2


def _route_after_planner(state: PostCallAgentState) -> str:
    """analysis_planner 직후 분기.

    - escalation_immediate: 즉시 종료 (분석/액션 모두 스킵)
    - planner 자체가 LLM 호출 실패 등으로 human_review_required=True 면 우회
    - 그 외 → reviewer
    """
    if state["trigger"] == "escalation_immediate":
        return "skip_review"
    if state.get("human_review_required"):  # type: ignore[call-overload]
        return "skip_review"
    return "review"


def _route_after_review(state: PostCallAgentState) -> str:
    verdict = state.get("review_verdict") or "fail"  # type: ignore[call-overload]
    if verdict in ("pass", "correctable"):
        return "save_reviewed"
    # verdict == "fail"
    retry_count = int(state.get("analysis_retry_count") or 0)  # type: ignore[call-overload]
    if retry_count < MAX_ANALYSIS_RETRIES:
        return "retry_analysis"
    return "notify_admin"


def build_post_call_graph():
    g: StateGraph = StateGraph(PostCallAgentState)

    g.add_node("load_context", load_context_node)
    g.add_node("analysis_planner_agent", analysis_planner_agent_node)
    g.add_node("reviewer_agent", reviewer_agent_node)
    g.add_node("save_reviewed_analysis", save_reviewed_analysis_node)
    g.add_node("increment_analysis_retry", increment_analysis_retry_node)
    g.add_node("notify_admin_review_failed", notify_admin_review_failed_node)
    g.add_node("action_executor", action_executor_node)
    g.add_node("human_queue", human_queue_node)
    g.add_node("auto_action_executor", auto_action_executor_node)
    g.add_node("save_final", save_final_node)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "analysis_planner_agent")

    # planner → reviewer (escalation_immediate / planner-fail 분기는 즉시 END)
    g.add_conditional_edges(
        "analysis_planner_agent",
        _route_after_planner,
        {
            "review": "reviewer_agent",
            "skip_review": END,
        },
    )

    g.add_conditional_edges(
        "reviewer_agent",
        _route_after_review,
        {
            "save_reviewed": "save_reviewed_analysis",
            "retry_analysis": "increment_analysis_retry",
            "notify_admin": "notify_admin_review_failed",
        },
    )

    # 정상 경로: save_reviewed → action_executor → save_final
    g.add_edge("save_reviewed_analysis", "action_executor")
    g.add_edge("action_executor", "save_final")

    # retry 경로: increment → planner (다시)
    g.add_edge("increment_analysis_retry", "analysis_planner_agent")

    # fail max 경로: notify_admin → human_queue → auto_action_executor → save_final
    g.add_edge("notify_admin_review_failed", "human_queue")
    g.add_edge("human_queue", "auto_action_executor")
    g.add_edge("auto_action_executor", "save_final")

    g.add_edge("save_final", END)

    return g.compile()
