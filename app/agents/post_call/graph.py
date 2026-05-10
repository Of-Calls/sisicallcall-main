"""Post-call 2-에이전트 그래프.

  load_context
    → analysis_planner_agent          (Agent 1: 분석 + 액션 후보 propose)
    → save_intermediate                (분석 결과 무조건 저장 — 대시보드 보장)
    → escalation_immediate? END
    → reviewer_agent                   (Agent 2: ReAct 루프 검증)
        ├ pass / correctable          → action_executor → save_final → END
        ├ fail (retry < MAX)          → increment_analysis_retry → analysis_planner_agent
        │                               (review_feedback 누적 + 분석 재실행)
        └ fail (retry ≥ MAX)          → human_queue → save_final → END

분석/요약은 reviewer 통과 여부와 무관하게 save_intermediate 단계에서 저장된다 —
reviewer 가 실패하더라도 call_summaries / voc_analyses 는 보장된다 (ON CONFLICT
upsert 이므로 retry 시 중복 INSERT 발생 안 함).

외부 MCP 액션은 reviewer 통과 시에만 실행된다.

retry 가드:
  - MAX_ANALYSIS_RETRIES = 2 (총 3회 = 최초 + 재시도 2회)
  - retry 한도 초과 시 강제 human_queue
  - retry 시 LLM 비용 증가 — telemetry 에 retry_count 기록
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.nodes.action_executor_node import action_executor_node
from app.agents.post_call.nodes.analysis_planner_agent_node import analysis_planner_agent_node
from app.agents.post_call.nodes.human_queue_node import human_queue_node
from app.agents.post_call.nodes.increment_analysis_retry_node import (
    increment_analysis_retry_node,
)
from app.agents.post_call.nodes.load_context_node import load_context_node
from app.agents.post_call.nodes.reviewer_agent_node import reviewer_agent_node
from app.agents.post_call.nodes.save_result_node import (
    save_final_node,
    save_intermediate_node,
)


# 분석 재시도 한도 — env 로 노출 안 함 (안전 마진).
MAX_ANALYSIS_RETRIES = 2


def _route_after_intermediate(state: PostCallAgentState) -> str:
    # escalation_immediate: reviewer/액션 스킵 → 분석만 저장하고 종료.
    if state["trigger"] == "escalation_immediate":
        return "skip_review"
    # 분석 자체가 실패해서 human_review_required 가 이미 True 면 reviewer 우회.
    if state.get("human_review_required"):  # type: ignore[call-overload]
        return "skip_review"
    return "review"


def _route_after_review(state: PostCallAgentState) -> str:
    verdict = state.get("review_verdict") or "fail"  # type: ignore[call-overload]
    if verdict in ("pass", "correctable"):
        return "execute"
    # verdict == "fail"
    retry_count = int(state.get("analysis_retry_count") or 0)  # type: ignore[call-overload]
    if retry_count < MAX_ANALYSIS_RETRIES:
        return "retry_analysis"
    return "human_queue"


def build_post_call_graph():
    g: StateGraph = StateGraph(PostCallAgentState)

    g.add_node("load_context", load_context_node)
    g.add_node("analysis_planner_agent", analysis_planner_agent_node)
    g.add_node("save_intermediate", save_intermediate_node)
    g.add_node("reviewer_agent", reviewer_agent_node)
    g.add_node("increment_analysis_retry", increment_analysis_retry_node)
    g.add_node("action_executor", action_executor_node)
    g.add_node("human_queue", human_queue_node)
    g.add_node("save_final", save_final_node)

    g.set_entry_point("load_context")
    g.add_edge("load_context", "analysis_planner_agent")
    g.add_edge("analysis_planner_agent", "save_intermediate")

    g.add_conditional_edges(
        "save_intermediate",
        _route_after_intermediate,
        {
            "review": "reviewer_agent",
            "skip_review": END,
        },
    )
    g.add_conditional_edges(
        "reviewer_agent",
        _route_after_review,
        {
            "execute": "action_executor",
            "retry_analysis": "increment_analysis_retry",
            "human_queue": "human_queue",
        },
    )
    # retry loop: increment → planner → save_intermediate → reviewer (다시)
    g.add_edge("increment_analysis_retry", "analysis_planner_agent")

    g.add_edge("action_executor", "save_final")
    g.add_edge("human_queue", "save_final")
    g.add_edge("save_final", END)

    return g.compile()
