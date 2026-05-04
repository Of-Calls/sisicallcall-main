from langgraph.graph import StateGraph, END

from app.agents.conversational.state import CallState
from app.agents.conversational.nodes.query_refine_node.query_refine_node import query_refine_node
from app.agents.conversational.nodes.intent_router_llm_node.intent_router_llm_node import intent_router_llm_node
from app.agents.conversational.nodes.faq_branch_node.faq_branch_node import faq_branch_node
from app.agents.conversational.nodes.task_branch_node.task_branch_node import task_branch_node
from app.agents.conversational.nodes.auth_branch_node.auth_branch_node import auth_branch_node
from app.agents.conversational.nodes.vision_branch_node.vision_branch_node import vision_branch_node
from app.agents.conversational.nodes.escalation_branch_node.escalation_branch_node import escalation_branch_node
from app.agents.conversational.nodes.clarify_branch_node.clarify_branch_node import clarify_branch_node
from app.agents.conversational.nodes.repeat_branch_node.repeat_branch_node import repeat_branch_node


def _route_by_clarity(state: CallState) -> str:
    return "intent_router" if state.get("is_clear") else "clarify"


def _route_by_intent(state: CallState) -> str:
    return state["intent"]


def build_graph():
    g = StateGraph(CallState)

    g.add_node("query_refine", query_refine_node)
    g.add_node("intent_router", intent_router_llm_node)
    g.add_node("faq", faq_branch_node)
    g.add_node("task", task_branch_node)
    g.add_node("auth", auth_branch_node)
    g.add_node("vision", vision_branch_node)
    g.add_node("escalation", escalation_branch_node)
    g.add_node("clarify", clarify_branch_node)
    g.add_node("repeat", repeat_branch_node)

    g.set_entry_point("query_refine")

    g.add_conditional_edges(
        "query_refine",
        _route_by_clarity,
        {
            "intent_router": "intent_router",
            "clarify": "clarify",
        },
    )

    g.add_conditional_edges(
        "intent_router",
        _route_by_intent,
        {
            "faq": "faq",
            "task": "task",
            "auth": "auth",
            "vision": "vision",
            "escalation": "escalation",
            "repeat": "repeat",
        },
    )

    g.add_edge("faq", END)
    g.add_edge("task", END)
    g.add_edge("auth", END)
    g.add_edge("vision", END)
    g.add_edge("escalation", END)
    g.add_edge("clarify", END)
    g.add_edge("repeat", END)

    return g.compile()
