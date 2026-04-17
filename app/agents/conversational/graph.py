from langgraph.graph import END, StateGraph

from app.agents.conversational.state import CallState
from app.agents.conversational.nodes.vad_node.vad_node import vad_node
from app.agents.conversational.nodes.speaker_verify_node.speaker_verify_node import speaker_verify_node
from app.agents.conversational.nodes.stt_node.stt_node import stt_node
from app.agents.conversational.nodes.norm_text_node.norm_text_node import norm_text_node
from app.agents.conversational.nodes.cache_node.cache_node import cache_node
from app.agents.conversational.nodes.knn_router_node.knn_router_node import knn_router_node
from app.agents.conversational.nodes.intent_router_llm_node.intent_router_llm_node import intent_router_llm_node
from app.agents.conversational.nodes.faq_branch_node.faq_branch_node import faq_branch_node
from app.agents.conversational.nodes.task_branch_node.task_branch_node import task_branch_node
from app.agents.conversational.nodes.auth_branch_node.auth_branch_node import auth_branch_node
from app.agents.conversational.nodes.escalation_branch_node.escalation_branch_node import escalation_branch_node
from app.agents.conversational.nodes.reviewer_node.reviewer_node import reviewer_node
from app.agents.conversational.nodes.tts_node.tts_node import tts_node

# 신용 연구 완료 후 app/utils/config.py 이관
KNN_CONFIDENCE_THRESHOLD = 0.85


# ── 조건부 엣지 함수 ──────────────────────────────────────────

def route_after_vad(state: CallState) -> str:
    return "pass" if state["is_speech"] else "skip"


def route_after_speaker_verify(state: CallState) -> str:
    return "pass" if state["is_speaker_verified"] else "reject"


def route_after_cache(state: CallState) -> str:
    return "hit" if state["cache_hit"] else "miss"


def route_after_knn(state: CallState) -> str:
    """KNN confidence 임계값 이상이면 primary_intent 확정 후 브랜치 직행."""
    if state["knn_confidence"] >= KNN_CONFIDENCE_THRESHOLD and state["knn_intent"]:
        return _intent_to_branch(state["primary_intent"])
    return "fallback_llm"


def route_to_branch(state: CallState) -> str:
    """IntentRouterLLM 확정 후 primary_intent → 브랜치."""
    return _intent_to_branch(state["primary_intent"])


def route_after_branch(state: CallState) -> str:
    return "review" if _is_high_risk(state) else "skip_review"


def _intent_to_branch(intent: str | None) -> str:
    mapping = {
        "intent_faq": "faq",
        "intent_task": "task",
        "intent_auth": "auth",
        "intent_escalation": "escalation",
    }
    return mapping.get(intent or "", "escalation")


def _is_high_risk(state: CallState) -> bool:
    # TODO(미배정): 담당자 지정 후 구현 — agents.md Reviewer 섹션 + R-09 결과 확정 후
    return False


# ── 그래프 빌더 ───────────────────────────────────────────────

def build_call_graph():
    graph = StateGraph(CallState)

    # 노드 등록
    graph.add_node("vad", vad_node)
    graph.add_node("speaker_verify", speaker_verify_node)
    graph.add_node("stt", stt_node)
    graph.add_node("norm_text", norm_text_node)
    graph.add_node("cache", cache_node)
    graph.add_node("knn_router", knn_router_node)
    graph.add_node("intent_router_llm", intent_router_llm_node)
    graph.add_node("faq_branch", faq_branch_node)
    graph.add_node("task_branch", task_branch_node)
    graph.add_node("auth_branch", auth_branch_node)
    graph.add_node("escalation_branch", escalation_branch_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("tts", tts_node)

    # 진입점
    graph.set_entry_point("vad")

    # 전처리 단계
    graph.add_conditional_edges("vad", route_after_vad,
        {"pass": "speaker_verify", "skip": END})
    graph.add_conditional_edges("speaker_verify", route_after_speaker_verify,
        {"pass": "stt", "reject": END})
    graph.add_edge("stt", "norm_text")
    graph.add_edge("norm_text", "cache")

    # Gate 1 분기
    graph.add_conditional_edges("cache", route_after_cache,
        {"hit": "tts", "miss": "knn_router"})

    # KNN → 브랜치 직행 또는 IntentRouterLLM fallback
    graph.add_conditional_edges("knn_router", route_after_knn, {
        "faq": "faq_branch",
        "task": "task_branch",
        "auth": "auth_branch",
        "escalation": "escalation_branch",
        "fallback_llm": "intent_router_llm",
    })

    # IntentRouterLLM → 브랜치
    graph.add_conditional_edges("intent_router_llm", route_to_branch, {
        "faq": "faq_branch",
        "task": "task_branch",
        "auth": "auth_branch",
        "escalation": "escalation_branch",
    })

    # 브랜치 → (조건부 Reviewer) → TTS
    for branch in ("faq_branch", "task_branch", "auth_branch", "escalation_branch"):
        graph.add_conditional_edges(branch, route_after_branch,
            {"review": "reviewer", "skip_review": "tts"})

    graph.add_edge("reviewer", "tts")
    graph.add_edge("tts", END)

    return graph.compile()
