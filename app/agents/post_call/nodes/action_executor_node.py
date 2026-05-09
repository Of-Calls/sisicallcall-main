"""approved_actions 를 ActionExecutor 로 위임하는 그래프 노드.

reviewer_agent 의 verdict=pass/correctable 분기에서 호출된다.
fail 분기는 human_queue_node 가 받는다.
"""
from __future__ import annotations

from app.agents.post_call.actions.executor import ActionExecutor
from app.agents.post_call.state import PostCallAgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)
_executor = ActionExecutor()


async def action_executor_node(state: PostCallAgentState) -> dict:
    call_id = state["call_id"]
    tenant_id: str = state.get("tenant_id", "") or ""  # type: ignore[call-overload]
    approved: list = list(state.get("approved_actions") or [])  # type: ignore[call-overload]

    plan_for_dashboard = {
        "action_required": len(approved) > 0,
        "actions": [dict(a) for a in approved],
        "rationale": "reviewer_approved",
    }

    if not approved:
        logger.info("action_executor: approved_actions 없음 call_id=%s — skip", call_id)
        return {"executed_actions": [], "action_plan": plan_for_dashboard}

    try:
        executed = await _executor.execute_actions(
            call_id=call_id,
            tenant_id=tenant_id,
            actions=approved,
        )
        failed = [a for a in executed if a.get("status") == "failed"]
        logger.info(
            "action_executor 완료 call_id=%s executed=%d failed=%d",
            call_id, len(executed), len(failed),
        )
        out: dict = {
            "executed_actions": executed,
            "action_plan": plan_for_dashboard,
        }
        if failed:
            out["partial_success"] = True
        return out
    except Exception as exc:
        logger.error("action_executor 실패 call_id=%s err=%s", call_id, exc)
        errors = list(state.get("errors", []))  # type: ignore[call-overload]
        errors.append({"node": "action_executor", "error": str(exc)})
        return {
            "executed_actions": [],
            "action_plan": plan_for_dashboard,
            "errors": errors,
            "partial_success": True,
        }
