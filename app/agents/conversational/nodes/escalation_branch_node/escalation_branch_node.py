from app.agents.conversational.state import CallState


async def escalation_branch_node(state: CallState) -> dict:
    print(f"[escalation_branch] 진입 user_text='{state['user_text']}'")
    return {"response_text": "[Escalation 분기 도달]"}
