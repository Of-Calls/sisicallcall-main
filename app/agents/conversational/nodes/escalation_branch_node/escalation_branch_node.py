from app.agents.conversational.state import CallState


_ESCALATION_MESSAGE = (
    "죄송합니다. 상담사가 직접 연락드릴 수 있도록 접수해드릴게요. "
    "통화 마치고 곧 연락드리겠습니다."
)


async def escalation_branch_node(state: CallState) -> dict:
    print(f"[escalation_branch] 진입 user_text='{state['user_text']}'")
    return {"response_text": _ESCALATION_MESSAGE}
