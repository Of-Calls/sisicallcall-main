from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 해제 조건: 담당자 지정 후 (M2 기능)
# 타임아웃: 4초 하드컷 (asyncio.wait_for 사용)

logger = get_logger(__name__)

FALLBACK_MESSAGE = "확인이 어려워 담당자에게 연결해 드리겠습니다."


async def task_branch_node(state: CallState) -> dict:
    # TODO: MCP + GPT-4o 업무 처리 구현
    return {
        "response_text": FALLBACK_MESSAGE,
        "response_path": "task",
        "is_timeout": False,
    }
