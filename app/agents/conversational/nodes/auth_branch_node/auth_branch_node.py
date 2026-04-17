from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 해제 조건: 담당자 지정 후
# 타임아웃: 3초 하드컷 (asyncio.wait_for 사용)

logger = get_logger(__name__)

FALLBACK_MESSAGE = "본인 확인을 위해 잠시 후 다시 안내해 드리겠습니다."


async def auth_branch_node(state: CallState) -> dict:
    # TODO: GPT-4o-mini 로 인증 안내 구현
    return {
        "response_text": FALLBACK_MESSAGE,
        "response_path": "auth",
        "is_timeout": False,
    }
