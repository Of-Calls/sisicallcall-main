from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 해제 조건: 담당자 지정 후
# 타임아웃: 2초 하드컷 (asyncio.wait_for 사용)

logger = get_logger(__name__)

FALLBACK_MESSAGE = "확인이 어려워 담당자에게 연결해 드리겠습니다."


async def faq_branch_node(state: CallState) -> dict:
    # TODO: RAG 검색 + GPT-4o-mini 응답 생성 구현
    return {
        "rag_results": [],
        "response_text": FALLBACK_MESSAGE,
        "response_path": "faq",
        "is_timeout": False,
    }
