import asyncio

from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(주미): VAD threshold 연구 완료 후 구현 — architecture.md 참조
# 해제 조건: 업종별 최적 threshold 도출 후 팀장 보고

logger = get_logger(__name__)


async def vad_node(state: CallState) -> dict:
    # TODO(주미): Silero VAD 로 is_speech 판정 구현
    # blocking 모델 추론은 아래 패턴 사용:
    # loop = asyncio.get_running_loop()
    # is_speech = await loop.run_in_executor(None, _vad_service.detect_sync, state["audio_chunk"])
    return {"is_speech": True}
