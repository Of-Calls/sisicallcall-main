from app.services.stt.base import BaseSTTService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DeepgramSTTService(BaseSTTService):
    def __init__(self):
        from deepgram import DeepgramClient
        self._client = DeepgramClient(settings.deepgram_api_key)

    async def transcribe(self, audio_chunk: bytes) -> str:
        # TODO: Deepgram Nova-2 Pre-recorded API 호출 구현
        # 실시간 스트리밍은 WebSocket 연결에서 직접 처리 고려
        raise NotImplementedError
