from app.services.tts.base import BaseTTSService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GoogleTTSService(BaseTTSService):
    def __init__(self):
        from google.cloud import texttospeech
        self._client = texttospeech.TextToSpeechAsyncClient()

    async def synthesize_and_stream(self, text: str) -> None:
        # TODO: Google TTS WaveNet 합성 후 Twilio WebSocket 으로 μ-law 8kHz 스트리밍
        raise NotImplementedError
