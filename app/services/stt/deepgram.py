from app.services.stt.base import BaseSTTService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

class DeepgramSTTService(BaseSTTService):
    def __init__(self):
        # 지연 로딩(Lazy Loading) 적용
        from deepgram import DeepgramClient, PrerecordedOptions
        
        self._client = DeepgramClient(settings.deepgram_api_key)
        
        # 핵심 옵션 추가: 16kHz 16-bit PCM 포맷 명시
        self._prerecordedOptions = PrerecordedOptions(
            model="nova-3",
            language="ko",
            smart_format=True,
            punctuate=True,
            encoding="mulaw",
            sample_rate=8000,
        )

    async def transcribe(self, audio_chunk: bytes) -> str:
        if not audio_chunk:
            return ""

        try:
            payload = {"buffer": audio_chunk}
            
            # 비동기 API 호출 (초기화 시 만들어둔 옵션 객체 재사용)
            response = await self._client.listen.asyncprerecorded.v("1").transcribe_file(
                payload, self._prerecordedOptions
            )

            transcript = response.results.channels[0].alternatives[0].transcript
            return transcript

        except Exception as e:
            logger.error(f"Deepgram API 호출 실패: {e}")
            raise e