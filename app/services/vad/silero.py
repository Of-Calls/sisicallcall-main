from app.services.vad.base import BaseVADService
from app.utils.logger import get_logger

# TODO(주미): VAD threshold 연구 완료 후 구현
# 해제 조건: 업종별 최적 threshold 도출 후 팀장 보고

logger = get_logger(__name__)


class SileroVADService(BaseVADService):
    async def detect(self, audio_chunk: bytes) -> bool:
        # TODO(주미): Silero VAD 모델 로드 및 추론 구현
        raise NotImplementedError
