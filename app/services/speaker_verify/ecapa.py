from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.utils.logger import get_logger

# TODO(대영): ECAPA-TDNN 파인튜닝 완료 후 구현
# 해제 조건: 파인튜닝 완료 보고 후

logger = get_logger(__name__)


class ECAPASpeakerVerifyService(BaseSpeakerVerifyService):
    async def verify(self, audio_chunk: bytes, call_id: str) -> bool:
        # TODO(대영): ECAPA-TDNN 추론 + Redis voiceprint 비교
        raise NotImplementedError

    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        # TODO(대영): voiceprint 추출 후 Redis call:{call_id}:voiceprint 저장
        raise NotImplementedError
