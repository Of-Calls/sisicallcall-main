from abc import ABC, abstractmethod


class BaseSpeakerVerifyService(ABC):
    @abstractmethod
    async def verify(self, audio_chunk: bytes, call_id: str) -> bool:
        """화자 검증 — voiceprint 와 비교해 동일 화자 여부 반환."""
        raise NotImplementedError

    @abstractmethod
    async def extract_and_store(self, audio_chunk: bytes, call_id: str) -> None:
        """voiceprint 추출 후 Redis 에 저장 (첫 발화 5초 누적 후 호출)."""
        raise NotImplementedError
