from abc import ABC, abstractmethod


class BaseVADService(ABC):
    @abstractmethod
    async def detect(self, audio_chunk: bytes) -> bool:
        """PCM 16kHz 오디오에서 발화 구간 여부 반환."""
        raise NotImplementedError
