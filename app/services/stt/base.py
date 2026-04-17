from abc import ABC, abstractmethod


class BaseSTTService(ABC):
    @abstractmethod
    async def transcribe(self, audio_chunk: bytes) -> str:
        """PCM 16kHz 오디오를 텍스트로 변환."""
        raise NotImplementedError
