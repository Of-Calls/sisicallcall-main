from abc import ABC, abstractmethod


class BaseTTSService(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """텍스트 → 음성 바이트 (mulaw 8kHz)."""
        raise NotImplementedError
