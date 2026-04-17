from abc import ABC, abstractmethod


class BaseTTSService(ABC):
    @abstractmethod
    async def synthesize_and_stream(self, text: str) -> None:
        """텍스트를 음성으로 합성해 Twilio WebSocket 으로 스트리밍."""
        raise NotImplementedError
