from abc import ABC, abstractmethod


class BaseChunkingService(ABC):
    @abstractmethod
    async def chunk(self, text: str) -> list[str]:
        """텍스트를 청크 리스트로 분할."""
        raise NotImplementedError
