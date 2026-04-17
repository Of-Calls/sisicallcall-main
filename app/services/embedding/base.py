from abc import ABC, abstractmethod


class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """텍스트를 BGE-M3 임베딩 벡터로 변환."""
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩."""
        raise NotImplementedError
