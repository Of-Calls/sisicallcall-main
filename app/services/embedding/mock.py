import random

from app.services.embedding.base import BaseEmbeddingService

EMBEDDING_DIM = 1024


class MockEmbeddingService(BaseEmbeddingService):
    """테스트 전용 — 랜덤 벡터 반환. BGE-M3 구현 완료 후 교체."""

    async def embed(self, text: str) -> list[float]:
        random.seed(hash(text) % (2**32))
        return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]
