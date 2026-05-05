from abc import ABC, abstractmethod


class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """텍스트 → 임베딩 벡터. 모델별 prefix 미적용."""
        raise NotImplementedError

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩. 모델별 prefix 미적용."""
        raise NotImplementedError

    # ── asymmetric retrieval API (BGE-M3 등) ────────────────────────────
    # query 측 / passage 측 임베딩 분리. 모델 구현체가 prefix/instruction 적용.
    # default 동작은 prefix 없이 embed/embed_batch 호출 (Mock 등 호환).

    async def embed_query(self, text: str) -> list[float]:
        """query 측 임베딩 — 검색 입력. 모델별 query instruction 적용 가능."""
        return await self.embed(text)

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """passage 측 임베딩 — 인덱싱 대상. 모델별 passage 처리 적용 가능."""
        return await self.embed_batch(texts)
