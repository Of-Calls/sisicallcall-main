"""Qwen3-Embedding-0.6B sentence-transformers 기반 임베딩 service.

asymmetric retrieval — query 측에 task instruction prepend (sentence-transformers prompt_name="query"),
passage 측은 plain (instruction 없음). 차원 1024 (BGE-M3 호환), context 32K.
"""
import asyncio
import time

from app.services.embedding.base import BaseEmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Qwen3EmbeddingService(BaseEmbeddingService):
    """Qwen3-Embedding-0.6B (HuggingFace, sentence-transformers).

    - embed/embed_batch: instruction 없이 plain (호환성 / passage 기본 동작)
    - embed_query: prompt_name="query" 로 Qwen3 default task instruction prepend
        ("Given a web search query, retrieve relevant passages that answer the query")
    - embed_passages: plain
    """

    _MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        t0 = time.monotonic()
        logger.info("Qwen3-Embedding 로딩 시작 model=%s device=%s ...", self._MODEL_NAME, device)
        self._model = SentenceTransformer(self._MODEL_NAME, device=device)
        elapsed = time.monotonic() - t0
        logger.info("Qwen3-Embedding 로딩 완료 elapsed=%.2fs", elapsed)

    def _encode_sync(self, texts: list[str], prompt_name: str | None = None) -> list[list[float]]:
        kwargs = {"normalize_embeddings": True}
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        vecs = self._model.encode(texts, **kwargs)
        return vecs.tolist()

    async def embed(self, text: str) -> list[float]:
        """plain encoding — instruction 없음. query 의도면 embed_query 권장."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode_sync, [text], None)
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """plain batch — passage 측 기본 동작."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, None)

    async def embed_query(self, text: str) -> list[float]:
        """query 측 — Qwen3 default task instruction prepend."""
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode_sync, [text], "query")
        return results[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """passage 측 — plain (Qwen3 권장: passage 는 instruction 없이 임베딩)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, None)
