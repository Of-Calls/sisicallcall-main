from app.services.embedding.base import BaseEmbeddingService
from app.utils.config import settings

_embedder: BaseEmbeddingService | None = None


def get_embedder() -> BaseEmbeddingService:
    """임베더 lazy singleton — settings.embedding_provider 따라 분기.

    - "bge-m3" (default): FlagEmbedding BGE-M3 (1024d, multilingual SOTA 2024)
    - "qwen3": sentence-transformers Qwen3-Embedding-0.6B (1024d, multilingual SOTA 2025)

    프로덕션은 app/main.py lifespan 에서 부팅 시 1회 호출 → 첫 요청 latency 0.
    """
    global _embedder
    if _embedder is None:
        if settings.embedding_provider == "qwen3":
            from app.services.embedding.qwen3 import Qwen3EmbeddingService
            _embedder = Qwen3EmbeddingService()
        else:
            from app.services.embedding.local import BGEM3LocalEmbeddingService
            _embedder = BGEM3LocalEmbeddingService()
    return _embedder
