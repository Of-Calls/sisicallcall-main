from app.services.embedding.base import BaseEmbeddingService
from app.services.embedding.local import BGEM3LocalEmbeddingService

_embedder: BaseEmbeddingService | None = None


def get_embedder() -> BaseEmbeddingService:
    """BGE-M3 임베더 lazy singleton.

    - 프로덕션: app/main.py lifespan 에서 부팅 시 1회 호출 → 첫 요청 latency 0
    - 테스트 (scripts/graph_test.py): 첫 호출 시 로딩 (테스트라 OK)
    """
    global _embedder
    if _embedder is None:
        _embedder = BGEM3LocalEmbeddingService()
    return _embedder
