from app.services.embedding.base import BaseEmbeddingService
from app.utils.logger import get_logger

# TODO(희영): BGE-M3 API vs 로컬 실험 결과 보고 후 구현
# 해제 조건: 로컬 방식 실험 결과 보고 후

logger = get_logger(__name__)


class BGEM3LocalEmbeddingService(BaseEmbeddingService):
    async def embed(self, text: str) -> list[float]:
        # TODO(희영): 로컬 BGE-M3 모델 추론 구현 (run_in_executor 필수)
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # TODO(희영): 배치 추론 구현
        raise NotImplementedError
