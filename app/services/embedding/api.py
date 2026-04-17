from app.services.embedding.base import BaseEmbeddingService
from app.utils.logger import get_logger

# TODO(희영): BGE-M3 API vs 로컬 실험 결과 보고 후 구현
# 해제 조건: API 방식 실험 결과 보고 후

logger = get_logger(__name__)


class BGEM3APIEmbeddingService(BaseEmbeddingService):
    async def embed(self, text: str) -> list[float]:
        # TODO(희영): BGE-M3 API 호출 구현
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # TODO(희영): 배치 API 호출 구현
        raise NotImplementedError
