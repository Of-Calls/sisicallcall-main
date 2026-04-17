from app.services.knn_router.base import BaseKNNRouterService
from app.utils.logger import get_logger

# TODO(신용): 연구 완료 후 구현
# 해제 조건: KNN Router 연구 결과 팀장 보고 후

logger = get_logger(__name__)


class KNNRouterService(BaseKNNRouterService):
    async def classify(
        self, embedding: list[float], tenant_id: str
    ) -> tuple[str | None, float]:
        # TODO(신용): KNN 분류 구현
        raise NotImplementedError
