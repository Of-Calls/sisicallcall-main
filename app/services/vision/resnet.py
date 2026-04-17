from app.services.vision.base import BaseVisionService
from app.utils.logger import get_logger

# TODO(수현): 모델 선정 후 구현
# 해제 조건: 비전 모델 선정 후

logger = get_logger(__name__)


class ResNetVisionService(BaseVisionService):
    async def classify(self, image_bytes: bytes) -> dict:
        # TODO(수현): ResNet 분류 구현
        raise NotImplementedError
