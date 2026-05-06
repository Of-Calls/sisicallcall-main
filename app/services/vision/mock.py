from app.services.vision.base import BaseVisionService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_FIXED_LABEL = "B1"
_FIXED_CONFIDENCE = 0.95


class MockVisionService(BaseVisionService):
    """학습 모델 미완성 단계용 mock 분류기.

    항상 (label="B1", confidence=0.95) 반환. 시연 흐름 디버깅 + 인프라 검증용.
    EfficientNet-B0 가중치 학습 완료 후 별도 구현 (efficientnet.py) 으로 swap.
    """

    async def classify(self, image_bytes: bytes) -> dict:
        size = len(image_bytes)
        logger.info("mock vision classify image_bytes=%d → %s", size, _FIXED_LABEL)
        return {"label": _FIXED_LABEL, "confidence": _FIXED_CONFIDENCE}
