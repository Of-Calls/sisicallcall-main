from abc import ABC, abstractmethod


class BaseVisionService(ABC):
    @abstractmethod
    async def classify(self, image_bytes: bytes) -> dict:
        """이미지 분류 — 제품 라벨 등 인식 결과 반환."""
        raise NotImplementedError
