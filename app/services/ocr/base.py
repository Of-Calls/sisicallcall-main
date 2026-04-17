from abc import ABC, abstractmethod


class BaseOCRService(ABC):
    @abstractmethod
    async def extract_text(self, image_bytes: bytes) -> str:
        """이미지에서 텍스트 추출 (제품 라벨 / 신분증 공통 사용)."""
        raise NotImplementedError
