from abc import ABC, abstractmethod


class BaseAuthService(ABC):
    @abstractmethod
    async def verify_face(self, image_bytes: bytes, tenant_id: str, user_id: str) -> bool:
        """얼굴 인증 — ArcFace + MediaPipe 기반."""
        raise NotImplementedError

    @abstractmethod
    async def register_face(self, image_bytes: bytes, tenant_id: str, user_id: str) -> None:
        """얼굴 임베딩 등록."""
        raise NotImplementedError
