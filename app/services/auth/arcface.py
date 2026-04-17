from app.services.auth.base import BaseAuthService
from app.utils.logger import get_logger

# TODO(희원): 구현 방식 확정 후 구현
# 해제 조건: 얼굴 인증 구현 방식 확정 후

logger = get_logger(__name__)


class ArcFaceAuthService(BaseAuthService):
    async def verify_face(self, image_bytes: bytes, tenant_id: str, user_id: str) -> bool:
        # TODO(희원): ArcFace + MediaPipe 얼굴 인증 구현
        raise NotImplementedError

    async def register_face(self, image_bytes: bytes, tenant_id: str, user_id: str) -> None:
        # TODO(희원): 얼굴 임베딩 추출 후 face_embeddings 테이블 저장
        raise NotImplementedError
