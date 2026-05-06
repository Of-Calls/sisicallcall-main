import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_VISION_SESSION_TTL = 600  # 10분


def _key(vision_id: str) -> str:
    return f"vision:session:{vision_id}"


class VisionSessionService:
    """vision 세션 Redis 저장소.

    status flow: pending → analyzing → analyzed | failed
    분석 결과 (label, confidence) 는 analyzed 시 저장.
    """

    def __init__(self) -> None:
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def create_session(
        self,
        *,
        tenant_id: str,
        customer_phone: str,
        call_id: str,
    ) -> str:
        vision_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._redis.hset(_key(vision_id), mapping={
            "vision_id": vision_id,
            "tenant_id": tenant_id,
            "customer_phone": customer_phone,
            "call_id": call_id,
            "status": "pending",
            "label": "",
            "confidence": "",
            "created_at": now,
        })
        await self._redis.expire(_key(vision_id), _VISION_SESSION_TTL)
        logger.info("vision session 생성 vision_id=%s tenant=%s", vision_id, tenant_id)
        return vision_id

    async def get_session(self, vision_id: str) -> dict | None:
        data = await self._redis.hgetall(_key(vision_id))
        return data if data else None

    async def set_analyzing(self, vision_id: str) -> None:
        await self._redis.hset(_key(vision_id), "status", "analyzing")

    async def set_analyzed(
        self, vision_id: str, label: str, confidence: float
    ) -> None:
        await self._redis.hset(_key(vision_id), mapping={
            "status": "analyzed",
            "label": label,
            "confidence": f"{confidence:.4f}",
        })

    async def set_failed(self, vision_id: str, reason: str = "") -> None:
        mapping = {"status": "failed"}
        if reason:
            mapping["fail_reason"] = reason
        await self._redis.hset(_key(vision_id), mapping=mapping)
