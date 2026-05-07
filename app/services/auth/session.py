import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_AUTH_SESSION_TTL = 600  # 10분


def _key(auth_id: str) -> str:
    return f"auth:session:{auth_id}"


class AuthSessionService:
    def __init__(self) -> None:
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def create_session(
        self,
        *,
        tenant_id: str,
        customer_ref: str,
        customer_phone: str,
        call_id: str,
    ) -> str:
        auth_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._redis.hset(_key(auth_id), mapping={
            "auth_id": auth_id,
            "tenant_id": tenant_id,
            "customer_ref": customer_ref,
            "customer_phone": customer_phone,
            "call_id": call_id,
            "status": "pending",
            "liveness_passed": "true",  # liveness 단계 미구현 — 항상 통과
            "face_verified": "false",
            "face_attempts": "0",
            "created_at": now,
        })
        await self._redis.expire(_key(auth_id), _AUTH_SESSION_TTL)
        logger.info("auth session 생성 auth_id=%s tenant=%s", auth_id, tenant_id)
        return auth_id

    async def get_session(self, auth_id: str) -> dict | None:
        data = await self._redis.hgetall(_key(auth_id))
        return data if data else None

    async def update_status(self, auth_id: str, status: str) -> None:
        await self._redis.hset(_key(auth_id), "status", status)

    async def set_liveness_passed(self, auth_id: str) -> None:
        await self._redis.hset(_key(auth_id), mapping={
            "liveness_passed": "true",
            "status": "liveness_passed",
        })

    async def increment_face_attempts(self, auth_id: str) -> int:
        return await self._redis.hincrby(_key(auth_id), "face_attempts", 1)

    async def set_face_verified(self, auth_id: str) -> None:
        await self._redis.hset(_key(auth_id), mapping={
            "face_verified": "true",
            "status": "verified",
        })

    async def set_blocked(self, auth_id: str) -> None:
        await self._redis.hset(_key(auth_id), "status", "blocked")
