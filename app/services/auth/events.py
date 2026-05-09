"""Auth state change pub/sub — 통화 측 listener 가 자율 발화 트리거 가능.

채널: auth:events:{auth_id}
메시지 형식 (JSON): {"event_type": str, "auth_id": str, "payload": dict}

이벤트 타입 (1단계 face 인증):
  verified     얼굴 인증 통과 → pending_task 자동 재실행
  blocked      여러 번 실패 차단 → 상담원 연결 안내
  face_failed  얼굴 인증 단일 실패 (재시도 가능) → 재시도 안내

publish 는 fire-and-forget. Redis publish 실패해도 통화는 진행.
listener 가 없으면 메시지는 silently drop.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import redis.asyncio as aioredis

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_redis = aioredis.from_url(settings.redis_url, decode_responses=True)


def _channel(auth_id: str) -> str:
    return f"auth:events:{auth_id}"


async def publish_auth_event(
    auth_id: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    """Auth 상태 변화 publish — fire-and-forget."""
    msg = {
        "event_type": event_type,
        "auth_id": auth_id,
        "payload": payload or {},
    }
    try:
        await _redis.publish(_channel(auth_id), json.dumps(msg, ensure_ascii=False))
        logger.info("auth event published auth_id=%s event=%s", auth_id, event_type)
    except Exception as exc:
        logger.warning(
            "auth event publish 실패 auth_id=%s event=%s err=%s",
            auth_id, event_type, exc,
        )


async def subscribe_auth_events(auth_id: str) -> AsyncIterator[dict]:
    """async generator — auth_id 채널 이벤트 yield. break/cancel 시 자동 정리.

    사용 예:
        async for event in subscribe_auth_events(auth_id):
            if event["event_type"] == "verified":
                ...
                break
    """
    pubsub = _redis.pubsub()
    await pubsub.subscribe(_channel(auth_id))
    try:
        async for raw in pubsub.listen():
            if raw.get("type") != "message":
                continue
            try:
                yield json.loads(raw["data"])
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "auth event JSON decode 실패 raw=%r err=%s",
                    raw.get("data"), exc,
                )
    finally:
        try:
            await pubsub.unsubscribe(_channel(auth_id))
            await pubsub.close()
        except Exception as exc:
            logger.warning("pubsub close 실패 auth_id=%s err=%s", auth_id, exc)
