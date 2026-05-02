import json
import time

import redis.asyncio as redis

from app.utils.config import settings

_TTL_SECONDS = 3600  # 1시간 후 세션 자동 만료


class RedisSessionService:
    """대화 세션을 Redis에 저장/조회. 한 통화 = 하나의 키.

    session_view 구조:
        {
            "conversation_history": [
                {"role": "user", "text": "...", "ts": 1234567890},
                {"role": "assistant", "text": "...", "ts": 1234567891},
                ...
            ]
        }
    """

    def __init__(self):
        self._client = redis.from_url(settings.redis_url, decode_responses=True)

    def _key(self, call_id: str) -> str:
        return f"session:{call_id}"

    async def load(self, call_id: str) -> dict:
        """세션 view 로드. 없으면 빈 구조 반환."""
        data = await self._client.get(self._key(call_id))
        if data:
            return json.loads(data)
        return {"conversation_history": []}

    async def append_turn(self, call_id: str, user_text: str, response_text: str) -> None:
        """이번 턴(사용자 발화 + AI 응답) 추가 후 저장."""
        view = await self.load(call_id)
        history = view.setdefault("conversation_history", [])
        ts = time.time()
        history.append({"role": "user", "text": user_text, "ts": ts})
        history.append({"role": "assistant", "text": response_text, "ts": ts})
        await self._client.set(
            self._key(call_id),
            json.dumps(view, ensure_ascii=False),
            ex=_TTL_SECONDS,
        )

    async def clear(self, call_id: str) -> None:
        """통화 종료 시 세션 삭제."""
        await self._client.delete(self._key(call_id))

    async def set_auth_id(self, call_id: str, auth_id: str) -> None:
        """auth_branch 가 SMS 발송 후 통화 세션에 auth_id 기록 — 재진입 시 폴링용."""
        view = await self.load(call_id)
        view["auth_id"] = auth_id
        await self._client.set(
            self._key(call_id),
            json.dumps(view, ensure_ascii=False),
            ex=_TTL_SECONDS,
        )

    async def get_auth_id(self, call_id: str) -> str | None:
        view = await self.load(call_id)
        return view.get("auth_id")
