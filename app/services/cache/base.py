from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CacheHit:
    response_text: str
    distance: float
    cached_at: float  # unix ts (디버깅용)


class BaseCacheService(ABC):
    @abstractmethod
    async def lookup(
        self, tenant_id: str, query_embedding: list[float]
    ) -> CacheHit | None:
        """캐시 조회 — distance threshold 이내 + 미만료 entry 반환. 없으면 None."""
        raise NotImplementedError

    @abstractmethod
    async def save(
        self,
        tenant_id: str,
        query_text: str,
        query_embedding: list[float],
        response_text: str,
    ) -> None:
        """캐시 저장 — TTL 메타 자동 계산."""
        raise NotImplementedError

    @abstractmethod
    async def clear(self, tenant_id: str) -> None:
        """테스트/관리용 — 매장 전체 캐시 비우기."""
        raise NotImplementedError
