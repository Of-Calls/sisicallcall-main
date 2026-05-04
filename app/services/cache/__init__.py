from app.services.cache.base import BaseCacheService, CacheHit
from app.services.cache.chroma_cache import ChromaCacheService

_cache: BaseCacheService | None = None


def get_cache() -> BaseCacheService:
    """ChromaCacheService lazy singleton — get_embedder 와 동일 패턴."""
    global _cache
    if _cache is None:
        _cache = ChromaCacheService()
    return _cache
