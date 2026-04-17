from app.agents.conversational.state import CallState
from app.services.cache.semantic_cache import SemanticCacheService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_cache_service = SemanticCacheService()


async def cache_node(state: CallState) -> dict:
    try:
        result = await _cache_service.lookup(
            text=state["normalized_text"],
            tenant_id=state["tenant_id"],
        )
        if result:
            return {
                "query_embedding": result["embedding"],
                "cache_hit": True,
                "response_text": result["response_text"],
                "response_path": "cache",
            }
        return {
            "query_embedding": result["embedding"] if result else [],
            "cache_hit": False,
        }
    except Exception as e:
        logger.error(f"Cache 조회 실패 call_id={state['call_id']}: {e}")
        return {"query_embedding": [], "cache_hit": False}
