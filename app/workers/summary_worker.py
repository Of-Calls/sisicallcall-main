from app.agents.summary.async_mode import AsyncSummaryAgent
from app.core.events import CALL_ENDED
from app.utils.logger import get_logger

# CALL_ENDED 이벤트 소비 → AsyncSummaryAgent.run() 호출
# Celery 또는 Redis Stream 기반 — 택일 미확정 (architecture.md §7.1 참조)

logger = get_logger(__name__)

_agent = AsyncSummaryAgent()


async def handle_call_ended(call_id: str, tenant_id: str) -> None:
    """CALL_ENDED 이벤트 핸들러 — 비동기 Summary 실행."""
    logger.info(f"[{CALL_ENDED}] summary 시작 call_id={call_id}")
    try:
        result = await _agent.run(call_id=call_id, tenant_id=tenant_id)
        logger.info(f"summary 완료 call_id={call_id}")
        return result
    except Exception as e:
        logger.error(f"summary 실패 call_id={call_id}: {e}")
        raise
