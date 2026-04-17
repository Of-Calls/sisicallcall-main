from app.agents.voc.orchestrator import VOCOrchestrator
from app.core.events import SUMMARY_READY
from app.utils.logger import get_logger

# SUMMARY_READY 이벤트 소비 → VOCOrchestrator.run() 호출 (M2)

logger = get_logger(__name__)

_orchestrator = VOCOrchestrator()


async def handle_summary_ready(call_id: str, tenant_id: str, summary: str) -> None:
    """SUMMARY_READY 이벤트 핸들러 — VOC 3서브 병렬 실행."""
    logger.info(f"[{SUMMARY_READY}] VOC 시작 call_id={call_id}")
    try:
        result = await _orchestrator.run(
            call_id=call_id, tenant_id=tenant_id, summary=summary
        )
        logger.info(f"VOC 완료 call_id={call_id} partial={result['partial_success']}")
        return result
    except Exception as e:
        logger.error(f"VOC 실패 call_id={call_id}: {e}")
        raise
