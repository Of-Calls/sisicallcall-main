import asyncio

from app.agents.voc.subagents.sentiment import SentimentSubagent
from app.agents.voc.subagents.intent import IntentSubagent
from app.agents.voc.subagents.priority import PrioritySubagent
from app.utils.logger import get_logger

# M2 기능 — SUMMARY_READY 이벤트 소비 후 3서브 병렬 실행
# partial_success: 일부 서브 실패 시 나머지 결과로 저장

logger = get_logger(__name__)

_sentiment = SentimentSubagent()
_intent = IntentSubagent()
_priority = PrioritySubagent()


class VOCOrchestrator:
    async def run(self, call_id: str, tenant_id: str, summary: str) -> dict:
        results = await asyncio.gather(
            _sentiment.analyze(call_id, summary),
            _intent.classify(call_id, summary),
            _priority.judge(call_id, summary),
            return_exceptions=True,
        )

        sentiment, intent, priority = results
        partial_success = any(isinstance(r, Exception) for r in results)

        if partial_success:
            logger.warning(f"VOC partial_success call_id={call_id}")

        return {
            "call_id": call_id,
            "sentiment": sentiment if not isinstance(sentiment, Exception) else None,
            "intent": intent if not isinstance(intent, Exception) else None,
            "priority": priority if not isinstance(priority, Exception) else None,
            "partial_success": partial_success,
        }
