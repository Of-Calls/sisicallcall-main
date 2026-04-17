from app.agents.summary.base import BaseSummaryAgent
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 호출 주체: app/workers/summary_worker.py (call_ended 이벤트 소비)
# 모델: GPT-4o | 타임아웃: 없음
# 출력: summary_long (500~1000자, 구조화 JSON) → call_summaries 테이블 UPDATE

logger = get_logger(__name__)


class AsyncSummaryAgent(BaseSummaryAgent):
    async def run(self, call_id: str, tenant_id: str) -> dict:
        # TODO: GPT-4o 로 심화 요약 생성 → call_summaries UPDATE → SUMMARY_READY 이벤트 발행
        return {"summary_long": ""}
