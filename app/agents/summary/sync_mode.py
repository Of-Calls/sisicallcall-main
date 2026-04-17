from app.agents.summary.base import BaseSummaryAgent
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 호출 주체: escalation_branch_node (immediate 확정 시 직접 호출)
# 모델: GPT-4o-mini | 타임아웃: 3초 하드컷
# 출력: summary_short (200자 이내, 상담원 인수인계용)

logger = get_logger(__name__)


class SyncSummaryAgent(BaseSummaryAgent):
    async def run(self, call_id: str, tenant_id: str) -> dict:
        # TODO: GPT-4o-mini 로 통화 요약 생성 (3초 하드컷)
        # Redis summary:sync:{call_id} 에 저장
        return {"summary_short": ""}
