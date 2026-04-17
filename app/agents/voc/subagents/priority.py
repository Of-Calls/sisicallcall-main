from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현 (M2)

logger = get_logger(__name__)


class PrioritySubagent:
    async def judge(self, call_id: str, summary: str) -> dict:
        # TODO: GPT-4o-mini 우선순위 판정 구현
        return {"level": None}
