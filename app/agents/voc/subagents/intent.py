from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현 (M2)

logger = get_logger(__name__)


class IntentSubagent:
    async def classify(self, call_id: str, summary: str) -> dict:
        # TODO: GPT-4o-mini 의도 분류 구현
        return {"label": None}
