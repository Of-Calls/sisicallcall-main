from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + R-09 결과 + agents.md 작성 후 구현
# 해제 조건: R-09 연구 결과 + 담당자 지정 후
# 타임아웃: 1.5초 하드컷 → 원본 응답 유지, reviewer_verdict="pass" 강제

logger = get_logger(__name__)


async def reviewer_node(state: CallState) -> dict:
    # TODO: GPT-4o-mini 로 고위험 응답 검토 구현
    return {
        "reviewer_applied": True,
        "reviewer_verdict": "pass",
    }
