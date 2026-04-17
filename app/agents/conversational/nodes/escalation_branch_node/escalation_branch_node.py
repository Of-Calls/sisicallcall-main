from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(미배정): 담당자 지정 + agents.md 작성 후 구현
# 해제 조건: 담당자 지정 후
# LLM 없음 — 운영시간·상담원 가용성·턴 카운트 기반 결정적 분기
# Escalation 3분할: immediate / callback / offhours

logger = get_logger(__name__)


async def escalation_branch_node(state: CallState) -> dict:
    # TODO: 운영시간 조회 → immediate/callback/offhours 분기
    # immediate 확정 시 Summary 에이전트 동기 모드 직접 호출
    return {
        "response_text": "상담원에게 연결해 드리겠습니다. 잠시만 기다려 주세요.",
        "response_path": "escalation",
    }
