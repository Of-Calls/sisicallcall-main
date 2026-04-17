from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

# TODO(대영): ECAPA-TDNN 파인튜닝 완료 후 구현 — architecture.md 참조
# 해제 조건: 파인튜닝 완료 보고 후

logger = get_logger(__name__)


async def speaker_verify_node(state: CallState) -> dict:
    # TODO(대영): ECAPA-TDNN 으로 화자 검증 구현
    return {"is_speaker_verified": True}
