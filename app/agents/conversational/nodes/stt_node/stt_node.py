from app.agents.conversational.state import CallState
from app.services.stt.base import BaseSTTService
from app.services.stt.deepgram import DeepgramSTTService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_stt_service: BaseSTTService = DeepgramSTTService()


async def stt_node(state: CallState) -> dict:
    try:
        transcript = await _stt_service.transcribe(state["audio_chunk"])
        return {"raw_transcript": transcript}
    except Exception as e:
        logger.error(f"STT 실패 call_id={state['call_id']}: {e}")
        return {"raw_transcript": "", "error": str(e)}
