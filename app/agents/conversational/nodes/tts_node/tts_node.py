from app.agents.conversational.state import CallState
from app.services.tts.base import BaseTTSService
from app.utils.logger import get_logger

logger = get_logger(__name__)

_tts_service: BaseTTSService | None = None


def _get_tts_service() -> BaseTTSService:
    global _tts_service
    if _tts_service is None:
        from app.services.tts.google import GoogleTTSService
        _tts_service = GoogleTTSService()
    return _tts_service


async def tts_node(state: CallState) -> dict:
    try:
        await _get_tts_service().synthesize_and_stream(state["response_text"])
    except Exception as e:
        logger.error(f"TTS 실패 call_id={state['call_id']}: {e}")
    return {"is_timeout": state.get("is_timeout", False)}
