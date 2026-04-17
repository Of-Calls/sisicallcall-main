import audioop

from app.utils.logger import get_logger

logger = get_logger(__name__)

_RESAMPLE_STATE = None


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Twilio μ-law 8kHz 오디오를 PCM 16kHz 16-bit로 변환."""
    global _RESAMPLE_STATE
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_16k, _RESAMPLE_STATE = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, _RESAMPLE_STATE)
    return pcm_16k


def reset_resample_state() -> None:
    """통화 종료 시 리샘플 상태 초기화."""
    global _RESAMPLE_STATE
    _RESAMPLE_STATE = None
