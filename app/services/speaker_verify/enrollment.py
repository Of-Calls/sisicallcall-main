"""Voiceprint enrollment 헬퍼 — call.py 가 graph 진입 전 직접 호출.

STT 성공 발화 PCM 만 누적해 settings.speaker_verify_enrollment_sec 도달 시
voiceprint 등록. 빈 STT (잡음) 오디오는 누적 자체 차단 — voiceprint 오염 방지.

cleanup() 은 call.py 의 stop event 에서 호출 (메모리 해제).
"""
from app.services.speaker_verify.titanet_onnx import get_speaker_verify_service
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PCM_BYTES_PER_SEC = 16000 * 2  # 16kHz 16-bit mono

# per-call enrollment 누적 buffer
_buffers: dict[str, bytearray] = {}


async def accumulate(call_id: str, pcm_16k: bytes, transcript: str) -> bool:
    """STT 성공 발화 누적 후 임계 도달 시 voiceprint 등록.

    Returns:
        True: 등록 완료 상태 (이번 호출 또는 이전). False: 미완료/STT 빈값.
    """
    if not transcript:
        return False  # 잡음 차단 — voiceprint 누적 거부

    svc = get_speaker_verify_service()
    if svc.is_enrolled(call_id):
        return True  # 이미 등록됨

    target = int(settings.speaker_verify_enrollment_sec * _PCM_BYTES_PER_SEC)
    buf = _buffers.setdefault(call_id, bytearray())
    buf.extend(pcm_16k)

    logger.debug(
        "[SpeakerVerify] enrollment %d/%d bytes call_id=%s",
        len(buf), target, call_id,
    )

    if len(buf) >= target:
        audio = bytes(_buffers.pop(call_id))
        await svc.extract_and_store(audio, call_id)
        return svc.is_enrolled(call_id)

    return False


def cleanup(call_id: str) -> None:
    """통화 종료 시 누적 buffer 메모리 해제."""
    if _buffers.pop(call_id, None) is not None:
        logger.info("[SpeakerVerify] enrollment buffer removed call_id=%s", call_id)
