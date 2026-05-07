"""TTS filler audio prewarm — STT 직후 즉시 송출용 짧은 멘트.

Latency 옵션 A: graph + 본 응답 TTS 동안 사용자 무음 → filler 음성으로 갭 채움.
startup 1회 합성, 메모리 dict 재사용 (cold start 회피).
"""
import random

from app.services.tts.base import BaseTTSService
from app.utils.logger import get_logger

_logger = get_logger(__name__)

_FILLER_TEXTS = [
    "잠시만요",
    "확인해드릴게요",
]
# 2단계 filler — graph 가 filler 1 끝나고도 진행 중일 때 추가 송출.
# 사용자 silence 분산으로 체감 latency ↓ (콜센터 상담원 자연스러움).
_FILLER_CONTINUATION_TEXTS = [
    "확인 중이에요",
    "잠시만 기다려주세요",
]
_filler_audios: list[bytes] = []
_filler_continuation_audios: list[bytes] = []


async def prewarm_fillers(tts: BaseTTSService) -> None:
    """startup 1회 — 1단계 + 2단계 filler 각각 mulaw 8kHz 합성하여 module 캐시.

    실패해도 startup 진행 (filler 없이 운용 가능). pick_filler*() 가 빈 cache 면 None 반환.
    """
    global _filler_audios, _filler_continuation_audios

    async def _synth_all(texts: list[str]) -> list[bytes]:
        audios: list[bytes] = []
        for text in texts:
            try:
                audio = await tts.synthesize(text)
                if audio:
                    audios.append(audio)
            except Exception as exc:
                _logger.warning("filler prewarm 실패 text=%r: %s", text, exc)
        return audios

    _filler_audios = await _synth_all(_FILLER_TEXTS)
    _filler_continuation_audios = await _synth_all(_FILLER_CONTINUATION_TEXTS)
    _logger.info(
        "filler ready primary=%d/%d continuation=%d/%d",
        len(_filler_audios), len(_FILLER_TEXTS),
        len(_filler_continuation_audios), len(_FILLER_CONTINUATION_TEXTS),
    )


def pick_filler() -> bytes | None:
    """1단계 filler — 랜덤 1개 mulaw 8kHz 음성 반환. 빈 cache 면 None."""
    if not _filler_audios:
        return None
    return random.choice(_filler_audios)


def pick_filler_continuation() -> bytes | None:
    """2단계 filler — graph 가 filler 1 끝나고도 진행 중일 때 호출. 빈 cache 면 None."""
    if not _filler_continuation_audios:
        return None
    return random.choice(_filler_continuation_audios)
