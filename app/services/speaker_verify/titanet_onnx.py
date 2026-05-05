"""TitaNet-L ONNX 화자검증 서비스.

per-call voiceprint dict — 통화 시작 후 첫 N초 누적 → 등록.
이후 모든 turn 에 대해 cosine similarity 비교.

ONNX I/O spec (titanet_large.onnx):
    inputs : audio_signal [batch, 80, T] float32 (log-mel),
             length       [batch]         int64
    outputs: logits [batch, 16681] (학습 시 분류용 — 무시),
             embs   [batch, 192]   (화자 임베딩 — 사용)

enabled=False or 모델 파일 미존재 → disabled 모드 (모든 verify bypass).
"""
import asyncio
from pathlib import Path

import numpy as np

from app.services.speaker_verify.base import BaseSpeakerVerifyService
from app.services.speaker_verify._mel import pcm16_to_logmel
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TitaNetOnnxService(BaseSpeakerVerifyService):
    def __init__(self) -> None:
        self._voiceprints: dict[str, np.ndarray] = {}
        self._session = None  # onnxruntime.InferenceSession | None

    async def warmup(self) -> None:
        """ONNX 세션 로드. enabled=False or 모델 파일 미존재 → disabled 모드."""
        if not settings.speaker_verify_enabled:
            logger.info("[SpeakerVerify] disabled by config — skip warmup")
            return

        model_path = Path(settings.speaker_verify_model_path)
        if not model_path.exists():
            logger.warning(
                "[SpeakerVerify] model not found at %s — disabled mode (verify bypass)",
                model_path,
            )
            return

        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            logger.info(
                "[SpeakerVerify] ONNX session ready provider=%s",
                self._session.get_providers()[0],
            )
        except Exception as e:
            logger.error("[SpeakerVerify] warmup failed: %s — disabled mode", e)
            self._session = None

    def _run_onnx_sync(self, pcm_16k: bytes) -> np.ndarray:
        """PCM16 → log-mel → ONNX → L2-normalized embedding (192-d).

        입력 길이 cap (settings.speaker_verify_enrollment_sec, 기본 3초):
            - TitaNet-L ONNX encoder 내부 max frame ≈1200 (≈12초). 초과 시 broadcast 충돌.
            - 화자검증은 1.5~3초로 충분 — 더 길면 잡음/잔향 누적으로 임베딩 분산 증가.
            - enrollment 와 verify 가 동일 길이 입력 → 임베딩 분포 안정.
        """
        assert self._session is not None
        max_bytes = int(settings.speaker_verify_enrollment_sec * 16000 * 2)
        if len(pcm_16k) > max_bytes:
            pcm_16k = pcm_16k[:max_bytes]
        log_mel = pcm16_to_logmel(pcm_16k)              # (1, 80, T) float32
        length = np.array([log_mel.shape[2]], dtype=np.int64)
        inputs = {"audio_signal": log_mel, "length": length}
        outputs = self._session.run(["embs"], inputs)   # embs only — logits 무시
        emb = np.asarray(outputs[0]).squeeze()          # (192,)
        norm = np.linalg.norm(emb) + 1e-9
        return (emb / norm).astype(np.float32)

    async def _embed(self, pcm_16k: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_onnx_sync, pcm_16k)

    async def extract_and_store(self, pcm_16k: bytes, call_id: str) -> None:
        if self._session is None:
            return  # disabled — 무동작
        try:
            emb = await self._embed(pcm_16k)
            self._voiceprints[call_id] = emb
            logger.info(
                "[SpeakerVerify] enrolled call_id=%s dim=%d", call_id, emb.shape[0],
            )
        except Exception as e:
            logger.error("[SpeakerVerify] enrollment failed call_id=%s: %s", call_id, e)

    async def verify(self, pcm_16k: bytes, call_id: str) -> tuple[bool, float]:
        if self._session is None or call_id not in self._voiceprints:
            return True, 1.0  # bypass (disabled or pre-enrollment)
        try:
            emb = await self._embed(pcm_16k)
            ref = self._voiceprints[call_id]
            sim = float(np.dot(emb, ref))  # 둘 다 L2 normalized → dot = cosine
            verified = sim >= settings.speaker_verify_threshold
            logger.info(
                "[SpeakerVerify] call_id=%s sim=%.4f thr=%.2f verified=%s",
                call_id, sim, settings.speaker_verify_threshold, verified,
            )
            return verified, sim
        except Exception as e:
            logger.error("[SpeakerVerify] verify failed call_id=%s: %s", call_id, e)
            return True, 0.0  # 실패 시 bypass — 사용자 경험 우선

    def is_enrolled(self, call_id: str) -> bool:
        return call_id in self._voiceprints

    def cleanup(self, call_id: str) -> None:
        if self._voiceprints.pop(call_id, None) is not None:
            logger.info("[SpeakerVerify] voiceprint removed call_id=%s", call_id)


_singleton: TitaNetOnnxService | None = None


def get_speaker_verify_service() -> TitaNetOnnxService:
    """call.py / enrollment 헬퍼가 공유하는 모듈 레벨 싱글톤."""
    global _singleton
    if _singleton is None:
        _singleton = TitaNetOnnxService()
    return _singleton
