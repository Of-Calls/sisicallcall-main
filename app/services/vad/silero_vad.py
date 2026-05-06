import numpy as np
import torch

from app.services.vad.base import BaseVADService

# Silero VAD 16kHz 모드 — 정확히 512 samples (1024 bytes linear16) 필요
_FRAME_SAMPLES = 512
_FRAME_BYTES = 1024


class SileroVADService(BaseVADService):

    def __init__(self, threshold: float = 0.5):
        model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self._model = model
        self._model.eval()
        self._threshold = threshold

    async def detect(self, audio_chunk: bytes) -> bool:
        """linear16 16kHz, 정확히 1024 bytes (512 samples) 청크 → speech 여부."""
        if len(audio_chunk) != _FRAME_BYTES:
            raise ValueError(
                f"VAD 프레임 크기 불일치: {len(audio_chunk)} bytes (필요: {_FRAME_BYTES})"
            )

        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)

        with torch.no_grad():
            confidence = self._model(tensor, 16000).item()

        return confidence > self._threshold
