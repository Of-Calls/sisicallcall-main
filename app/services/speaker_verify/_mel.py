"""TitaNet-L ONNX 입력 log-mel 전처리.

ONNX 도착 후 input spec (n_mels / hop / window 등) 보고 미세 조정.
TitaNet 기본 spec: 16kHz, n_fft=512, hop=160 (10ms), n_mels=80, log scale.

torchaudio MelSpectrogram 한 번만 생성해 재사용 (모듈 전역).
"""
import numpy as np
import torch
import torchaudio


_SAMPLE_RATE = 16000
_N_FFT = 512
_HOP = 160
_N_MELS = 80


_mel_transform: torchaudio.transforms.MelSpectrogram | None = None


def _get_mel_transform() -> torchaudio.transforms.MelSpectrogram:
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=_SAMPLE_RATE,
            n_fft=_N_FFT,
            hop_length=_HOP,
            n_mels=_N_MELS,
        )
    return _mel_transform


def pcm16_to_logmel(pcm_16k: bytes) -> np.ndarray:
    """16kHz mono PCM16 bytes → log-mel spectrogram (1, n_mels, T) float32."""
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, num_samples)
    mel = _get_mel_transform()(waveform)               # (1, n_mels, T)
    log_mel = torch.log(mel + 1e-6)
    return log_mel.numpy().astype(np.float32)
