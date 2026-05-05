"""TitaNet-L ONNX 입력 log-mel 전처리.

학습 시점 NeMo `AudioToMelSpectrogramPreprocessor` 와 동일한 spec 으로 맞춤:
    sr=16000, n_fft=512, win=400, hop=160, n_mels=80, fmin=0, fmax=8000, power=2.0
    log clamp + per-feature normalize (시간축 mean/std)

per-feature normalize 가 없으면 매 발화의 볼륨/노이즈 차이가 임베딩에 그대로 흡수되어
실 통화에서 sim 분산이 폭증함 (NeMo `normalize: per_feature` 동치).
"""
import numpy as np
import torch
import torchaudio


_SAMPLE_RATE = 16000
_N_FFT = 512
_WIN = 400
_HOP = 160
_N_MELS = 80
_F_MIN = 0.0
_F_MAX = 8000.0


_mel_transform: torchaudio.transforms.MelSpectrogram | None = None


def _get_mel_transform() -> torchaudio.transforms.MelSpectrogram:
    global _mel_transform
    if _mel_transform is None:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=_SAMPLE_RATE,
            n_fft=_N_FFT,
            win_length=_WIN,
            hop_length=_HOP,
            n_mels=_N_MELS,
            f_min=_F_MIN,
            f_max=_F_MAX,
            power=2.0,
            center=True,
        )
    return _mel_transform


def pcm16_to_logmel(pcm_16k: bytes) -> np.ndarray:
    """16kHz mono PCM16 bytes → log-mel + per-feature normalized (1, n_mels, T) float32."""
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, num_samples)
    mel = _get_mel_transform()(waveform)               # (1, n_mels, T)
    log_mel = torch.log(mel.clamp(min=1e-10))
    # per-feature normalization (시간축 기준) — NeMo `normalize: per_feature`
    mean = log_mel.mean(dim=2, keepdim=True)
    std = log_mel.std(dim=2, keepdim=True).clamp(min=1e-5)
    log_mel = (log_mel - mean) / std
    return log_mel.numpy().astype(np.float32)
