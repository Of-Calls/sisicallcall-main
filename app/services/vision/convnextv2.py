"""ConvNeXtV2-Femto TorchScript 기반 정수기 모델 분류기.

학습 측에서 timm 으로 학습 후 torch.jit 으로 export. 추론 시 timm 불필요,
PyTorch + Pillow + torchvision (transforms) 만으로 동작.

metadata JSON 에 input_size / normalize_mean / normalize_std / classes 정의.
classes 배열의 인덱스 = 모델 logits 인덱스 (학습 시 ImageFolder 알파벳 순).
"""
import asyncio
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.services.vision.base import BaseVisionService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_device(setting: str) -> str:
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


class ConvNeXtV2VisionService(BaseVisionService):
    def __init__(self) -> None:
        meta_path = Path(settings.vision_metadata_path)
        model_path = Path(settings.vision_model_path)

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self._classes: list[str] = meta["classes"]
        # input_size: [C, H, W] — 보통 [3, 256, 256]
        _, h, w = meta["input_size"]
        mean = meta["normalize_mean"]
        std = meta["normalize_std"]

        self._device = _resolve_device(settings.vision_device)
        self._model = torch.jit.load(str(model_path), map_location=self._device)
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        logger.info(
            "ConvNeXtV2VisionService 로드 완료 device=%s classes=%s input=%dx%d",
            self._device, self._classes, h, w,
        )

    def _classify_sync(self, image_bytes: bytes) -> dict:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        label = self._classes[int(idx.item())]
        confidence = float(conf.item())
        logger.info(
            "vision classify label=%s confidence=%.4f", label, confidence,
        )
        return {"label": label, "confidence": confidence}

    async def classify(self, image_bytes: bytes) -> dict:
        # 동기 추론 (~50ms GPU / ~200ms CPU) 을 별도 스레드로 → 이벤트 루프 차단 방지.
        return await asyncio.to_thread(self._classify_sync, image_bytes)
