import asyncio
import time

from app.services.embedding.base import BaseEmbeddingService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BGEM3LocalEmbeddingService(BaseEmbeddingService):
    """BGE-M3 로컬 추론 구현체 —"""

    def __init__(self):
        import torch
        from FlagEmbedding import BGEM3FlagModel

        # GPU 자동 감지 — CUDA 가용 시 GPU + fp16, 아니면 CPU + fp32
        # fp16은 GPU에서만 효과적이며 CPU에서는 오히려 느려질 수 있음
        # FlagEmbedding 1.x 의 devices 파라미터는 multi-process pool 용으로만 적용되며
        # 기본 single-device inference 시 모델이 CPU+float32 로 남는 버그/사양 존재.
        # 따라서 모델 인스턴스의 내부 transformers 모델을 명시적으로 .to(device) + half() 로
        # GPU 와 fp16 으로 강제 이동시킴.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_fp16 = device.startswith("cuda")
        t0 = time.monotonic()
        logger.info("BGE-M3 로컬 모델 로딩 시작 device=%s use_fp16=%s ...", device, use_fp16)
        self._model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=use_fp16,
            devices=[device],
        )
        # 명시적 GPU 이동 — FlagEmbedding 자동 이동 우회
        if device.startswith("cuda"):
            self._model.model.to(device)
            if use_fp16:
                self._model.model.half()
            logger.info("BGE-M3 모델을 GPU+fp16 으로 강제 이동 완료")
        elapsed = time.monotonic() - t0
        logger.info("BGE-M3 로컬 모델 로딩 완료 elapsed=%.2fs", elapsed)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(texts, batch_size=12, max_length=512)
        return output["dense_vecs"].tolist()

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode_sync, [text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts)
