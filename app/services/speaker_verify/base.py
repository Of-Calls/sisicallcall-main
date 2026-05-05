from abc import ABC, abstractmethod


class BaseSpeakerVerifyService(ABC):
    """화자검증 Provider 인터페이스.

    동작 원칙:
        - voiceprint 미등록 또는 disabled 상태에서 verify 는 항상 (True, 1.0) bypass.
        - per-call dict 로 voiceprint 관리. cleanup 으로 메모리 해제.
    """

    @abstractmethod
    async def warmup(self) -> None:
        """ONNX 세션 미리 로드 — main.py lifespan 에서 1회 호출."""
        raise NotImplementedError

    @abstractmethod
    async def verify(self, pcm_16k: bytes, call_id: str) -> tuple[bool, float]:
        """화자검증. (verified, cosine_similarity) 반환.

        voiceprint 없거나 disabled → (True, 1.0) bypass.
        """
        raise NotImplementedError

    @abstractmethod
    async def extract_and_store(self, pcm_16k: bytes, call_id: str) -> None:
        """voiceprint 등록 — enrollment 헬퍼가 임계 도달 시 호출."""
        raise NotImplementedError

    @abstractmethod
    def is_enrolled(self, call_id: str) -> bool:
        """call_id 의 voiceprint 가 등록됐는지."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, call_id: str) -> None:
        """통화 종료 시 voiceprint 메모리 해제."""
        raise NotImplementedError
