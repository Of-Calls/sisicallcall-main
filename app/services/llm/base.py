from abc import ABC, abstractmethod


class BaseLLMService(ABC):
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """LLM 호출 — temperature ≤ 0.2 고정 권장."""
        raise NotImplementedError
