from app.services.llm.base import BaseLLMService
from app.utils.config import settings
from app.utils.logger import get_logger

# 사용처: Task 브랜치 / Summary 비동기 모드
# temperature ≤ 0.2 고정

logger = get_logger(__name__)


class GPT4OService(BaseLLMService):
    MODEL = "gpt-4o"

    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        temperature = min(temperature, 0.2)
        response = await self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
