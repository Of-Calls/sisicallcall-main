import json

from app.services.llm.base import BaseLLMService
from app.utils.config import settings
from app.utils.logger import get_logger

# 공통 사용처: Intent Router / FAQ / Auth / Reviewer / Summary 동기 / VOC 서브
# 수정 시 전체 영향 확인 후 팀장 승인 필수
# temperature ≤ 0.2 고정

logger = get_logger(__name__)


class GPT4OMiniService(BaseLLMService):
    MODEL = "gpt-4o-mini"

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

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
        tool_choice: str = "auto",
    ) -> dict:
        """OpenAI Function Calling 응답.

        반환:
            {"tool_call": {"name": str, "arguments": dict} | None,
             "text": str | None}

        tool_choice:
        - "auto" (기본): LLM 이 도구 호출 vs 텍스트 응답 자체 결정.
        - "required": 도구 호출 강제 — 인자 부족해도 호출. caller 가 게이트로 처리.
        - "none": 도구 호출 금지.
        """
        temperature = min(temperature, 0.2)
        response = await self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            return {
                "tool_call": {"name": call.function.name, "arguments": arguments},
                "text": None,
            }
        return {"tool_call": None, "text": message.content or ""}
