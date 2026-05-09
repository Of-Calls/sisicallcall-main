import json

from app.services.llm.base import BaseLLMService
from app.utils.config import settings
from app.utils.logger import get_logger

# 사용처: Task 브랜치 / Summary 비동기 모드 / Post-call analysis_planner_agent
# temperature ≤ 0.2 고정

logger = get_logger(__name__)


class GPT4OService(BaseLLMService):
    MODEL = "gpt-4o"

    def __init__(self):
        from openai import AsyncOpenAI
        from app.services.llm._http import get_openai_http_client
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            http_client=get_openai_http_client(),
        )

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
        messages: list[dict] | None = None,
    ) -> dict:
        """OpenAI Function Calling 응답.

        반환: {"tool_calls": [{"id": str, "name": str, "arguments": dict}, ...],
               "text": str | None,
               "raw_message": dict,
               "usage": {"prompt_tokens": int, "completion_tokens": int,
                         "total_tokens": int, "model": str}}

        post-call analysis_planner_agent / reviewer_agent 가 멀티 tool_call 을
        다루므로 단일 tool_call 만 노출하지 않고 전체 리스트를 반환한다.
        usage 는 OpenAI 응답에서 추출 — 텔레메트리 집계용.
        """
        temperature = min(temperature, 0.2)
        msgs = messages if messages is not None else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        response = await self._client.chat.completions.create(
            model=self.MODEL,
            messages=msgs,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        tool_calls: list[dict] = []
        for call in (message.tool_calls or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {"id": call.id, "name": call.function.name, "arguments": args}
            )
        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            "model": self.MODEL,
        }
        return {
            "tool_calls": tool_calls,
            "text": message.content or "",
            "raw_message": message.model_dump() if hasattr(message, "model_dump") else None,
            "usage": usage,
        }
