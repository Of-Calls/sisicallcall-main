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

        반환:
            {"tool_call": {"name": str, "arguments": dict} | None,
             "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...],
             "text": str | None,
             "raw_message": dict | None}

        tool_choice:
        - "auto" (기본): LLM 이 도구 호출 vs 텍스트 응답 자체 결정.
        - "required": 도구 호출 강제 — 인자 부족해도 호출. caller 가 게이트로 처리.
        - "none": 도구 호출 금지.

        messages 가 주어지면 system/user 대신 그 리스트를 그대로 전달한다 —
        reviewer_agent 처럼 multi-turn ReAct 루프에서 누적 메시지를 전달할 때 사용.
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
        all_calls: list[dict] = []
        for call in (message.tool_calls or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            all_calls.append(
                {"id": call.id, "name": call.function.name, "arguments": args}
            )
        first = all_calls[0] if all_calls else None
        usage_obj = getattr(response, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            "model": self.MODEL,
        }
        return {
            "tool_call": (
                {"name": first["name"], "arguments": first["arguments"]} if first else None
            ),
            "tool_calls": all_calls,
            "text": (None if all_calls else (message.content or "")),
            "raw_message": message.model_dump() if hasattr(message, "model_dump") else None,
            "usage": usage,
        }
