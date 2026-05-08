"""
MCPProtocolClient — 진짜 MCP protocol client (stdio transport).

stdio_client + ClientSession 으로 MCP Server process 를 띄우고
list_tools / call_tool 을 호출한다.

특징:
  - 매 호출마다 server process 를 새로 띄우는 ``one-shot`` 모드와
    여러 호출을 같은 process 에서 처리하는 ``persistent`` 모드 지원.
  - timeout 처리 (MCP_CLIENT_TIMEOUT_SEC, 기본 30s)
  - transport 오류와 tool 결과 오류를 명확히 구분 (transport 오류 → MCPClientTransportError)
  - tool 결과는 가능한 dict 로 정규화

Post-call 흐름은 한 번에 액션 여러 개를 처리하므로 기본은 ``persistent``.
``MCPGatewayConnector`` 가 process lifecycle 을 관리한다.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from app.services.mcp.protocol_client.transport import build_stdio_params
from app.utils.logger import get_logger

logger = get_logger(__name__)


_DEFAULT_TIMEOUT_SEC = 30.0


def _client_timeout_sec() -> float:
    raw = os.getenv("MCP_CLIENT_TIMEOUT_SEC", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SEC
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TIMEOUT_SEC


class MCPClientTransportError(RuntimeError):
    """MCP server process / stdio transport 가 실패했을 때만 발생.

    Tool 실행 결과 (status=failed) 는 정상 응답이므로 이 예외를 쓰지 않는다.
    MCP-only ActionExecutor 는 이 예외를 잡아 status=failed +
    error=mcp_transport_failed:* 로 변환하며, direct fallback 은 하지 않는다.
    """


def _normalize_tool_result(call_result) -> dict[str, Any]:
    """CallToolResult 를 가능한 dict 로 정규화한다.

    FastMCP 의 tool 가 dict 를 반환하면, MCP는 보통 structured_output 에
    그대로 dict 를 담아 보낸다 (그리고 fallback 으로 content[0] 에 JSON
    문자열도 함께 넣는다). 둘 다 시도해서 dict 로 바꾼다.
    """
    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP 가 dict 반환을 wrapping 한다면 ``result`` 키에 들어 있을 수
        # 있다 — 그렇지 않으면 그대로 사용.
        if set(structured.keys()) >= {"action_type", "tool", "status"}:
            return structured
        if "result" in structured and isinstance(structured["result"], dict):
            inner = structured["result"]
            if {"action_type", "tool", "status"}.issubset(inner.keys()):
                return inner
        return structured

    content = getattr(call_result, "content", None) or []
    for item in content:
        text = getattr(item, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}

    is_err = bool(getattr(call_result, "isError", False))
    return {
        "raw": [getattr(c, "text", str(c)) for c in content],
        "is_error": is_err,
    }


class MCPProtocolClient:
    """진짜 MCP Client — stdio transport.

    사용 패턴:
        async with MCPProtocolClient() as cli:
            tools = await cli.list_tools()
            result = await cli.call_tool("slack.send_slack_alert", payload)

    또는 일회성:
        result = await MCPProtocolClient.one_shot_call_tool(name, payload)
    """

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        extra_env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self._command = command
        self._args = args
        self._cwd = cwd
        self._extra_env = extra_env
        self._timeout_sec = timeout_sec or _client_timeout_sec()

        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "MCPProtocolClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is not None:
            return
        params = build_stdio_params(
            command=self._command,
            args=self._args,
            cwd=self._cwd,
            extra_env=self._extra_env,
        )
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=self._timeout_sec)
        except asyncio.TimeoutError:
            await stack.aclose()
            raise MCPClientTransportError(
                f"mcp_initialize_timeout:{self._timeout_sec}s"
            )
        except Exception as exc:
            await stack.aclose()
            raise MCPClientTransportError(
                f"mcp_transport_failed:{type(exc).__name__}:{exc}"
            ) from exc
        self._stack = stack
        self._session = session

    async def close(self) -> None:
        if self._stack is None:
            return
        try:
            await self._stack.aclose()
        except Exception as exc:
            logger.warning("mcp client close 중 예외 무시: %s", type(exc).__name__)
        finally:
            self._stack = None
            self._session = None

    # ── operations ───────────────────────────────────────────────────────────

    async def list_tools(self) -> list[dict[str, Any]]:
        """MCP Server 의 tool 목록을 ``[{"name", "description"}, ...]`` 으로 반환."""
        if self._session is None:
            await self.start()
        assert self._session is not None
        try:
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError:
            raise MCPClientTransportError(
                f"mcp_list_tools_timeout:{self._timeout_sec}s"
            )

        tools: list[dict[str, Any]] = []
        for t in getattr(result, "tools", []) or []:
            tools.append({
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", "") or "",
            })
        return tools

    async def call_tool(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """tool 을 호출하고 dict result 를 반환한다.

        transport / 시간 초과 → MCPClientTransportError 발생.
        Tool 이 status=failed dict 를 돌려준 경우는 정상 반환 (예외 아님).
        """
        if self._session is None:
            await self.start()
        assert self._session is not None
        try:
            call_result = await asyncio.wait_for(
                self._session.call_tool(
                    name,
                    payload or {},
                    read_timeout_seconds=timedelta(seconds=self._timeout_sec),
                ),
                timeout=self._timeout_sec + 1,
            )
        except asyncio.TimeoutError:
            raise MCPClientTransportError(
                f"mcp_call_tool_timeout:{self._timeout_sec}s tool={name}"
            )
        except Exception as exc:
            # call_tool 자체가 raise 했다면 transport 또는 server 내부 에러.
            raise MCPClientTransportError(
                f"mcp_call_tool_failed:{type(exc).__name__}:{exc}"
            ) from exc

        return _normalize_tool_result(call_result)

    # ── 일회성 호출 ──────────────────────────────────────────────────────────

    @classmethod
    async def one_shot_call_tool(
        cls,
        name: str,
        payload: dict[str, Any] | None = None,
        *,
        command: str | None = None,
        args: list[str] | None = None,
    ) -> dict[str, Any]:
        async with cls(command=command, args=args) as cli:
            return await cli.call_tool(name, payload)


# ── module-level default helper ──────────────────────────────────────────────

_default: MCPProtocolClient | None = None


def get_default_protocol_client() -> MCPProtocolClient:
    """프로세스 lifetime 동안 재사용 가능한 lazy singleton.

    실제 ``start()`` 는 처음 ``call_tool`` / ``list_tools`` 시점에 일어난다.
    """
    global _default
    if _default is None:
        _default = MCPProtocolClient()
    return _default


async def shutdown_default_protocol_client() -> None:
    global _default
    if _default is not None:
        try:
            await _default.close()
        finally:
            _default = None
