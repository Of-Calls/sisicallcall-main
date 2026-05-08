"""
MCP Server tool 표준 결과 shape 빌더.

모든 MCP Server tool 은 아래 shape 의 JSON 직렬화 가능한 dict 를 반환한다.

  {
    "action_type":    "string",
    "tool":           "string",          # provider tool 키 (slack, calendar, ...)
    "status":         "success|failed|skipped",
    "external_id":    "string|null",
    "error":          "string|null",
    "result":         {...},
    "latency_ms":     int,
    "source":         "mcp_server",
    "via_mcp":        true,
    "execution_mode": "mcp",
    "mcp_tool":       "slack.send_slack_alert"
  }

규칙:
  - 외부 API 실패 → status=failed
  - 토큰/설정 없음 → status=skipped
  - 필수 payload 없음 → status=skipped 또는 failed (error 명확히)
  - 예외를 server 밖으로 던져 Post-call pipeline 을 죽이면 안 됨
"""
from __future__ import annotations

from typing import Any


_SOURCE = "mcp_server"
_EXEC_MODE = "mcp"


def _envelope(
    *,
    tool: str,
    action_type: str,
    mcp_tool: str,
    status: str,
    external_id: str | None,
    error: str | None,
    result: dict[str, Any] | None,
    latency_ms: int,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "tool": tool,
        "status": status,
        "external_id": external_id,
        "error": error,
        "result": result or {},
        "latency_ms": int(latency_ms),
        "source": _SOURCE,
        "via_mcp": True,
        "execution_mode": _EXEC_MODE,
        "mcp_tool": mcp_tool,
    }


def success(
    *,
    tool: str,
    action_type: str,
    mcp_tool: str,
    external_id: str | None,
    result: dict[str, Any] | None = None,
    latency_ms: int = 0,
) -> dict[str, Any]:
    return _envelope(
        tool=tool,
        action_type=action_type,
        mcp_tool=mcp_tool,
        status="success",
        external_id=external_id,
        error=None,
        result=result,
        latency_ms=latency_ms,
    )


def failed(
    *,
    tool: str,
    action_type: str,
    mcp_tool: str,
    error: str,
    result: dict[str, Any] | None = None,
    latency_ms: int = 0,
) -> dict[str, Any]:
    return _envelope(
        tool=tool,
        action_type=action_type,
        mcp_tool=mcp_tool,
        status="failed",
        external_id=None,
        error=error,
        result=result,
        latency_ms=latency_ms,
    )


def skipped(
    *,
    tool: str,
    action_type: str,
    mcp_tool: str,
    reason: str,
    result: dict[str, Any] | None = None,
    latency_ms: int = 0,
) -> dict[str, Any]:
    return _envelope(
        tool=tool,
        action_type=action_type,
        mcp_tool=mcp_tool,
        status="skipped",
        external_id=None,
        error=reason,
        result=result,
        latency_ms=latency_ms,
    )
