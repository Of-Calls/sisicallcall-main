"""
단일 MCP Server tool 호출 디버그 스크립트.

사용:
  python scripts/call_mcp_tool.py <tool_name> --payload-json '{"tenant_id": "...", "call_id": "..."}'
  python scripts/call_mcp_tool.py slack.send_slack_alert --payload-json "{\"tenant_id\":\"...\",\"call_id\":\"...\",\"params\":{\"message\":\"hi\"}}"

실제 MCP Client transport 를 통해 MCP Server tool 을 호출한다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.services.mcp.protocol_client import (  # noqa: E402
    MCPClientTransportError,
    MCPProtocolClient,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a single MCP Server tool.")
    parser.add_argument("tool", help="Tool name, e.g. slack.send_slack_alert")
    parser.add_argument(
        "--payload-json",
        default="{}",
        help="Tool 호출 인자를 JSON 문자열로. 기본 {}",
    )
    return parser.parse_args(argv)


async def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        payload = json.loads(args.payload_json)
    except ValueError as exc:
        print(f"[call_mcp_tool] payload-json 파싱 실패: {exc}", file=sys.stderr)
        return 2

    print(f"[call_mcp_tool] calling {args.tool} ...")
    try:
        async with MCPProtocolClient() as cli:
            result = await cli.call_tool(args.tool, payload)
    except MCPClientTransportError as exc:
        print(f"[call_mcp_tool] transport error: {exc}", file=sys.stderr)
        return 2

    print("[call_mcp_tool] result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
