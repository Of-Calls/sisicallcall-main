"""
MCP Server tool list 조회 스크립트.

이 스크립트는 진짜 MCP Client transport 로 MCP Server 를 띄워
list_tools 를 호출한 결과를 출력한다. Python 내부 list/dict 를 출력하면
안 되고, 반드시 stdio_client → ClientSession.list_tools() 결과를 사용해야 한다.

사용:
  python scripts/check_mcp_tools.py
"""
from __future__ import annotations

import asyncio
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


async def main() -> int:
    print("[check_mcp_tools] connecting to MCP Server via stdio transport ...")
    try:
        async with MCPProtocolClient() as cli:
            tools = await cli.list_tools()
    except MCPClientTransportError as exc:
        print(f"[check_mcp_tools] transport error: {exc}", file=sys.stderr)
        return 2

    print(f"[check_mcp_tools] {len(tools)} tool(s) registered:")
    for t in tools:
        name = t.get("name", "")
        desc = (t.get("description") or "").splitlines()[0] if t.get("description") else ""
        print(f"  - {name}{(' — ' + desc) if desc else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
