"""``python -m app.services.mcp.server`` 진입점.

stdio transport 가 stdout 을 JSON-RPC 전용으로 쓰므로, 프로젝트 import
보다 먼저 ``MCP_STDIO_MODE=true`` 를 set 한다.
"""
import os

os.environ.setdefault("MCP_STDIO_MODE", "true")

from app.services.mcp.server.main import main  # noqa: E402

if __name__ == "__main__":
    main()
