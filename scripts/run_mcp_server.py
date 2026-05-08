"""
Sisicallcall MCP Server entry script.

stdio transport 로 진짜 MCP Server 를 실행한다.

사용:
  python scripts/run_mcp_server.py

자식 process 로 띄울 때 (MCP Client 가 사용):
  StdioServerParameters(command="python", args=["scripts/run_mcp_server.py"])

중요: stdio transport 에서 stdout 은 JSON-RPC 메시지 전용이다.
프로젝트 logger 가 stdout 을 쓰지 않게 ``MCP_STDIO_MODE=true`` 를
**프로젝트 import 보다 먼저** 설정한다.
"""
from __future__ import annotations

import os
import sys

# ── 프로젝트 import 전에 stdio 환경 강제 ─────────────────────────────────────
# MCP_STDIO_MODE=true → app.utils.logger 가 StreamHandler 를 stderr 로 향하게 한다.
# MCP_STDIO_LOG_LEVEL 미설정이면 WARNING 이 기본 — INFO/DEBUG 로 키우려면
# MCP_STDIO_LOG_LEVEL=INFO 환경변수로 override.
os.environ.setdefault("MCP_STDIO_MODE", "true")

# 프로젝트 루트를 sys.path 에 강제 추가 — 다른 cwd 에서 실행해도 동작.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# stdout 은 절대 일반 텍스트로 쓰지 않는다 — MCP server 가 binary JSON-RPC 만 쓴다.
# 그래도 print() 같은 사고 출력을 방지하기 위해 stderr 로 redirect 하지는 않는다
# (FastMCP 내부 SDK 가 정확히 stdout 을 사용한다).

from app.services.mcp.server.main import main  # noqa: E402

if __name__ == "__main__":
    main()
