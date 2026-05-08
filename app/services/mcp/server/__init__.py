"""
Sisicallcall MCP Server package — KDT-101.

진짜 MCP protocol Server. FastMCP 기반으로 외부 앱 action 을 tool 로
노출하고, MCP Client (stdio transport) 가 호출하면 tenant_id 기준으로
token/config 를 조회해 실제 provider API 를 호출한다.

Server 진입점:
  python -m app.services.mcp.server.main
  또는
  python scripts/run_mcp_server.py

이 패키지는 import 시점에 외부 네트워크 connection 을 만들지 않는다.
tool 호출 시점에서만 tenant_integrations / provider API 를 조회한다.
"""
