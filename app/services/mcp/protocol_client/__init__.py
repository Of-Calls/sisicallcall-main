"""
진짜 MCP protocol Client — KDT-101.

Post-call ActionExecutor 는 MCP-only 다 — 이 패키지가 stdio transport 로
자체 MCP Server 와 통신한다.

기존 ``app/services/mcp/client.py`` (MCPClient + Connector registry) 는
실시간 통화 흐름 (auth_branch_node / task_branch_node) 전용으로 보존되며,
Post-call 경로에서는 호출되지 않는다.

엔트리:
  from app.services.mcp.protocol_client.client import (
      MCPProtocolClient,
      get_default_protocol_client,
  )
"""

from app.services.mcp.protocol_client.client import (  # noqa: F401
    MCPProtocolClient,
    MCPClientTransportError,
    get_default_protocol_client,
)
