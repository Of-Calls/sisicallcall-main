"""
Sisicallcall MCP Server entrypoint — KDT-101.

진짜 MCP protocol 서버다. FastMCP 기반으로 모든 외부 앱 action 을 tool
로 등록한 뒤, stdio transport 로 들어온 MCP Client 요청을 처리한다.

직접 실행:
  python -m app.services.mcp.server.main
  python scripts/run_mcp_server.py
  python -m app.services.mcp.server          # __main__.py 가 main() 호출

이 모듈은 import 만으로 server 를 실행하지 않는다 — main() 호출 시에만
``mcp.run("stdio")`` 가 동작한다 (테스트에서 import 만 하고 list_tools
검증할 수 있도록).

등록 tool (12개):
  slack.send_slack_alert
  calendar.schedule_callback
  gmail.send_manager_email
  jira.create_jira_issue
  notion.create_notion_call_record
  sms.send_voc_receipt_sms
  sms.send_callback_sms
  company_db.create_voc_issue
  company_db.add_priority_queue
  company_db.mark_faq_candidate
  internal_dashboard.add_priority_queue
  internal_dashboard.mark_faq_candidate
"""
from __future__ import annotations

import os
import sys

# ── stdio transport pre-setup ────────────────────────────────────────────────
# main() 가 호출되기 전에 import 만 일어나도 logger 가 만들어질 수 있으므로,
# 모듈 로드 시점에 MCP_STDIO_MODE 를 set 한다 — 이미 set 되어 있다면 보존.
# entrypoint (scripts/run_mcp_server.py 또는 __main__.py) 가 먼저 set 하지만
# 직접 import 케이스 (테스트의 build_server() 등) 에서는 비활성으로 두기 위해
# 여기서는 강제 set 하지 않는다. 단 main() 안에서는 강제한다.

from mcp.server.fastmcp import FastMCP

from app.services.mcp.server.providers import (
    calendar_tools,
    company_db_tools,
    gmail_tools,
    internal_dashboard_tools,
    jira_tools,
    notion_tools,
    slack_tools,
    sms_tools,
)

SERVER_NAME = "sisicallcall-mcp-server"


def build_server() -> FastMCP:
    """FastMCP 인스턴스를 만들어 모든 provider tool 을 등록한다.

    테스트는 이 함수를 호출해 list_tools / call_tool 동작을 검증한다.
    server.run() 은 호출하지 않는다.
    """
    mcp = FastMCP(name=SERVER_NAME)

    slack_tools.register(mcp)
    calendar_tools.register(mcp)
    gmail_tools.register(mcp)
    jira_tools.register(mcp)
    notion_tools.register(mcp)
    sms_tools.register(mcp)
    company_db_tools.register(mcp)
    internal_dashboard_tools.register(mcp)

    return mcp


def main() -> None:
    """stdio transport 로 MCP server 를 실행한다.

    process 의 stdin/stdout 이 MCP transport 로 사용되므로, 일반 application
    log 는 반드시 stderr 로 가야 한다. ``MCP_STDIO_MODE=true`` 를 강제하고,
    혹시 import 가 먼저 일어나 stdout handler 가 만들어졌다면 stderr 로
    safety-net 재라우팅한다.
    """
    # 1) Windows console stdout 이 UTF-16 으로 잡혀 있으면 stdio transport 가
    #    깨질 수 있다 — UTF-8 로 강제 (Python 3.7+).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    # 2) MCP_STDIO_MODE 강제 + 이미 만들어진 logger 들의 stream 을 stderr 로 옮김.
    os.environ["MCP_STDIO_MODE"] = "true"
    try:
        from app.utils.logger import reroute_existing_loggers_to_stderr
        reroute_existing_loggers_to_stderr()
    except Exception:
        # logger reroute 실패가 server 실행을 막아서는 안 된다.
        pass

    os.environ.setdefault("MCP_SERVER_NAME", SERVER_NAME)
    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
