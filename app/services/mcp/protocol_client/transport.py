"""
MCP Server stdio transport 파라미터 빌더.

환경변수 (기본값):
  MCP_SERVER_COMMAND   = sys.executable   (현재 파이썬 인터프리터)
  MCP_SERVER_ARGS      = scripts/run_mcp_server.py
  MCP_SERVER_ARGS_JSON = (선택) JSON list. 우선순위 높음.
  MCP_SERVER_CWD       = (선택) server process 의 cwd. 기본은 현재 cwd.

Windows CMD/PowerShell 안전성을 위해 기본 args 는 단일 문자열을 공백으로
분리한 list 로 사용한다. JSON 형태로 명시적으로 넘기려면
``MCP_SERVER_ARGS_JSON='["scripts/run_mcp_server.py", "--debug"]'`` 처럼 지정.
"""
from __future__ import annotations

import json
import os
import shlex
import sys

from mcp import StdioServerParameters


_DEFAULT_COMMAND = sys.executable or "python"
_DEFAULT_SCRIPT = "scripts/run_mcp_server.py"


def build_stdio_params(
    *,
    command: str | None = None,
    args: list[str] | None = None,
    cwd: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> StdioServerParameters:
    """MCP Server 를 stdio 로 띄우기 위한 StdioServerParameters 를 만든다.

    args 우선순위:
        1. 인자로 전달된 args
        2. MCP_SERVER_ARGS_JSON  (JSON list)
        3. MCP_SERVER_ARGS       (shlex split)
        4. [_DEFAULT_SCRIPT]
    """
    cmd = command or os.getenv("MCP_SERVER_COMMAND") or _DEFAULT_COMMAND

    if args is None:
        json_args = os.getenv("MCP_SERVER_ARGS_JSON", "").strip()
        if json_args:
            try:
                parsed = json.loads(json_args)
                if isinstance(parsed, list):
                    args = [str(x) for x in parsed]
            except (ValueError, TypeError):
                args = None

    if args is None:
        raw = os.getenv("MCP_SERVER_ARGS", "").strip()
        if raw:
            # Windows CMD 호환을 위해 posix=False 로 split — backslash 보존
            try:
                args = shlex.split(raw, posix=False)
            except ValueError:
                args = raw.split()
        else:
            args = [_DEFAULT_SCRIPT]

    server_cwd = cwd or os.getenv("MCP_SERVER_CWD") or os.getcwd()

    # 자식 프로세스가 부모 환경변수 (TENANT_INTEGRATION_STORAGE,
    # MCP_USE_TENANT_OAUTH, MCP_ALLOW_ENV_FALLBACK, OAuth client secret 등)
    # 를 그대로 보도록 현재 env 를 통째로 전달한다.
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    # MCP server 가 tool 안에서 print() 같은 걸로 stdout 을 더럽히지 않게
    # 한국어 콘솔 환경에서 UTF-8 강제.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    return StdioServerParameters(
        command=cmd,
        args=args,
        cwd=server_cwd,
        env=env,
    )
