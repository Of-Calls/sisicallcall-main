"""시연용 Post-call 실행 스크립트.

demo-call-critical 컨텍스트를 시드한 뒤 PostCallAgent를 실행해 MCP 액션 결과를 출력한다.

동작 모드:
  기본 (--real-actions 없음): 모든 connector를 mock으로 강제 실행
  --real-actions:             .env의 각 *_MCP_REAL 설정을 그대로 따름
  --real-actions --only-tool notion:
                              notion만 .env 설정을 따르고, 나머지는 mock으로 강제

사용 예:
    python scripts/run_post_call_demo.py
    python scripts/run_post_call_demo.py --real-actions
    python scripts/run_post_call_demo.py --real-actions --only-tool notion
    python scripts/run_post_call_demo.py --real-actions --only-tool sms
    python scripts/run_post_call_demo.py --tenant-id my-tenant --call-id my-call-001
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── .env 로드 (실제 환경 변수보다 우선도가 낮으므로 override=False) ────────────
from dotenv import load_dotenv  # noqa: E402
load_dotenv(override=False)

from tests.fixtures.demo_post_call_context import (  # noqa: E402
    DEMO_POST_CALL_CONTEXT,
)
from app.agents.post_call.context_provider import seed_test_context  # noqa: E402
from app.agents.post_call.completed_call_runner import run_post_call_for_completed_call  # noqa: E402

# ── ANSI 색상 ─────────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _status_color(status: str) -> str:
    mapping = {"success": _GREEN, "failed": _RED, "skipped": _YELLOW}
    color = mapping.get(status, "")
    return _c(color + _BOLD, status.upper())


# ── Tool → env var 매핑 ───────────────────────────────────────────────────────
_VALID_TOOLS = ("calendar", "slack", "sms", "notion", "gmail", "jira", "company_db")

_TOOL_ENV_VARS: dict[str, list[str]] = {
    "calendar":   ["CALENDAR_MCP_REAL"],
    "slack":      ["SLACK_MCP_REAL"],
    "sms":        ["SMS_MCP_REAL"],
    "notion":     ["NOTION_MCP_REAL"],
    "gmail":      ["GMAIL_MCP_REAL"],
    "jira":       ["JIRA_MCP_REAL"],
    "company_db": ["COMPANY_DB_MCP_REAL", "MCP_COMPANY_DB_REAL"],
}


def _apply_connector_modes(real_actions: bool, only_tool: str | None) -> dict[str, bool]:
    """connector별 real mode를 결정하고 os.environ에 적용한다.

    반환값: {tool_name: is_real} — 출력용
    """
    effective: dict[str, bool] = {}

    for tool, env_vars in _TOOL_ENV_VARS.items():
        if not real_actions:
            # mock 강제
            for ev in env_vars:
                os.environ[ev] = "false"
            effective[tool] = False
        elif only_tool is not None and tool != only_tool:
            # --only-tool X: 지정 도구 외는 mock 강제
            for ev in env_vars:
                os.environ[ev] = "false"
            effective[tool] = False
        else:
            # .env / 기존 환경변수 그대로 사용 (첫 번째 env var 기준으로 상태 읽기)
            val = os.environ.get(env_vars[0], "false").lower() in ("1", "true")
            effective[tool] = val

    return effective


# ── 출력 헬퍼 ─────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{_c(_BOLD, '─' * 60)}")
    print(_c(_CYAN + _BOLD, f"  {title}"))
    print(_c(_BOLD, "─" * 60))


def _print_connector_modes(modes: dict[str, bool]) -> None:
    _print_section("Connector 실행 모드")
    for tool, is_real in modes.items():
        tag = _c(_GREEN + _BOLD, "REAL") if is_real else _c(_YELLOW, "mock")
        print(f"    {tool:12s}: {tag}")


def _print_result(result: dict) -> None:
    _print_section("Post-call 분석 결과")

    summary = result.get("summary") or {}
    print(f"  summary_short      : {summary.get('summary_short', '—')}")
    print(f"  customer_emotion   : {_c(_BOLD, str(summary.get('customer_emotion', '—')))}")
    print(f"  resolution_status  : {summary.get('resolution_status', '—')}")

    priority = result.get("priority_result") or {}
    print(f"  priority           : {_c(_BOLD, str(priority.get('priority', '—')))}")
    print(f"  action_required    : {priority.get('action_required', False)}")

    review_verdict = result.get("review_verdict") or "—"
    human_review = result.get("human_review_required", False)
    blocked = result.get("blocked_actions") or []
    retry_count = result.get("review_retry_count", 0)
    verdict_color = _GREEN if review_verdict == "pass" else (_YELLOW if review_verdict == "correctable" else _RED)
    print(f"  review_verdict     : {_c(verdict_color + _BOLD, review_verdict)}")
    print(f"  human_review_req   : {human_review}")
    if blocked:
        print(f"  blocked_actions    : {blocked}")
    if retry_count:
        print(f"  review_retry_count : {retry_count}")

    _print_section("Action Plan")
    plan = result.get("action_plan") or {}
    actions = plan.get("actions") or []
    if actions:
        for a in actions:
            print(f"    · {a.get('action_type'):35s} tool={a.get('tool')}")
    else:
        print("    (계획된 액션 없음)")

    _print_section("실행된 액션")
    executed = result.get("executed_actions") or []
    if executed:
        for a in executed:
            status_str = _status_color(a.get("status", "unknown"))
            ext_id = a.get("external_id") or "—"
            err    = a.get("error") or ""
            line = (
                f"    [{status_str}] "
                f"{a.get('action_type', '?'):35s} "
                f"tool={a.get('tool', '?'):20s} "
                f"external_id={ext_id}"
            )
            if err:
                line += f"  {_c(_RED, 'err=' + err)}"
            print(line)
    else:
        print("    (실행된 액션 없음)")

    errors = result.get("errors") or []
    partial = result.get("partial_success", False)

    if errors:
        _print_section("오류")
        for e in errors:
            print(f"    · {_c(_RED, str(e))}")

    _print_section("요약")
    failed_cnt  = sum(1 for a in executed if a.get("status") == "failed")
    skipped_cnt = sum(1 for a in executed if a.get("status") == "skipped")
    success_cnt = sum(1 for a in executed if a.get("status") == "success")
    print(f"  partial_success : {partial}")
    print(
        f"  액션 결과       : "
        f"{_c(_GREEN, str(success_cnt) + ' success')}  "
        f"{_c(_YELLOW, str(skipped_cnt) + ' skipped')}  "
        f"{_c(_RED, str(failed_cnt) + ' failed')}"
    )


# ── Demo LLM mode — 신규 2-에이전트 그래프는 POST_CALL_LLM_MODE=mock 으로 강제 ────

def _patch_llm_nodes() -> None:
    """신규 2-에이전트 그래프에서는 mock 모드로 강제하기만 하면 된다.

    analysis_planner_agent_node 와 reviewer_agent_node 는
    POST_CALL_LLM_MODE != 'real' 일 때 자체 결정론적 mock LLM 을 사용한다.
    """
    os.environ.setdefault("POST_CALL_LLM_MODE", "mock")
    import app.agents.post_call.nodes.analysis_planner_agent_node as _planner
    import app.agents.post_call.nodes.reviewer_agent_node as _reviewer
    _planner._llm = None  # type: ignore[attr-defined]
    _reviewer._llm = None  # type: ignore[attr-defined]


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

async def _run(tenant_id: str, call_id: str, real_actions: bool, only_tool: str | None) -> None:
    ctx = DEMO_POST_CALL_CONTEXT

    print(_c(_BOLD, "\n시시콜콜 Post-call MCP 시연 스크립트"))
    print(f"  call_id   : {call_id}")
    print(f"  tenant_id : {tenant_id}")

    if not real_actions:
        print(f"  mode      : {_c(_YELLOW + _BOLD, 'MOCK 모드')} (모든 connector mock — --real-actions로 .env 설정 적용)")
    elif only_tool:
        print(
            f"  mode      : {_c(_GREEN + _BOLD, 'REAL 모드')} "
            f"({only_tool}만 .env 설정 적용, 나머지 mock)"
        )
    else:
        print(f"  mode      : {_c(_GREEN + _BOLD, 'REAL 모드')} (.env 각 *_MCP_REAL 설정 적용)")

    # connector real mode 결정 및 적용
    connector_modes = _apply_connector_modes(real_actions, only_tool)
    _print_connector_modes(connector_modes)

    use_real_llm = os.environ.get("POST_CALL_USE_REAL_LLM", "").lower() == "true"
    if use_real_llm:
        print(f"\n  LLM       : {_c(_GREEN, '실제 LLM (POST_CALL_USE_REAL_LLM=true)')}")
    else:
        print(f"\n  LLM       : {_c(_YELLOW, 'Demo Mock LLM (angry/critical 시나리오 고정)')}")
        _patch_llm_nodes()

    # 시연 컨텍스트 seed
    await seed_test_context(
        call_id=call_id,
        tenant_id=tenant_id,
        transcripts=ctx.get("transcripts"),
        call_metadata={**ctx.get("metadata", {}), "call_id": call_id, "tenant_id": tenant_id},
        branch_stats=ctx.get("branch_stats"),
    )

    print("\n  컨텍스트 시드 완료 — PostCallAgent 실행 중...")

    outcome = await run_post_call_for_completed_call(
        call_id=call_id,
        tenant_id=tenant_id,
        trigger="call_ended",
    )

    if not outcome.get("ok"):
        print(_c(_RED + _BOLD, f"\n[오류] PostCallAgent 실행 실패: {outcome.get('error')}"))
        sys.exit(1)

    _print_result(outcome["result"])
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="시연용 Post-call MCP 실행 스크립트 (기본: Mock 모드)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""예시:
  python scripts/run_post_call_demo.py                          # 전체 mock
  python scripts/run_post_call_demo.py --real-actions           # .env 설정 전체 적용
  python scripts/run_post_call_demo.py --real-actions --only-tool notion  # Notion만 real
  python scripts/run_post_call_demo.py --real-actions --only-tool sms     # SMS만 real
""",
    )
    parser.add_argument(
        "--tenant-id",
        default="demo-tenant",
        help="테넌트 ID (기본값: demo-tenant)",
    )
    parser.add_argument(
        "--call-id",
        default="demo-call-critical",
        help="통화 ID (기본값: demo-call-critical)",
    )
    parser.add_argument(
        "--real-actions",
        action="store_true",
        help=".env의 각 *_MCP_REAL 설정을 따름 (없으면 전체 mock 강제)",
    )
    parser.add_argument(
        "--only-tool",
        choices=list(_VALID_TOOLS),
        default=None,
        metavar="TOOL",
        help=(
            "--real-actions와 함께 사용. 지정한 도구만 .env real 설정을 따르고 "
            f"나머지는 mock 강제. 선택: {', '.join(_VALID_TOOLS)}"
        ),
    )
    args = parser.parse_args()

    if args.only_tool and not args.real_actions:
        parser.error("--only-tool은 --real-actions와 함께 사용해야 합니다.")

    asyncio.run(_run(args.tenant_id, args.call_id, args.real_actions, args.only_tool))


if __name__ == "__main__":
    main()
