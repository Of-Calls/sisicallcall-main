"""Run the completed-call post-call pipeline from Postgres context only.

This script does not seed in-memory call context. It calls
run_post_call_for_completed_call(), while patching that runner's context lookup
so the DB lookup can use a Twilio SID even when the CLI tenant label is not the
UUID stored in calls.tenant_id.

Run manually:
    python scripts/run_post_call_from_db.py --call-id demo-db-call-critical --tenant-id demo-tenant
    python scripts/run_post_call_from_db.py --call-id demo-db-call-critical --tenant-id demo-tenant --real-actions
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

from app.agents.post_call import completed_call_runner as runner_mod  # noqa: E402
from app.agents.post_call.llm_caller import (  # noqa: E402
    describe_post_call_llm,
    get_post_call_llm_mode,
    post_call_openai_key_available,
)
from app.services.db.transcripts import get_completed_call_context_from_db  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from scripts.run_post_call_demo import (  # noqa: E402
    _GREEN,
    _RED,
    _VALID_TOOLS,
    _YELLOW,
    _apply_connector_modes,
    _c,
    _patch_llm_nodes,
    _print_connector_modes,
    _print_result,
)

DEFAULT_CALL_ID = "demo-db-call-critical"
DEFAULT_TENANT_ID = "demo-tenant"

logger = get_logger(__name__)


async def _get_db_context_only(call_id: str, tenant_id: str | None = None) -> dict | None:
    raw = await get_completed_call_context_from_db(call_id, tenant_id=None)
    if raw is None:
        logger.warning("context_provider: DB context not found call_id=%s", call_id)
        return None

    ctx = copy.deepcopy(raw)
    metadata = ctx.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["call_id"] = call_id
    if tenant_id:
        metadata["tenant_id"] = tenant_id
    ctx["metadata"] = metadata

    if ctx.get("transcripts") is None:
        ctx["transcripts"] = []
    if ctx.get("branch_stats") is None:
        ctx["branch_stats"] = {}

    transcript_count = len(ctx.get("transcripts") or [])
    logger.info(
        "context_provider: DB context 사용 call_id=%s transcripts=%d",
        call_id,
        transcript_count,
    )
    print(f"context_provider: DB context 사용 call_id={call_id} transcripts={transcript_count}")
    return ctx


def _patch_runner_context_lookup() -> None:
    runner_mod.get_call_context_for_post_call = _get_db_context_only  # type: ignore[assignment]


def _reset_llm_nodes() -> None:
    import app.agents.post_call.nodes.post_call_analysis_node as _analysis
    import app.agents.post_call.nodes.review_node as _review
    import app.agents.post_call.nodes.summary_node as _summary
    import app.agents.post_call.nodes.voc_analysis_node as _voc
    import app.agents.post_call.nodes.priority_node as _priority

    _analysis._caller = None  # type: ignore[attr-defined]
    _review._caller = None  # type: ignore[attr-defined]
    _summary._caller = None  # type: ignore[attr-defined]
    _voc._caller = None  # type: ignore[attr-defined]
    _priority._caller = None  # type: ignore[attr-defined]


def _apply_llm_mode(llm_mode: str | None) -> str:
    if llm_mode is not None:
        os.environ["POST_CALL_LLM_MODE"] = llm_mode
        os.environ["POST_CALL_USE_REAL_LLM"] = "true" if llm_mode == "real" else "false"
    return get_post_call_llm_mode()


def _print_clean_summary(result: dict) -> None:
    executed = result.get("executed_actions") or []
    success_cnt = sum(1 for action in executed if action.get("status") == "success")
    skipped_cnt = sum(1 for action in executed if action.get("status") == "skipped")
    failed_cnt = sum(1 for action in executed if action.get("status") == "failed")
    plan = result.get("action_plan") or {}
    actions = plan.get("actions") or []

    print("\nDB runner summary")
    print(f"  review_verdict        : {result.get('review_verdict')}")
    print(f"  human_review_required : {result.get('human_review_required', False)}")
    print(f"  action_plan_count     : {len(actions)}")
    print(f"  executed_actions      : {len(executed)}")
    print(
        "  action_results        : "
        f"{_c(_GREEN, str(success_cnt) + ' success')}  "
        f"{_c(_YELLOW, str(skipped_cnt) + ' skipped')}  "
        f"{_c(_RED, str(failed_cnt) + ' failed')}"
    )


async def _run(
    *,
    call_id: str,
    tenant_id: str,
    trigger: str,
    real_actions: bool,
    only_tool: str | None,
    llm_mode: str | None,
) -> None:
    print("\nPost-call DB runner")
    print(f"  call_id   : {call_id}")
    print(f"  tenant_id : {tenant_id}")
    print(f"  trigger   : {trigger}")
    print("  context   : Postgres calls/transcripts")

    connector_modes = _apply_connector_modes(real_actions, only_tool)
    _print_connector_modes(connector_modes)

    if real_actions:
        print(
            f"\n  {_c(_YELLOW, 'Real actions enabled.')} "
            f"Make sure tenant integrations are connected."
        )
        print(
            f"  Run: python scripts/check_post_call_integrations.py "
            f"--tenant-id {tenant_id} --show-actions"
        )

    effective_llm_mode = _apply_llm_mode(llm_mode)
    if effective_llm_mode == "real":
        print(f"\n  LLM       : {_c(_GREEN, describe_post_call_llm())}")
        if not post_call_openai_key_available():
            print(
                f"  LLM warn  : {_c(_YELLOW, 'OPENAI_API_KEY is missing; real LLM will fall back to mock')}"
            )
        _reset_llm_nodes()
    else:
        print(f"\n  LLM       : {_c(_YELLOW, 'Demo Mock LLM')}")
        _patch_llm_nodes()

    _patch_runner_context_lookup()

    outcome = await runner_mod.run_post_call_for_completed_call(
        call_id=call_id,
        tenant_id=tenant_id,
        trigger=trigger,
    )

    if not outcome.get("ok"):
        print(f"\nPostCallAgent failed: {outcome.get('error')}")
        sys.exit(1)

    result = outcome["result"]
    _print_clean_summary(result)
    _print_result(result)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-call completed-call processing from Postgres context.",
    )
    parser.add_argument("--call-id", default=DEFAULT_CALL_ID)
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument(
        "--trigger",
        default="call_ended",
        choices=["call_ended", "escalation_immediate", "manual"],
    )
    parser.add_argument(
        "--real-actions",
        action="store_true",
        help="Use the *_MCP_REAL values already present in the environment.",
    )
    parser.add_argument(
        "--only-tool",
        choices=list(_VALID_TOOLS),
        default=None,
        metavar="TOOL",
        help="With --real-actions, only this tool follows its real-mode env setting.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["mock", "real"],
        default=None,
        help="Override POST_CALL_LLM_MODE for this run. Default: env value, then mock.",
    )
    args = parser.parse_args()

    if args.only_tool and not args.real_actions:
        parser.error("--only-tool requires --real-actions")

    asyncio.run(
        _run(
            call_id=args.call_id,
            tenant_id=args.tenant_id,
            trigger=args.trigger,
            real_actions=args.real_actions,
            only_tool=args.only_tool,
            llm_mode=args.llm_mode,
        )
    )


if __name__ == "__main__":
    main()
