"""E2E post-call agent 실 LLM + 실 MCP connector 검증.

사용 예:
    python scripts/run_e2e_post_call.py --tenant-id ba2bf499-... --call-id e2e-001

각 시나리오:
  - run_post_call_agent_safely(call_id, "call_ended", tenant_id)
  - 실행 전후 mcp_action_logs SELECT 비교
  - 결과 .local/e2e_results/<call_id>.json 에 저장
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJ / ".env", override=False)

# 실 LLM 모드 강제 — POST_CALL_LLM_MODE 가 .env 에 없을 수 있으므로 명시.
os.environ.setdefault("POST_CALL_LLM_MODE", "real")

import asyncpg  # noqa: E402

from app.utils.config import settings  # noqa: E402
from app.agents.post_call.runner import run_post_call_agent_safely  # noqa: E402


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


_LOG_COLUMNS = (
    "call_id, tenant_id, action_type, tool_name, status, "
    "external_id, error_message, created_at"
)


async def _fetch_action_logs_by_uuid(db_call_uuid: str, tenant_id: str) -> list[dict]:
    conn = await asyncpg.connect(_database_url())
    try:
        rows = await conn.fetch(
            f"""
            SELECT {_LOG_COLUMNS}
            FROM mcp_action_logs
            WHERE tenant_id = $1 AND call_id = $2
            ORDER BY created_at ASC
            """,
            tenant_id, db_call_uuid,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


async def _fetch_call_summary_by_uuid(db_call_uuid: str) -> dict | None:
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            """
            SELECT call_id::text AS call_id_uuid, tenant_id::text AS tenant_id_uuid,
                   summary_short, customer_emotion, resolution_status,
                   model_used, generation_mode
            FROM call_summaries
            WHERE call_id = $1::uuid
            """,
            db_call_uuid,
        )
        return dict(row) if row else None
    finally:
        await conn.close()


def _serializable(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serializable(v) for v in obj]
    return obj


async def _resolve_db_call_uuid(twilio_call_sid: str, tenant_id: str) -> str | None:
    """e2e-001 같은 twilio_call_sid → calls.id (UUID) 로 변환.

    PostCallAgent 가 받는 call_id 는 UUID 여야 call_summaries / voc_analyses
    repository 가 DB 저장한다 (non_uuid_call_id warning 방지).
    """
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            "SELECT id::text AS id FROM calls "
            "WHERE twilio_call_sid = $1 AND tenant_id = $2::uuid",
            twilio_call_sid, tenant_id,
        )
        return row["id"] if row else None
    finally:
        await conn.close()


async def run_one(call_id: str, tenant_id: str, output_dir: Path) -> dict:
    print(f"\n=== {call_id} (tenant={tenant_id}) ===")

    db_call_uuid = await _resolve_db_call_uuid(call_id, tenant_id)
    if db_call_uuid is None:
        raise RuntimeError(f"calls row not found for twilio_call_sid={call_id}")
    print(f"  resolved db_call_uuid = {db_call_uuid}")

    logs_before = await _fetch_action_logs_by_uuid(db_call_uuid, tenant_id)
    print(f"  mcp_action_logs before: {len(logs_before)} rows")

    t0 = datetime.now(timezone.utc)
    # production call.py:502 패턴 — db_call_id (UUID) 를 PostCallAgent 에 전달
    outcome = await run_post_call_agent_safely(
        call_id=db_call_uuid,
        trigger="call_ended",
        tenant_id=tenant_id,
    )
    t1 = datetime.now(timezone.utc)
    latency_s = (t1 - t0).total_seconds()
    print(f"  agent finished ok={outcome.get('ok')} error={outcome.get('error')} latency={latency_s:.1f}s")

    logs_after = await _fetch_action_logs_by_uuid(db_call_uuid, tenant_id)
    new_logs = logs_after[len(logs_before):]
    print(f"  mcp_action_logs after:  {len(logs_after)} rows  (new={len(new_logs)})")
    for log in new_logs:
        ext = log.get("external_id") or ""
        err = log.get("error_message") or ""
        print(f"    [{log['status']:8s}] {log['action_type']:32s} via {log['tool_name']:18s} "
              f"external_id={ext[:60]!r} err={err[:60]!r}")

    summary = await _fetch_call_summary_by_uuid(db_call_uuid)
    if summary:
        print(f"  call_summary: emotion={summary['customer_emotion']} status={summary['resolution_status']} model={summary['model_used']}")

    state = outcome.get("result") or {}
    out = {
        "call_id_label": call_id,
        "db_call_uuid": db_call_uuid,
        "tenant_id": tenant_id,
        "ok": outcome.get("ok"),
        "error": outcome.get("error"),
        "latency_s": round(latency_s, 3),
        "state_keys": {
            "review_verdict": state.get("review_verdict"),
            "human_review_required": state.get("human_review_required"),
            "approved_actions_count": len(state.get("approved_actions") or []),
            "executed_actions_count": len(state.get("executed_actions") or []),
            "proposed_actions_count": len(state.get("proposed_actions") or []),
            "reviewer_steps": state.get("reviewer_steps"),
            "errors": state.get("errors") or [],
            "partial_success": state.get("partial_success"),
        },
        "summary": (state.get("summary") or {}),
        "voc_analysis_priority": ((state.get("voc_analysis") or {}).get("priority_result") or {}),
        "proposed_actions": [
            {
                "action_type": a.get("action_type"),
                "tool": a.get("tool"),
                "priority": a.get("priority"),
                "params": {k: v for k, v in (a.get("params") or {}).items() if k != "tenant_id"},
            }
            for a in (state.get("proposed_actions") or [])
        ],
        "executed_actions": [
            {
                "action_type": a.get("action_type"),
                "tool": a.get("tool"),
                "status": a.get("status"),
                "external_id": a.get("external_id"),
                "error": a.get("error"),
                "result_keys": list((a.get("result") or {}).keys()),
            }
            for a in (state.get("executed_actions") or [])
        ],
        "review_result": {
            "verdict": (state.get("review_result") or {}).get("verdict"),
            "approved_count": len((state.get("review_result") or {}).get("approved_actions") or []),
            "rejected": [
                {"action_type": r.get("action_type"), "reject_reason": r.get("reject_reason")}
                for r in ((state.get("review_result") or {}).get("rejected_actions") or [])
            ],
            "corrections_to_analysis": (state.get("review_result") or {}).get("corrections_to_analysis") or {},
            "corrections_dropped": (state.get("review_result") or {}).get("corrections_dropped") or [],
        },
        "telemetry": {
            "analysis_planner": state.get("analysis_planner_telemetry"),
            "reviewer": state.get("reviewer_telemetry"),
        },
        "mcp_action_logs": {
            "before_count": len(logs_before),
            "after_count": len(logs_after),
            "new": [_serializable(l) for l in new_logs],
        },
        "call_summary_db": _serializable(summary),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{call_id}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  saved {out_path}")
    return out


async def main_async(tenant_id: str, call_ids: list[str]) -> None:
    output_dir = PROJ / ".local" / "e2e_results"
    print(f"POST_CALL_LLM_MODE={os.environ.get('POST_CALL_LLM_MODE')}")
    print(f"output_dir = {output_dir}")
    for call_id in call_ids:
        try:
            await run_one(call_id, tenant_id, output_dir)
        except Exception as exc:
            print(f"  FAILED {call_id}: {type(exc).__name__}: {exc}")
        await asyncio.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default="ba2bf499-6fcc-4340-b3dd-9341f8bcc915")
    parser.add_argument("--call-id", action="append", default=None,
                        help="repeatable; default e2e-001/002/003")
    args = parser.parse_args()

    call_ids = args.call_id or ["e2e-001", "e2e-002", "e2e-003"]
    asyncio.run(main_async(args.tenant_id, call_ids))


if __name__ == "__main__":
    main()
