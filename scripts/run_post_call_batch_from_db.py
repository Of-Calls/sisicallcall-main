"""Batch Post-call runner — DB에서 completed calls를 조회해 일괄 실행.

사용 예:
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 2 --llm-mode real
    python scripts/run_post_call_batch_from_db.py --all-tenants --limit 10 --llm-mode mock
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock --dry-run
    python scripts/run_post_call_batch_from_db.py --all-tenants --limit 10 --llm-mode mock --only-missing-results --dry-run

export 예:
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock --output .local/reports/batch.json
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock --output .local/reports/batch.csv --output-format csv
    python scripts/run_post_call_batch_from_db.py --tenant-id <uuid> --limit 5 --llm-mode mock --output .local/reports/batch.md --output-format md
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

import asyncpg  # noqa: E402

from app.agents.post_call import completed_call_runner as runner_mod  # noqa: E402
from app.agents.post_call.llm_caller import (  # noqa: E402
    compute_estimated_cost_usd,
    describe_post_call_llm,
    get_post_call_llm_mode,
    post_call_openai_key_available,
)
from app.utils.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from scripts.run_post_call_demo import (  # noqa: E402
    _BOLD,
    _CYAN,
    _GREEN,
    _RED,
    _RESET,
    _YELLOW,
    _apply_connector_modes,
    _c,
    _patch_llm_nodes,
)
from scripts.run_post_call_from_db import (  # noqa: E402
    _apply_llm_mode,
    _patch_runner_context_lookup,
    _reset_llm_nodes,
)

logger = get_logger(__name__)

_SEP = "─" * 56


# ── DB helpers ────────────────────────────────────────────────────────────────

def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


# ── Target call query ────────────────────────────────────────────────────────

def build_target_calls_sql(
    tenant_id: str | None,
    limit: int,
    offset: int,
    only_missing: bool,
) -> tuple[str, list]:
    """Build SQL + params for querying completed calls.

    Returns a (sql, params) tuple suitable for asyncpg.fetch(*params).
    """
    sql = """
        SELECT
          c.id::text                              AS call_id,
          c.tenant_id::text                       AS tenant_id,
          c.twilio_call_sid,
          c.status,
          c.started_at,
          c.ended_at,
          COUNT(t.id)::int                        AS transcript_count,
          (MAX(cs.call_id::text) IS NOT NULL)     AS has_summary,
          (MAX(va.call_id::text) IS NOT NULL)     AS has_voc
        FROM calls c
        LEFT JOIN transcripts     t  ON t.call_id  = c.id
        LEFT JOIN call_summaries  cs ON cs.call_id = c.id
        LEFT JOIN voc_analyses    va ON va.call_id = c.id
        WHERE c.status = 'completed'"""

    params: list = []

    if tenant_id is not None:
        params.append(tenant_id)
        sql += f"\n  AND c.tenant_id = ${len(params)}::uuid"

    sql += "\nGROUP BY c.id, c.tenant_id, c.twilio_call_sid, c.status, c.started_at, c.ended_at"

    if only_missing:
        sql += "\nHAVING MAX(cs.call_id::text) IS NULL OR MAX(va.call_id::text) IS NULL"

    sql += "\nORDER BY COUNT(t.id) DESC"

    params.append(limit)
    sql += f"\nLIMIT ${len(params)}"
    params.append(offset)
    sql += f"\nOFFSET ${len(params)}"

    return sql, params


async def fetch_target_calls(
    conn,
    tenant_id: str | None,
    limit: int,
    offset: int,
    only_missing: bool,
) -> list[dict]:
    sql, params = build_target_calls_sql(tenant_id, limit, offset, only_missing)
    rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


# ── Per-call result extraction ────────────────────────────────────────────────

def _usage_tokens(usage: dict | None) -> tuple[int | None, int | None, int | None]:
    """Return (prompt_tokens, completion_tokens, total_tokens) from a usage dict."""
    if not isinstance(usage, dict):
        return None, None, None
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _sum_optional(*values: int | None) -> int | None:
    """Sum integers ignoring None. Returns None if all values are None."""
    if all(v is None for v in values):
        return None
    return sum(int(v) for v in values if v is not None)


def extract_call_result(
    call_id: str,
    tenant_id: str,
    transcript_count: int,
    outcome: dict,
) -> dict:
    """Convert a runner outcome dict into a flat result record."""
    base = {
        "call_id": call_id,
        "tenant_id": str(tenant_id),
        "transcript_count": transcript_count,
    }
    if not outcome.get("ok"):
        return {**base, "status": "fail", "error": outcome.get("error")}

    result = outcome.get("result") or {}
    summary = result.get("summary") or {}
    priority_result = result.get("priority_result") or {}
    voc_analysis = result.get("voc_analysis") or {}
    intent_result = voc_analysis.get("intent_result") or {}
    sentiment_result = voc_analysis.get("sentiment_result") or {}
    review_result = result.get("review_result") or {}
    executed = result.get("executed_actions") or []
    actions = (result.get("action_plan") or {}).get("actions") or []

    # ── LLM usage 추출 (real LLM 모드에서만 채워짐) ───────────────────────────
    analysis_usage = result.get("analysis_llm_usage")
    review_usage = result.get("review_llm_usage")

    a_pt, a_ct, a_tt = _usage_tokens(analysis_usage)
    r_pt, r_ct, r_tt = _usage_tokens(review_usage)

    total_pt = _sum_optional(a_pt, r_pt)
    total_ct = _sum_optional(a_ct, r_ct)
    total_tt = _sum_optional(a_tt, r_tt)

    model = None
    for usage in (analysis_usage, review_usage):
        if isinstance(usage, dict) and usage.get("model"):
            model = usage["model"]
            break

    estimated_cost = compute_estimated_cost_usd(total_pt, total_ct, model)

    fallback = bool(
        (isinstance(analysis_usage, dict) and analysis_usage.get("fallback"))
        or (isinstance(review_usage, dict) and review_usage.get("fallback"))
        or (review_result.get("llm_fallback"))
    )

    return {
        **base,
        "status": "ok",
        "partial_success": result.get("partial_success", False),
        "review_verdict": result.get("review_verdict") or "—",
        "review_confidence": review_result.get("confidence"),
        "review_confidence_source": review_result.get("confidence_source"),
        "review_corrected_keys": review_result.get("corrected_keys"),
        "human_review_required": result.get("human_review_required", False),
        "action_plan_count": len(actions),
        "executed_count": len(executed),
        "action_success": sum(1 for a in executed if a.get("status") == "success"),
        "action_skipped": sum(1 for a in executed if a.get("status") == "skipped"),
        "action_failed": sum(1 for a in executed if a.get("status") == "failed"),
        "summary_short": summary.get("summary_short") or "—",
        "customer_emotion": summary.get("customer_emotion") or "—",
        "resolution_status": summary.get("resolution_status") or "—",
        "primary_category": intent_result.get("primary_category") or "—",
        "priority": priority_result.get("priority") or "—",
        "sentiment": sentiment_result.get("sentiment") or "—",
        # ── LLM usage / cost / fallback ─────────────────────────────────────
        "llm_model": model,
        "llm_fallback": fallback,
        "analysis_prompt_tokens": a_pt,
        "analysis_completion_tokens": a_ct,
        "analysis_total_tokens": a_tt,
        "review_prompt_tokens": r_pt,
        "review_completion_tokens": r_ct,
        "review_total_tokens": r_tt,
        "total_prompt_tokens": total_pt,
        "total_completion_tokens": total_ct,
        "total_tokens": total_tt,
        "estimated_cost_usd": estimated_cost,
        "error": None,
    }


# ── Batch execution ──────────────────────────────────────────────────────────

async def run_batch(
    calls: list[dict],
    trigger: str,
    dry_run: bool = False,
) -> list[dict]:
    """Run post-call processing for each call in *calls*.

    - transcript_count == 0: skip (reason=transcripts_missing)
    - dry_run == True: record status=dry_run without calling the runner
    - Any single call exception is caught and recorded; the batch continues.
    - KeyboardInterrupt is re-raised immediately.
    """
    results: list[dict] = []

    for row in calls:
        call_id = str(row["call_id"])
        tenant_id = str(row["tenant_id"])
        transcript_count = int(row.get("transcript_count") or 0)

        if transcript_count == 0:
            results.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "transcript_count": 0,
                "status": "skip",
                "skip_reason": "transcripts_missing",
            })
            continue

        if dry_run:
            results.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "transcript_count": transcript_count,
                "status": "dry_run",
            })
            continue

        try:
            outcome = await runner_mod.run_post_call_for_completed_call(
                call_id=call_id,
                tenant_id=tenant_id,
                trigger=trigger,
            )
            results.append(extract_call_result(call_id, tenant_id, transcript_count, outcome))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error("batch call failed call_id=%s err=%s", call_id, exc)
            results.append({
                "call_id": call_id,
                "tenant_id": tenant_id,
                "transcript_count": transcript_count,
                "status": "fail",
                "error": str(exc),
            })

    return results


# ── Report SQL queries ────────────────────────────────────────────────────────

async def fetch_tenant_report(conn, tenant_id: str) -> dict:
    """Query distribution metrics for a single tenant from the DB."""

    async def _count(sql: str, *params) -> int:
        row = await conn.fetchrow(sql, *params)
        return int(row[0]) if row else 0

    async def _dist(sql: str, *params) -> dict[str, int]:
        rows = await conn.fetch(sql, *params)
        return {str(r[0] or "—"): int(r[1]) for r in rows}

    call_type = await _dist(
        """
        SELECT intent_result->>'primary_category' AS call_type, COUNT(*)
        FROM voc_analyses
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    emotion = await _dist(
        """
        SELECT customer_emotion, COUNT(*)
        FROM call_summaries
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    priority = await _dist(
        """
        SELECT priority_result->>'priority' AS priority, COUNT(*)
        FROM voc_analyses
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    resolution = await _dist(
        """
        SELECT resolution_status, COUNT(*)
        FROM call_summaries
        WHERE tenant_id = $1::uuid
        GROUP BY 1 ORDER BY 2 DESC
        """,
        tenant_id,
    )

    missing_category = await _count(
        """
        SELECT COUNT(*) FROM voc_analyses
        WHERE tenant_id = $1::uuid
          AND NULLIF(intent_result->>'primary_category', '') IS NULL
        """,
        tenant_id,
    )

    mismatch_summary = await _count(
        """
        SELECT COUNT(*) FROM call_summaries cs
        JOIN calls c ON c.id = cs.call_id
        WHERE cs.tenant_id <> c.tenant_id
          AND cs.tenant_id = $1::uuid
        """,
        tenant_id,
    )

    mismatch_voc = await _count(
        """
        SELECT COUNT(*) FROM voc_analyses va
        JOIN calls c ON c.id = va.call_id
        WHERE va.tenant_id <> c.tenant_id
          AND va.tenant_id = $1::uuid
        """,
        tenant_id,
    )

    mismatch_action_logs = await _count(
        """
        SELECT COUNT(*) FROM mcp_action_logs ml
        JOIN calls c ON c.id::text = ml.call_id
        WHERE ml.tenant_id IS DISTINCT FROM c.tenant_id::text
          AND c.tenant_id::text = $1::text
        """,
        tenant_id,
    )

    return {
        "call_type": call_type,
        "emotion": emotion,
        "priority": priority,
        "resolution": resolution,
        "missing_primary_category": missing_category,
        "tenant_mismatch_summary": mismatch_summary,
        "tenant_mismatch_voc": mismatch_voc,
        "tenant_mismatch_action_logs": mismatch_action_logs,
    }


# ── Export helpers ────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "status",
    "call_id",
    "tenant_id",
    "transcript_count",
    "review_verdict",
    "review_confidence",
    "review_confidence_source",
    "human_review_required",
    "primary_category",
    "customer_emotion",
    "resolution_status",
    "priority",
    "sentiment",
    "action_plan_count",
    "executed_count",
    "action_success",
    "action_skipped",
    "action_failed",
    # ── LLM usage / cost / fallback ─────────────────────────────────────────
    "llm_model",
    "llm_mode",
    "llm_fallback",
    "analysis_prompt_tokens",
    "analysis_completion_tokens",
    "analysis_total_tokens",
    "review_prompt_tokens",
    "review_completion_tokens",
    "review_total_tokens",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "error",
]


def compute_usage_summary(records: list[dict]) -> dict:
    """Aggregate per-call usage into a batch-wide summary.

    Only ``status == "ok"`` records contribute. Mock-mode records have
    ``total_tokens == None`` and contribute zero to the totals while still
    counting toward ``calls_with_usage`` only when actual tokens are present.
    """
    a_pt = a_ct = a_tt = 0
    r_pt = r_ct = r_tt = 0
    calls_with_usage = 0
    fallback_calls = 0
    model: str | None = None

    for r in records:
        if r.get("status") != "ok":
            continue
        if r.get("llm_fallback"):
            fallback_calls += 1
        if r.get("total_tokens") is not None:
            calls_with_usage += 1
        if not model and r.get("llm_model"):
            model = r["llm_model"]

        a_pt += r.get("analysis_prompt_tokens") or 0
        a_ct += r.get("analysis_completion_tokens") or 0
        a_tt += r.get("analysis_total_tokens") or 0
        r_pt += r.get("review_prompt_tokens") or 0
        r_ct += r.get("review_completion_tokens") or 0
        r_tt += r.get("review_total_tokens") or 0

    total_pt = a_pt + r_pt
    total_ct = a_ct + r_ct
    total_tt = a_tt + r_tt

    estimated_cost = (
        compute_estimated_cost_usd(total_pt, total_ct, model) if model else None
    )

    return {
        "model": model,
        "calls_with_usage": calls_with_usage,
        "fallback_calls": fallback_calls,
        "analysis": {
            "prompt_tokens": a_pt,
            "completion_tokens": a_ct,
            "total_tokens": a_tt,
        },
        "review": {
            "prompt_tokens": r_pt,
            "completion_tokens": r_ct,
            "total_tokens": r_tt,
        },
        "total": {
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "total_tokens": total_tt,
            "estimated_cost_usd": estimated_cost,
        },
    }


def _infer_output_format(path: str, explicit_format: str | None) -> str:
    """Return the effective export format string (json | csv | md)."""
    if explicit_format:
        return explicit_format.lower()
    ext = os.path.splitext(path)[1].lower()
    return {".json": "json", ".csv": "csv", ".md": "md", ".markdown": "md"}.get(ext, "json")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _serialize_row(row: dict) -> dict:
    """Convert asyncpg row values to JSON-serializable primitives."""
    result: dict = {}
    for k, v in row.items():
        result[k] = v.isoformat() if hasattr(v, "isoformat") else v
    return result


def _format_report_for_json(report: dict) -> dict:
    return {
        "call_type": report.get("call_type", {}),
        "emotion": report.get("emotion", {}),
        "priority": report.get("priority", {}),
        "resolution": report.get("resolution", {}),
        "data_quality": {
            "missing_primary_category": report.get("missing_primary_category", 0),
            "tenant_mismatch_summary": report.get("tenant_mismatch_summary", 0),
            "tenant_mismatch_voc": report.get("tenant_mismatch_voc", 0),
            "tenant_mismatch_action_logs": report.get("tenant_mismatch_action_logs", 0),
        },
    }


def export_json(
    path: str,
    metadata: dict,
    targets: list[dict],
    records: list[dict],
    tenant_reports: dict[str, dict],
    usage_summary: dict | None = None,
) -> None:
    payload = {
        "metadata": metadata,
        "targets": [_serialize_row(t) for t in targets],
        "records": records,
        "tenant_reports": {
            tid: _format_report_for_json(rep)
            for tid, rep in tenant_reports.items()
        },
        "usage_summary": usage_summary or compute_usage_summary(records),
    }
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def export_csv(path: str, records: list[dict]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore", restval="")
        writer.writeheader()
        for r in records:
            row = {
                col: ("" if r.get(col) is None else r.get(col, ""))
                for col in _CSV_COLUMNS
            }
            writer.writerow(row)


def export_markdown(
    path: str,
    metadata: dict,
    targets: list[dict],
    records: list[dict],
    tenant_reports: dict[str, dict],
    usage_summary: dict | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Post-call Batch Report\n")

    # ── Metadata ─────────────────────────────────────────────────────────────
    lines.append("## Metadata\n")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k, v in metadata.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # ── Call Results ──────────────────────────────────────────────────────────
    lines.append("## Call Results\n")
    lines.append(
        "| Status | Call ID | Tenant | Category | Priority | Review | Tokens | Cost | Fallback |"
    )
    lines.append("|---|---|---|---|---|---|---:|---:|---|")
    for r in records:
        conf = r.get("review_confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else "—"
        review_combined = (
            f"{r.get('review_verdict', '—')} / {conf_str}"
            if r.get("review_verdict")
            else "—"
        )
        tenant_short = str(r.get("tenant_id", "—"))[:8] + "..."

        tokens = r.get("total_tokens")
        tokens_str = f"{tokens:,}" if isinstance(tokens, int) else "—"
        cost = r.get("estimated_cost_usd")
        cost_str = f"${cost:.4f}" if isinstance(cost, float) else "—"
        fallback_str = "true" if r.get("llm_fallback") else "false"

        lines.append(
            f"| {r.get('status', '—')} "
            f"| {r.get('call_id', '—')} "
            f"| {tenant_short} "
            f"| {r.get('primary_category', '—')} "
            f"| {r.get('priority', '—')} "
            f"| {review_combined} "
            f"| {tokens_str} "
            f"| {cost_str} "
            f"| {fallback_str} |"
        )
    lines.append("")

    # ── LLM Usage Summary ─────────────────────────────────────────────────────
    summary = usage_summary if usage_summary is not None else compute_usage_summary(records)
    lines.append("## LLM Usage Summary\n")
    if summary.get("calls_with_usage", 0) == 0 and (metadata.get("llm_mode") != "real"):
        lines.append("usage unavailable in mock mode")
        lines.append("")
    else:
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| model | {summary.get('model') or '—'} |")
        lines.append(f"| calls_with_usage | {summary.get('calls_with_usage', 0)} |")
        lines.append(f"| fallback_calls | {summary.get('fallback_calls', 0)} |")
        lines.append("")

        for section in ("analysis", "review"):
            data = summary.get(section, {}) or {}
            lines.append(f"### {section}\n")
            lines.append("| Tokens | Count |")
            lines.append("|---|---:|")
            lines.append(f"| prompt_tokens | {data.get('prompt_tokens', 0):,} |")
            lines.append(f"| completion_tokens | {data.get('completion_tokens', 0):,} |")
            lines.append(f"| total_tokens | {data.get('total_tokens', 0):,} |")
            lines.append("")

        total = summary.get("total", {}) or {}
        cost = total.get("estimated_cost_usd")
        cost_str = f"${cost:.4f} (estimated)" if isinstance(cost, float) else "—"
        lines.append("### total\n")
        lines.append("| Field | Value |")
        lines.append("|---|---:|")
        lines.append(f"| prompt_tokens | {total.get('prompt_tokens', 0):,} |")
        lines.append(f"| completion_tokens | {total.get('completion_tokens', 0):,} |")
        lines.append(f"| total_tokens | {total.get('total_tokens', 0):,} |")
        lines.append(f"| estimated_cost_usd | {cost_str} |")
        lines.append("")

    # ── Tenant Reports ────────────────────────────────────────────────────────
    if tenant_reports:
        lines.append("## Tenant Reports\n")
        for tid, report in tenant_reports.items():
            lines.append(f"### Tenant: {tid}\n")

            for section_label, key in [
                ("Call Type", "call_type"),
                ("Emotion", "emotion"),
                ("Priority", "priority"),
                ("Resolution", "resolution"),
            ]:
                data = report.get(key, {})
                lines.append(f"#### {section_label}\n")
                lines.append("| Type | Count |")
                lines.append("|---|---:|")
                for label, cnt in data.items():
                    lines.append(f"| {label} | {cnt} |")
                lines.append("")

            lines.append("#### Data Quality\n")
            lines.append("| Check | Count |")
            lines.append("|---|---:|")
            for check in [
                "missing_primary_category",
                "tenant_mismatch_summary",
                "tenant_mismatch_voc",
                "tenant_mismatch_action_logs",
            ]:
                lines.append(f"| {check} | {report.get(check, 0)} |")
            lines.append("")

    _ensure_parent_dir(path)
    # utf-8-sig: Windows CMD/메모장/Excel 계열 도구에서 한국어 BOM 인식 향상.
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


def write_report(
    output: str,
    output_format: str | None,
    metadata: dict,
    targets: list[dict],
    records: list[dict],
    tenant_reports: dict[str, dict],
    usage_summary: dict | None = None,
) -> None:
    fmt = _infer_output_format(output, output_format)
    try:
        if fmt == "csv":
            export_csv(output, records)
        elif fmt == "md":
            export_markdown(output, metadata, targets, records, tenant_reports, usage_summary)
        else:
            export_json(output, metadata, targets, records, tenant_reports, usage_summary)
        print(f"\nReport written: {output}")
    except Exception as exc:
        print(f"\n{_c(_RED, 'Failed to write batch report')} output={output} err={exc}")
        sys.exit(1)


# ── Console output helpers ────────────────────────────────────────────────────

def _print_sep(title: str = "") -> None:
    print()
    print(_SEP)
    if title:
        print(title)
        print(_SEP)


def _started_str(started_at) -> str:
    if started_at is None:
        return "—"
    if hasattr(started_at, "isoformat"):
        return started_at.isoformat()[:19]
    return str(started_at)[:19]


def _print_result_row(r: dict) -> None:
    call_id = r["call_id"]
    status = r["status"]

    if status == "skip":
        print(f"[SKIP] call_id={call_id}  reason={r.get('skip_reason', '—')}")
        return

    if status in ("fail", "dry_run"):
        tag = _c(_YELLOW, "DRY") if status == "dry_run" else _c(_RED, "FAIL")
        detail = "" if status == "dry_run" else f"  error={r.get('error', '—')}"
        print(f"[{tag}] call_id={call_id}{detail}")
        return

    conf = r.get("review_confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, float) else "—"
    verdict = r.get("review_verdict", "—")
    v_color = _GREEN if verdict == "pass" else (_YELLOW if verdict == "correctable" else _RED)

    print(
        f"[{_c(_GREEN, 'OK')}] "
        f"call_id={call_id}  "
        f"tenant={str(r['tenant_id'])[:8]}...  "
        f"category={r.get('primary_category', '—')}  "
        f"emotion={r.get('customer_emotion', '—')}  "
        f"priority={r.get('priority', '—')}  "
        f"review={_c(v_color, verdict)}  "
        f"confidence={conf_str}  "
        f"actions={r.get('action_plan_count', 0)}  "
        f"success={r.get('action_success', 0)}  "
        f"skipped={r.get('action_skipped', 0)}  "
        f"failed={r.get('action_failed', 0)}"
    )


def _print_dist(label: str, data: dict[str, int]) -> None:
    print(f"  {label}:")
    if data:
        for k, v in data.items():
            print(f"    {k}  {v}")
    else:
        print("    (데이터 없음)")


def _print_usage_summary(metadata: dict, summary: dict) -> None:
    print()
    print(_SEP)
    print("LLM Usage Summary")
    print(_SEP)

    if summary.get("calls_with_usage", 0) == 0 and metadata.get("llm_mode") != "real":
        print("  usage unavailable in mock mode")
        return

    print(f"  model              : {summary.get('model') or '—'}")
    print(f"  calls with usage   : {summary.get('calls_with_usage', 0)}")
    print(f"  fallback calls     : {summary.get('fallback_calls', 0)}")

    for section in ("analysis", "review"):
        data = summary.get(section, {}) or {}
        print(f"\n  {section}:")
        print(f"    prompt_tokens     : {data.get('prompt_tokens', 0):,}")
        print(f"    completion_tokens : {data.get('completion_tokens', 0):,}")
        print(f"    total_tokens      : {data.get('total_tokens', 0):,}")

    total = summary.get("total", {}) or {}
    cost = total.get("estimated_cost_usd")
    cost_str = f"${cost:.4f} (estimated)" if isinstance(cost, float) else "—"
    print("\n  total:")
    print(f"    prompt_tokens     : {total.get('prompt_tokens', 0):,}")
    print(f"    completion_tokens : {total.get('completion_tokens', 0):,}")
    print(f"    total_tokens      : {total.get('total_tokens', 0):,}")
    print(f"    estimated_cost    : {cost_str}")


def _print_tenant_report(
    tenant_id: str,
    report: dict,
    ok_results: list[dict],
) -> None:
    print(f"\ntenant_id={tenant_id}")

    _print_dist("call_type", report["call_type"])
    _print_dist("emotion", report["emotion"])
    _print_dist("priority", report["priority"])
    _print_dist("resolution", report["resolution"])

    verdict_counts = Counter(r.get("review_verdict", "—") for r in ok_results)
    print("  review (이번 batch 기준):")
    if verdict_counts:
        for verdict, cnt in verdict_counts.items():
            print(f"    {verdict}  {cnt}")
    else:
        print("    (실행 결과 없음)")

    print("  data quality:")
    print(f"    missing_primary_category    {report['missing_primary_category']}")
    print(f"    tenant_mismatch_summary     {report['tenant_mismatch_summary']}")
    print(f"    tenant_mismatch_voc         {report['tenant_mismatch_voc']}")
    print(f"    tenant_mismatch_action_logs {report['tenant_mismatch_action_logs']}")


# ── Main async flow ───────────────────────────────────────────────────────────

async def _main(
    *,
    tenant_id: str | None,
    all_tenants: bool,
    limit: int,
    offset: int,
    llm_mode: str | None,
    dry_run: bool,
    only_missing_results: bool,
    trigger: str,
    output: str | None = None,
    output_format: str | None = None,
) -> None:
    effective_llm = _apply_llm_mode(llm_mode)
    if effective_llm == "real":
        _reset_llm_nodes()
    else:
        _patch_llm_nodes()

    _apply_connector_modes(real_actions=False, only_tool=None)
    _patch_runner_context_lookup()

    # ── Header ───────────────────────────────────────────────────────────────
    print("\nPost-call Batch Runner")
    llm_label = (
        _c(_GREEN, describe_post_call_llm())
        if effective_llm == "real"
        else _c(_YELLOW, "Demo Mock LLM")
    )
    print(f"  mode      : {llm_label}")
    if all_tenants:
        print(f"  tenant_id : (all tenants — 개발/운영자 진단용)")
    else:
        print(f"  tenant_id : {tenant_id}")
    print(f"  limit     : {limit}  offset : {offset}")
    if only_missing_results:
        print(f"  filter    : only-missing-results")
    if dry_run:
        print(f"\n  {_c(_YELLOW + _BOLD, 'DRY RUN: no post-call execution will be performed')}")
    if output:
        fmt = _infer_output_format(output, output_format)
        print(f"  output    : {output}  ({fmt})")
    if effective_llm == "real" and not post_call_openai_key_available():
        print(
            f"  LLM warn  : {_c(_YELLOW, 'OPENAI_API_KEY is missing; real LLM will fall back to mock')}"
        )

    # ── Query target calls ───────────────────────────────────────────────────
    try:
        conn = await asyncpg.connect(_database_url())
    except Exception as exc:
        print(f"\n{_c(_RED, 'DB connection failed:')} {exc}")
        sys.exit(1)

    try:
        calls = await fetch_target_calls(
            conn=conn,
            tenant_id=None if all_tenants else tenant_id,
            limit=limit,
            offset=offset,
            only_missing=only_missing_results,
        )
    finally:
        await conn.close()

    print(f"  targets   : {len(calls)}")

    # ── Metadata for export ──────────────────────────────────────────────────
    metadata = {
        "llm_mode": effective_llm,
        "tenant_id": tenant_id,
        "all_tenants": all_tenants,
        "limit": limit,
        "offset": offset,
        "dry_run": dry_run,
        "only_missing_results": only_missing_results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── Print target list ────────────────────────────────────────────────────
    _print_sep("대상 calls")
    if not calls:
        print("  (대상 call 없음)")
    else:
        for i, row in enumerate(calls, 1):
            line = (
                f"  {i:3d}. call_id={row['call_id']}  "
                f"tenant={str(row['tenant_id'])[:8]}...  "
                f"transcripts={row['transcript_count']}  "
                f"started={_started_str(row.get('started_at'))}"
            )
            print(line)
            if dry_run:
                print(
                    f"       has_summary={row.get('has_summary', False)}  "
                    f"has_voc={row.get('has_voc', False)}  "
                    f"twilio_sid={row.get('twilio_call_sid') or '—'}"
                )

    if not calls:
        if output:
            write_report(output, output_format, metadata, calls, [], {})
        return

    # ── Execute (or dry-run) ─────────────────────────────────────────────────
    results = await run_batch(calls=calls, trigger=trigger, dry_run=dry_run)

    # llm_mode is metadata, not a runner output — stamp it on each record so
    # CSV/JSON consumers can correlate token counts with the active mode.
    for r in results:
        r["llm_mode"] = effective_llm

    if dry_run:
        if output:
            write_report(output, output_format, metadata, calls, results, {})
        return

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skip")
    fail_count = sum(1 for r in results if r["status"] == "fail")

    _print_sep("실행 결과")
    for r in results:
        _print_result_row(r)
    print(f"\n  합계: {ok_count} ok  {skip_count} skip  {fail_count} fail")

    # ── LLM usage summary ────────────────────────────────────────────────────
    usage_summary = compute_usage_summary(results)
    _print_usage_summary(metadata, usage_summary)

    # ── Tenant report ────────────────────────────────────────────────────────
    all_tenant_reports: dict[str, dict] = {}
    tenant_ids: list[str] = _collect_tenant_ids(tenant_id, all_tenants, results)

    if tenant_ids:
        _print_sep("tenant별 dashboard 원천 데이터 요약")

        try:
            conn = await asyncpg.connect(_database_url())
        except Exception as exc:
            print(f"\n{_c(_RED, 'DB connection failed for report:')} {exc}")
            if output:
                write_report(
                    output, output_format, metadata, calls, results, {},
                    usage_summary=usage_summary,
                )
            return

        try:
            for tid in tenant_ids:
                ok_for_tenant = [
                    r for r in results
                    if r.get("tenant_id") == tid and r["status"] == "ok"
                ]
                try:
                    report = await fetch_tenant_report(conn, tid)
                    all_tenant_reports[tid] = report
                    _print_tenant_report(tid, report, ok_for_tenant)
                except Exception as exc:
                    print(f"  tenant_id={tid}  report_error={exc}")
        finally:
            await conn.close()

    # ── Export ────────────────────────────────────────────────────────────────
    if output:
        write_report(
            output, output_format, metadata, calls, results, all_tenant_reports,
            usage_summary=usage_summary,
        )


def _collect_tenant_ids(
    tenant_id: str | None,
    all_tenants: bool,
    results: list[dict],
) -> list[str]:
    if not all_tenants and tenant_id:
        return [tenant_id]
    seen: set[str] = set()
    ordered: list[str] = []
    for r in results:
        tid = r.get("tenant_id") or ""
        if tid and tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DB completed calls를 batch로 실행하고 tenant별 리포트를 출력한다.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tenant-id", help="특정 tenant의 completed call만 실행")
    group.add_argument(
        "--all-tenants",
        action="store_true",
        help="모든 tenant 대상 (개발/운영자 진단용; 출력은 tenant별로 구분)",
    )
    parser.add_argument("--limit",  type=int, default=5, help="실행 call 수 제한 (기본값 5)")
    parser.add_argument("--offset", type=int, default=0, help="조회 offset (기본값 0)")
    parser.add_argument(
        "--llm-mode",
        choices=["mock", "real"],
        default=None,
        help="POST_CALL_LLM_MODE 오버라이드. 기본값: env 설정 또는 mock",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 대상 call 목록만 출력",
    )
    parser.add_argument(
        "--only-missing-results",
        action="store_true",
        help="call_summaries 또는 voc_analyses가 없는 call만 대상",
    )
    parser.add_argument(
        "--trigger",
        default="call_ended",
        choices=["call_ended", "escalation_immediate", "manual"],
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="report 저장 경로 (.json / .csv / .md). 부모 디렉터리가 없으면 자동 생성",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "csv", "md"],
        default=None,
        help="저장 형식. 미지정 시 --output 확장자로 자동 추론, 기본값 json",
    )
    args = parser.parse_args()

    asyncio.run(
        _main(
            tenant_id=args.tenant_id,
            all_tenants=args.all_tenants,
            limit=args.limit,
            offset=args.offset,
            llm_mode=args.llm_mode,
            dry_run=args.dry_run,
            only_missing_results=args.only_missing_results,
            trigger=args.trigger,
            output=args.output,
            output_format=args.output_format,
        )
    )


if __name__ == "__main__":
    main()
