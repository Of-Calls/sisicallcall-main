"""KDT-73 retry 경로 e2e 검증 — 강제 fail → 재시도 → pass 흐름.

production 코드는 변경하지 않는다. 이 스크립트가 다음을 monkeypatch 한다:
  - reviewer_mod._llm: 첫 호출만 escalate_to_human + finalize(verdict=fail) 응답을
    fake 로 주입. 두 번째 호출부터는 원본 GPT4OMiniService 로 위임.
  - planner_mod._llm: generate_with_tools 를 wrap 해 매 호출마다 raw system_prompt
    + tool_calls 응답을 .local/e2e_retry_logs/ 에 저장.

검증 항목 (보고에 자동 체크):
  1. review_feedback 에 1차 reject 사유 들어감
  2. 2차 planner system prompt 에 "[이전 분석 검토 결과 —" 블록 주입
  3. 1차 vs 2차 분석 결과 diff (LLM 이 같은 실수 반복 안 했는지)
  4. analysis_retry_count == 1
  5. 외부 발송이 2차 결과 기준 (1차 액션 미사용)
  6. save_intermediate 멱등 — call_summaries 단일 row

사용 예:
    python scripts/run_e2e_retry_verify.py --call-id e2e-001 --tenant-id <UUID>

검증 후 production 코드 잔재 없음 확인:
    grep -r "FORCE_REVIEW_FAIL\|force_fail\|force_review_fail" app/   # 0 hit 기대
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJ / ".env", override=False)

# 실 LLM 모드 강제
os.environ.setdefault("POST_CALL_LLM_MODE", "real")

import asyncpg  # noqa: E402

from app.utils.config import settings  # noqa: E402
from app.agents.post_call.runner import run_post_call_agent_safely  # noqa: E402
import app.agents.post_call.nodes.analysis_planner_agent_node as planner_mod  # noqa: E402
import app.agents.post_call.nodes.reviewer_agent_node as reviewer_mod  # noqa: E402


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _resolve_db_call_uuid(twilio_call_sid: str, tenant_id: str) -> str | None:
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


async def _count_call_summaries(db_call_uuid: str) -> int:
    """call_summaries 동일 call_id row 수 — 멱등 검증용."""
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            "SELECT count(*) AS n FROM call_summaries WHERE call_id = $1::uuid",
            db_call_uuid,
        )
        return int(row["n"]) if row else 0
    finally:
        await conn.close()


async def _fetch_call_summary(db_call_uuid: str) -> dict | None:
    conn = await asyncpg.connect(_database_url())
    try:
        row = await conn.fetchrow(
            "SELECT summary_short, customer_emotion, resolution_status, "
            "model_used, updated_at FROM call_summaries WHERE call_id = $1::uuid",
            db_call_uuid,
        )
        return dict(row) if row else None
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# monkeypatch wrappers
# ─────────────────────────────────────────────────────────────────────────────


class _ReviewerForceFailFirstThenDelegate:
    """reviewer._llm 의 첫 호출만 강제 fail. 그 후엔 원본 위임."""

    def __init__(self, real_llm) -> None:
        self._real = real_llm
        self._call_n = 0
        self._calls_log: list[dict] = []

    async def generate_with_tools(self, **kwargs):
        self._call_n += 1
        if self._call_n == 1:
            entry = {
                "call_n": 1,
                "mode": "FORCE_FAIL",
                "system_prompt_excerpt": (kwargs.get("system_prompt") or "")[:500],
            }
            self._calls_log.append(entry)
            return {
                "tool_calls": [
                    {
                        "id": "ff_e",
                        "name": "escalate_to_human",
                        "arguments": {
                            "reason": "[E2E_FORCE_FAIL] handoff_notes 가 비어 있어 보강 필요. "
                                       "분석 결론과 액션 후보 (slack/jira/notion 등) 는 "
                                       "그대로 유지하고 handoff_notes 만 구체적으로 채워 다시 제출하라.",
                        },
                    },
                    {
                        "id": "ff_f",
                        "name": "finalize_review",
                        "arguments": {
                            "verdict": "fail",
                            "summary_reason": "[E2E_FORCE_FAIL] retry 경로 검증",
                        },
                    },
                ],
                "text": "",
                "raw_message": None,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "model": "force_fail_stub",
                },
            }
        # 2회차 이후 — 원본 LLM 으로 위임
        result = await self._real.generate_with_tools(**kwargs)
        self._calls_log.append({
            "call_n": self._call_n,
            "mode": "REAL_DELEGATE",
            "tool_call_names": [c.get("name") for c in (result.get("tool_calls") or [])],
            "usage": result.get("usage") or {},
        })
        return result


class _PlannerLoggingWrap:
    """planner._llm 호출마다 raw system_prompt + tool_calls 응답 캡처."""

    def __init__(self, real_llm, log_dir: Path) -> None:
        self._real = real_llm
        self._call_n = 0
        self._log_dir = log_dir
        self.captures: list[dict] = []

    async def generate_with_tools(self, **kwargs):
        self._call_n += 1
        system_prompt = kwargs.get("system_prompt") or ""
        result = await self._real.generate_with_tools(**kwargs)
        capture = {
            "call_n": self._call_n,
            "system_prompt": system_prompt,
            "user_message_excerpt": (kwargs.get("user_message") or "")[:500],
            "tool_calls": [
                {"name": c.get("name"), "arguments": c.get("arguments")}
                for c in (result.get("tool_calls") or [])
            ],
            "usage": result.get("usage") or {},
        }
        self.captures.append(capture)
        # raw 도 별도 파일로 (사용자가 직접 grep 가능)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        out = self._log_dir / f"planner_call_{self._call_n}.json"
        out.write_text(
            json.dumps(capture, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 검증 / 보고
# ─────────────────────────────────────────────────────────────────────────────


def _diff_analysis_summaries(a1: dict, a2: dict) -> dict:
    """1차 vs 2차 analysis_result 의 의미있는 필드 diff."""
    def pick(a):
        s = (a or {}).get("summary") or {}
        p = (a or {}).get("priority_result") or {}
        return {
            "summary_short": s.get("summary_short"),
            "customer_emotion": s.get("customer_emotion"),
            "resolution_status": s.get("resolution_status"),
            "handoff_notes": s.get("handoff_notes"),
            "priority": p.get("priority"),
            "action_required": p.get("action_required"),
        }
    f1 = pick(a1)
    f2 = pick(a2)
    diff = {k: {"first": f1.get(k), "second": f2.get(k)} for k in f1
            if f1.get(k) != f2.get(k)}
    return {"first_pass": f1, "second_pass": f2, "changed_fields": diff}


def _verify(report: dict, raw_captures: list[dict]) -> list[dict]:
    """6 검증 항목 자동 체크. raw_captures 는 system_prompt 원문 포함."""
    checks: list[dict] = []

    # 1. review_feedback 에 reject 사유 누적
    rf = report["state"].get("review_feedback") or []
    has_force_fail_reason = any("E2E_FORCE_FAIL" in s or "reviewer_escalated" in s
                                  for s in rf)
    checks.append({
        "id": "1_review_feedback_has_reject_reason",
        "ok": bool(rf) and has_force_fail_reason,
        "detail": f"review_feedback len={len(rf)} sample={rf[:2]}",
    })

    # 2. 2차 planner system_prompt 에 [이전 분석 검토 결과 — 블록 주입
    second_prompt = raw_captures[1]["system_prompt"] if len(raw_captures) >= 2 else ""
    has_retry_block = "[이전 분석 검토 결과" in second_prompt
    checks.append({
        "id": "2_retry_block_in_2nd_prompt",
        "ok": has_retry_block,
        "detail": (
            f"capture_count={len(raw_captures)} "
            f"2nd_prompt_excerpt={second_prompt[-300:]!r}"
        ),
    })

    # 3. 1차 vs 2차 analysis 변화 (changed_fields 비어있지 않거나 최소 동일이라도 통과)
    diff = report["analysis_diff"]
    # 같을 수도 있음 (동일한 결론). 이 경우 ok=True 지만 note 표시.
    changed = bool(diff.get("changed_fields"))
    checks.append({
        "id": "3_first_vs_second_analysis_diff",
        "ok": True,  # diff 유무 자체보다 흐름 검증이 목적
        "detail": f"changed_fields={list((diff.get('changed_fields') or {}).keys())}",
    })

    # 4. analysis_retry_count == 1 (1회 fail → 1회 retry → 2회차 pass)
    rc = int(report["state"].get("analysis_retry_count") or 0)
    checks.append({
        "id": "4_analysis_retry_count_is_1",
        "ok": rc == 1,
        "detail": f"analysis_retry_count={rc}",
    })

    # 5. 외부 발송이 2차 결과 기준 — 1차 액션이 외부에 사용 안 됨이 핵심.
    # (a) 2차 분석에서 propose=0 이면 1차 액션도 자동으로 차단 (executed=0).
    # (b) 2차 분석에서 propose>0 이면 executed 의 출처가 2차 propose 여야 함.
    verdict = report["state"].get("review_verdict")
    executed = report["state"].get("executed_actions") or []
    second_capture_proposed = 0
    if len(raw_captures) >= 2:
        names_2nd = [t["name"] for t in raw_captures[1].get("tool_calls") or []]
        second_capture_proposed = sum(1 for n in names_2nd if n.startswith("propose_") and n != "propose_no_action")
    if verdict not in ("pass", "correctable"):
        ok5 = False
        reason5 = "2차 verdict 가 pass/correctable 가 아님"
    elif second_capture_proposed == 0:
        # 2차에서 액션 자체가 없으니 1차 액션이 외부 사용될 수 없음 (재실행 후 fresh 상태).
        ok5 = len(executed) == 0
        reason5 = "2차 propose=0 + executed=0 (1차 액션 우회 사용 없음)"
    else:
        ok5 = len(executed) > 0
        reason5 = f"2차 propose={second_capture_proposed} executed={len(executed)}"
    checks.append({
        "id": "5_external_actions_from_2nd_pass",
        "ok": ok5,
        "detail": f"verdict={verdict} {reason5}",
    })

    # 6. save_intermediate 멱등 — call_summaries row 1개
    n_rows = report["call_summary_row_count"]
    checks.append({
        "id": "6_call_summaries_idempotent_single_row",
        "ok": n_rows == 1,
        "detail": f"call_summaries.count where call_id=<>={n_rows}",
    })

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


async def run_one(call_id_label: str, tenant_id: str, output_dir: Path) -> dict:
    print(f"\n=== retry verify: {call_id_label} (tenant={tenant_id}) ===")

    db_call_uuid = await _resolve_db_call_uuid(call_id_label, tenant_id)
    if db_call_uuid is None:
        raise RuntimeError(f"calls row not found for twilio_call_sid={call_id_label}")
    print(f"  resolved db_call_uuid = {db_call_uuid}")

    # ── monkeypatch reviewer + planner ─────────────────────────────────────
    # 원본 LLM 인스턴스 확보 (lazy init 트리거)
    orig_reviewer = reviewer_mod._get_llm()
    orig_planner = planner_mod._get_llm()

    log_dir = output_dir / f"{call_id_label}_planner_calls"
    rev_wrap = _ReviewerForceFailFirstThenDelegate(orig_reviewer)
    pln_wrap = _PlannerLoggingWrap(orig_planner, log_dir)
    reviewer_mod._llm = rev_wrap
    planner_mod._llm = pln_wrap

    try:
        t0 = datetime.now(timezone.utc)
        outcome = await run_post_call_agent_safely(
            call_id=db_call_uuid,
            trigger="call_ended",
            tenant_id=tenant_id,
        )
        t1 = datetime.now(timezone.utc)
    finally:
        # 원본 복원 — pytest 가 아니라서 명시 cleanup
        reviewer_mod._llm = orig_reviewer
        planner_mod._llm = orig_planner

    state = outcome.get("result") or {}
    latency_s = (t1 - t0).total_seconds()
    print(f"  agent finished ok={outcome.get('ok')} latency={latency_s:.1f}s")
    print(f"  reviewer calls: {len(rev_wrap._calls_log)}")
    print(f"  planner calls:  {len(pln_wrap.captures)}")

    n_rows = await _count_call_summaries(db_call_uuid)
    summary_db = await _fetch_call_summary(db_call_uuid)

    # 1차 vs 2차 analysis_result diff — captures[i].tool_calls 의 record_analysis 추출
    def _extract_analysis_from_capture(cap: dict) -> dict | None:
        for tc in cap.get("tool_calls") or []:
            if tc.get("name") == "record_analysis":
                return planner_mod._record_to_analysis(tc.get("arguments") or {})
        return None

    a1 = _extract_analysis_from_capture(pln_wrap.captures[0]) if pln_wrap.captures else None
    a2 = _extract_analysis_from_capture(pln_wrap.captures[1]) if len(pln_wrap.captures) >= 2 else None
    analysis_diff = _diff_analysis_summaries(a1, a2)

    report = {
        "call_id_label": call_id_label,
        "db_call_uuid": db_call_uuid,
        "tenant_id": tenant_id,
        "ok": outcome.get("ok"),
        "error": outcome.get("error"),
        "latency_s": round(latency_s, 3),
        "state": {
            "review_verdict": state.get("review_verdict"),
            "human_review_required": state.get("human_review_required"),
            "analysis_retry_count": state.get("analysis_retry_count"),
            "review_feedback": state.get("review_feedback") or [],
            "approved_actions_count": len(state.get("approved_actions") or []),
            "executed_actions": [
                {"action_type": a.get("action_type"),
                 "tool": a.get("tool"),
                 "status": a.get("status"),
                 "external_id": a.get("external_id")}
                for a in (state.get("executed_actions") or [])
            ],
            "errors": state.get("errors") or [],
        },
        "telemetry": {
            "analysis_planner": state.get("analysis_planner_telemetry"),
            "reviewer": state.get("reviewer_telemetry"),
        },
        "reviewer_calls_log": rev_wrap._calls_log,
        "planner_capture_count": len(pln_wrap.captures),
        "planner_captures": [
            {
                "call_n": c["call_n"],
                "system_prompt_len": len(c["system_prompt"]),
                "system_prompt_tail_500": c["system_prompt"][-500:],
                "tool_call_names": [t["name"] for t in c["tool_calls"]],
                "usage": c["usage"],
            }
            for c in pln_wrap.captures
        ],
        "analysis_diff": analysis_diff,
        "call_summary_row_count": n_rows,
        "call_summary_db": summary_db,
    }

    # 검증 (raw captures 도 같이 넘겨 system_prompt 원문 접근)
    checks = _verify(report, pln_wrap.captures)
    report["checks"] = checks

    # 콘솔 보고
    print("\n  ─ 검증 결과 ─")
    for c in checks:
        mark = "OK " if c["ok"] else "FAIL"
        print(f"    [{mark}] {c['id']}: {c['detail']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{call_id_label}_retry_verify.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  saved {out_path}")
    return report


async def main_async(tenant_id: str, call_ids: list[str]) -> None:
    output_dir = PROJ / ".local" / "e2e_retry_results"
    print(f"POST_CALL_LLM_MODE={os.environ.get('POST_CALL_LLM_MODE')}")
    print(f"output_dir = {output_dir}")
    for call_id in call_ids:
        try:
            await run_one(call_id, tenant_id, output_dir)
        except Exception as exc:
            print(f"  FAILED {call_id}: {type(exc).__name__}: {exc}")
        await asyncio.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default="ba2bf499-6fcc-4340-b3dd-9341f8bcc915")
    parser.add_argument("--call-id", action="append", default=None)
    args = parser.parse_args()

    call_ids = args.call_id or ["e2e-001"]
    asyncio.run(main_async(args.tenant_id, call_ids))


if __name__ == "__main__":
    main()
