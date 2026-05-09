"""실 OpenAI API 검증용 smoke test.

3 transcript x 2 tenant = 6 run, POST_CALL_LLM_MODE=real.
모든 외부 호출(DB / MCP) 은 stub 으로 차단. LLM 만 실 호출.

산출:
  .local/smoke_test_results/run_N_<scenario>_<tenant>.json
  .local/smoke_test_results/summary.json
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv
load_dotenv(PROJ / ".env", override=False)

# ── 실 LLM 모드 강제 ────────────────────────────────────────────────────────
os.environ["POST_CALL_LLM_MODE"] = "real"
# tenant_integrations 는 db-only repository 가 됐으므로 별도 storage 강제 불필요.
# 이 스크립트는 list_integrations 를 monkeypatch 로 fake 하므로 DB 조회도 안 한다.
os.environ.setdefault("MCP_ACTION_LOG_STORE", "file")
os.environ.setdefault("MCP_ACTION_LOG_FILE", str(PROJ / ".local" / "smoke_test_results" / "_action_log.json"))


import app.agents.post_call.nodes.analysis_planner_agent_node as planner_mod  # noqa: E402
import app.agents.post_call.nodes.reviewer_agent_node as reviewer_mod  # noqa: E402
import app.agents.post_call.tools.action_catalog as catalog_mod  # noqa: E402
import app.agents.post_call.nodes.load_context_node as load_mod  # noqa: E402
import app.agents.post_call.nodes.save_result_node as save_mod  # noqa: E402
import app.agents.post_call.nodes.action_executor_node as exec_mod  # noqa: E402
from app.services.llm.gpt4o import GPT4OService  # noqa: E402
from app.services.llm.gpt4o_mini import GPT4OMiniService  # noqa: E402
from app.models.tenant_integration import IntegrationStatus  # noqa: E402


# ── 합성 transcript ────────────────────────────────────────────────────────

TRANSCRIPTS: dict[str, list[dict]] = {
    "angry_unresolved": [
        {"role": "customer", "text": "지난주 주문한 가방, 사진이랑 색상이 완전히 다른데 환불 처리해주세요."},
        {"role": "agent", "text": "고객님, 단순 변심으로는 환불이 어렵습니다."},
        {"role": "customer", "text": "단순 변심이 아니라 상품 설명이랑 다르다고요. 이거 사기 아닌가요?"},
        {"role": "agent", "text": "정책상 7일 이내라도 사용 흔적이 있으면 환불이 어렵습니다."},
        {"role": "customer", "text": "사용도 안 했어요. 박스 그대로 있는데 사용 흔적이 어디 있어요? 정말 화가 납니다."},
        {"role": "agent", "text": "내부에서 검토 후 다시 연락드리겠습니다."},
        {"role": "customer", "text": "검토 같은 소리 하지 마세요. 소비자보호원 신고하고 민원 넣을 거예요."},
        {"role": "agent", "text": "죄송합니다. 상부에 보고드리겠습니다."},
    ],
    "simple_inquiry": [
        {"role": "customer", "text": "거기 영업시간이 어떻게 되나요?"},
        {"role": "agent", "text": "평일 오전 9시부터 오후 6시까지이고, 주말은 휴무입니다."},
        {"role": "customer", "text": "위치도 알려주실 수 있나요?"},
        {"role": "agent", "text": "강남역 3번 출구에서 도보 5분 거리입니다."},
        {"role": "customer", "text": "감사합니다."},
    ],
    "callback_request": [
        {"role": "customer", "text": "지금 회의 중이라 통화가 어려운데, 내일 오후 3시에 다시 전화 주실 수 있나요?"},
        {"role": "agent", "text": "네, 내일 오후 3시에 콜백 예약 도와드리겠습니다. 연락처 확인 부탁드립니다."},
        {"role": "customer", "text": "010-1234-5678입니다. 잘 부탁드립니다."},
        {"role": "agent", "text": "010-1234-5678로 내일 15:00 콜백 예약 완료했습니다."},
    ],
}


# ── tenant 카탈로그 fixture ────────────────────────────────────────────────

TENANT_INTEGRATIONS: dict[str, list[str]] = {
    "tenant-A": ["slack", "google_calendar", "jira", "gmail"],   # 모두 연결
    "tenant-B": ["google_calendar", "jira", "gmail"],            # slack 미연결
}


def fake_list_integrations(tenant_id: str):
    providers = TENANT_INTEGRATIONS.get(tenant_id, [])
    return [
        SimpleNamespace(provider=p, status=IntegrationStatus.connected)
        for p in providers
    ]


# ── stub repos ──────────────────────────────────────────────────────────────

class _FakeContextRepo:
    def __init__(self, transcripts: list[dict], tenant_id: str):
        self._transcripts = transcripts
        self._tenant_id = tenant_id

    async def get_call_context(self, call_id: str) -> dict:
        return {
            "metadata": {"call_id": call_id, "tenant_id": self._tenant_id, "customer_phone": "01012345678"},
            "transcripts": self._transcripts,
            "branch_stats": {},
        }


class _StubExecutor:
    """실 MCP 호출 차단 — approved_actions 가 들어오면 success 만 반환."""
    def __init__(self):
        self.calls: list[dict] = []

    async def execute_actions(self, *, call_id: str, tenant_id: str, actions: list[dict]) -> list[dict]:
        self.calls.append({"call_id": call_id, "tenant_id": tenant_id, "count": len(actions)})
        return [
            {
                **a,
                "status": "success",
                "external_id": f"smoke-{call_id}-{i}",
                "error": None,
                "result": {"smoke_test": True, "via_mcp": False},
            }
            for i, a in enumerate(actions)
        ]


# ── 인스트루멘티드 LLM 래퍼 ─────────────────────────────────────────────────

class InstrumentedLLM:
    def __init__(self, inner, label: str):
        self._inner = inner
        self.label = label
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.tool_log: list[dict] = []   # [{step, name, arguments}]
        self.text_responses: list[str] = []
        self.errors: list[str] = []
        self._tool_counter: Counter = Counter()
        self._client_create = inner._client.chat.completions.create

    async def generate_with_tools(self, **kwargs):
        captured_usage = {"value": None}
        orig_create = self._inner._client.chat.completions.create

        async def wrapped_create(*a, **kw):
            response = await orig_create(*a, **kw)
            usage = getattr(response, "usage", None)
            if usage:
                captured_usage["value"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                }
            return response

        # patch in
        self._inner._client.chat.completions.create = wrapped_create
        step_idx = self.calls + 1
        try:
            result = await self._inner.generate_with_tools(**kwargs)
        except Exception as exc:
            self.errors.append(f"step {step_idx}: {type(exc).__name__}: {exc}")
            raise
        finally:
            self._inner._client.chat.completions.create = orig_create

        self.calls += 1
        if captured_usage["value"]:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                self.usage[k] += captured_usage["value"][k]

        for tc in (result.get("tool_calls") or []):
            self._tool_counter[tc["name"]] += 1
            self.tool_log.append({
                "step": step_idx,
                "name": tc["name"],
                "arguments": tc.get("arguments", {}),
            })
        if result.get("text"):
            self.text_responses.append(result["text"])
        return result

    def snapshot(self) -> dict:
        return {
            "label": self.label,
            "calls": self.calls,
            "usage": self.usage,
            "tool_counts": dict(self._tool_counter),
            "tool_log": self.tool_log,
            "text_responses": self.text_responses,
            "errors": self.errors,
        }


# ── 직렬화 헬퍼 ─────────────────────────────────────────────────────────────

def _serializable_state(state) -> dict | None:
    if not state:
        return None
    keys = (
        "call_id", "tenant_id", "trigger",
        "summary", "voc_analysis", "priority_result",
        "analysis_result", "proposed_actions",
        "review_result", "review_verdict", "approved_actions",
        "corrections_to_analysis", "escalate_reason", "reviewer_steps",
        "executed_actions", "human_review_required",
        "errors", "partial_success",
        "analysis_planner_rationale",
        # v3: telemetry 키
        "analysis_planner_telemetry", "reviewer_telemetry",
    )
    out = {k: state.get(k) for k in keys}
    # dashboard_payload.telemetry 도 별도 보관
    dp = state.get("dashboard_payload") or {}
    out["dashboard_telemetry"] = dp.get("telemetry") if isinstance(dp, dict) else None
    return out


# ── 단일 run ────────────────────────────────────────────────────────────────

async def run_single(
    *,
    scenario_name: str,
    transcripts: list[dict],
    tenant_id: str,
    run_n: int,
    output_dir: Path,
    retries: int = 2,
) -> dict:
    # singleton 리셋
    planner_mod._llm = None
    reviewer_mod._llm = None

    # 인스트루멘티드 LLM
    instr_planner = InstrumentedLLM(GPT4OService(), "planner")
    instr_reviewer = InstrumentedLLM(GPT4OMiniService(), "reviewer")
    planner_mod._llm = instr_planner
    reviewer_mod._llm = instr_reviewer

    # 카탈로그 필터 patch
    catalog_mod.list_integrations = fake_list_integrations

    # context repo 교체
    fake_context_repo = _FakeContextRepo(transcripts, tenant_id)
    load_mod._repo = fake_context_repo

    # save_result repos 무효화 (DB 안 건드리게)
    async def _noop(*a, **kw):
        return None

    save_mod._summary_repo.save_summary = _noop
    save_mod._voc_repo.save_voc_analysis = _noop
    save_mod._action_log_repo.save_action_log = _noop
    save_mod._dashboard_repo.upsert_dashboard = _noop

    # executor stub
    stub_exec = _StubExecutor()
    exec_mod._executor = stub_exec

    # tenant 카탈로그 스냅샷 — 이 시점에서 평가
    catalog = catalog_mod.get_action_catalog(tenant_id)
    catalog_names = [e["name"] for e in catalog]

    # PostCallAgent 빌드
    from app.agents.post_call.agent import PostCallAgent
    from app.agents.post_call.graph import build_post_call_graph
    agent = PostCallAgent()
    agent._graph = build_post_call_graph()

    call_id = f"smoke-r{run_n}-{scenario_name}-{tenant_id}"

    # 실행 (rate limit 대비 retry)
    state = None
    error = None
    last_exc = None
    t0 = time.perf_counter()
    for attempt in range(retries):
        try:
            state = await agent.run(call_id=call_id, trigger="call_ended", tenant_id=tenant_id)
            error = None
            break
        except Exception as exc:
            last_exc = exc
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [retry {attempt+1}] {error}")
            if "rate" in str(exc).lower() or "429" in str(exc):
                await asyncio.sleep(15)
            else:
                await asyncio.sleep(2)
    t1 = time.perf_counter()

    if state is None and last_exc is not None:
        traceback.print_exception(type(last_exc), last_exc, last_exc.__traceback__)

    out = {
        "run": run_n,
        "scenario": scenario_name,
        "tenant_id": tenant_id,
        "tenant_catalog": catalog_names,
        "call_id": call_id,
        "latency_s": round(t1 - t0, 3),
        "error": error,
        "executor_stub_calls": stub_exec.calls,
        "planner_telemetry": instr_planner.snapshot(),
        "reviewer_telemetry": instr_reviewer.snapshot(),
        "state": _serializable_state(state),
    }

    out_path = output_dir / f"run_{run_n}_{scenario_name}_{tenant_id}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out


async def main():
    sub = os.environ.get("SMOKE_TEST_DIR", "smoke_test_results")
    output_dir = PROJ / ".local" / sub
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"output dir: {output_dir}")
    print(f"OPENAI_API_KEY: {'present' if os.environ.get('OPENAI_API_KEY') else 'MISSING'}")
    print(f"POST_CALL_LLM_MODE: {os.environ.get('POST_CALL_LLM_MODE')}")

    results: list[dict] = []
    run_n = 0
    for scenario_name, transcripts in TRANSCRIPTS.items():
        for tenant_id in TENANT_INTEGRATIONS.keys():
            run_n += 1
            print(f"\n=== Run {run_n}: {scenario_name} on {tenant_id} ===")
            try:
                r = await run_single(
                    scenario_name=scenario_name,
                    transcripts=transcripts,
                    tenant_id=tenant_id,
                    run_n=run_n,
                    output_dir=output_dir,
                )
                results.append(r)
                state = r.get("state") or {}
                print(
                    f"  verdict={state.get('review_verdict')} "
                    f"latency={r['latency_s']}s "
                    f"planner_calls={r['planner_telemetry']['calls']} "
                    f"reviewer_steps={r['reviewer_telemetry']['calls']} "
                    f"planner_tok={r['planner_telemetry']['usage']['total_tokens']} "
                    f"reviewer_tok={r['reviewer_telemetry']['usage']['total_tokens']} "
                    f"approved={len(state.get('approved_actions') or [])} "
                    f"executed={len(state.get('executed_actions') or [])}"
                )
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                results.append({"run": run_n, "scenario": scenario_name, "tenant_id": tenant_id, "error": str(exc)})
            await asyncio.sleep(2)  # 양보

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {len(results)} runs to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
