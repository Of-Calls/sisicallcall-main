"""Agent 2 — reviewer_agent.

analysis_planner_agent 의 출력 (analysis_result + proposed_actions) 를
transcript 에 대조 검증한다.

ReAct 루프:
  - 매 step 마다 LLM 호출 → tool_calls observe → messages append
  - finalize_review 호출 시 종료
  - max_steps 초과 시 누적 결정으로 강제 종료, 결정 안 된 액션은 reject

verdict:
  pass        : 모든 액션 approve
  correctable : 일부 액션 보정 후 approve, 일부 reject 가능
  fail        : 액션 모두 차단, escalate_to_human
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

from app.agents.post_call.state import PostCallAgentState
from app.agents.post_call.tools.review_catalog import REVIEW_TOOLS_OPENAI
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 테스트에서 monkeypatch 로 교체.
_llm: Any = None

_MAX_STEPS = 5

_SYSTEM_PROMPT = """당신은 콜센터 후처리 분석/액션 검증 전문가입니다. [POST_CALL_REVIEWER]

[목적]
1. analysis_planner_agent 가 생성한 분석 결과가 transcript 에 충분히 근거하는지 확인
2. proposed_actions 가 통화 내용에 적절한지 검토
3. 잘못된 분석은 correct_analysis 로 교정, 부적절한 액션은 reject_action / correct_action 으로 처리
4. 모든 결정이 끝나면 반드시 finalize_review 를 호출하여 종료

[보정 규칙 — 매우 중요]
- 분석 결과가 transcript 와 모순되지 않으면 correct_analysis 를 호출하지 말고 바로
  approve_action 하세요. "확인 차원에서 보정"은 금지.
- neutral / low / resolved 는 강한 반대 증거 부재 시 정당한 기본값입니다.
  transcript 에 angry/high/escalated 등 다른 값에 부합하는 직접적 발화가 있을 때만
  변경하세요. "transcript 에 감정/우선순위가 명시되어 있지 않다" 같은 사유로 null
  이나 다른 값으로 변경하지 마세요 — 명시되지 않으면 기본값 그대로가 정답입니다.
- handoff_notes 가 분석에서 transcript 발화의 자연스러운 paraphrase 면 정당합니다.
  paraphrase 라는 이유로 null 처리하지 마세요. transcript 에 명백히 어긋날 때만 보정.
- correct_analysis 호출 시 반드시 transcript_evidence 에 transcript 원문 그대로의
  연속된 substring 을 인용하세요 (paraphrase 금지). 시스템이 raw transcript 에
  문자 그대로 존재하는지 검증 후 적용합니다. 인용할 게 없으면 보정 호출 금지.
- 같은 field 에 대해 보정이 한 번 drop 되면 다시 시도하지 마세요 — 다음 도구로
  넘어가서 approve_action / finalize_review 진행.
- correct_analysis 의 new_value 는 JSON null 사용 (문자열 "null" 금지).
- priority 는 priority_result.priority 가 단일 source 입니다. priority 변경은
  correct_analysis(field='priority_result.priority') 만 사용. 액션의 priority 를
  따로 바꾸려 하지 마세요 (자동 sync 됩니다).

[도구 사용 가이드]
- 의심스러운 필드는 re_read_transcript 또는 verify_field_grounding 로 근거 확인
- 분석에 명백한 오류 (transcript 에 없는 handoff_notes 등) → correct_analysis (반드시 evidence 인용)
- 정당한 액션 → approve_action
- 부적절한 액션 → reject_action
- 액션 params 일부만 잘못 → correct_action
- 위험하거나 자동 판단 불가 → escalate_to_human + finalize_review(fail)
- 정상이면 모두 approve_action 후 finalize_review(pass)

[reason 작성 규칙]
- approve_action / reject_action 의 reason 은 transcript 사실관계 또는 분석 결과
  인용에 기반한 한 문장. 일반 정책 문구("단순 문의이므로 부적절") 금지.
  예: "고객이 '환불 안 해주면 민원 넣을게요' 발화 — 강한 escalation, slack 알림 정당".

[빠른 종료 가이드 — V3-1/3]
- proposed_actions 가 비어있고 분석 결과에 transcript 와 어긋나는 명백한 오류가 없다면,
  첫 step 에서 finalize_review 호출 후 즉시 종료하세요.
- 검토할 액션이 없는데 verify_field_grounding 으로 시간을 낭비하지 마세요.

도구 호출 횟수는 step 당 여러 개 가능하지만 효율적으로. 최대 5 step 안에 finalize 하세요."""


def _format_transcripts(transcripts: list[dict]) -> str:
    if not transcripts:
        return "(녹취 없음)"
    return "\n".join(f"[{t.get('role','?')}] {t.get('text','')}" for t in transcripts)


def _set_path(d: dict, path: str, value: Any) -> None:
    """dot-path 로 dict 안 값을 설정."""
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_path(d: dict, path: str) -> Any:
    parts = path.split(".")
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ── D-3: "null" 같은 sentinel 문자열을 실제 None 으로 coerce ────────────────
_NULL_SENTINELS = frozenset({"null", "none", "undefined", ""})


def _coerce_sentinel(v):
    if isinstance(v, str) and v.strip().lower() in _NULL_SENTINELS:
        return None
    return v


def _coerce_params(d: dict) -> dict:
    return {k: _coerce_sentinel(v) for k, v in (d or {}).items()}


# ── D-2: 기본값 류 보정은 evidence 검증 강화 ───────────────────────────────
_DEFAULT_VALUES_BY_FIELD = {
    "summary.customer_emotion": frozenset({"neutral"}),
    "summary.resolution_status": frozenset({"resolved"}),
    "voc_analysis.sentiment_result.sentiment": frozenset({"neutral"}),
    "priority_result.priority": frozenset({"low"}),
    "voc_analysis.priority_result.priority": frozenset({"low"}),
}


def _is_default_class_change(field: str, current_value, new_value) -> bool:
    """current → new 가 'default 로의 변경 (또는 null/문자열 null)' 인지.

    이런 변경은 transcript_evidence 가 transcript 에 실제로 존재할 때만 허용.
    """
    if new_value is None or (isinstance(new_value, str) and new_value.strip().lower() in _NULL_SENTINELS):
        return True
    defaults = _DEFAULT_VALUES_BY_FIELD.get(field)
    if defaults and isinstance(new_value, str) and new_value.strip().lower() in defaults:
        return True
    return False


def _evidence_in_transcripts(evidence: str, transcripts: list[dict]) -> bool:
    """transcript_evidence 가 실제 transcript 발화에 substring 으로 존재하는지."""
    if not evidence:
        return False
    needle = evidence.strip()
    if len(needle) < 4:
        return False
    haystack = " ".join(str(t.get("text") or "") for t in transcripts)
    return needle in haystack


# ── D-4: priority sync — corrections 가 priority 를 바꾸면 actions 도 sync ──
def _sync_action_priorities(actions: list[dict], new_priority: str) -> list[dict]:
    out = []
    for a in actions:
        a2 = copy.deepcopy(a)
        a2["priority"] = new_priority
        # params.priority 와 jira labels 도 sync (있을 때만)
        params = a2.get("params") or {}
        if isinstance(params, dict):
            if "priority" in params:
                params["priority"] = new_priority
            labels = params.get("labels")
            if isinstance(labels, list):
                params["labels"] = [
                    new_priority if (isinstance(l, str) and l in {"low", "medium", "high", "critical"}) else l
                    for l in labels
                ]
            a2["params"] = params
        out.append(a2)
    return out


def _post_call_llm_mode() -> str:
    raw = (os.environ.get("POST_CALL_LLM_MODE") or "").strip().lower()
    if raw in {"mock", "real"}:
        return raw
    legacy = (os.environ.get("POST_CALL_USE_REAL_LLM") or "").strip().lower()
    return "real" if legacy in {"1", "true", "yes", "on"} else "mock"


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    if _post_call_llm_mode() == "real":
        from app.services.llm.gpt4o_mini import GPT4OMiniService
        _llm = GPT4OMiniService()
    else:
        _llm = _MockReviewerLLM()
    return _llm


class _MockReviewerLLM:
    """결정론적 reviewer mock — 모든 propose 를 approve 하고 첫 step 에서 finalize."""

    async def generate_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tool_choice: str = "auto",
        messages: list[dict] | None = None,
    ) -> dict:
        # messages 의 가장 최근 user payload 에서 proposed action_id 추출
        action_ids: list[str] = []
        verdict_pass = True
        text_blob = ""
        if messages:
            for m in reversed(messages):
                content = m.get("content")
                if isinstance(content, str):
                    text_blob = content
                    break
        else:
            text_blob = user_message or ""

        # 마커: "ACTION_ID: <id>"
        for line in text_blob.splitlines():
            line = line.strip()
            if line.startswith("ACTION_ID:"):
                aid = line.split(":", 1)[1].strip()
                if aid:
                    action_ids.append(aid)

        # mock 정책: transcript 에 'jira critical' 같은 단어가 단순 문의에 등장하면 reject 한 개
        is_simple_inquiry = "단순 문의" in text_blob or "운영시간" in text_blob

        calls: list[dict] = []
        for idx, aid in enumerate(action_ids):
            if is_simple_inquiry and "jira" in aid.lower():
                calls.append({
                    "id": f"call_rej_{idx}",
                    "name": "reject_action",
                    "arguments": {"action_id": aid, "reason": "[MOCK] 단순 문의에 부적절"},
                })
                verdict_pass = False
            else:
                calls.append({
                    "id": f"call_app_{idx}",
                    "name": "approve_action",
                    "arguments": {"action_id": aid},
                })

        calls.append({
            "id": "call_final",
            "name": "finalize_review",
            "arguments": {
                "verdict": "pass" if verdict_pass else "correctable",
                "summary_reason": "[MOCK] 결정론적 reviewer mock",
            },
        })
        return {
            "tool_calls": calls,
            "text": "",
            "raw_message": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": "mock"},
        }


def _make_action_id(idx: int, action: dict) -> str:
    return f"a{idx}_{action.get('action_type', 'unknown')}_{action.get('tool', 'none')}"


async def reviewer_agent_node(state: PostCallAgentState) -> dict:
    call_id: str = state["call_id"]
    tenant_id: str = state.get("tenant_id") or ""  # type: ignore[call-overload]
    transcripts: list = state.get("transcripts") or []  # type: ignore[call-overload]
    analysis: dict = dict(state.get("analysis_result") or {})  # type: ignore[call-overload]
    proposed: list = list(state.get("proposed_actions") or [])  # type: ignore[call-overload]
    errors: list = list(state.get("errors", []))  # type: ignore[call-overload]

    t0 = time.perf_counter()
    # ── 빠른 출구: transcript 또는 분석 결과 없음 → escalate ────────────────────
    if not transcripts or not analysis:
        logger.warning(
            "reviewer: transcript/analysis 부재 call_id=%s — escalate",
            call_id,
        )
        review_result = {
            "verdict": "fail",
            "approved_actions": [],
            "corrections_to_analysis": {},
            "corrections_dropped": [],
            "escalate_reason": "missing_transcript_or_analysis",
            "steps": 0,
            "rejected_actions": [],
        }
        return {
            "review_result": review_result,
            "review_verdict": "fail",
            "approved_actions": [],
            "corrections_to_analysis": {},
            "escalate_reason": "missing_transcript_or_analysis",
            "reviewer_steps": 0,
            "human_review_required": True,
            "reviewer_telemetry": {
                "calls": 0,
                "tokens": {"prompt": 0, "completion": 0, "total": 0, "model": ""},
                "tool_counts": {},
                "steps": 0,
                "max_steps_reached": False,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            },
            "errors": errors,
        }

    # ── 액션 후보 인덱싱 ─────────────────────────────────────────────────────────
    indexed: dict[str, dict] = {}
    for i, act in enumerate(proposed):
        indexed[_make_action_id(i, act)] = copy.deepcopy(act)

    decisions: dict[str, str] = {}     # action_id → "approve" | "reject" | "correct"
    correction_payloads: dict[str, dict] = {}  # action_id → patched action
    reject_reasons: dict[str, str] = {}
    analysis_corrections: dict[str, Any] = {}  # field path → new value (메타 only)
    corrections_dropped: list[dict] = []  # D-2: evidence 부족으로 drop 된 보정
    corrected_analysis = copy.deepcopy(analysis)
    escalate_reason: str | None = None
    finalize_called = False
    explicit_verdict: str | None = None
    finalize_reason: str = ""

    transcripts_text = _format_transcripts(transcripts)
    analysis_text = json.dumps(analysis, ensure_ascii=False, indent=2)
    proposed_text_lines = []
    for aid, act in indexed.items():
        proposed_text_lines.append(
            f"ACTION_ID: {aid}\n  action_type={act.get('action_type')} tool={act.get('tool')} "
            f"priority={act.get('priority')} params={json.dumps(act.get('params', {}), ensure_ascii=False)}"
        )
    proposed_text = "\n".join(proposed_text_lines) if proposed_text_lines else "(액션 후보 없음)"

    # V3-1/3: actions 0 케이스에는 명시적 fast-finalize 힌트
    fast_path_hint = ""
    if not indexed:
        fast_path_hint = (
            "\n\n[참고] 이 통화는 액션 후보가 0 개. 분석에 명백한 오류 없으면 "
            "verify/correct 호출 자제하고 첫 step 에서 finalize_review 호출."
        )

    initial_user = (
        f"[통화 녹취]\n{transcripts_text}\n\n"
        f"[분석 결과]\n{analysis_text}\n\n"
        f"[액션 후보 — 각 ACTION_ID 별로 approve/reject/correct]\n{proposed_text}"
        f"{fast_path_hint}"
    )
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": initial_user},
    ]

    llm = _get_llm()
    steps_used = 0
    # 텔레메트리 누적자
    tokens_total = {"prompt": 0, "completion": 0, "total": 0, "model": ""}
    tool_counts: dict[str, int] = {}

    for step in range(_MAX_STEPS):
        steps_used = step + 1
        try:
            response = await llm.generate_with_tools(
                system_prompt=_SYSTEM_PROMPT,
                user_message="",
                tools=REVIEW_TOOLS_OPENAI,
                temperature=0.0,
                max_tokens=1200,
                tool_choice="auto",
                messages=messages,
            )
        except Exception as exc:
            logger.error("reviewer LLM 실패 call_id=%s step=%d err=%s", call_id, step, exc)
            errors.append({"node": "reviewer_agent", "error": f"llm_failed_step_{step}: {exc}"})
            escalate_reason = f"llm_failed: {exc}"
            break

        tool_calls = list(response.get("tool_calls") or [])
        text = response.get("text") or ""
        # 토큰 누적
        usage = response.get("usage") or {}
        tokens_total["prompt"] += int(usage.get("prompt_tokens", 0) or 0)
        tokens_total["completion"] += int(usage.get("completion_tokens", 0) or 0)
        tokens_total["total"] += int(usage.get("total_tokens", 0) or 0)
        if usage.get("model"):
            tokens_total["model"] = str(usage["model"])
        # 도구 호출 카운트
        for tc in tool_calls:
            n = tc.get("name") or ""
            if n:
                tool_counts[n] = tool_counts.get(n, 0) + 1

        # raw assistant message 추가 (다음 turn 에서 OpenAI 가 tool_call_id 매칭 가능)
        raw_msg = response.get("raw_message")
        if raw_msg:
            messages.append(raw_msg)
        elif tool_calls:
            messages.append({
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                        },
                    }
                    for c in tool_calls
                ],
            })
        else:
            messages.append({"role": "assistant", "content": text})

        if not tool_calls:
            # tool_call 없음 — LLM 이 텍스트로 응답. 종료 시그널로 본다.
            logger.info("reviewer: tool_call 없음 — 종료 call_id=%s step=%d", call_id, step)
            break

        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("arguments") or {}
            obs: str = ""

            if name == "re_read_transcript":
                query = (args.get("query") or "").strip()
                obs = _do_re_read(transcripts, query)
            elif name == "verify_field_grounding":
                obs = _do_verify_grounding(transcripts, args.get("field", ""), args.get("value", ""))
            elif name == "approve_action":
                aid = str(args.get("action_id") or "")
                if aid in indexed:
                    decisions[aid] = "approve"
                    obs = f"approved {aid}"
                else:
                    obs = f"unknown_action_id: {aid}"
            elif name == "reject_action":
                aid = str(args.get("action_id") or "")
                if aid in indexed:
                    decisions[aid] = "reject"
                    reject_reasons[aid] = str(args.get("reason") or "")
                    obs = f"rejected {aid}"
                else:
                    obs = f"unknown_action_id: {aid}"
            elif name == "correct_action":
                aid = str(args.get("action_id") or "")
                new_params = args.get("new_params") or {}
                if aid in indexed and isinstance(new_params, dict):
                    # D-3: sentinel coerce
                    coerced = _coerce_params(new_params)
                    base = correction_payloads.get(aid) or copy.deepcopy(indexed[aid])
                    base["params"] = _deep_merge(base.get("params", {}), coerced)
                    correction_payloads[aid] = base
                    decisions[aid] = "correct"
                    obs = f"corrected {aid}"
                else:
                    obs = f"unknown_action_id_or_invalid_params: {aid}"
            elif name == "correct_analysis":
                field = str(args.get("field") or "")
                # D-3: sentinel coerce
                new_value = _coerce_sentinel(args.get("new_value"))
                reason = str(args.get("reason") or "")
                evidence = str(args.get("transcript_evidence") or "").strip()

                if not field:
                    obs = "missing_field"
                else:
                    # 같은 필드에 이미 drop 된 적이 있으면 즉시 strong stop signal
                    already_dropped = any(d.get("field") == field for d in corrections_dropped)
                    if already_dropped:
                        obs = (
                            f"STOP: correction on '{field}' already dropped earlier. "
                            f"The current value is acceptable. Move on to approve_action / finalize_review."
                        )
                    else:
                        current_value = _get_path(corrected_analysis, field)
                        is_default_change = _is_default_class_change(field, current_value, new_value)
                        has_evidence = _evidence_in_transcripts(evidence, transcripts)

                        # D-2: 기본값 류 변경 + evidence 미존재 → drop
                        if is_default_change and not has_evidence:
                            drop_reason = (
                                "default_class_change_without_evidence"
                                if not evidence
                                else "evidence_not_in_transcript"
                            )
                            corrections_dropped.append({
                                "field": field,
                                "new_value": new_value,
                                "reason": reason,
                                "transcript_evidence": evidence,
                                "drop_reason": drop_reason,
                            })
                            obs = (
                                f"DROPPED: correction on '{field}' rejected — {drop_reason}. "
                                f"The current value is the valid default. STOP correcting this field; "
                                f"proceed with approve_action / finalize_review."
                            )
                            logger.info(
                                "reviewer correction dropped call_id=%s field=%s new=%r drop=%s",
                                call_id, field, new_value, drop_reason,
                            )
                        else:
                            _set_path(corrected_analysis, field, new_value)
                            analysis_corrections[field] = {
                                "new_value": new_value,
                                "reason": reason,
                                "transcript_evidence": evidence,
                            }
                            obs = f"analysis corrected: {field}"
            elif name == "escalate_to_human":
                escalate_reason = str(args.get("reason") or "escalated")
                obs = f"escalated: {escalate_reason}"
            elif name == "finalize_review":
                finalize_called = True
                explicit_verdict = args.get("verdict")
                finalize_reason = str(args.get("summary_reason") or "")
                obs = "review_finalized"
            else:
                obs = f"unknown_tool: {name}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": name,
                "content": obs,
            })

        if finalize_called:
            break

    # ── max_steps 초과 ── 결정 안 된 액션은 reject 처리 ────────────────────────
    forced_close = not finalize_called
    if forced_close:
        logger.warning(
            "reviewer: max_steps 도달 또는 비정상 종료 call_id=%s steps=%d",
            call_id, steps_used,
        )

    for aid in indexed.keys():
        if aid not in decisions:
            decisions[aid] = "reject"
            reject_reasons.setdefault(
                aid,
                "max_steps_reached_no_decision" if forced_close else "no_decision",
            )

    # ── verdict 산출 ────────────────────────────────────────────────────────────
    approved_actions: list[dict] = []
    rejected_actions: list[dict] = []
    any_corrections = bool(correction_payloads) or bool(analysis_corrections)
    any_rejected = False

    for aid, action in indexed.items():
        d = decisions[aid]
        if d == "approve":
            approved_actions.append(action)
        elif d == "correct":
            patched = correction_payloads.get(aid, action)
            approved_actions.append(patched)
        else:  # reject
            rejected = copy.deepcopy(action)
            rejected["status"] = "rejected"
            rejected["reject_reason"] = reject_reasons.get(aid, "rejected")
            rejected_actions.append(rejected)
            any_rejected = True

    if escalate_reason:
        verdict = "fail"
    elif explicit_verdict in ("pass", "correctable", "fail"):
        verdict = explicit_verdict
        if verdict == "fail" and not escalate_reason:
            escalate_reason = finalize_reason or "verdict_fail_no_reason"
    elif any_rejected and not approved_actions:
        # 모두 reject → fail
        verdict = "fail"
        escalate_reason = "all_actions_rejected"
    elif any_corrections or any_rejected:
        verdict = "correctable"
    else:
        verdict = "pass"

    if verdict == "fail":
        approved_actions = []  # fail 시 외부 액션 차단

    # ── D-4: priority single source of truth — final analysis priority 로 sync ──
    final_priority = (
        (corrected_analysis.get("priority_result") or {}).get("priority")
        or "low"
    )
    approved_actions = _sync_action_priorities(approved_actions, final_priority)
    rejected_actions = _sync_action_priorities(rejected_actions, final_priority)

    review_result = {
        "verdict": verdict,
        "approved_actions": approved_actions,
        "rejected_actions": rejected_actions,
        "corrections_to_analysis": analysis_corrections,
        "corrections_dropped": corrections_dropped,
        "escalate_reason": escalate_reason,
        "steps": steps_used,
        "finalize_reason": finalize_reason,
        "forced_close": forced_close,
    }

    latency_ms = int((time.perf_counter() - t0) * 1000)
    telemetry = {
        "calls": steps_used,
        "tokens": tokens_total,
        "tool_counts": tool_counts,
        "steps": steps_used,
        "max_steps_reached": forced_close,
        "latency_ms": latency_ms,
    }

    logger.info(
        "post_call telemetry node=reviewer call_id=%s tenant=%s "
        "verdict=%s steps=%d max_reached=%s tokens=%d latency_ms=%d "
        "approved=%d rejected=%d corrections=%d dropped=%d",
        call_id, tenant_id,
        verdict, steps_used, forced_close,
        tokens_total["total"], latency_ms,
        len(approved_actions), len(rejected_actions),
        len(analysis_corrections), len(corrections_dropped),
    )

    return {
        "review_result": review_result,
        "review_verdict": verdict,
        "approved_actions": approved_actions,
        "corrections_to_analysis": analysis_corrections,
        "escalate_reason": escalate_reason,
        "reviewer_steps": steps_used,
        "reviewer_telemetry": telemetry,
        "human_review_required": verdict == "fail",
        # 분석 보정 결과 반영
        "analysis_result": corrected_analysis,
        "summary": corrected_analysis.get("summary"),
        "voc_analysis": corrected_analysis.get("voc_analysis"),
        "priority_result": corrected_analysis.get("priority_result"),
        "errors": errors,
    }


def _do_re_read(transcripts: list[dict], query: str) -> str:
    if not query:
        return "(empty_query)"
    q = query.lower()
    matches = []
    for t in transcripts:
        text = str(t.get("text") or "")
        if q in text.lower():
            matches.append(f"[{t.get('role','?')}] {text}")
    if not matches:
        return f"(no_match_for: {query})"
    return "\n".join(matches[:5])


def _do_verify_grounding(transcripts: list[dict], field: str, value: str) -> str:
    if not value:
        return json.dumps({"ok": False, "reason": "empty_value"}, ensure_ascii=False)
    haystack = " ".join(str(t.get("text") or "") for t in transcripts).lower()
    needle = str(value).lower()
    found = needle in haystack or _has_emotion_signal(needle, haystack)
    return json.dumps(
        {"ok": found, "reason": f"transcript {'contains' if found else 'lacks'} value '{value}'"},
        ensure_ascii=False,
    )


_EMOTION_KEYWORDS = {
    "angry": ("화", "짜증", "최악", "어이없", "분노"),
    "negative": ("불만", "실망", "별로"),
    "positive": ("감사", "좋", "고맙"),
}


def _has_emotion_signal(value: str, haystack: str) -> bool:
    """value 가 감정 enum 일 때 대응되는 키워드가 transcript 에 있으면 True."""
    keys = _EMOTION_KEYWORDS.get(value.strip().lower())
    if not keys:
        return False
    return any(kw in haystack for kw in keys)
