import os
from datetime import datetime

from app.agents.conversational.state import CallState
from app.agents.conversational.tool_catalog import (
    get_available_actions,
    to_openai_tools,
)
from app.services.auth.session import AuthSessionService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.mcp.client import mcp_client
from app.services.session.redis_session import RedisSessionService
from app.utils.korean_time import format_korean_friendly

_llm = GPT4OMiniService()
_auth_session_svc = AuthSessionService()
_call_session_svc = RedisSessionService()
_HISTORY_TURN_LIMIT = 6

_KOREAN_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def _today_label() -> str:
    """task/humanize prompt 에 주입할 '오늘' 라벨 — 'YYYY-MM-DD (요일)'."""
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d')} ({_KOREAN_WEEKDAYS[now.weekday()]})"


_POLITE_NO_TOOLS = "이 매장은 자동 업무 처리가 지원되지 않아요. 매장으로 직접 문의해주세요."
_POLITE_AUTH = "본인 인증이 필요한 작업이에요. 인증 진행해드릴까요?"
_POLITE_BLOCKED = "본인 인증이 여러 번 실패해 더 이상 진행이 어려워요. 상담원으로 연결해드릴게요."
_POLITE_MISSING_INFO = "처리에 필요한 정보를 조금 더 알려주시겠어요?"
_POLITE_TOOL_FAILED = "처리 중 문제가 생겼어요. 잠시 후 다시 시도해주시거나 매장으로 문의해주세요."

_SELECT_SYSTEM_PROMPT_TEMPLATE = """당신은 매장 전화 상담 AI 의 업무 처리 도구 선택기입니다.

[현재 날짜] {today}

[지침]
- 사용자 요청에 가장 잘 맞는 도구를 선택해 호출하세요. 인자가 부족해도 일단 호출하세요 — 시스템이 부족한 인자나 인증 필요 여부를 처리합니다.
- 환각 금지 — 사용자가 명시하지 않은 시간/번호/이름을 추측해서 채우지 마세요. 모르면 빈 문자열로 두세요.
- 시간 인자 처리 (매우 중요):
  · [재작성된 의도] 에 이미 절대 날짜 (YYYY-MM-DD HH:MM 형식) 가 있으면 그 값을 **글자 그대로** args 에 넣으세요.
    사용자 발화의 요일 표현 ("이번주 토요일" 등) 과 다르더라도 [재작성된 의도] 의 절대 날짜를 신뢰합니다.
    날짜를 임의로 다시 계산하지 마세요.
  · [재작성된 의도] 에 절대 날짜가 없고 사용자 발화에만 상대 표현이 있으면, [현재 날짜] 기준으로 계산해서 절대 날짜로 채우세요.
- 도구로 처리할 수 없는 요청 (예약/조회/문자 같은 도구 작업이 아닌 일반 안내) 일 때만 호출 없이 "이 작업은 매장으로 직접 문의 부탁드려요" 라고 답하세요. 따옴표/머릿말 금지."""

_ASK_MISSING_SYSTEM_PROMPT = """당신은 매장 전화 상담 AI 입니다.
사용자가 요청한 작업에 필요한 정보가 일부 부족합니다.
부족한 정보의 의미를 보고, 자연스러운 한국어 한 문장으로 사용자에게 물어보세요.

[지침]
- 친절한 어조 ("혹시", "죄송하지만" 같은 부드러운 표현)
- 한 문장만. 따옴표/머릿말 금지."""

_HUMANIZE_SYSTEM_PROMPT_TEMPLATE = """당신은 매장 전화 상담 AI 입니다. MCP 도구 호출 결과를 사용자에게 친절한 음성 안내로 전달하세요.

[현재 날짜] {today}

[지침]
- 한두 문장으로 자연스럽게.
- 결과 데이터에 있는 사실만 사용. 없는 정보 추측 금지.
- "도구", "API" 같은 메타 표현 금지. 매장 직원처럼 응답.
- 음성 출력이므로 URL/링크/마크다운 ([텍스트](URL))/이메일/event_id 등 내부 식별자 절대 출력 금지.
- 결과 데이터의 날짜는 사실로 받아들이세요. 결과 데이터의 날짜와 사용자 발화의 요일이 다르면 결과 데이터를 신뢰하고, 그 날짜의 실제 요일을 [현재 날짜] 기준으로 직접 계산하세요.
- 결과 데이터의 날짜가 'YYYY-MM-DD HH:MM' 형식이면 음성 친화적 한국어 ("5월 8일 금요일 오후 3시") 로 변환해서 안내하세요.
- [코드 계산된 한국어 시각] 섹션이 있으면 그 표현 한 번만 그대로 안내에 사용하세요. 자체 요일/시각 계산 절대 금지. "내일", "다음 주 X요일", "2026년" 같은 추가 시간 표현/연도 절대 추가하지 말 것 — 코드 결과 한 표현만 깔끔하게 음성으로 자연스럽게.
- 출력은 응답 텍스트만. 따옴표/머릿말 금지."""


def _format_user_message(rewritten: str, user_text: str, history: list) -> str:
    """LLM Function Calling 입력 포맷.

    [현재 사용자 발화] (원본) + [재작성된 의도] (참고용) 둘 다 노출.
    query_refine 이 정보 빠뜨린 경우 LLM 이 원본에서 보완 가능.
    """
    sections = []
    if history:
        lines = []
        for entry in history[-_HISTORY_TURN_LIMIT:]:
            role = "사용자" if entry.get("role") == "user" else "AI"
            lines.append(f"{role}: {entry.get('text', '')}")
        sections.append("[이전 대화]\n" + "\n".join(lines))
    sections.append(f"[현재 사용자 발화]\n{user_text}")
    if rewritten and rewritten != user_text:
        sections.append(f"[재작성된 의도 (참고용)]\n{rewritten}")
    return "\n\n".join(sections)


async def ask_for_missing(
    query: str, action_type: str, spec: dict, missing: list[str]
) -> str:
    """누락된 required 인자에 대해 자연스러운 역질문 생성.

    TOOL_CATALOG 의 parameters.properties[k].description 을 LLM 에게 넘겨
    음성 친화적 한 문장 응답을 만든다. 실패 시 _POLITE_MISSING_INFO fallback.

    auth_branch 자동 재실행 (D-C) 의 args 부족 분기에서도 재사용되므로 module-public.
    """
    properties = (spec.get("parameters") or {}).get("properties") or {}
    descs = [
        properties[k]["description"]
        for k in missing
        if k in properties and "description" in properties[k]
    ]
    if not descs:
        return _POLITE_MISSING_INFO

    user_message = (
        f"[사용자 요청]\n{query}\n\n"
        f"[처리하려는 작업]\n{action_type}\n\n"
        f"[부족한 정보]\n" + "\n".join(f"- {d}" for d in descs)
    )
    try:
        text = await _llm.generate(
            system_prompt=_ASK_MISSING_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.2,
            max_tokens=80,
        )
        return text.strip().strip('"').strip("'") or _POLITE_MISSING_INFO
    except Exception as exc:
        print(f"[task_branch] ask_for_missing 실패: {exc}")
        return _POLITE_MISSING_INFO


async def humanize_tool_result(query: str, action_type: str, mcp_result: dict) -> str:
    """MCP 도구 결과를 음성 친화적 한두 문장으로 변환.

    auth_branch 자동 재실행 (D-C) 에서도 재사용되므로 module-public.

    시간 필드 (preferred_time, scheduled_time, start_time) 가 있으면
    코드로 한국어 친화 표현 ("5월 9일 토요일 오후 7시") 미리 계산해
    LLM 한테 hint 로 넘김 — LLM 의 요일/시각 비결정성 차단.
    """
    result_data = mcp_result.get("result")
    formatted_time_hint = ""
    # 시간 필드는 코드 hint 로 분리 + raw 는 결과 데이터에서 제거.
    # LLM 이 ISO/사용자 발화를 mix 해서 "내일 5월 8일", "2026년 5월 12일" 같이
    # 음성에 redundant 표기하는 것 차단.
    if isinstance(result_data, dict):
        result_data = dict(result_data)
        for time_field in ("preferred_time", "scheduled_time", "start_time", "datetime"):
            time_str = result_data.get(time_field)
            if isinstance(time_str, str) and time_str.strip():
                try:
                    dt = datetime.fromisoformat(time_str.strip().replace(" ", "T"))
                except (ValueError, AttributeError):
                    continue
                formatted_time_hint = (
                    f"\n\n[코드 계산된 한국어 시각 — 반드시 이 표현만 그대로 사용]\n"
                    f"{format_korean_friendly(dt)}"
                )
                result_data.pop(time_field, None)
                break

    user_message = (
        f"[사용자 요청]\n{query}\n\n"
        f"[처리한 작업]\n{action_type}\n\n"
        f"[결과 데이터]\n{result_data}"
        f"{formatted_time_hint}"
    )
    try:
        text = await _llm.generate(
            system_prompt=_HUMANIZE_SYSTEM_PROMPT_TEMPLATE.format(today=_today_label()),
            user_message=user_message,
            temperature=0.2,
            max_tokens=200,
        )
        return text.strip().strip('"').strip("'") or _POLITE_TOOL_FAILED
    except Exception as exc:
        print(f"[task_branch] humanize 실패: {exc}")
        return _POLITE_TOOL_FAILED


async def task_branch_node(state: CallState) -> dict:
    user_text = state["user_text"]
    rewritten = state.get("rewritten_query") or ""
    tenant_id = state["tenant_id"]
    call_id = state["call_id"]
    history = state.get("session_view", {}).get("conversation_history", [])

    # 1. 매장 가용 도구
    available = await get_available_actions(tenant_id)
    print(f"[task_branch] tenant={tenant_id} available={list(available.keys())}")

    if not available:
        return {"response_text": _POLITE_NO_TOOLS}

    # 2. LLM Function Calling
    tools = to_openai_tools(available)
    user_message = _format_user_message(rewritten, user_text, history)

    try:
        result = await _llm.generate_with_tools(
            system_prompt=_SELECT_SYSTEM_PROMPT_TEMPLATE.format(today=_today_label()),
            user_message=user_message,
            tools=tools,
            temperature=0.1,
            max_tokens=300,
            tool_choice="required",  # LLM 이 인자 부족해도 무조건 tool_call → 게이트가 처리
        )
    except Exception as exc:
        print(f"[task_branch] LLM 실패: {exc}")
        return {"response_text": _POLITE_TOOL_FAILED}

    # (A) tool_call 없음 — LLM 자체 텍스트 응답 (ask 또는 거부)
    if result.get("tool_call") is None:
        text = (result.get("text") or "").strip().strip('"').strip("'")
        print(f"[task_branch] LLM ask/refuse text='{text}'")
        return {"response_text": text or _POLITE_MISSING_INFO}

    tool_call = result["tool_call"]
    action_type = tool_call["name"]
    arguments = tool_call.get("arguments") or {}

    # 방어: LLM 이 카탈로그 외 도구 환각
    if action_type not in available:
        print(f"[task_branch] 알 수 없는 action_type={action_type}")
        return {"response_text": _POLITE_NO_TOOLS}

    spec = available[action_type]
    print(f"[task_branch] selected={action_type} args={arguments}")

    # required 인자 사전 계산 — auth 게이트의 pending_task 저장 정책에 사용.
    required = (spec.get("parameters") or {}).get("required") or []
    missing = [k for k in required if not arguments.get(k)]

    # 3. requires_auth 게이트 (required 검증보다 먼저) — verified=우회, blocked=상담원,
    #    그 외(pending/세션없음)=pending_task 저장 + polite_auth. args 부족해도 항상 저장 —
    #    auth_branch 가 verified 시점에 spec 재조회해서 missing 있으면 ask_for_missing 처리.
    #    인증 필요한 도구는 인자 묻기 전에 인증부터 — phone 같은 인자 응답하다가
    #    auth 로 새는 흐름 차단.
    if spec.get("requires_auth"):
        auth_id = await _call_session_svc.get_auth_id(call_id)
        auth_session = (
            await _auth_session_svc.get_session(auth_id) if auth_id else None
        )
        status = auth_session.get("status") if auth_session else None
        if status == "verified":
            print(f"[task_branch] requires_auth=True, auth verified → 게이트 우회")
        elif status == "blocked":
            print(f"[task_branch] requires_auth=True, auth blocked → polite_blocked")
            return {"response_text": _POLITE_BLOCKED}
        else:
            await _call_session_svc.set_pending_task(call_id, {
                "tool": spec["tool"],
                "action_type": action_type,
                "arguments": arguments,
                "user_text": user_text,
            })
            print(f"[task_branch] requires_auth=True, status={status}, missing={missing} → pending_task 저장 + polite_auth")
            return {"response_text": _POLITE_AUTH}

    # 4. required 인자 사후 검증 (LLM 환각/생략 안전망)
    if missing:
        print(f"[task_branch] required 부족 missing={missing}")
        text = await ask_for_missing(user_text, action_type, spec, missing)
        return {"response_text": text}

    # 5. mcp_client 호출 (sms 인 경우 수신자 자동 주입 — 시연용 안전망)
    if spec["tool"] == "sms":
        test_recipient = os.getenv("SMS_TEST_RECIPIENT", "")
        if test_recipient:
            arguments["customer_phone"] = test_recipient
            print(f"[task_branch] sms customer_phone 자동 주입: {test_recipient}")

    print(f"[task_branch] mcp_client.call_tool(tool={spec['tool']}, action={action_type})")
    mcp_result = await mcp_client.call_tool(
        spec["tool"],
        action_type,
        arguments,
        call_id=call_id,
        tenant_id=tenant_id,
    )
    print(f"[task_branch] mcp_result status={mcp_result.get('status')}")

    if mcp_result.get("status") != "success":
        return {"response_text": _POLITE_TOOL_FAILED}

    # 6. 결과 humanize
    text = await humanize_tool_result(user_text, action_type, mcp_result)
    return {"response_text": text}
