import os
import re
from datetime import datetime

from app.agents.conversational.prompts.fallback_phrases import (
    get_contact_channel,
    get_inquiry_phrase,
)
from app.agents.conversational.prompts.task import (
    build_ask_missing_prompt,
    build_humanize_prompt,
    build_select_prompt,
)
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


_POLITE_AUTH = (
    "본인 인증이 필요한 작업이에요. 인증을 진행해드릴까요?"
)
_POLITE_BLOCKED = "본인 인증이 여러 번 실패해 더 이상 진행이 어려워요. 상담원으로 연결해드릴게요."
_POLITE_MISSING_INFO = "처리에 필요한 정보를 조금 더 알려주시겠어요?"
_POLITE_DECLINE_BOOKING = "네, 예약은 진행하지 않을게요. 더 필요하신게 있으실까요?"


def _polite_no_tools(industry: str) -> str:
    return f"이곳은 자동 업무 처리가 지원되지 않아요. {get_inquiry_phrase(industry)}."


def _polite_tool_failed(industry: str) -> str:
    return f"처리 중 문제가 생겼어요. 잠시 후 다시 시도해주시거나 {get_inquiry_phrase(industry)}."


def _format_schedule_complete(time_kor: str, sms_status: str) -> str:
    """예약 등록 성공 메시지 — 결정적 조립 (humanize 우회).

    sms_status: 'sent' = 확인 문자 자동 발송 성공 / 'failed' = 발송 실패.
    """
    base = f"{time_kor}에 예약이 완료되었습니다."
    if sms_status == "sent":
        return base + " 확인 문자도 발송해드렸어요. 더 필요하신게 있으실까요?"
    return base + " 다만 확인 문자 발송에 문제가 생겼어요. 더 필요하신게 있으실까요?"


async def _send_confirm_sms_inline(
    call_id: str, tenant_id: str, sms_body: str
) -> bool:
    """예약 등록 완료 직후 자동 SMS 발송. 성공 여부 반환.

    SMS_TEST_RECIPIENT env 가 있으면 customer_phone 자동 주입 (시연용).
    """
    arguments: dict = {"message": sms_body}
    test_recipient = os.getenv("SMS_TEST_RECIPIENT", "")
    if test_recipient:
        arguments["customer_phone"] = test_recipient
    try:
        result = await mcp_client.call_tool(
            "sms", "send_confirm_sms", arguments,
            call_id=call_id, tenant_id=tenant_id,
        )
        ok = result.get("status") == "success"
        print(f"[task_branch] auto SMS status={result.get('status')}")
        return ok
    except Exception as exc:
        print(f"[task_branch] auto SMS 예외: {exc}")
        return False


def _build_sms_body(tenant_name: str, tenant_industry: str, time_kor: str) -> str:
    """예약 확정 SMS 본문 조립 — industry 별 contact channel 자동."""
    name = tenant_name or "고객센터"
    channel = get_contact_channel(tenant_industry)
    return (
        f"[{name}] 예약 확정: {time_kor}\n"
        f"변경이나 문의 사항이 있으시면 {channel} 연락 주세요."
    )


def _topic_marker(s: str) -> str:
    """한국어 받침 여부에 따라 "은/는" 선택. 받침 있으면 "은", 없으면 "는"."""
    if not s:
        return "는"
    last = s[-1]
    if not ("가" <= last <= "힣"):
        return "는"
    return "은" if (ord(last) - ord("가")) % 28 != 0 else "는"


def _is_short_affirm(user_text: str) -> bool:
    """짧은 긍정 발화 — '네', '네 보내주세요', '예 부탁해요' 등.

    pending 동의 detection 안전망: query_refine 이 핵심 규칙 3 패턴 못 만든
    경우의 fallback. 부정/변경 의도 단어가 보이면 False (false-positive 차단).
    """
    s = user_text.strip()
    if not s or len(s) > 30:
        return False
    if not s.startswith(("네", "예", "응", "좋", "넵", "오케이")):
        return False
    if any(kw in s for kw in ("말고", "변경", "취소", "안해", "안 해", "다른", "그런데", "근데")):
        return False
    return True


def _is_explicit_decline(user_text: str) -> bool:
    """명시 거절어 detection — pending 거절 안전망.

    query_refine 이 거절을 동의로 잘못 분류하는 경우의 fallback. 거절어가
    발화 시작 또는 substring 으로 보이면 True. "아니요 X로 해주세요" 같은
    하이브리드 발화도 거절 우선 처리.
    """
    s = user_text.lstrip()
    if s.startswith(("아니", "안 ", "안해", "싫", "괜찮")):
        return True
    if "말고" in s:
        return True
    return False


def _format_check_response(result_data: dict) -> str:
    """check_availability 결과 → 음성 응답 텍스트.

    LLM 우회 — 결정적 한국어 시간 표현으로 직접 조립.
    LLM 의 비결정성 + latency 모두 제거. format_korean_friendly 한 표현만.
    """
    status = result_data.get("status", "")
    requested = result_data.get("requested_time", "")
    suggestions = result_data.get("suggested_slots") or []

    try:
        req_dt = datetime.fromisoformat(requested.replace(" ", "T"))
        req_kor = format_korean_friendly(req_dt)
    except (ValueError, AttributeError):
        req_kor = requested

    sug_kor = ""
    if suggestions:
        try:
            s_dt = datetime.fromisoformat(suggestions[0].replace(" ", "T"))
            sug_kor = format_korean_friendly(s_dt)
        except (ValueError, AttributeError):
            sug_kor = ""

    req_marker = _topic_marker(req_kor)
    sug_marker = _topic_marker(sug_kor) if sug_kor else "는"

    if status == "available":
        return f"{req_kor}{req_marker} 예약 가능합니다. 진행해드릴까요?"
    if status == "closed_day":
        if sug_kor:
            return f"{req_kor}{req_marker} 휴무라 예약이 어려워요. 가장 빠른 예약 가능 시간은 {sug_kor}입니다. 진행해드릴까요?"
        return f"{req_kor}{req_marker} 휴무라 예약이 어려워요. 다른 날짜 알려주실 수 있을까요?"
    if status == "conflict":
        if sug_kor:
            return f"{req_kor}{req_marker} 다른 예약이 있어요. 같은 날 {sug_kor} 예약 가능합니다. 진행해드릴까요?"
        return f"{req_kor}{req_marker} 다른 예약이 있어요. 다른 시간 알려주실 수 있을까요?"
    return _POLITE_MISSING_INFO

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
    query: str,
    action_type: str,
    spec: dict,
    missing: list[str],
    tenant_industry: str = "",
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
        f"[처리하려는 작업]\n{spec.get('description', action_type)}\n\n"
        f"[부족한 정보]\n" + "\n".join(f"- {d}" for d in descs)
    )
    try:
        text = await _llm.generate(
            system_prompt=build_ask_missing_prompt(tenant_industry),
            user_message=user_message,
            temperature=0.2,
            max_tokens=80,
        )
        return text.strip().strip('"').strip("'") or _POLITE_MISSING_INFO
    except Exception as exc:
        print(f"[task_branch] ask_for_missing 실패: {exc}")
        return _POLITE_MISSING_INFO


async def humanize_tool_result(
    query: str,
    action_type: str,
    mcp_result: dict,
    tenant_industry: str = "",
) -> str:
    """MCP 도구 결과를 음성 친화적 한두 문장으로 변환.

    auth_branch 자동 재실행 (D-C) 에서도 재사용되므로 module-public.
    tenant_industry 는 fallback (humanize 실패 시 polite tool failed) 동적화 용.

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
            system_prompt=build_humanize_prompt(tenant_industry, _today_label()),
            user_message=user_message,
            temperature=0.2,
            max_tokens=200,
        )
        return text.strip().strip('"').strip("'") or _polite_tool_failed(tenant_industry)
    except Exception as exc:
        print(f"[task_branch] humanize 실패: {exc}")
        return _polite_tool_failed(tenant_industry)


async def _resume_schedule_callback(call_id: str, tenant_id: str, pending: dict) -> dict:
    """availability_confirmed 동의 후 schedule_callback 자동 재실행.

    등록 성공 시: 확인 SMS 자동 발송 + 결정적 완료 메시지 (sent/failed 분기).
    실패 시: polite_tool_failed.
    """
    arguments = pending.get("arguments") or {}
    user_text = pending.get("user_text", "")
    tenant_name = pending.get("tenant_name", "")
    tenant_industry = pending.get("tenant_industry", "")
    print(f"[task_branch] availability_confirmed 동의 → schedule_callback 자동 재실행 args={arguments}")

    try:
        mcp_result = await mcp_client.call_tool(
            "calendar",
            "schedule_callback",
            arguments,
            call_id=call_id,
            tenant_id=tenant_id,
        )
    except Exception as exc:
        print(f"[task_branch] resume mcp 실패: {exc}")
        return {"response_text": _polite_tool_failed(tenant_industry)}

    if mcp_result.get("status") != "success":
        return {"response_text": _polite_tool_failed(tenant_industry)}

    # 결정적 완료 메시지 + SMS 제안 — preferred_time 기반으로 한국어 시간 조립
    time_kor = ""
    preferred = arguments.get("preferred_time", "")
    try:
        dt = datetime.fromisoformat(preferred.replace(" ", "T"))
        time_kor = format_korean_friendly(dt)
    except (ValueError, AttributeError):
        pass

    if not time_kor:
        # preferred_time 파싱 실패 — fallback 으로 humanize 사용
        text = await humanize_tool_result(
            user_text, "schedule_callback", mcp_result, tenant_industry
        )
        return {"response_text": text}

    # 예약 등록 직후 확인 문자 자동 발송 — 별도 동의 turn 없이 즉시 처리.
    sms_body = _build_sms_body(tenant_name, tenant_industry, time_kor)
    sms_ok = await _send_confirm_sms_inline(call_id, tenant_id, sms_body)
    return {"response_text": _format_schedule_complete(time_kor, "sent" if sms_ok else "failed")}


async def _force_check_then_confirm(
    call_id: str,
    tenant_id: str,
    tenant_industry: str,
    tenant_name: str,
    arguments: dict,
    user_text: str,
) -> dict:
    """schedule_callback 직전 강제 2단계: check_availability → 동의 요청.

    arguments 에 tenant_industry/tenant_name 주입 — connector 가 영업시간 lookup
    + 캘린더 이벤트 title ("한밭식당 예약" 등) 조립에 사용.

    available=True → pending_task 저장 (kind=availability_confirmed) + "진행해드릴까요?"
    available=False (conflict/closed_day) → 빈 슬롯 안내 + pending 저장 X
    check 실패 → fallback 으로 직접 schedule_callback 진행
    """
    enriched_args = dict(arguments)
    enriched_args["tenant_industry"] = tenant_industry
    enriched_args["tenant_name"] = tenant_name
    # customer_phone — 캘린더 이벤트 description 에 노출 (시연 + audit 가시성).
    # 시연용 SMS_TEST_RECIPIENT 자동 주입. 운영은 Twilio caller ID 로 교체.
    test_recipient = os.getenv("SMS_TEST_RECIPIENT", "")
    if test_recipient and not enriched_args.get("customer_phone"):
        enriched_args["customer_phone"] = test_recipient

    try:
        avail_result = await mcp_client.call_tool(
            "calendar",
            "check_availability",
            enriched_args,
            call_id=call_id,
            tenant_id=tenant_id,
        )
    except Exception as exc:
        print(f"[task_branch] check_availability 예외: {exc} → 직접 schedule_callback fallback")
        return await _direct_schedule(call_id, tenant_id, enriched_args, tenant_name, tenant_industry, user_text)

    if avail_result.get("status") != "success":
        print(f"[task_branch] check_availability {avail_result.get('status')} → 직접 schedule_callback fallback")
        return await _direct_schedule(call_id, tenant_id, enriched_args, tenant_name, tenant_industry, user_text)

    result_data = avail_result.get("result") or {}
    available = bool(result_data.get("available"))
    print(f"[task_branch] check_availability available={available} status={result_data.get('status')}")

    if available:
        await _call_session_svc.set_pending_task(call_id, {
            "tool": "calendar",
            "action_type": "schedule_callback",
            "arguments": enriched_args,
            "tenant_name": tenant_name,
            "tenant_industry": tenant_industry,
            "user_text": user_text,
            "kind": "availability_confirmed",
        })
        return {"response_text": _format_check_response(result_data)}

    # closed_day / conflict + 검증된 추천 슬롯 → pending 저장 (preferred_time 을 추천으로 swap).
    # 사용자 "네" 한 마디로 추천 슬롯 자동 예약. fallback 흐름 답답함 제거.
    suggestions = result_data.get("suggested_slots") or []
    if suggestions:
        swapped_args = dict(enriched_args)
        swapped_args["preferred_time"] = suggestions[0]
        await _call_session_svc.set_pending_task(call_id, {
            "tool": "calendar",
            "action_type": "schedule_callback",
            "arguments": swapped_args,
            "tenant_name": tenant_name,
            "tenant_industry": tenant_industry,
            "user_text": user_text,
            "kind": "availability_confirmed",
        })
        print(f"[task_branch] {result_data.get('status')} + suggestion={suggestions[0]} → pending 저장")
        return {"response_text": _format_check_response(result_data)}

    # 추천 슬롯 없음 (만석) — 사용자에게 다른 시간/날짜 요청
    return {"response_text": _format_check_response(result_data)}


async def _direct_schedule(
    call_id: str,
    tenant_id: str,
    arguments: dict,
    tenant_name: str,
    tenant_industry: str,
    user_text: str,
) -> dict:
    """check_availability 실패 시 fallback — 직접 등록 + (성공 시) SMS 제안.

    pending 메커니즘 우회 안 하고 _resume_schedule_callback 와 동일한 후처리 적용.
    """
    try:
        mcp_result = await mcp_client.call_tool(
            "calendar", "schedule_callback", arguments,
            call_id=call_id, tenant_id=tenant_id,
        )
    except Exception:
        return {"response_text": _polite_tool_failed(tenant_industry)}
    if mcp_result.get("status") != "success":
        return {"response_text": _polite_tool_failed(tenant_industry)}

    time_kor = ""
    try:
        dt = datetime.fromisoformat(arguments.get("preferred_time", "").replace(" ", "T"))
        time_kor = format_korean_friendly(dt)
    except (ValueError, AttributeError):
        pass

    if not time_kor:
        text = await humanize_tool_result(
            user_text, "schedule_callback", mcp_result, tenant_industry
        )
        return {"response_text": text}

    # 예약 등록 직후 확인 문자 자동 발송 — _resume_schedule_callback 와 동일.
    sms_body = _build_sms_body(tenant_name, tenant_industry, time_kor)
    sms_ok = await _send_confirm_sms_inline(call_id, tenant_id, sms_body)
    return {"response_text": _format_schedule_complete(time_kor, "sent" if sms_ok else "failed")}


async def task_branch_node(state: CallState) -> dict:
    user_text = state["user_text"]
    rewritten = state.get("rewritten_query") or ""
    tenant_id = state["tenant_id"]
    tenant_industry = state.get("tenant_industry", "")
    tenant_name = state.get("tenant_name", "")
    call_id = state["call_id"]
    history = state.get("session_view", {}).get("conversation_history", [])

    # 0. pending 처리 — 직전 turn 의 제안 (예약 가능 안내 / SMS 안내) 에 대한
    #    동의/거절. 1차: query_refine 핵심 규칙 3 의 "사용자가 ... 동의함/거절함" 패턴.
    #    2차: 짧은 긍정 발화 + kind 별 키워드 (LLM rewrite 가 패턴 안 만든 경우 안전망).
    #    안전망: 명시 거절어 ("아니요" 등) 보이면 LLM 이 잘못 동의 패턴화해도 거절 강제.
    pending = await _call_session_svc.get_pending_task(call_id)
    if pending:
        kind = pending.get("kind", "")
        decision_src = f"{rewritten} {user_text}"
        explicit_decline = _is_explicit_decline(user_text)
        agreed = (
            "동의" in decision_src
            or "진행" in rewritten
            or _is_short_affirm(user_text)
        )
        # 거절어 우선 — "아니요 X로 해주세요" 같은 하이브리드 발화는 거절로 처리
        if explicit_decline:
            agreed = False
        declined = "거절" in decision_src or explicit_decline

        if kind == "availability_confirmed":
            if agreed:
                await _call_session_svc.clear_pending_task(call_id)
                return await _resume_schedule_callback(call_id, tenant_id, pending)
            if declined:
                await _call_session_svc.clear_pending_task(call_id)
                print(f"[task_branch] availability_confirmed 거절 → polite_decline")
                return {"response_text": _POLITE_DECLINE_BOOKING}
            # 다른 의도 (예: "내일로 변경") → pending clear 하고 정상 흐름
            await _call_session_svc.clear_pending_task(call_id)
            print(f"[task_branch] availability_confirmed 다른 의도 → pending clear")

    # 1. 매장 가용 도구 — OAuth + industry 화이트리스트 둘 다 통과한 것만 노출
    available = await get_available_actions(tenant_id, tenant_industry)
    print(f"[task_branch] tenant={tenant_id} industry={tenant_industry} available={list(available.keys())}")

    if not available:
        return {"response_text": _polite_no_tools(tenant_industry)}

    # 2. LLM Function Calling
    tools = to_openai_tools(available)
    user_message = _format_user_message(rewritten, user_text, history)

    try:
        result = await _llm.generate_with_tools(
            system_prompt=build_select_prompt(tenant_industry, _today_label()),
            user_message=user_message,
            tools=tools,
            temperature=0.1,
            max_tokens=300,
            tool_choice="auto",  # LLM 이 도구 부적합 판단 시 호출 안 하고 polite refuse 가능
        )
    except Exception as exc:
        print(f"[task_branch] LLM 실패: {exc}")
        return {"response_text": _polite_tool_failed(tenant_industry)}

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
        return {"response_text": _polite_no_tools(tenant_industry)}

    spec = available[action_type]
    print(f"[task_branch] selected={action_type} args={arguments}")

    # 안전망 — schedule_callback preferred_time 형식 검증.
    # LLM 이 시간 누락된 'YYYY-MM-DD' 만 채워도 빈 값 강제 → ask_missing 발동.
    # calendar connector 의 silent fallback (자정 00:00) 차단.
    if action_type == "schedule_callback":
        pt = arguments.get("preferred_time", "")
        if pt and not re.fullmatch(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", pt):
            print(f"[task_branch] preferred_time 형식 invalid '{pt}' → 빈 값 강제")
            arguments["preferred_time"] = ""

    # company_db 도구의 phone_number 자동 주입 (시연용 안전망).
    # required 사전 계산 + requires_auth 게이트 보다 먼저 실행 — pending_task 저장 시점에도
    # phone_number 가 채워져 있어야 verified 후 자동 재실행 흐름이 안전하게 작동.
    if spec["tool"] == "company_db" and "phone_number" in spec["parameters"].get("properties", {}):
        test_recipient = os.getenv("SMS_TEST_RECIPIENT", "")
        if test_recipient and not arguments.get("phone_number"):
            arguments["phone_number"] = test_recipient
            print(f"[task_branch] company_db phone_number 자동 주입: {test_recipient}")

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
        text = await ask_for_missing(user_text, action_type, spec, missing, tenant_industry)
        return {"response_text": text}

    # 5a. schedule_callback 강제 2단계 — check_availability 먼저, pending_task 저장 후 동의 요청.
    #     사용자 동의는 다음 turn 의 query_refine 핵심 규칙 3 + 위 step 0 으로 처리.
    if action_type == "schedule_callback":
        return await _force_check_then_confirm(
            call_id, tenant_id, tenant_industry, tenant_name, arguments, user_text
        )

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
        return {"response_text": _polite_tool_failed(tenant_industry)}

    # 6. 결과 humanize
    text = await humanize_tool_result(user_text, action_type, mcp_result, tenant_industry)
    return {"response_text": text}
