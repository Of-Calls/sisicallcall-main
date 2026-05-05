import asyncio
import audioop
import base64
import json
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.state import CallState
from app.api.v1.admin_auth import get_current_admin_user
from app.api.v1._call_session import CallSession, spawn_background
from app.api.v1._tenant_helpers import (
    get_greeting,
    get_tenant_name,
    resolve_tenant_id,
)
from app.core.events import CALL_ENDED, CALL_STARTED
from app.repositories.call_repo import (
    get_call_by_id_for_tenant,
    insert_call,
    list_calls_for_tenant,
)
from app.repositories.transcript_repo import get_transcripts_by_call_id, insert_transcript
from app.services.session.redis_session import RedisSessionService
from app.services.stt.deepgram import DeepgramSTTService
from app.services.stt.deepgram_streaming import DeepgramStreamingSTTService
from app.services.stt.keyterm_cache import get_tenant_keyterms
from app.services.tts.channel import tts_channel
from app.services.vad.silero_vad import SileroVADService
from app.utils.audio import mulaw_to_pcm16
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def _current_admin_tenant_id(current_admin: dict[str, Any]) -> str:
    user = current_admin.get("user") or {}
    tenant_id = str(user.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin tenant",
        )
    return tenant_id


def _validate_query_tenant_id(query_tenant_id: str | None, jwt_tenant_id: str) -> None:
    if not query_tenant_id:
        return
    if query_tenant_id.strip().lower() != jwt_tenant_id.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Query tenant mismatch",
        )

# 그래프 싱글톤 (앱 기동 시 1회 컴파일) — 텍스트 워크플로우 전용.
# audio 처리 (VAD / 화자검증 / STT / enrollment) 는 graph 진입 전 본 파일에서 처리.
def _get_voiceprint_enrollment():
    from app.services.speaker_verify import enrollment as voiceprint_enrollment

    return voiceprint_enrollment


def _get_titanet_service():
    from app.services.speaker_verify.titanet import get_titanet_service

    return get_titanet_service()


_graph = build_call_graph()
_session_service = RedisSessionService()
_streaming_stt = DeepgramStreamingSTTService()
_prerecorded_stt = DeepgramSTTService()
# Silero VAD — 메인 발화 게이트 + barge-in verify 1차. 같은 인스턴스 공유 가능 (stateless).
_silero_vad = SileroVADService()
_bargein_vad = _silero_vad  # alias — barge-in 코드 가독성용

# ── 발화 감지 파라미터 ────────────────────────────────────────────────────────
# Silero VAD 를 메인 게이트로 사용. RMS 는 cheap pre-filter (완전 침묵/정적 차단만).
_VAD_RMS_PREFILTER = 200           # RMS < 200 이면 Silero 호출 생략 (완전 침묵)
_VAD_WINDOW_BYTES = 5120           # Silero 슬라이딩 윈도우 (~160ms, 5 frames @ 16kHz)
                                   # 5 frames 중 min_speech_frames(3) 이상 speech 이면 True
_BARGEIN_RMS_THRESHOLD = 2400
_BARGEIN_GRACE_SEC = 3.0
_SILENCE_CHUNKS_TO_END = 65        # 발화 종료 묵음 청크 수 (~1300ms). UtteranceEnd 가 먼저
                                   # 오면 즉시 trigger, 본 카운팅은 fallback 안전망.
_MIN_UTTERANCE_BYTES = 6400        # 최소 발화 길이 (~200ms) — "네"/"예" 짧은 응답 허용
_MAX_UTTERANCE_BYTES = 320000      # 최대 발화 길이 (~10s) 강제 처리

# ── 침묵 감지 파라미터 ────────────────────────────────────────────────────────
_SILENCE_FIRST_SEC = 10.0
_SILENCE_SECOND_SEC = 10.0
_KO_TTS_CHARS_PER_SEC = 3.0
_GREETING_PLAY_BUFFER_SEC = 13.0
_MSG_SILENCE_CHECK = "통화 중이십니까? 불편한 점이 있으시면 말씀해 주세요."
_MSG_SILENCE_ESCALATION = "전화 연결이 원활하지 않은 것 같습니다. 상담원에게 연결해 드리겠습니다."

# ── 빈 STT escalation ─────────────────────────────────────────────────────────
_EMPTY_STT_ESCALATION_THRESHOLD = 3   # 연속 빈 STT 이 임계 도달 시 상담원 연결
_MSG_EMPTY_STT_ESCALATION = "말씀하신 내용을 잘 듣지 못했습니다. 상담원에게 연결해 드리겠습니다."

# ── helpers ──────────────────────────────────────────────────────────────────

def _reset_utterance_state(session: CallSession) -> None:
    """발화 처리 완료 후 버퍼/플래그 초기화."""
    session.pcm_buffer.clear()
    session.silence_chunk_count = 0
    session.in_speech = False
    session.bargein_verify_attempted = False
    session.vad_window.clear()


# ── turn 실행 ─────────────────────────────────────────────────────────────────

async def _run_turn(session: CallSession, state: CallState, streaming_transcript: str) -> None:
    """그래프 실행 + 결과 후처리. asyncio.create_task 로 백그라운드 실행."""
    try:
        result = await _graph.ainvoke(state)
    except asyncio.CancelledError:
        logger.info("call_id=%s turn cancelled (barge-in)", session.call_id)
        raise
    except Exception as e:
        logger.error("call_id=%s turn 실행 오류: %s", session.call_id, e)
        return

    _result = result if isinstance(result, dict) else {}
    if _result.get("raw_transcript"):
        session.empty_stt_count = 0
        session.silence_alert_count = 0
    else:
        session.empty_stt_count = _result.get("empty_stt_count", session.empty_stt_count)

    _resp_text = _result.get("response_text") or ""
    _play_buffer = len(_resp_text) / _KO_TTS_CHARS_PER_SEC
    session.last_activity_at = time.monotonic() + _play_buffer

    if session.db_call_id:
        customer_text = _result.get("normalized_text") or streaming_transcript
        if customer_text:
            await insert_transcript(
                db_call_id=session.db_call_id,
                turn_index=state["turn_index"],
                speaker="customer",
                text=customer_text,
                is_barge_in=bool(state.get("is_bargein")),
            )
        if _resp_text:
            await insert_transcript(
                db_call_id=session.db_call_id,
                turn_index=state["turn_index"],
                speaker="agent",
                text=_resp_text,
                response_path=_result.get("response_path"),
                reviewer_applied=bool(_result.get("reviewer_applied")),
                reviewer_verdict=_result.get("reviewer_verdict"),
                is_barge_in=False,
            )

    _new_intent = _result.get("primary_intent")
    session.session_view["turn_count"] += 1
    session.session_view["last_intent"] = _new_intent
    session.session_view["last_question"] = (
        _result.get("normalized_text") or streaming_transcript
    )
    if _resp_text:
        session.session_view["last_assistant_text"] = _resp_text[:200]
    if _new_intent == "intent_clarify":
        session.session_view["clarify_count"] += 1
    else:
        session.session_view["clarify_count"] = 0

    if _result.get("response_path") == "faq":
        session.session_view["rag_miss_count"] = _result.get("rag_miss_count", 0)
    else:
        session.session_view["rag_miss_count"] = 0

    session.session_view["auth_pending"] = _result.get("auth_pending", False)


# ── barge-in ──────────────────────────────────────────────────────────────────

async def _attempt_bargein_verify(session: CallSession) -> None:
    """VAD + TitaNet 으로 BARGE-IN 검증.

    spawn_background() 로 비블로킹 실행 — 메인 루프 inbound 처리 차단 없음.
    session.bargein_verify_attempted 는 호출 전 메인 루프에서 True 로 설정.
    """
    if not settings.bargein_verify_enabled:
        return

    chunk = bytes(session.pcm_buffer[: settings.bargein_verify_chunk_bytes])

    try:
        is_speech = await _bargein_vad.detect(chunk)
    except Exception as exc:
        logger.warning("call_id=%s bargein VAD 실패: %s — skip", session.call_id, exc)
        return
    if not is_speech:
        logger.debug("call_id=%s bargein verify skip — VAD non-speech", session.call_id)
        return

    try:
        is_verified, similarity = await _get_titanet_service().verify(chunk, session.call_id)
    except Exception as exc:
        logger.error("call_id=%s bargein verify 실패: %s — skip", session.call_id, exc)
        return

    channel_speaking = session.channel_opened and tts_channel.is_speaking(session.call_id)
    turn_running = session.turn_task is not None and not session.turn_task.done()

    # TTS 재생 중 echo 로 인한 유사도 급락 보정
    _ECHO_BARGEIN_THRESHOLD = 0.20
    enrollment_done = similarity < 1.0
    if channel_speaking and enrollment_done:
        effective_verified = similarity >= _ECHO_BARGEIN_THRESHOLD
    else:
        effective_verified = is_verified

    logger.info(
        "call_id=%s BARGE-IN verify sim=%.3f verified=%s speaking=%s turn_running=%s",
        session.call_id, similarity, effective_verified, channel_speaking, turn_running,
    )
    if not effective_verified:
        return

    logger.info("call_id=%s BARGE-IN 감지 (verified)", session.call_id)
    if session.channel_opened:
        session.interrupted_response_text = tts_channel.current_text(session.call_id) or ""
        await tts_channel.cancel(session.call_id)
    await session.cancel_turn_task()
    session.last_activity_at = time.monotonic() + _BARGEIN_GRACE_SEC
    session.silence_alert_count = 0


# ── utterance processing ──────────────────────────────────────────────────────

async def _process_utterance(session: CallSession, utterance_end_signal: bool) -> None:
    """발화 완성 시 STT → 화자검증 → enrollment → graph dispatch."""
    chunk = bytes(session.pcm_buffer)
    trigger_reason = (
        "utterance_end" if utterance_end_signal
        else ("max_bytes" if len(session.pcm_buffer) >= _MAX_UTTERANCE_BYTES else "silence")
    )
    logger.info(
        "call_id=%s | 발화 완성 %d bytes (%.1fs) trigger=%s → 오디오 처리",
        session.call_id, len(chunk), len(chunk) / 32000, trigger_reason,
    )

    # 1) STT — streaming 결과 회수, 비어있으면 prerecorded fallback
    stt_t0 = time.monotonic()
    transcript = await _streaming_stt.flush_transcript(session.call_id)
    if not transcript:
        try:
            transcript = await _prerecorded_stt.transcribe(chunk)
            if transcript:
                logger.info(
                    "call_id=%s | STT prerecorded fallback '%s'", session.call_id, transcript,
                )
        except Exception as exc:
            logger.warning("call_id=%s | STT prerecorded 실패: %s", session.call_id, exc)
    transcript_norm = " ".join(transcript.split()) if transcript else ""
    logger.info(
        "[pre-graph:stt] elapsed=%.0fms call_id=%s len=%d",
        (time.monotonic() - stt_t0) * 1000, session.call_id, len(transcript_norm),
    )

    if not transcript_norm:
        session.empty_stt_count += 1
        logger.debug(
            "call_id=%s | 빈 STT %d회 → graph skip",
            session.call_id, session.empty_stt_count,
        )
        if session.empty_stt_count >= _EMPTY_STT_ESCALATION_THRESHOLD:
            logger.info("call_id=%s | 빈 STT %d회 → escalation", session.call_id, session.empty_stt_count)
            spawn_background(
                tts_channel.push_response(
                    call_id=session.call_id,
                    text=_MSG_EMPTY_STT_ESCALATION,
                    response_path="escalation",
                )
            )
            session.empty_stt_count = 0
        return

    session.empty_stt_count = 0

    # 2) 화자 검증 (텔레메트리 — STT 결과 있으면 graph 진입 자체는 막지 않음)
    verify_t0 = time.monotonic()
    try:
        await _get_titanet_service().verify(chunk, session.call_id)
    except Exception as exc:
        logger.error("call_id=%s | 화자 검증 실패: %s", session.call_id, exc)
    logger.info(
        "[pre-graph:verify] elapsed=%.0fms call_id=%s",
        (time.monotonic() - verify_t0) * 1000, session.call_id,
    )

    # 3) Enrollment — STT 성공 발화만 누적해 voiceprint 등록
    enroll_t0 = time.monotonic()
    await _get_voiceprint_enrollment().accumulate(session.call_id, chunk, transcript_norm)
    logger.info(
        "[pre-graph:enroll] elapsed=%.0fms call_id=%s",
        (time.monotonic() - enroll_t0) * 1000, session.call_id,
    )

    # 4) Graph state 빌드 + dispatch
    state: CallState = {
        "call_id": session.call_id,
        "tenant_id": session.tenant_id,
        "turn_index": session.turn_index,
        "raw_transcript": transcript,
        "normalized_text": transcript_norm,
        "query_embedding": [],
        "cache_hit": False,
        "primary_intent": None,
        "session_view": session.session_view,
        "rag_results": [],
        "response_text": "",
        "response_path": "",
        "reviewer_applied": False,
        "reviewer_verdict": None,
        "is_timeout": False,
        "empty_stt_count": 0,
    }
    if session.interrupted_response_text:
        state["is_bargein"] = True
        state["interrupted_response_text"] = session.interrupted_response_text
        session.interrupted_response_text = ""
    state["rag_miss_count"] = session.session_view.get("rag_miss_count", 0)
    state["available_categories"] = session.session_view.get("rag_categories", [])
    state["auth_pending"] = session.session_view.get("auth_pending", False)

    # barge-in 단순 모델 — turn cancel 은 오직 _attempt_bargein_verify(TitaNet 통과) 만 책임.
    # 새 발화 완성 자체로는 이전 turn 을 cancel 하지 않음.
    session.turn_task = asyncio.create_task(_run_turn(session, state, transcript_norm))
    session.turn_index += 1


# ── silence check ─────────────────────────────────────────────────────────────

def _check_silence(session: CallSession) -> None:
    """침묵 타이머 확인 및 알림 발화 (in_speech=False 구간에서 매 청크 호출)."""
    turn_running_now = session.turn_task is not None and not session.turn_task.done()
    channel_speaking_now = session.channel_opened and tts_channel.is_speaking(session.call_id)
    if turn_running_now or channel_speaking_now:
        session.last_activity_at = time.monotonic()
        return
    if not session.channel_opened or session.silence_alert_count >= 2:
        return
    now = time.monotonic()
    elapsed = now - session.last_activity_at
    if session.silence_alert_count == 1 and elapsed >= _SILENCE_SECOND_SEC:
        logger.info("call_id=%s 침묵 escalation (2차)", session.call_id)
        spawn_background(
            tts_channel.push_response(
                call_id=session.call_id,
                text=_MSG_SILENCE_ESCALATION,
                response_path="escalation",
            )
        )
        session.silence_alert_count = 2
        session.last_activity_at = now
    elif session.silence_alert_count == 0 and elapsed >= _SILENCE_FIRST_SEC:
        logger.info("call_id=%s 침묵 확인 멘트 (1차, %.1f초)", session.call_id, elapsed)
        spawn_background(
            tts_channel.push_response(
                call_id=session.call_id,
                text=_MSG_SILENCE_CHECK,
                response_path="silence_check",
            )
        )
        session.silence_alert_count = 1
        session.last_activity_at = now


# ── event handlers ────────────────────────────────────────────────────────────

async def _handle_start(session: CallSession, msg: dict) -> None:
    """Twilio 'start' 이벤트 — TTS channel, STT, tenant meta 초기화."""
    stream_sid = msg.get("streamSid", "")
    custom_params = msg.get("start", {}).get("customParameters", {})
    session.tenant_id = custom_params.get("tenant_id", session.tenant_id)
    logger.info(
        "call_id=%s stream_sid=%s tenant_id=%s",
        session.call_id, stream_sid, session.tenant_id,
    )

    try:
        await tts_channel.open(
            call_id=session.call_id,
            tenant_id=session.tenant_id,
            websocket=session.websocket,
            stream_sid=stream_sid,
        )
        session.channel_opened = True
    except TypeError:
        # MockTTSOutputChannel 은 websocket/stream_sid 인자 없음 — 기본 시그니처 재시도
        await tts_channel.open(call_id=session.call_id, tenant_id=session.tenant_id)
        session.channel_opened = True

    (
        tenant_keyterms,
        rag_categories,
        within_hours,
        tenant_name,
    ) = await asyncio.gather(
        get_tenant_keyterms(session.tenant_id),
        _session_service.get_rag_categories(session.tenant_id),
        _session_service.is_within_business_hours(session.tenant_id),
        get_tenant_name(session.tenant_id),
    )
    session.session_view["rag_categories"] = rag_categories
    session.session_view["is_within_hours"] = within_hours
    session.session_view["tenant_name"] = tenant_name
    session.session_view["tenant_keyterms"] = tenant_keyterms

    greeting = await get_greeting(session.tenant_id, within_hours)

    try:
        await _streaming_stt.open(session.call_id, keyterms=tenant_keyterms)
    except Exception as _stt_err:
        logger.warning(
            "STT 스트리밍 연결 실패 call_id=%s: %s (prerecorded 폴백)",
            session.call_id, _stt_err,
        )

    # Greeting 송신 백그라운드 — 메인 루프가 즉시 inbound 처리 시작
    spawn_background(
        tts_channel.push_response(
            call_id=session.call_id, text=greeting, response_path="greeting"
        )
    )
    logger.info(
        "call_id=%s greeting 발송 within_hours=%s tenant_name=%s",
        session.call_id, within_hours, tenant_name,
    )
    session.last_activity_at = time.monotonic() + _GREETING_PLAY_BUFFER_SEC

    session.db_call_id = await insert_call(
        tenant_id=session.tenant_id,
        twilio_call_sid=session.call_id,
        caller_number=None,
    )
    session.call_started_at_monotonic = time.monotonic()


async def _handle_media(session: CallSession, msg: dict) -> None:
    """Twilio 'media' 이벤트 — Silero VAD 게이트 → STT → 화자검증 → graph dispatch."""
    track = msg["media"].get("track", "inbound")
    if track != "inbound":
        return

    mulaw_bytes = base64.b64decode(msg["media"]["payload"])
    pcm_bytes = mulaw_to_pcm16(mulaw_bytes, session.call_id)
    rms = audioop.rms(pcm_bytes, 2)

    # Silero 슬라이딩 윈도우 갱신 (~160ms 유지)
    session.vad_window.extend(pcm_bytes)
    if len(session.vad_window) > _VAD_WINDOW_BYTES:
        del session.vad_window[: len(session.vad_window) - _VAD_WINDOW_BYTES]

    # 모든 청크를 Deepgram 으로 전송 (VAD 게이트와 무관)
    await _streaming_stt.send(session.call_id, pcm_bytes)

    # 발화 게이트: RMS cheap pre-filter → Silero ML 판정
    # vad_window 가 최소 1024 bytes (2 frames) 이상일 때만 Silero 호출
    if rms >= _VAD_RMS_PREFILTER and len(session.vad_window) >= 1024:
        is_speech = await _silero_vad.detect(bytes(session.vad_window))
    else:
        is_speech = False

    if is_speech:
        session.pcm_buffer.extend(pcm_bytes)
        session.silence_chunk_count = 0
        session.in_speech = True
        # 최대 발화 길이 초과 — 강제 처리 (매우 긴 발화 안전망)
        if len(session.pcm_buffer) >= _MAX_UTTERANCE_BYTES:
            await _process_utterance(session, utterance_end_signal=False)
            _reset_utterance_state(session)

    elif session.in_speech:
        # 발화 직후 묵음 — trailing silence 포함 누적
        session.pcm_buffer.extend(pcm_bytes)
        session.silence_chunk_count += 1

        utterance_end_signal = _streaming_stt.consume_utterance_end(session.call_id)
        trigger = (
            session.silence_chunk_count >= _SILENCE_CHUNKS_TO_END
            or len(session.pcm_buffer) >= _MAX_UTTERANCE_BYTES
            or utterance_end_signal
        )
        if trigger:
            if len(session.pcm_buffer) >= _MIN_UTTERANCE_BYTES:
                await _process_utterance(session, utterance_end_signal)
            else:
                logger.debug(
                    "call_id=%s | 발화 무시 (too short: %d bytes)",
                    session.call_id, len(session.pcm_buffer),
                )
            _reset_utterance_state(session)

    else:
        # in_speech=False, is_speech=False → 진짜 침묵 구간
        _check_silence(session)

    # ── BARGE-IN verify trigger (발화당 1회, 비블로킹) ────────────────────────
    # spawn_background 로 메인 루프 차단 없이 TitaNet verify 실행.
    # bargein_verify_attempted 를 즉시 True 로 설정해 중복 실행 방지.
    channel_speaking = session.channel_opened and tts_channel.is_speaking(session.call_id)
    turn_running = session.turn_task is not None and not session.turn_task.done()
    if (
        (channel_speaking or turn_running)
        and not session.bargein_verify_attempted
        and len(session.pcm_buffer) >= settings.bargein_verify_chunk_bytes
    ):
        session.bargein_verify_attempted = True
        spawn_background(_attempt_bargein_verify(session))


# ── WebSocket 엔드포인트 (디스패처) ──────────────────────────────────────────

@router.get("")
async def list_calls(
    status_filter: str | None = Query(default=None, alias="status"),
    started_from: datetime | None = Query(default=None),
    started_to: datetime | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1),
    tenant_id: str | None = Query(default=None),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _current_admin_tenant_id(current_admin)
    _validate_query_tenant_id(tenant_id, jwt_tenant_id)
    effective_limit = min(limit, 100)

    result = await list_calls_for_tenant(
        tenant_id=jwt_tenant_id,
        status=status_filter,
        started_from=started_from,
        started_to=started_to,
        offset=offset,
        limit=effective_limit,
    )

    return {
        "data": {
            "items": result["items"],
            "total": result["total"],
            "offset": offset,
            "limit": effective_limit,
        },
        "request_id": _request_id(),
    }


@router.post("/incoming")
async def incoming_call(request: Request):
    """Twilio 전화 수신 webhook — TwiML 반환."""
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    to_number = form.get("To", "unknown")
    tenant_id = await resolve_tenant_id(to_number)

    logger.info(f"[{CALL_STARTED}] call_sid={call_sid} to={to_number} tenant_id={tenant_id}")

    ws_url = f"wss://{settings.base_url.removeprefix('https://').removeprefix('http://')}/call/ws/{call_sid}"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="tenant_id" value="{tenant_id}"/>
    </Stream>
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@router.get("/{call_id}/transcripts")
async def get_call_transcripts(
    call_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    tenant_id = _current_admin_tenant_id(current_admin)
    transcripts = await get_transcripts_by_call_id(call_id, tenant_id)
    if transcripts is None:
        raise HTTPException(
            status_code=404,
            detail=f"transcripts not found: {call_id!r}",
        )

    return {
        "data": {
            "items": transcripts,
            "total": len(transcripts),
        },
        "request_id": _request_id(),
    }


@router.get("/{call_id}")
async def get_call_detail(
    call_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    tenant_id = _current_admin_tenant_id(current_admin)
    record = await get_call_by_id_for_tenant(call_id, tenant_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"call not found: {call_id!r}",
        )

    return {
        "data": record,
        "request_id": _request_id(),
    }


@router.websocket("/ws/{call_id}")
async def call_websocket(
    websocket: WebSocket,
    call_id: str,
    tenant_id: str = Query(default="unknown"),
) -> None:
    """Twilio Media Streams WebSocket 엔드포인트.

    메시지 루프 디스패처 — 실제 로직은 _handle_start / _handle_media 에 위임.
    종료 경로 (stop / disconnect / exception) 모두 session.teardown() 으로 단일화.
    """
    await websocket.accept()
    logger.info("WebSocket 연결 수락 call_id=%s", call_id)
    session = CallSession(websocket, call_id, tenant_id)
    exit_status = "error"
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                logger.info("call_id=%s Twilio Media Stream connected", call_id)
            elif event == "start":
                await _handle_start(session, msg)
            elif event == "media":
                await _handle_media(session, msg)
            elif event == "stop":
                logger.info("[%s] call_id=%s", CALL_ENDED, call_id)
                exit_status = "completed"
                break

    except WebSocketDisconnect:
        logger.info("WebSocket 종료 call_id=%s", call_id)
        exit_status = "abandoned"
    except Exception:
        logger.exception("call_id=%s WebSocket 오류", call_id)
        exit_status = "error"
    finally:
        await _streaming_stt.close(session.call_id)
        await session.teardown(exit_status)
