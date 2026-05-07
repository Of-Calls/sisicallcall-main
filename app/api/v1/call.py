import asyncio
import audioop
import base64
import json
import os
import time
import traceback
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.rest import Client as TwilioRestClient

from app.agents.conversational.graph import build_graph
from app.agents.post_call.runner import run_post_call_agent_safely
from app.repositories.call_repo import finalize_call, insert_call
from app.repositories.transcript_repo import insert_transcript
from app.services.session.redis_session import RedisSessionService
from app.services.speaker_verify import enrollment as voiceprint_enrollment
from app.services.speaker_verify import get_speaker_verify_service
from app.services.stt.deepgram import DeepgramSTTService
from app.services.tenant import DEFAULT_INDUSTRY, DEFAULT_NAME, get_greeting, get_tenant_meta, resolve_tenant_id
from app.services.tts.azure import AzureTTSService
from app.services.tts.filler import pick_filler, pick_filler_continuation
from app.services.vad.silero_vad import SileroVADService
from app.utils.config import settings

router = APIRouter()
_stt = DeepgramSTTService()
_tts = AzureTTSService()
_vad = SileroVADService()
_verifier = get_speaker_verify_service()
_graph = build_graph()
_session = RedisSessionService()
_twilio_rest = (
    TwilioRestClient(settings.twilio_account_sid, settings.twilio_auth_token)
    if settings.twilio_account_sid and settings.twilio_auth_token
    else None
)

# Stage 4a — graph 통합 (echo 회귀 환경변수)
_GRAPH_ENABLED = os.getenv("GRAPH_INTEGRATION_ENABLED", "false").lower() in ("1", "true", "yes")

_VAD_FRAME_BYTES = 1024     # linear16 16kHz, 512 samples
_SILENCE_THRESHOLD = 45     # 연속 침묵 VAD 프레임 수 (~1440ms)
_TWILIO_CHUNK_BYTES = 160   # 20ms mulaw 8kHz — Twilio 권장 단위


def _extract_caller_number(caller: str) -> str:
    """SIP URI 의 user part 또는 e164 그대로. caller_number VARCHAR(20) 제약."""
    if caller.startswith("sip:"):
        user = caller[4:].split("@", 1)[0].split(";", 1)[0]
        return user[:20]
    return caller[:20]


def _intent_to_response_path(intent: str | None) -> str | None:
    """graph intent → transcripts.response_path (CHECK 'cache','faq','task','auth','escalation').
    매핑 외 (clarify/repeat/vision/echo/None) → NULL.
    """
    if intent in ("faq", "task", "auth", "escalation"):
        return intent
    return None


@router.post("/incoming")
async def incoming_call(request: Request):
    form = await request.form()
    to_field = form.get("To", "")
    twilio_call_sid = form.get("CallSid", "")
    caller_raw = form.get("Caller", "") or form.get("From", "")
    caller_number = _extract_caller_number(caller_raw)
    # SIP URI / e164 / single digit 모두 처리 — 매칭 실패 시 raw 값 반환됨 (UUID 아님 → echo).
    tenant_id = await resolve_tenant_id(to_field)
    print(f"[INCOMING] to={to_field!r} caller={caller_number!r} call_sid={twilio_call_sid!r} tenant_id={tenant_id!r}")

    host = request.headers.get("host", "")
    ws_url = f"wss://{host}/call/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="tenant_id" value="{tenant_id}" />
      <Parameter name="twilio_call_sid" value="{twilio_call_sid}" />
      <Parameter name="caller_number" value="{caller_number}" />
    </Stream>
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


async def _send_audio_to_twilio(websocket: WebSocket, stream_sid: str, audio: bytes) -> None:
    """Send mulaw 8kHz audio to Twilio Media Streams in 20ms chunks."""
    for i in range(0, len(audio), _TWILIO_CHUNK_BYTES):
        chunk = audio[i:i + _TWILIO_CHUNK_BYTES]
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode()},
        }
        await websocket.send_text(json.dumps(msg))


async def _run_conversational_graph(
    transcript: str,
    tenant_id: str,
    tenant_name: str,
    tenant_industry: str,
    call_id: str,
) -> tuple[str, str | None, bool]:
    """graph 호출 + session 저장. (response_text, intent, should_hangup) 반환.

    barge-in (Stage 3) 도입 시 호출 측에서 task.cancel() 가능하도록 별도 함수로 분리.
    """
    session_view = await _session.load(call_id)
    graph_state = {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "tenant_industry": tenant_industry,
        "user_text": transcript,
        "intent": "",
        "response_text": "",
        "session_view": session_view,
        "rewritten_query": "",
        "is_clear": False,
        "missing_info": "",
        "is_goodbye": False,
        "should_hangup": False,
    }
    t_graph = time.perf_counter()
    result = await _graph.ainvoke(graph_state)
    graph_ms = (time.perf_counter() - t_graph) * 1000
    graph_resp = result.get("response_text", "")
    graph_intent = result.get("intent") or None
    should_hangup = bool(result.get("should_hangup", False))
    print(f"[GRAPH] intent={graph_intent} resp='{graph_resp[:60]}' hangup={should_hangup} ({graph_ms:.0f}ms)")
    if graph_resp:
        await _session.append_turn(call_id, transcript, graph_resp)
    return graph_resp, graph_intent, should_hangup


@router.websocket("/ws")
async def call_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Twilio connected")

    audio_buffer = bytearray()    # STT용 (mulaw 8kHz 누적)
    pcm_buffer = bytearray()      # VAD frame 잘라쓰는 용 (linear16 16kHz 임시)
    utterance_pcm = bytearray()   # 화자검증/enrollment 용 (linear16 16kHz, 발화 단위 누적)
    ratecv_state = None           # audioop.ratecv state — 8k→16k 보간 연속성 유지
    silence_count = 0
    had_speech = False
    stream_sid = None
    is_speaking = False  # TTS 송출 중 플래그 — Mode A (Stage 1b) 자기루프 방지
    # Stage 4a — graph 컨텍스트 (start 시 로드)
    tenant_id = ""
    tenant_name = "고객센터"
    tenant_industry = "unknown"
    # Stage 4c-1 — DB hook 컨텍스트
    db_call_id = ""               # calls.id (UUID, INSERT 후 받음). 빈값 = DB hook 비활성
    call_started_at = 0.0         # time.perf_counter() — duration 계산용
    twilio_call_sid = ""
    caller_number = ""
    # Stage 4c-2 — transcripts turn 카운터 (customer + agent 한 쌍이 같은 turn_index 공유)
    turn_index = 0

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                print("[WS] connected")

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                custom_params = msg["start"].get("customParameters") or {}
                received_tenant_id = custom_params.get("tenant_id", "")
                twilio_call_sid = custom_params.get("twilio_call_sid", "")
                caller_number = custom_params.get("caller_number", "")
                print(f"[WS] start streamSid={stream_sid} tenant_id={received_tenant_id!r} call_sid={twilio_call_sid!r}")
                audio_buffer.clear()
                pcm_buffer.clear()
                utterance_pcm.clear()
                ratecv_state = None
                silence_count = 0
                had_speech = False
                is_speaking = False
                tenant_id = ""  # 매칭 실패/graph 비활성 시 echo 모드 신호
                db_call_id = ""
                call_started_at = time.perf_counter()
                turn_index = 0

                # Stage 4b — Twilio Parameter 로 받은 tenant_id 검증 + 메타 로드
                if _GRAPH_ENABLED and received_tenant_id:
                    try:
                        tenant_name, tenant_industry = await get_tenant_meta(received_tenant_id)
                        if tenant_name == DEFAULT_NAME and tenant_industry == DEFAULT_INDUSTRY:
                            print(f"[GRAPH] tenant 매칭 실패 (received={received_tenant_id!r}) → echo")
                        else:
                            tenant_id = received_tenant_id
                            print(f"[GRAPH] tenant={tenant_name} ({tenant_industry})")
                            # Stage 4c-1 — calls INSERT (graph 활성 + tenant 매칭 시)
                            if twilio_call_sid:
                                inserted = await insert_call(
                                    tenant_id=tenant_id,
                                    twilio_call_sid=twilio_call_sid,
                                    caller_number=caller_number or None,
                                )
                                if inserted:
                                    db_call_id = inserted
                                    print(f"[DB] calls INSERT db_call_id={db_call_id}")

                            # Stage 4d — greeting 자동 송출 (사용자 첫 발화 기다리지 않음)
                            try:
                                greeting = await get_greeting(tenant_id, within_hours=True)
                                print(f"[GREETING] '{greeting[:60]}'")
                                tts_audio = await _tts.synthesize(greeting)
                                is_speaking = True
                                await _send_audio_to_twilio(websocket, stream_sid, tts_audio)
                                play_sec = len(tts_audio) / 8000
                                await asyncio.sleep(play_sec)
                                is_speaking = False
                                print(f"[GREETING] 재생 끝 ({play_sec:.2f}s)")
                                # transcripts INSERT — turn 0 agent (사용자 첫 발화는 turn 1 부터)
                                if db_call_id:
                                    await insert_transcript(
                                        db_call_id, turn_index, "agent", greeting,
                                        response_path=None,
                                    )
                                    turn_index += 1
                                # buffer reset — greeting 잔향이 enrollment 에 섞이지 않게
                                audio_buffer.clear()
                                pcm_buffer.clear()
                                utterance_pcm.clear()
                                ratecv_state = None
                                silence_count = 0
                                had_speech = False
                            except Exception as e:
                                is_speaking = False
                                print(f"[GREETING] failed: {e} — silent")
                    except Exception as e:
                        print(f"[GRAPH] tenant_meta load failed: {e} → echo fallback this call")

            elif event == "media":
                if is_speaking:
                    continue  # TTS 송출 중엔 사용자 발화 무시 — barge-in 은 Stage 3 에서 도입

                mulaw = base64.b64decode(msg["media"]["payload"])
                audio_buffer.extend(mulaw)

                pcm_8k = audioop.ulaw2lin(mulaw, 2)
                pcm_16k, ratecv_state = audioop.ratecv(
                    pcm_8k,
                    2,
                    1,
                    8000,
                    16000,
                    ratecv_state,
                )
                pcm_buffer.extend(pcm_16k)

                while len(pcm_buffer) >= _VAD_FRAME_BYTES:
                    frame = bytes(pcm_buffer[:_VAD_FRAME_BYTES])
                    del pcm_buffer[:_VAD_FRAME_BYTES]

                    is_speech = await _vad.detect(frame)

                    if is_speech:
                        if not had_speech:
                            print("[VAD] 발화 시작")
                            utterance_pcm.clear()  # 새 발화 — 이전 잔여/silence prefix 폐기
                        utterance_pcm.extend(frame)  # 발화 시작 후만 누적 (verify/enrollment 입력 품질)
                        silence_count = 0
                        had_speech = True
                    else:
                        if had_speech:
                            utterance_pcm.extend(frame)  # 발화 중 짧은 pause 도 포함
                            if silence_count == 0:
                                print("[VAD] 침묵 카운트 시작")
                        silence_count += 1
                        if silence_count >= _SILENCE_THRESHOLD and had_speech:
                            pcm_for_verify = bytes(utterance_pcm)
                            print(f"[VERIFY-DBG] utterance_pcm={len(pcm_for_verify)}B")

                            # Stage B: voiceprint 등록 후 — STT 전 화자검증 게이트
                            # TitaNet 짧은 발화 한계 → 1.5초 미만은 verify 스킵 (본인 reject 방지)
                            min_verify_bytes = int(settings.speaker_verify_min_audio_sec * 16000 * 2)
                            if stream_sid and _verifier.is_enrolled(stream_sid):
                                if len(pcm_for_verify) < min_verify_bytes:
                                    dur_ms = len(pcm_for_verify) * 1000 // 32000
                                    print(f"[VERIFY] short ({dur_ms}ms < {settings.speaker_verify_min_audio_sec}s) — skip verify")
                                else:
                                    verified, sim = await _verifier.verify(pcm_for_verify, stream_sid)
                                    if not verified:
                                        print(f"[VERIFY] reject sim={sim:.3f} — STT skip")
                                        audio_buffer.clear()
                                        utterance_pcm.clear()
                                        silence_count = 0
                                        had_speech = False
                                        continue

                            print(f"[VAD] 침묵 임계 도달 — {len(audio_buffer)}B → STT")

                            t_stt = time.perf_counter()
                            transcript = await _stt.transcribe(bytes(audio_buffer))
                            stt_ms = (time.perf_counter() - t_stt) * 1000
                            print(f"[STT] '{transcript}' ({stt_ms:.0f}ms, in={len(audio_buffer)}B)")

                            audio_buffer.clear()
                            silence_count = 0
                            had_speech = False

                            # Stage A: voiceprint 미등록 — STT 후 enrollment 누적 (빈 STT 차단)
                            if stream_sid and not _verifier.is_enrolled(stream_sid):
                                await voiceprint_enrollment.accumulate(
                                    stream_sid, pcm_for_verify, transcript
                                )

                            utterance_pcm.clear()

                            if transcript and stream_sid:
                                # Latency 옵션 A — graph background + filler 즉시 송출 병렬.
                                # barge-in (Stage 3) 도입 시 graph_task.cancel() 한 줄 추가하면 됨.
                                graph_task: asyncio.Task | None = None
                                if _GRAPH_ENABLED and tenant_id:
                                    graph_task = asyncio.create_task(
                                        _run_conversational_graph(
                                            transcript, tenant_id, tenant_name, tenant_industry, stream_sid,
                                        )
                                    )

                                # filler 즉시 송출 — graph 진행 중 사용자 무음 갭 채움.
                                # cache 비어있으면 (startup 합성 실패) skip — 기존 흐름 그대로.
                                filler = pick_filler()
                                if filler:
                                    is_speaking = True
                                    try:
                                        await _send_audio_to_twilio(websocket, stream_sid, filler)
                                        filler_play = len(filler) / 8000
                                        await asyncio.sleep(filler_play)
                                        print(f"[FILLER] 재생 끝 ({filler_play:.2f}s)")
                                    except Exception as exc:
                                        print(f"[FILLER] 송출 실패 (continue): {exc}")

                                # 2단계 filler — filler 1 끝났는데 graph 아직 진행 중이면
                                # 짧은 자연 호흡 갭 (~2.2s) 후 추가 송출. graph 가 빠르면 skip
                                # 으로 응답 지연 방지. silence 분산으로 체감 latency ↓.
                                if graph_task is not None and not graph_task.done():
                                    await asyncio.sleep(2.2)
                                    if not graph_task.done():
                                        filler2 = pick_filler_continuation()
                                        if filler2:
                                            is_speaking = True
                                            try:
                                                await _send_audio_to_twilio(websocket, stream_sid, filler2)
                                                filler2_play = len(filler2) / 8000
                                                await asyncio.sleep(filler2_play)
                                                print(f"[FILLER2] 재생 끝 ({filler2_play:.2f}s)")
                                            except Exception as exc:
                                                print(f"[FILLER2] 송출 실패 (continue): {exc}")

                                # graph 결과 await — 실패/empty 시 echo fallback
                                response_text = transcript
                                graph_intent: str | None = None
                                should_hangup = False
                                if graph_task is not None:
                                    try:
                                        graph_resp, graph_intent, should_hangup = await graph_task
                                        if graph_resp:
                                            response_text = graph_resp
                                        else:
                                            print("[GRAPH] empty response — echo fallback")
                                    except Exception as e:
                                        print(f"[GRAPH] error: {e} → echo fallback")

                                # Stage 4c-2 — transcripts INSERT 양방향 (TTS 송출 전).
                                # DB 실패해도 통화 지속 (DB 누락만, traceback 명시).
                                if db_call_id:
                                    try:
                                        await insert_transcript(
                                            db_call_id, turn_index, "customer", transcript,
                                        )
                                        await insert_transcript(
                                            db_call_id, turn_index, "agent", response_text,
                                            response_path=_intent_to_response_path(graph_intent),
                                        )
                                        turn_index += 1
                                    except Exception as exc:
                                        print(f"[DB] transcripts INSERT 실패 (통화 지속): {type(exc).__name__}: {exc}")
                                        traceback.print_exc()

                                # TTS synth — 실패 시 polite fallback 멘트로 1회 재시도, 그것도 실패면 silent skip.
                                tts_audio = b""
                                try:
                                    t_tts = time.perf_counter()
                                    tts_audio = await _tts.synthesize(response_text)
                                    tts_ms = (time.perf_counter() - t_tts) * 1000
                                    print(f"[TTS] synth {tts_ms:.0f}ms, out={len(tts_audio)}B")
                                except Exception as exc:
                                    print(f"[TTS] synth 실패: {type(exc).__name__}: {exc}")
                                    traceback.print_exc()
                                    try:
                                        tts_audio = await _tts.synthesize(
                                            "잠시 문제가 생겼어요. 다시 말씀해주세요."
                                        )
                                        print(f"[TTS] fallback synth ok out={len(tts_audio)}B")
                                    except Exception as exc2:
                                        print(f"[TTS] fallback 도 실패 (silent skip): {type(exc2).__name__}: {exc2}")
                                        tts_audio = b""

                                if not tts_audio:
                                    # silent skip — filler 가 송출됐다면 is_speaking 풀어줘야
                                    # 다음 사용자 발화가 차단되지 않음.
                                    is_speaking = False
                                    continue

                                is_speaking = True
                                # _send_audio_to_twilio 실패 = WebSocket 손상 신호. raise 해서
                                # outer except Exception 분기가 정리/traceback 처리.
                                t_send = time.perf_counter()
                                await _send_audio_to_twilio(websocket, stream_sid, tts_audio)
                                send_ms = (time.perf_counter() - t_send) * 1000
                                print(f"[TTS] 송출 {send_ms:.0f}ms")

                                # 실제 재생 시간만큼 대기 — Twilio 큐가 비워질 때까지 is_speaking 유지
                                # (송출 ≠ 재생. mulaw 8kHz = 8000B/s)
                                play_sec = len(tts_audio) / 8000
                                await asyncio.sleep(play_sec)
                                is_speaking = False
                                print(f"[TTS] 재생 끝 ({play_sec:.2f}s)")

                                # TTS 후 buffer / VAD state 일괄 리셋 — 잔향/echo 누적 방지
                                audio_buffer.clear()
                                pcm_buffer.clear()
                                utterance_pcm.clear()
                                ratecv_state = None
                                silence_count = 0
                                had_speech = False

                                # Polish — goodbye 분기에서 should_hangup=True → Twilio REST API hangup
                                if should_hangup and twilio_call_sid and _twilio_rest is not None:
                                    try:
                                        await asyncio.to_thread(
                                            _twilio_rest.calls(twilio_call_sid).update,
                                            status="completed",
                                        )
                                        print(f"[HANGUP] Twilio call terminated: {twilio_call_sid}")
                                    except Exception as e:
                                        print(f"[HANGUP] failed: {e}")

            elif event == "stop":
                print("[WS] stop")
                if stream_sid:
                    _verifier.cleanup(stream_sid)
                    voiceprint_enrollment.cleanup(stream_sid)
                    if _GRAPH_ENABLED:
                        try:
                            await _session.clear(stream_sid)
                        except Exception as e:
                            print(f"[GRAPH] session clear failed: {e}")
                # Stage 4c-1 — calls UPDATE finalize (Twilio 정상 종료).
                # finalize 실패해도 break 는 진행 — outer 가 잡지 않게.
                if db_call_id:
                    duration_sec = int(time.perf_counter() - call_started_at) if call_started_at else None
                    try:
                        await finalize_call(db_call_id, "completed", duration_sec)
                        print(f"[DB] finalize_call db_call_id={db_call_id} status=completed dur={duration_sec}s")
                    except Exception as e:
                        print(f"[DB] finalize_call failed: {e}")
                        traceback.print_exc()
                    # Stage 4c-3 — post_call agent fire-and-forget
                    if tenant_id:
                        asyncio.create_task(
                            run_post_call_agent_safely(db_call_id, "call_ended", tenant_id)
                        )
                        print(f"[POST_CALL] triggered db_call_id={db_call_id} tenant_id={tenant_id}")
                break

    except WebSocketDisconnect:
        print("[WS] 연결 끊김 (사용자/Twilio)")
        if stream_sid:
            _verifier.cleanup(stream_sid)
            voiceprint_enrollment.cleanup(stream_sid)
            if _GRAPH_ENABLED:
                try:
                    await _session.clear(stream_sid)
                except Exception as e:
                    print(f"[GRAPH] session clear failed: {e}")
        # Stage 4c-1 — calls UPDATE finalize (사용자 먼저 끊음)
        if db_call_id:
            duration_sec = int(time.perf_counter() - call_started_at) if call_started_at else None
            try:
                await finalize_call(db_call_id, "abandoned", duration_sec)
                print(f"[DB] finalize_call db_call_id={db_call_id} status=abandoned dur={duration_sec}s")
            except Exception as e:
                print(f"[DB] finalize_call failed: {e}")
                traceback.print_exc()
            # Stage 4c-3 — post_call agent fire-and-forget (abandoned 통화도 분석 트리거)
            if tenant_id:
                asyncio.create_task(
                    run_post_call_agent_safely(db_call_id, "call_ended", tenant_id)
                )
                print(f"[POST_CALL] triggered db_call_id={db_call_id} tenant_id={tenant_id}")
    except Exception as exc:
        # WebSocketDisconnect 외 unhandled 예외 — Twilio 입장에선 우리가 close = Error 31921.
        # traceback 을 stdout 으로 명시 출력해 logs/{date}/stdout_*.log 에 잡히게.
        print(f"[WS] 비정상 종료: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if stream_sid:
            try:
                _verifier.cleanup(stream_sid)
                voiceprint_enrollment.cleanup(stream_sid)
                if _GRAPH_ENABLED:
                    await _session.clear(stream_sid)
            except Exception as e:
                print(f"[WS] cleanup failed: {e}")
        if db_call_id:
            duration_sec = int(time.perf_counter() - call_started_at) if call_started_at else None
            try:
                await finalize_call(db_call_id, "abandoned", duration_sec)
                print(f"[DB] finalize_call db_call_id={db_call_id} status=abandoned (server error) dur={duration_sec}s")
            except Exception as e:
                print(f"[DB] finalize_call failed: {e}")
                traceback.print_exc()
            if tenant_id:
                asyncio.create_task(
                    run_post_call_agent_safely(db_call_id, "call_ended", tenant_id)
                )
                print(f"[POST_CALL] triggered (server error) db_call_id={db_call_id} tenant_id={tenant_id}")
