import asyncio
import audioop
import base64
import json
import os
import time
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_graph
from app.services.session.redis_session import RedisSessionService
from app.services.speaker_verify import enrollment as voiceprint_enrollment
from app.services.speaker_verify import get_speaker_verify_service
from app.services.stt.deepgram import DeepgramSTTService
from app.services.tenant import DEFAULT_INDUSTRY, DEFAULT_NAME, get_tenant_meta, resolve_tenant_id
from app.services.tts.azure import AzureTTSService
from app.services.vad.silero_vad import SileroVADService
from app.utils.config import settings

router = APIRouter()
_stt = DeepgramSTTService()
_tts = AzureTTSService()
_vad = SileroVADService()
_verifier = get_speaker_verify_service()
_graph = build_graph()
_session = RedisSessionService()

# Stage 4a — graph 통합 (echo 회귀 환경변수)
_GRAPH_ENABLED = os.getenv("GRAPH_INTEGRATION_ENABLED", "false").lower() in ("1", "true", "yes")

_VAD_FRAME_BYTES = 1024     # linear16 16kHz, 512 samples
_SILENCE_THRESHOLD = 30     # 연속 침묵 VAD 프레임 수 (~960ms)
_TWILIO_CHUNK_BYTES = 160   # 20ms mulaw 8kHz — Twilio 권장 단위


@router.post("/incoming")
async def incoming_call(request: Request):
    form = await request.form()
    to_field = form.get("To", "")
    # SIP URI / e164 / single digit 모두 처리 — 매칭 실패 시 raw 값 반환됨 (UUID 아님 → echo).
    tenant_id = await resolve_tenant_id(to_field)
    print(f"[INCOMING] to={to_field!r} tenant_id={tenant_id!r}")

    host = request.headers.get("host", "")
    ws_url = f"wss://{host}/call/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="tenant_id" value="{tenant_id}" />
    </Stream>
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


async def _send_audio_to_twilio(websocket: WebSocket, stream_sid: str, audio: bytes) -> None:
    """mulaw 8kHz 음성을 20ms 청크 단위로 Twilio Media Stream에 송출."""
    for i in range(0, len(audio), _TWILIO_CHUNK_BYTES):
        chunk = audio[i:i + _TWILIO_CHUNK_BYTES]
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(chunk).decode()},
        }
        await websocket.send_text(json.dumps(msg))


@router.websocket("/ws")
async def call_ws(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Twilio 연결됨")

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
                print(f"[WS] start streamSid={stream_sid} tenant_id={received_tenant_id!r}")
                audio_buffer.clear()
                pcm_buffer.clear()
                utterance_pcm.clear()
                ratecv_state = None
                silence_count = 0
                had_speech = False
                is_speaking = False
                tenant_id = ""  # 매칭 실패/graph 비활성 시 echo 모드 신호

                # Stage 4b — Twilio Parameter 로 받은 tenant_id 검증 + 메타 로드
                if _GRAPH_ENABLED and received_tenant_id:
                    try:
                        tenant_name, tenant_industry = await get_tenant_meta(received_tenant_id)
                        if tenant_name == DEFAULT_NAME and tenant_industry == DEFAULT_INDUSTRY:
                            print(f"[GRAPH] tenant 매칭 실패 (received={received_tenant_id!r}) → echo")
                        else:
                            tenant_id = received_tenant_id
                            print(f"[GRAPH] tenant={tenant_name} ({tenant_industry})")
                    except Exception as e:
                        print(f"[GRAPH] tenant_meta load failed: {e} → echo fallback this call")

            elif event == "media":
                if is_speaking:
                    continue  # TTS 송출 중엔 사용자 발화 무시 — barge-in 은 Stage 3 에서 도입

                mulaw = base64.b64decode(msg["media"]["payload"])
                audio_buffer.extend(mulaw)

                pcm_8k = audioop.ulaw2lin(mulaw, 2)
                pcm_16k, ratecv_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, ratecv_state)
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
                                # Stage 4a — graph 호출 분기 (실패 시 echo 회귀)
                                response_text = transcript  # 기본 echo
                                if _GRAPH_ENABLED and tenant_id:
                                    try:
                                        session_view = await _session.load(stream_sid)
                                        graph_state = {
                                            "call_id": stream_sid,
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
                                        }
                                        t_graph = time.perf_counter()
                                        result = await _graph.ainvoke(graph_state)
                                        graph_ms = (time.perf_counter() - t_graph) * 1000
                                        graph_resp = result.get("response_text", "")
                                        print(f"[GRAPH] intent={result.get('intent')} resp='{graph_resp[:60]}' ({graph_ms:.0f}ms)")
                                        if graph_resp:
                                            response_text = graph_resp
                                            await _session.append_turn(stream_sid, transcript, graph_resp)
                                        else:
                                            print("[GRAPH] empty response — echo fallback")
                                    except Exception as e:
                                        print(f"[GRAPH] error: {e} → echo fallback")

                                t_tts = time.perf_counter()
                                tts_audio = await _tts.synthesize(response_text)
                                tts_ms = (time.perf_counter() - t_tts) * 1000
                                print(f"[TTS] synth {tts_ms:.0f}ms, out={len(tts_audio)}B")

                                is_speaking = True
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
                break

    except WebSocketDisconnect:
        print("[WS] 연결 끊김")
        if stream_sid:
            _verifier.cleanup(stream_sid)
            voiceprint_enrollment.cleanup(stream_sid)
            if _GRAPH_ENABLED:
                try:
                    await _session.clear(stream_sid)
                except Exception as e:
                    print(f"[GRAPH] session clear failed: {e}")
