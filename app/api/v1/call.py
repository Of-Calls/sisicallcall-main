import asyncio
import audioop
import base64
import json
import time
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.services.speaker_verify import enrollment as voiceprint_enrollment
from app.services.speaker_verify import get_speaker_verify_service
from app.services.stt.deepgram import DeepgramSTTService
from app.services.tts.azure import AzureTTSService
from app.services.vad.silero_vad import SileroVADService

router = APIRouter()
_stt = DeepgramSTTService()
_tts = AzureTTSService()
_vad = SileroVADService()
_verifier = get_speaker_verify_service()

_VAD_FRAME_BYTES = 1024     # linear16 16kHz, 512 samples
_SILENCE_THRESHOLD = 30     # 연속 침묵 VAD 프레임 수 (~960ms)
_TWILIO_CHUNK_BYTES = 160   # 20ms mulaw 8kHz — Twilio 권장 단위


@router.post("/incoming")
async def incoming_call(request: Request):
    host = request.headers.get("host", "")
    ws_url = f"wss://{host}/call/ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" />
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

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                print("[WS] connected")

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                print(f"[WS] start streamSid={stream_sid}")
                audio_buffer.clear()
                pcm_buffer.clear()
                utterance_pcm.clear()
                ratecv_state = None
                silence_count = 0
                had_speech = False
                is_speaking = False

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
                    utterance_pcm.extend(frame)  # verify/enrollment 용 16kHz 평행 누적

                    is_speech = await _vad.detect(frame)

                    if is_speech:
                        if not had_speech:
                            print("[VAD] 발화 시작")
                        silence_count = 0
                        had_speech = True
                    else:
                        if had_speech and silence_count == 0:
                            print("[VAD] 침묵 카운트 시작")
                        silence_count += 1
                        if silence_count >= _SILENCE_THRESHOLD and had_speech:
                            pcm_for_verify = bytes(utterance_pcm)

                            # Stage B: voiceprint 등록 후 — STT 전 화자검증 게이트
                            if stream_sid and _verifier.is_enrolled(stream_sid):
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
                                t_tts = time.perf_counter()
                                tts_audio = await _tts.synthesize(transcript)
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
                break

    except WebSocketDisconnect:
        print("[WS] 연결 끊김")
        if stream_sid:
            _verifier.cleanup(stream_sid)
            voiceprint_enrollment.cleanup(stream_sid)
