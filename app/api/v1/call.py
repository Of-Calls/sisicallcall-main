import audioop
import base64
import json

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.services.stt.deepgram import DeepgramSTTService
from app.services.tts.azure import AzureTTSService
from app.services.vad.silero_vad import SileroVADService

router = APIRouter()
_stt = DeepgramSTTService()
_tts = AzureTTSService()
_vad = SileroVADService()

_VAD_FRAME_BYTES = 1024     # linear16 16kHz, 512 samples
_SILENCE_THRESHOLD = 30     # consecutive silent VAD frames, about 960ms
_TWILIO_CHUNK_BYTES = 160   # 20ms mulaw 8kHz Twilio chunk


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
    """Send mulaw 8kHz audio to Twilio Media Streams in 20ms chunks."""
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
    print("[WS] Twilio connected")

    audio_buffer = bytearray()
    pcm_buffer = bytearray()
    ratecv_state = None
    silence_count = 0
    had_speech = False
    stream_sid = None

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
                ratecv_state = None
                silence_count = 0
                had_speech = False

            elif event == "media":
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
                        silence_count = 0
                        had_speech = True
                    else:
                        silence_count += 1
                        if silence_count >= _SILENCE_THRESHOLD and had_speech:
                            print(f"[VAD] silence detected, {len(audio_buffer)} bytes -> STT")
                            transcript = await _stt.transcribe(bytes(audio_buffer))
                            print(f"[STT] '{transcript}'")
                            audio_buffer.clear()
                            silence_count = 0
                            had_speech = False

                            if transcript and stream_sid:
                                tts_audio = await _tts.synthesize(transcript)
                                print(f"[TTS] {len(tts_audio)} bytes -> Twilio")
                                await _send_audio_to_twilio(websocket, stream_sid, tts_audio)

            elif event == "stop":
                print("[WS] stop")
                break

    except WebSocketDisconnect:
        print("[WS] disconnected")
