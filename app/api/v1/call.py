import base64
import json

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.agents.conversational.graph import build_call_graph
from app.agents.conversational.state import CallState
from app.core.events import CALL_ENDED, CALL_STARTED
from app.utils.audio import mulaw_to_pcm16, reset_resample_state
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# 그래프 싱글톤 (앱 기동 시 1회 컴파일)
_graph = build_call_graph()


@router.post("/incoming")
async def incoming_call(request: Request):
    """
    Twilio가 전화 수신 시 호출하는 webhook.
    TwiML을 반환해 Twilio Media Streams WebSocket 연결을 지시한다.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    tenant_id = form.get("To", "unknown")  # Twilio 수신 번호 → tenant 식별자 (임시)

    logger.info(f"[{CALL_STARTED}] call_sid={call_sid} tenant_id={tenant_id}")

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


@router.websocket("/ws/{call_id}")
async def call_websocket(
    websocket: WebSocket,
    call_id: str,
    tenant_id: str = Query(default="unknown"),
):
    """
    Twilio Media Streams WebSocket 엔드포인트.
    오디오 청크를 수신해 LangGraph 파이프라인에 투입한다.
    """
    await websocket.accept()
    logger.info(f"WebSocket 연결 수락 call_id={call_id}")

    turn_index = 0
    audio_buffer = bytearray()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                logger.info(f"call_id={call_id} Twilio Media Stream connected")

            elif event == "start":
                stream_sid = msg.get("streamSid", "")
                custom_params = msg.get("start", {}).get("customParameters", {})
                tenant_id = custom_params.get("tenant_id", tenant_id)
                logger.info(
                    f"call_id={call_id} stream_sid={stream_sid} tenant_id={tenant_id}"
                )

            elif event == "media":
                track = msg["media"].get("track", "inbound")
                if track != "inbound":
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes = mulaw_to_pcm16(mulaw_bytes)
                audio_buffer.extend(pcm_bytes)

                # 320ms 분량(16kHz, 16-bit mono = 10240 bytes) 누적 시 그래프 투입
                if len(audio_buffer) >= 10240:
                    chunk = bytes(audio_buffer)
                    audio_buffer.clear()

                    state: CallState = {
                        "call_id": call_id,
                        "tenant_id": tenant_id,
                        "turn_index": turn_index,
                        "audio_chunk": chunk,
                        "is_speech": False,
                        "is_speaker_verified": False,
                        "raw_transcript": "",
                        "normalized_text": "",
                        "query_embedding": [],
                        "cache_hit": False,
                        "knn_intent": None,
                        "knn_confidence": 0.0,
                        "primary_intent": None,
                        "secondary_intents": [],
                        "routing_reason": None,
                        "session_view": {},
                        "rag_results": [],
                        "response_text": "",
                        "response_path": "",
                        "reviewer_applied": False,
                        "reviewer_verdict": None,
                        "is_timeout": False,
                        "error": None,
                    }

                    await _graph.ainvoke(state)
                    turn_index += 1

            elif event == "stop":
                logger.info(f"[{CALL_ENDED}] call_id={call_id}")
                reset_resample_state()
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket 종료 call_id={call_id}")
        reset_resample_state()
    except Exception as e:
        logger.error(f"call_id={call_id} WebSocket 오류: {e}")
        reset_resample_state()
