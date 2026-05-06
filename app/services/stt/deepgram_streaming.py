from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from app.utils.config import settings


class DeepgramStreamingSTTService:

    def __init__(self):
        self._client = DeepgramClient(settings.deepgram_api_key)
        self._connections: dict[str, object] = {}

    async def open(self, call_id: str) -> None:
        connection = self._client.listen.asyncwebsocket.v("1")

        async def _on_transcript(self_dg, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if result.is_final and sentence.strip():
                    print(f"[STT-STREAM] is_final: '{sentence}'")
            except Exception:
                pass

        connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)

        options = LiveOptions(
            model="nova-3",
            language="ko",
            encoding="mulaw",
            sample_rate=8000,
            smart_format=True,
            punctuate=True,
            interim_results=True,
            endpointing=400,
        )

        started = await connection.start(options)
        if not started:
            raise RuntimeError(f"Deepgram 스트리밍 연결 실패 call_id={call_id}")

        self._connections[call_id] = connection
        print(f"[STT-STREAM] 연결됨 call_id={call_id}")

    async def send(self, call_id: str, audio: bytes) -> None:
        conn = self._connections.get(call_id)
        if conn:
            await conn.send(audio)

    async def close(self, call_id: str) -> None:
        conn = self._connections.pop(call_id, None)
        if conn:
            await conn.finish()
            print(f"[STT-STREAM] 종료 call_id={call_id}")
