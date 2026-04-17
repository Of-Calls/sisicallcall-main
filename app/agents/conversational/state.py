from typing import Optional, TypedDict


class CallState(TypedDict):
    # 식별자
    call_id: str
    tenant_id: str
    turn_index: int

    # 오디오 입력
    audio_chunk: bytes

    # VAD / 화자 검증
    is_speech: bool
    is_speaker_verified: bool

    # STT 결과
    raw_transcript: str
    normalized_text: str

    # 임베딩 및 Cache 분기
    query_embedding: list[float]
    cache_hit: bool

    # 라우팅
    knn_intent: Optional[str]
    knn_confidence: float
    primary_intent: Optional[str]       # "intent_faq" | "intent_task" | "intent_auth" | "intent_escalation"
    secondary_intents: list[str]
    routing_reason: Optional[str]

    # 세션 view (Redis 에서 로드한 당 턴 관점 정보)
    session_view: dict

    # FAQ 브랜치 내부용
    rag_results: list[str]

    # 최종 응답
    response_text: str
    response_path: str                  # "cache" | "faq" | "task" | "auth" | "escalation"

    # Reviewer
    reviewer_applied: bool
    reviewer_verdict: Optional[str]     # "pass" | "revise"

    # 에러 / 타임아웃
    is_timeout: bool
    error: Optional[str]
