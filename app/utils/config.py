from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    base_url: str = "http://localhost:8000"

    # OpenAI
    openai_api_key: str = ""

    # Deepgram
    deepgram_api_key: str = ""

    # PostgreSQL
    postgres_user: str = "sisicallcall"
    postgres_password: str = "changeme"
    postgres_db: str = "sisicallcall"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql://sisicallcall:changeme@localhost:5432/sisicallcall"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str = "redis://localhost:6379"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Embedding provider — "bge-m3" (default, FlagEmbedding) | "qwen3" (sentence-transformers).
    # swap 시 ChromaDB 의 모든 컬렉션 reseed 필수 (벡터 공간이 모델별로 다름).
    embedding_provider: str = "qwen3"

    # FAQ RAG distance threshold (ChromaDB default L2). 임베딩 모델에 따라 분포 다름.
    # - BGE-M3: 0.85 (정답 0.6~0.85, 무관 1.0+)
    # - Qwen3:  1.15 (정답 0.5~1.20, 무관 1.21+, gap 우월)
    faq_distance_threshold: float = 1.20
    # vision 게이트는 일반 humanize 보다 느슨 (model_spec 청크 top_k 진입 자체가 신호).
    faq_vision_gate_threshold: float = 0.95

    # App
    env: str = "development"
    log_level: str = "INFO"

    # Admin JWT
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # TTS Output Channel 모드 — "mock" (기본, 테스트/유닛) | "twilio" (프로덕션 WebSocket)
    tts_channel_mode: str = "mock"

    # Speaker Verification (TitaNet-L ONNX, runtime enrollment)
    # ONNX 도착 전엔 enabled=False 권장 (회귀 안전망 — verify 자동 bypass).
    # 모델 도착 시 .env 에서 SPEAKER_VERIFY_ENABLED=true.
    # threshold/enrollment_sec 는 ONNX 도착 후 실통화로 튜닝 (보통 cosine 0.4~0.6).
    speaker_verify_enabled: bool = False
    speaker_verify_model_path: str = "models/speech_verification/titanet_large.onnx"
    speaker_verify_threshold: float = 0.4
    speaker_verify_enrollment_sec: float = 3.0
    # TitaNet 짧은 발화 한계 — 2.0초 미만 발화는 임베딩 신뢰성 낮아 본인 reject 위험.
    # 미만은 verify 스킵하고 통과 (짧은 응답 보호 + 시연 안정성 우선).
    # 임베딩 신뢰성 곡선: <1s 매우 불안정, 1~1.5s 불안정, 1.5~2s 경계, 2s+ 안정.
    speaker_verify_min_audio_sec: float = 2.0

    # Silero VAD (v6.2+, 2026-04-30 채택 — 짧은 발화 + 긴 trailing silence reject 해결).
    # logs/2026-04-30/server_100651.log Turn 4/5 사례: "예약은어떻게해요" 0.5s + trailing 1.3s
    # → WebRTC bulk ratio 28~38% reject → graph END. Silero per-frame 누적으로 해결.
    silero_threshold: float = 0.5            # speech 확률 임계값 (Silero 기본). 낮추면 잡음 통과↑
    silero_min_speech_frames: int = 3        # 청크 내 speech frame (32ms each) 최소 개수.
                                              # 3 = 96ms — 짧은 단어 ("응", "예") 까지 통과
    silero_use_onnx: bool = False            # ONNX runtime 가속 (~2x faster). PyTorch JIT 가
                                              # default — 첫 운영 안정화 후 True 전환 검토.

    # TTS 합성 엔진 — "azure" (Azure Speech SDK, μ-law 8kHz 네이티브 출력) 단일화
    tts_provider: str = "azure"
    # Azure Speech (TTS) — Korean Neural Voice
    azure_speech_key: str = ""
    azure_speech_region: str = ""  # e.g. "koreacentral", "eastus"
    azure_tts_voice: str = "ko-KR-SunHiNeural"

    # TTS Throttle (barge-in 정확도용)
    # Twilio jitter buffer 가 11~14초간 음성 재생하므로 송신 속도 ≈ 재생 속도 로 맞춰
    # cancel 즉시 효과 + is_speaking 정확도 자연 확보. 음성 끊김 발생 시 enabled=False
    # 로 즉시 끄고 재시작 가능. Linux 운영 정상, Windows 개발은 chunk_interval 0.015 보정.
    tts_throttle_enabled: bool = True
    tts_preroll_chunks: int = 20            # 처음 N 청크는 즉시 송신 (시작 latency + 지터 흡수)
    tts_chunk_interval_sec: float = 0.020   # 청크 사이 throttle (160B / 8kHz = 20ms)
    tts_play_tail_margin_sec: float = 0.15  # 송신 후 jitter buffer 잔여 재생 마진

    # Barge-in verify (Phase B — VAD + 화자검증 게이트)
    # TTS 송신 중 사용자 발화로 보이는 신호가 들어오면 첫 0.8초를 추출해
    # WebRTC VAD (음성 vs 잡음) + TitaNet (등록 화자 vs 타인/echo) 통과한 경우에만
    # BARGE-IN 트리거. enrollment 미완료 시 TitaNet 가 자동 bypass(True) 반환 →
    # RMS-only 동작으로 자연스럽게 fallback. 문제 시 enabled=false 로 즉시 PR1~3 동작.
    bargein_verify_enabled: bool = True
    bargein_rms_pre_threshold: int = 1500   # verify 게이트 진입 RMS (echo 임계값 2400 보다 낮음)
    bargein_verify_chunk_bytes: int = 25600 # 0.8s × 16kHz × 2byte (PCM16 mono)
    bargein_verify_chunk_sec: float = 0.8   # 디버그/로그용

    # SMS Provider — "solapi" (기본) | "twilio"
    sms_provider: str = "solapi"
    solapi_api_key: str = ""
    solapi_api_secret: str = ""
    solapi_sender_number: str = ""

    # Face Auth (M3+)
    arcface_model_name: str = "buffalo_l"
    arcface_similarity_threshold: float = 0.6
    arcface_max_retries: int = 3
    liveness_instruction_count: int = 3
    liveness_hmac_secret: str = "change-me-in-production"
    auth_session_ttl_sec: int = 600
    auth_enable_test_register: bool = False
    auth_web_base_url: str = "http://localhost:3000"

    # Vision (정수기 모델 분류 — TorchScript 단일 파일)
    # metadata JSON 안에 input_size, normalize_mean/std, classes 정의.
    # device="auto" 시 cuda 가능하면 cuda, 아니면 cpu 자동 선택.
    vision_model_path: str = "models/water_purifier_convnextv2_femto_scripted.pt"
    vision_metadata_path: str = "models/water_purifier_convnextv2_femto_metadata.json"
    vision_device: str = "auto"

    # FAQ 시맨틱 캐시 (faq_branch 전용)
    # ChromaDB L2 squared distance (BGE-M3 normalized, L2sq = 2(1-cos_sim)).
    # 0.04 (cos_sim ≥ 0.98) — 진단 결과 (8 paraphrase + 8 unrelated) 에서
    # false hit 0%, paraphrase 25% hit. 짧은 의문문/도메인 단어 묶임 발화는
    # BGE-M3 가 표면 매칭으로 unrelated 도 cos 0.97 까지 끌어올려서 위험.
    # 긴 task/예약 발화 (cos 0.99+) 위주로 캐시 효과. miss 시 RAG fallthrough 정답 보장.
    cache_distance_threshold: float = 0.04
    cache_ttl_seconds: int = 86400  # 24h

    # extra="ignore" — .env 에 코드에서 제거된 잔여 키(예: 과거 GOOGLE_APPLICATION_CREDENTIALS)
    # 가 있어도 ValidationError 로 죽지 않게. 신규 키는 위 클래스 필드로 명시 정의 필요.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
