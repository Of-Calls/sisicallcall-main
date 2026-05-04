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

    # App
    env: str = "development"
    log_level: str = "INFO"
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # TTS Output Channel 모드 — "mock" (기본, 테스트/유닛) | "twilio" (프로덕션 WebSocket)
    tts_channel_mode: str = "mock"

    # TitaNet 화자 검증 (대영 R-01 연구 결과 — titanet_large 채택)
    # threshold 변경 이력:
    #   0.40 → 0.30 (barge-in false negative 줄이려)
    #   0.30 → 0.45 (한뼘통화 echo/잔향이 0.30~0.40 회색지대 통과해 거짓 cancel — 2026-04-28)
    #   0.45 → 0.30 (2026-04-29): 짧은 발화 (1초 미만) sim 0.31 → verified=False → graph
    #     EOF 사례 다수 (server_220342.log Turn 4). 본인 음성도 짧으면 임베딩 거리가 멀어지는
    #     TitaNet 한계. echo 위험 재증가하지만 STT fallback (graph.py route_after_speaker_verify)
    #     이중 안전망과 함께 적용. 실통화 측정 후 0.35 등으로 재조정 가능.
    titanet_model_name: str = "titanet_large"
    titanet_similarity_threshold: float = 0.30
    titanet_enrollment_sec: float = 3.0   # voiceprint 등록에 사용할 첫 발화 누적 시간

    # WebRTC VAD (주미 연구 결과 — webrtc_vad 채택). 2026-04-30 Silero 로 교체.
    # 보존 사유: rollback 안전판 (call.py / vad_node.py 의 import 1줄 환원으로 즉시 복구).
    webrtc_mode: int = 3                        # aggressiveness 0~3 (3: 최대 잡음 제거)
    webrtc_frame_ms: int = 30                   # 프레임 크기 ms (10/20/30 중 택일)
    webrtc_speech_ratio_threshold: float = 0.5  # 프레임 중 발화 비율 임계값
    webrtc_energy_fallback_threshold: int = 1200 # webrtcvad 미설치 시 energy fallback RMS 임계값

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

    # extra="ignore" — .env 에 코드에서 제거된 잔여 키(예: 과거 GOOGLE_APPLICATION_CREDENTIALS)
    # 가 있어도 ValidationError 로 죽지 않게. 신규 키는 위 클래스 필드로 명시 정의 필요.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
