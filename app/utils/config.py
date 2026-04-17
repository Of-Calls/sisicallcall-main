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

    # Google Cloud TTS
    google_application_credentials: str = ""

    # PostgreSQL
    database_url: str = "postgresql://sisicollcoll:password@localhost:5432/sisicollcoll"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # App
    env: str = "development"
    log_level: str = "INFO"

    # KNN Router (신용 연구 완료 후 확정)
    knn_confidence_threshold: float = 0.85

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
