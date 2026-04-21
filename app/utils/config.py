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

    # KNN Router (신용 연구 완료 후 확정)
    knn_confidence_threshold: float = 0.85

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
