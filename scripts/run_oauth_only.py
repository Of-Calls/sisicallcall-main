"""
OAuth 전용 FastAPI 서버 — NeMo/STT/TTS 의존 없이 OAuth 플로우만 로컬 테스트.

실행:
  uvicorn scripts.run_oauth_only:app --reload --port 8000

필수 .env 설정:
  TOKEN_ENCRYPTION_KEY         Fernet 키 (python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
  GOOGLE_OAUTH_CLIENT_ID       Google Cloud Console OAuth 2.0 클라이언트 ID
  GOOGLE_OAUTH_CLIENT_SECRET   Google Cloud Console OAuth 2.0 클라이언트 Secret
  GOOGLE_CALENDAR_REDIRECT_URI http://localhost:8000/api/v1/oauth/google_calendar/callback
  GOOGLE_GMAIL_REDIRECT_URI    http://localhost:8000/api/v1/oauth/google_gmail/callback
  DATABASE_URL                 Postgres 연결 (tenant_integrations 테이블 저장)

Google Cloud Console 승인된 리디렉션 URI 등록:
  http://localhost:8000/api/v1/oauth/google_calendar/callback
  http://localhost:8000/api/v1/oauth/google_gmail/callback

테스트 흐름:
  1. 브라우저에서 http://localhost:8000/api/v1/oauth/google_calendar/authorize?tenant_id=my-tenant 열기
  2. Google 로그인 후 콜백 자동 처리 → JSON 응답 확인
  3. http://localhost:8000/api/v1/oauth/google_calendar/status?tenant_id=my-tenant 로 상태 확인
"""
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.api.v1.oauth import router as oauth_router
from app.repositories.tenant_integration_repo import tenant_integration_repo

app = FastAPI(
    title="Sisicallcall OAuth Only",
    description="OAuth 연동 테스트 전용 서버 — NeMo/STT/TTS 불필요",
)
app.include_router(oauth_router, prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
def health():
    return {"status": "ok"}
