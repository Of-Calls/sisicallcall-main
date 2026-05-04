import asyncio
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 서브모듈 import 이전에 .env 로드 — Azure/Twilio/Deepgram 등 외부 서비스 자격증명이
# os.environ 경유로 읽히므로 반드시 최상단에서 수행해야 한다.
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import admin_auth, auth, call, dashboard, post_call, summary, tenant
from app.api.v1.oauth import router as oauth_router
from app.utils.logger import get_logger

_logger = get_logger(__name__)


async def _warmup_titanet() -> None:
    """TitaNet 모델 백그라운드 warm-up — startup 차단 안 함.

    첫 통화 시점까지 보통 끝남. 만약 warm-up 끝나기 전 첫 통화가 들어오면
    enrollment_node 가 lazy load (~5~10초) 로 fallback. 통화 진행 자체는 가능.
    """
    t0 = time.monotonic()
    try:
        _logger.info("TitaNet 모델 warm-up 시작 (background)")
        from app.services.speaker_verify.titanet import get_titanet_service
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_titanet_service)
        elapsed = time.monotonic() - t0
        _logger.info("TitaNet 모델 warm-up 완료 elapsed=%.2fs — 화자 검증 준비됨", elapsed)
    except Exception as e:
        elapsed = time.monotonic() - t0
        _logger.error("TitaNet warm-up 실패 elapsed=%.2fs — 첫 발화 시 lazy load 됨: %s", elapsed, e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Azure TTS 는 클라우드 SDK 호출 (200~400ms) — 모델 로딩/사전 합성 캐시 불필요.
    # cache_node 의 BGE-M3 는 module-level 인스턴스화로 이미 자동 warm-up 됨 (elapsed 로그 참조).

    # TitaNet warm-up 을 background task 로 띄워 startup 즉시 yield → uvicorn 가
    # health check 받을 준비. await 하지 않음 — 첫 통화 시점까지 백그라운드 진행.
    asyncio.create_task(_warmup_titanet())

    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(call.router,      prefix="/call",       tags=["call"])
app.include_router(post_call.router, prefix="/post-call",  tags=["post-call"])
app.include_router(summary.router,   prefix="/summary",    tags=["summary"])
app.include_router(tenant.router,    prefix="/tenant",     tags=["tenant"])
app.include_router(dashboard.router, prefix="/dashboard",  tags=["dashboard"])
app.include_router(admin_auth.router, prefix="/auth",      tags=["admin-auth"])
app.include_router(auth.router,      prefix="/auth",       tags=["auth"])
app.include_router(oauth_router,     prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
