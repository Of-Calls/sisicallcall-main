from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

from app.core.config import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from app.core.middleware import RequestLoggingMiddleware
from app.api.v1 import auth, call, post_call, summary, tenant, dashboard, vision
from app.api.v1.oauth import router as oauth_router
from app.services.embedding import get_embedder
from app.utils.logger import get_logger

_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _logger.info("startup: loading BGE-M3 embedding model...")
    get_embedder()
    _logger.info("startup: embedding model ready")

    _logger.info("startup: warming up speaker verify (ONNX)...")
    from app.services.speaker_verify import get_speaker_verify_service
    await get_speaker_verify_service().warmup()
    _logger.info("startup: speaker verify ready")

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
app.include_router(auth.router,      prefix="/auth",       tags=["auth"])
app.include_router(vision.router,    prefix="/vision",     tags=["vision"])
app.include_router(oauth_router,     prefix="/api/v1/oauth", tags=["oauth"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": APP_TITLE}
