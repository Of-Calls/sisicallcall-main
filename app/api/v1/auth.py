from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.schemas.auth import (
    AuthInitiateRequest,
    AuthInitiateResponse,
    AuthStatusResponse,
    FaceRegisterResponse,
    FaceVerifyResponse,
    LivenessCompleteRequest,
    LivenessCompleteResponse,
    LivenessInstructionsResponse,
)
from app.services.auth.arcface import ArcFaceAuthService
from app.services.auth.liveness import LivenessService
from app.services.auth.session import AuthSessionService
from app.services.sms import get_sms_service
from app.utils.config import settings
from app.utils.logger import get_logger

_AUTH_PAGE_HTML = (
    Path(__file__).parent.parent.parent / "static" / "auth_face.html"
).read_text(encoding="utf-8")

logger = get_logger(__name__)

router = APIRouter()

_session_svc = AuthSessionService()
_liveness_svc = LivenessService()
_auth_svc = ArcFaceAuthService()
_sms_svc = get_sms_service()


@router.post("/verify", response_model=AuthInitiateResponse)
async def initiate_auth(body: AuthInitiateRequest):
    """인증 세션 생성 + 고객에게 얼굴 인증 링크 SMS 발송."""
    auth_id = await _session_svc.create_session(
        tenant_id=body.tenant_id,
        customer_ref=body.customer_ref,
        customer_phone=body.customer_phone,
        call_id=body.call_id,
    )
    auth_url = f"{settings.auth_web_base_url}/auth/{auth_id}"
    sms_body = f"[시시콜콜] 본인인증을 위해 아래 링크를 열어주세요.\n{auth_url}"
    sent = await _sms_svc.send_sms(to=body.customer_phone, body=sms_body)
    if not sent:
        logger.error("SMS 발송 실패 auth_id=%s phone=%s", auth_id, body.customer_phone)
    return AuthInitiateResponse(
        auth_id=auth_id,
        status="pending",
        message="SMS 발송 완료" if sent else "SMS 발송 실패 — 인증 세션은 유효",
    )


@router.get("/{auth_id}/liveness", response_model=LivenessInstructionsResponse)
async def get_liveness_instructions(auth_id: str):
    """Liveness 지시 시퀀스 발급 — 프론트엔드 MediaPipe 초기화 시 호출."""
    session = await _session_svc.get_session(auth_id)
    if not session:
        raise HTTPException(status_code=404, detail="인증 세션이 없거나 만료됨")
    if session.get("status") not in ("pending", "liveness_pending"):
        raise HTTPException(status_code=409, detail=f"잘못된 상태: {session.get('status')}")

    await _session_svc.update_status(auth_id, "liveness_pending")
    result = await _liveness_svc.generate_instructions(auth_id)
    return LivenessInstructionsResponse(
        auth_id=auth_id,
        instructions=result["instructions"],
        token=result["token"],
    )


@router.post("/{auth_id}/liveness", response_model=LivenessCompleteResponse)
async def complete_liveness(auth_id: str, body: LivenessCompleteRequest):
    """Liveness 완료 보고 — 서버 HMAC 토큰 검증."""
    session = await _session_svc.get_session(auth_id)
    if not session:
        raise HTTPException(status_code=404, detail="인증 세션이 없거나 만료됨")
    if session.get("status") != "liveness_pending":
        raise HTTPException(status_code=409, detail=f"잘못된 상태: {session.get('status')}")

    valid = await _liveness_svc.validate_token(auth_id, body.token)
    if not valid:
        raise HTTPException(status_code=400, detail="Liveness 토큰 검증 실패 — 다시 시도하세요")

    await _session_svc.set_liveness_passed(auth_id)
    return LivenessCompleteResponse(auth_id=auth_id, liveness_passed=True)


@router.post("/{auth_id}/face", response_model=FaceVerifyResponse)
async def verify_face(auth_id: str, file: UploadFile = File(...)):
    """얼굴 이미지 인증 — ArcFace cosine similarity.

    multipart/form-data: file (JPEG/PNG, 정면 얼굴 1장)
    3회 초과 실패 시 세션 blocked → 상담원 Escalation 전환.
    """
    session = await _session_svc.get_session(auth_id)
    if not session:
        raise HTTPException(status_code=404, detail="인증 세션이 없거나 만료됨")
    if session.get("status") == "blocked":
        raise HTTPException(status_code=403, detail="인증 차단됨 — 상담원 연결로 전환됩니다")
    if session.get("face_verified") == "true":
        raise HTTPException(status_code=409, detail="이미 인증 완료된 세션입니다")

    image_bytes = await file.read()
    attempts = await _session_svc.increment_face_attempts(auth_id)
    max_retries = settings.arcface_max_retries

    verified, similarity = await _auth_svc.verify_face(
        image_bytes=image_bytes,
        tenant_id=session["tenant_id"],
        customer_ref=session["customer_ref"],
    )

    if verified:
        await _session_svc.set_face_verified(auth_id)
        return FaceVerifyResponse(
            auth_id=auth_id,
            verified=True,
            similarity_score=round(similarity, 4),
            attempts_remaining=max_retries - attempts,
        )

    remaining = max_retries - attempts
    if remaining <= 0:
        await _session_svc.set_blocked(auth_id)
        logger.warning("얼굴 인증 최대 재시도 초과 → 차단 auth_id=%s", auth_id)
        raise HTTPException(status_code=403, detail="인증 실패 — 상담원 연결로 전환됩니다")

    return FaceVerifyResponse(
        auth_id=auth_id,
        verified=False,
        similarity_score=round(max(similarity, 0.0), 4),
        attempts_remaining=remaining,
    )


@router.get("/{auth_id}/status", response_model=AuthStatusResponse)
async def get_auth_status(auth_id: str):
    """인증 세션 상태 폴링 — 전화 응대 에이전트가 결과 확인 시 사용."""
    session = await _session_svc.get_session(auth_id)
    if not session:
        raise HTTPException(status_code=404, detail="인증 세션이 없거나 만료됨")
    return AuthStatusResponse(
        auth_id=auth_id,
        status=session.get("status", "unknown"),
        liveness_passed=session.get("liveness_passed") == "true",
        face_verified=session.get("face_verified") == "true",
        created_at=session.get("created_at"),
    )


@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    tenant_id: str,
    customer_ref: str,
    file: UploadFile = File(...),
):
    """임시 얼굴 등록 엔드포인트.

    AUTH_ENABLE_TEST_REGISTER=true 환경변수 설정 시에만 활성.
    운영 배포에서는 절대 활성화하지 말 것.
    """
    if not settings.auth_enable_test_register:
        raise HTTPException(status_code=403, detail="임시 등록 엔드포인트 비활성 — AUTH_ENABLE_TEST_REGISTER=true 필요")
    image_bytes = await file.read()
    await _auth_svc.register_face(
        image_bytes=image_bytes,
        tenant_id=tenant_id,
        customer_ref=customer_ref,
    )
    return FaceRegisterResponse(
        tenant_id=tenant_id,
        customer_ref=customer_ref,
        registered_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/{auth_id}", response_class=HTMLResponse)
async def auth_page(auth_id: str) -> HTMLResponse:
    """SMS 링크에서 진입하는 얼굴 인증 페이지.

    HTML 안 JS 가 /auth/{auth_id}/liveness, /face, /status 를 직접 호출.
    더 구체적인 라우트들(/liveness, /face, /status, /register, /verify)이
    위에 먼저 선언돼 있으므로 catch-all 처럼 동작해도 충돌 없음.
    """
    return HTMLResponse(content=_AUTH_PAGE_HTML)
