from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.services.vision.convnextv2 import ConvNeXtV2VisionService
from app.services.vision.session import VisionSessionService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_session_svc = VisionSessionService()
_classifier = ConvNeXtV2VisionService()

_VISION_PAGE_HTML = (
    Path(__file__).parent.parent.parent / "static" / "vision_upload.html"
).read_text(encoding="utf-8")


@router.post("/{vision_id}/predict")
async def predict_image(vision_id: str, file: UploadFile = File(...)):
    """폰에서 업로드한 사진 분류 — Mock(B1) 또는 학습된 모델."""
    session = await _session_svc.get_session(vision_id)
    if not session:
        raise HTTPException(status_code=404, detail="vision 세션이 없거나 만료됨")
    if session.get("status") not in ("pending", "analyzing"):
        raise HTTPException(status_code=409, detail=f"잘못된 상태: {session.get('status')}")

    await _session_svc.set_analyzing(vision_id)
    image_bytes = await file.read()
    try:
        result = await _classifier.classify(image_bytes)
    except Exception as exc:
        logger.exception("vision classify 실패: %s", exc)
        await _session_svc.set_failed(vision_id, reason=str(exc))
        raise HTTPException(status_code=500, detail="이미지 분석 실패")

    label = str(result.get("label", ""))
    confidence = float(result.get("confidence", 0.0))
    await _session_svc.set_analyzed(vision_id, label, confidence)
    logger.info("vision analyzed vision_id=%s label=%s conf=%.4f", vision_id, label, confidence)
    return {
        "vision_id": vision_id,
        "status": "analyzed",
        "label": label,
        "confidence": confidence,
    }


@router.get("/{vision_id}/status")
async def get_vision_status(vision_id: str):
    """vision 세션 상태 폴링."""
    session = await _session_svc.get_session(vision_id)
    if not session:
        raise HTTPException(status_code=404, detail="vision 세션이 없거나 만료됨")
    return {
        "vision_id": vision_id,
        "status": session.get("status", "unknown"),
        "label": session.get("label", ""),
        "confidence": session.get("confidence", ""),
    }


@router.get("/{vision_id}", response_class=HTMLResponse)
async def vision_page(vision_id: str) -> HTMLResponse:
    """SMS 링크에서 진입하는 사진 업로드 페이지.

    더 구체적인 라우트(/predict, /status)가 위에 먼저 선언돼 있으므로
    catch-all 처럼 동작해도 충돌 없음.
    """
    return HTMLResponse(content=_VISION_PAGE_HTML)
