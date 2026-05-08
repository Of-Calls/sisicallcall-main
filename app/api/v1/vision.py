from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.services.vision.session import VisionSessionService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_session_svc = VisionSessionService()

_VISION_PAGE_HTML = (
    Path(__file__).parent.parent.parent / "static" / "vision_upload.html"
).read_text(encoding="utf-8")


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
    """SMS 링크에서 진입하는 사진 업로드 페이지."""
    return HTMLResponse(content=_VISION_PAGE_HTML)
