import asyncio

from app.services.ocr.base import BaseOCRService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TesseractOCRService(BaseOCRService):
    async def extract_text(self, image_bytes: bytes) -> str:
        import pytesseract
        from PIL import Image
        import io

        loop = asyncio.get_running_loop()

        def _run():
            image = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(image, lang="kor+eng")

        return await loop.run_in_executor(None, _run)
