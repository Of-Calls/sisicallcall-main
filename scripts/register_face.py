"""finance tenant 에 희원 얼굴 1회 등록.

실행: python -m scripts.register_face
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.services.auth.arcface import ArcFaceAuthService

FINANCE_TENANT_ID = "22cde253-370d-4f2b-9a6b-93f633cb059a"
CUSTOMER_REF = "01047722480"
IMAGE_PATH = Path(__file__).resolve().parent / "myface.jpg"


async def main():
    if not IMAGE_PATH.exists():
        print(f"❌ 이미지 없음: {IMAGE_PATH}")
        sys.exit(1)
    img = IMAGE_PATH.read_bytes()
    svc = ArcFaceAuthService()
    await svc.register_face(
        image_bytes=img,
        tenant_id=FINANCE_TENANT_ID,
        customer_ref=CUSTOMER_REF,
    )
    print(f"✅ 등록 완료 tenant_id={FINANCE_TENANT_ID} customer_ref={CUSTOMER_REF}")


if __name__ == "__main__":
    asyncio.run(main())
