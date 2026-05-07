"""SIP 별 1 tenant PDF reseed — chunking + 임베딩 효과 측정용.

기존 ChromaDB 컬렉션 + rag_documents 삭제 후 새 PDFProcessor 로 재인덱싱.
지정한 tenant 외에는 건드리지 않음 (다른 baseline 보존).

실행 (default — 한밭식당):
    venv\\Scripts\\python.exe scripts/reseed_one_tenant.py
실행 (병원 / 구청):
    venv\\Scripts\\python.exe scripts/reseed_one_tenant.py --sip 5
    venv\\Scripts\\python.exe scripts/reseed_one_tenant.py --sip 4
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import asyncpg
import chromadb

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# SIP 번호 → (industry, pdf 파일명) 매핑.
SIP_TO_INFO: dict[str, tuple[str, str]] = {
    "1": ("finance", "financial_service_guide.pdf"),  # 금융
    "3": ("restaurant", "store_maual.pdf"),  # 한밭식당
    "5": ("hospital", "hospital_manual.pdf"),  # 병원
    "4": ("government", "district_office_manual.pdf"),  # 강남구청
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sip", default="3",
        help=f"tenant SIP. 지원: {sorted(SIP_TO_INFO.keys())} (default: 3=한밭식당)",
    )
    args = parser.parse_args()

    if args.sip not in SIP_TO_INFO:
        logger.error("지원 안 하는 SIP=%s. 가능: %s", args.sip, sorted(SIP_TO_INFO.keys()))
        sys.exit(1)
    sip_number = args.sip
    industry, pdf_filename = SIP_TO_INFO[sip_number]
    pdf_path = ROOT / "tests" / pdf_filename

    if not pdf_path.exists():
        logger.error("PDF 없음: %s", pdf_path)
        sys.exit(1)

    # tenant_id 조회
    conn = await asyncpg.connect(settings.database_url)
    try:
        row = await conn.fetchrow(
            "SELECT id, name FROM tenants WHERE twilio_number = $1",
            sip_number,
        )
    finally:
        await conn.close()
    if not row:
        logger.error("tenant SIP=%s 없음", sip_number)
        sys.exit(1)
    tenant_id = str(row["id"])
    logger.info(
        "reseed 시작 tenant=%s (%s) sip=%s industry=%s pdf=%s",
        tenant_id[:8], row["name"], sip_number, industry, pdf_filename,
    )

    # ChromaDB 컬렉션 삭제
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    col_name = f"tenant_{tenant_id.replace('-', '')}_docs"
    try:
        client.delete_collection(col_name)
        logger.info("chroma 컬렉션 삭제: %s", col_name)
    except Exception as e:
        logger.info("chroma 컬렉션 없음 (skip): %s — %s", col_name, e)

    # rag_documents 삭제
    conn = await asyncpg.connect(settings.database_url)
    try:
        deleted = await conn.execute(
            "DELETE FROM rag_documents WHERE tenant_id = $1::uuid",
            tenant_id,
        )
        logger.info("rag_documents 삭제 result=%s", deleted)
    finally:
        await conn.close()

    # 재인덱싱 — settings.embedding_provider 따라 자동 분기 (bge-m3 / qwen3)
    from app.services.embedding import get_embedder
    from app.services.rag.chroma import ChromaRAGService
    from app.services.chunking.pdf_processor import PDFProcessor

    embedder = get_embedder()
    logger.info("embedder=%s (provider=%s)", type(embedder).__name__, settings.embedding_provider)
    rag = ChromaRAGService()
    processor = PDFProcessor(embedder=embedder, rag=rag)

    doc_id = await processor.process(
        pdf_path=str(pdf_path),
        tenant_id=tenant_id,
        file_name=pdf_path.name,
        industry=industry,
    )
    logger.info("인덱싱 완료 doc_id=%s", doc_id)

    # 결과 청크 수 확인
    col = client.get_collection(col_name)
    logger.info("chroma 청크 수: %d", col.count())


if __name__ == "__main__":
    asyncio.run(main())
