"""정수기 매장 4모델 사양 시드 스크립트.

희원이 매장 tenant 를 DB 에 등록한 후 1회 실행. 4모델 (A1/B1/C1/D1) 사양 텍스트를
PDFProcessor 의 polish/enrich 인프라로 가공한 뒤 ChromaDB 에 upsert.
메타: doc_type="model_spec", model_id, is_vision=true.

향후 대시보드 폼 입력으로 대체될 부분의 시연용 시드.

실행:
    venv/Scripts/python.exe scripts/seed_samsong_models.py
"""
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import asyncpg

from app.services.chunking.pdf_processor import (
    _enrich_chunks_with_llm,
    _polish_chunks_for_embedding,
)
from app.services.embedding.local import BGEM3LocalEmbeddingService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TENANT_NAME = "샘솟정수기"
_INDUSTRY = "appliance"

_MODEL_SPECS: dict[str, str] = {
    "A1": (
        "샘솟 A1 정수기는 1인 가구를 위한 컴팩트 카운터탑 모델입니다. "
        "정격 용량 3L, 냉수 전용 출수 방식입니다. 본체 크기는 가로 18cm "
        "세로 35cm 깊이 25cm 로 좁은 공간에도 설치할 수 있습니다. "
        "RO 멤브레인과 카본 복합 필터를 사용하며 필터 교체 주기는 6개월입니다. "
        "소비전력 50W, 색상은 화이트와 그레이 두 가지로 제공되고 가격은 39만원입니다."
    ),
    "B1": (
        "샘솟 B1 정수기는 4인 가구에 적합한 데스크탑 겸 카운터탑 겸용 모델입니다. "
        "정격 용량 5L, 냉수와 온수 모두 제공하는 출수 방식입니다. 본체 크기는 "
        "가로 25cm 세로 42cm 깊이 32cm 입니다. RO 멤브레인과 카본 필터에 UV "
        "살균 모듈이 추가되었으며 필터 교체 주기는 6개월입니다. 소비전력 130W, "
        "색상은 화이트와 실버 두 가지이고 가격은 79만원입니다."
    ),
    "C1": (
        "샘솟 C1 정수기는 빌트인 또는 프리스탠딩 설치가 가능한 4인에서 6인 가구용 "
        "모델입니다. 정격 용량 7L, 냉수와 온수 정수 모두 출수합니다. 본체 크기는 "
        "가로 28cm 세로 45cm 깊이 35cm 입니다. RO 멤브레인 카본 미네랄 UV 살균 "
        "복합 필터를 사용하며 자동 살균 기능을 탑재했습니다. 필터 교체 주기는 "
        "8개월, 소비전력 180W 입니다. 색상은 화이트와 스테인리스 두 가지로 "
        "제공되고 가격은 119만원입니다."
    ),
    "D1": (
        "샘솟 D1 정수기는 6인 이상 대가족을 위한 프리미엄 스탠드 모델입니다. "
        "정격 용량 10L, 냉수 온수 정수와 함께 탄산수 출수까지 지원합니다. 본체 "
        "크기는 가로 32cm 세로 75cm 깊이 38cm 입니다. RO 멤브레인 카본 미네랄 "
        "UV 살균에 탄산 모듈이 통합되었고 AI 필터 잔여량 알림 기능을 제공합니다. "
        "필터 교체 주기는 12개월, 소비전력 220W 입니다. 색상은 화이트와 블랙 두 "
        "가지로 제공되고 가격은 189만원입니다."
    ),
}


async def _resolve_tenant_id(name: str) -> str:
    conn = await asyncpg.connect(settings.database_url)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE name = $1 LIMIT 1", name
        )
    finally:
        await conn.close()
    if not row:
        print(f"ERROR: tenants 테이블에 name='{name}' 없음. INSERT 먼저 진행하세요.")
        sys.exit(1)
    return str(row["id"])


async def main() -> None:
    tenant_id = await _resolve_tenant_id(_TENANT_NAME)
    logger.info("tenant_id=%s name=%s", tenant_id, _TENANT_NAME)

    embedder = BGEM3LocalEmbeddingService()
    rag = ChromaRAGService()
    llm = GPT4OMiniService()

    model_ids = list(_MODEL_SPECS.keys())
    chunks = list(_MODEL_SPECS.values())

    polished = await _polish_chunks_for_embedding(chunks, llm)
    logger.info("polish 완료 cnt=%d", len(polished))

    metas = await _enrich_chunks_with_llm(chunks, llm)
    logger.info("enrich 완료 cnt=%d", len(metas))

    embeddings = await embedder.embed_passages(polished)
    logger.info("임베딩 완료 cnt=%d", len(embeddings))

    document_id = str(uuid.uuid4())
    for i, (mid, chunk, polished_chunk, embedding, llm_meta) in enumerate(
        zip(model_ids, chunks, polished, embeddings, metas)
    ):
        keywords_str = ", ".join(llm_meta.get("keywords") or [])[:200]
        await rag.upsert(
            doc_id=f"samsong_model_{mid}",
            content=chunk,
            embedding=embedding,
            tenant_id=tenant_id,
            metadata={
                "tenant_id": tenant_id,
                "document_id": document_id,
                "file_name": f"samsong_{mid}.spec",
                "chunk_index": i,
                "industry": _INDUSTRY,
                "llm_title": llm_meta.get("title", ""),
                "llm_summary": llm_meta.get("summary", ""),
                "llm_keywords": keywords_str,
                "llm_topic": llm_meta.get("topic", "기타"),
                "is_auth": False,
                "is_vision": True,
                "doc_type": "model_spec",
                "model_id": mid,
                "polished_text": polished_chunk[:800],
            },
        )
        logger.info(
            "upsert model_id=%s title='%s' keywords='%s'",
            mid, llm_meta.get("title", ""), keywords_str,
        )

    print(f"done — {len(model_ids)} models seeded for tenant={tenant_id}")


if __name__ == "__main__":
    asyncio.run(main())
