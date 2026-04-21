"""
PDF 청킹 파이프라인 통합 테스트
실행: python -m pytest tests/test_chunking.py -v -s
또는: python tests/test_chunking.py

사전 조건:
  - docker compose up -d 실행 중
  - .env에 DATABASE_URL 설정
  - tests/ 폴더에 sample.pdf 파일 존재

LLM 분류 테스트:
  - .env에 OPENAI_API_KEY 설정 시 실제 GPT-4o-mini 사용
  - 미설정 시 Mock LLM으로 대체
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.services.chunking.paragraph import ParagraphChunkingService
from app.services.chunking.pdf_processor import PDFProcessor
from app.services.embedding.mock import MockEmbeddingService
from app.services.rag.chroma import ChromaRAGService


# ==============================================================================
# 단위 테스트: 단락 청킹
# ==============================================================================

async def test_paragraph_chunking():
    print("\n[1] 단락 청킹 테스트")
    chunker = ParagraphChunkingService()
    sample = """서울중앙병원 진료 안내입니다. 저희 병원은 내과, 외과, 정형외과, 신경과, 산부인과 등 다양한 진료과목을 운영하고 있으며 최신 의료 장비를 갖추고 있습니다.

MRI 검사는 사전 예약이 필요합니다. 검사 전 6시간 금식이 필요하며 금속 물질 제거 후 입실하셔야 합니다. 폐소공포증이 있으신 분은 사전에 담당 의사와 상담하시기 바랍니다. 검사 소요 시간은 부위에 따라 30분에서 1시간 정도입니다.

영업시간은 평일 오전 9시부터 오후 6시까지이며 토요일은 오전 9시부터 오후 1시까지 운영합니다. 일요일 및 공휴일은 휴진입니다. 야간 응급실은 24시간 운영되며 응급 환자는 언제든지 내원하실 수 있습니다.

주차장은 지하 1층부터 지하 3층까지 운영되며 외래 진료 시 2시간 무료 주차가 가능합니다. 장애인 전용 주차 구역은 지하 1층 출입구 근처에 마련되어 있습니다. 주차 요금은 시간당 2천원이며 하루 최대 1만원입니다."""

    chunks = await chunker.chunk(sample)
    print(f"  청크 수: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)}자) {c[:60]}...")
    assert len(chunks) >= 2, "청크가 2개 이상이어야 합니다"
    print("  ✅ 단락 청킹 통과")


# ==============================================================================
# 통합 테스트: PDF → ChromaDB
# ==============================================================================

async def test_pdf_pipeline():
    from app.utils.config import settings
    use_real_llm = bool(settings.openai_api_key)
    llm_label = "실제 GPT-4o-mini" if use_real_llm else "Mock LLM"
    print(f"\n[2] PDF 파이프라인 테스트 ({llm_label}, Mock 임베딩)")

    pdf_path = os.path.join(os.path.dirname(__file__), "sample.pdf")
    if not os.path.exists(pdf_path):
        print(f"  ⚠️  {pdf_path} 없음 — PDF 파이프라인 테스트 건너뜀")
        return

    import asyncpg

    conn = await asyncpg.connect(settings.database_url)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE twilio_number = $1", "+821000000001"
        )
    finally:
        await conn.close()

    if not row:
        print("  ❌ tenant 없음 — seed_postgres.py 먼저 실행하세요")
        return

    tenant_id = str(row["id"])
    print(f"  tenant_id: {tenant_id}")

    llm = _make_real_llm() if use_real_llm else _make_mock_llm()

    processor = PDFProcessor(
        chunker=ParagraphChunkingService(),
        embedder=MockEmbeddingService(),
        rag=ChromaRAGService(),
        llm=llm,
    )

    doc_id = await processor.process(
        pdf_path=pdf_path,
        tenant_id=tenant_id,
        file_name="sample.pdf",
        industry="hospital",
    )
    print(f"  document_id: {doc_id}")

    # ChromaDB 저장 확인
    rag = ChromaRAGService()
    col_name = rag._collection_name(tenant_id)
    import chromadb
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    col = client.get_collection(col_name)
    count = col.count()
    print(f"  ChromaDB 컬렉션 '{col_name}' 문서 수: {count}")
    assert count > 0, "ChromaDB에 문서가 저장되어야 합니다"
    print("  ✅ PDF 파이프라인 통과")

    return tenant_id


# ==============================================================================
# 검색 테스트: 텍스트 입력 → ChromaDB 유사 청크 조회
# ==============================================================================

async def test_search(tenant_id: str):
    print("\n[3] ChromaDB 검색 테스트 (Mock 임베딩)")
    print("  ⚠️  Mock 임베딩은 난수 기반이라 의미 유사도가 없습니다.")
    print("      파이프라인 동작 확인 용도로만 사용하세요.")
    print("      BGE-M3 연결 후 실제 유사 검색이 가능합니다.\n")

    query = input("  검색할 텍스트를 입력하세요: ").strip()
    if not query:
        print("  입력 없음 — 검색 테스트 건너뜀")
        return

    embedder = MockEmbeddingService()
    rag = ChromaRAGService()

    query_embedding = await embedder.embed(query)
    results = await rag.search(query_embedding, tenant_id, top_k=3)

    print(f"\n  검색 결과 ({len(results)}개):")
    for i, chunk in enumerate(results):
        print(f"\n  [{i+1}] {chunk[:120]}{'...' if len(chunk) > 120 else ''}")

    print("\n  ✅ 검색 테스트 완료")


# ==============================================================================
# LLM 팩토리
# ==============================================================================

def _make_real_llm():
    from app.services.llm.gpt4o_mini import GPT4OMiniService
    return GPT4OMiniService()


def _make_mock_llm():
    from app.services.llm.base import BaseLLMService
    import json

    class MockLLMService(BaseLLMService):
        async def generate(self, system_prompt, user_message, temperature=0.1, max_tokens=512):
            chunks = json.loads(user_message)
            return json.dumps(
                [{"category": "테스트", "product_name": f"항목{i}"} for i in range(len(chunks))],
                ensure_ascii=False,
            )

    return MockLLMService()


# ==============================================================================
# 실행
# ==============================================================================

async def main():
    await test_paragraph_chunking()
    tenant_id = await test_pdf_pipeline()
    if tenant_id:
        await test_search(tenant_id)
    print("\n🎉 모든 테스트 완료")


if __name__ == "__main__":
    asyncio.run(main())
