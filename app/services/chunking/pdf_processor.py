import asyncio
import json
import uuid

import asyncpg

from app.services.chunking.base import BaseChunkingService
from app.services.embedding.base import BaseEmbeddingService
from app.services.llm.base import BaseLLMService
from app.services.rag.chroma import ChromaRAGService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFY_SYSTEM = """당신은 문서 청크를 분류하는 전문가입니다.
주어진 텍스트 청크를 분석하여 JSON 객체로 응답하세요.
응답 형식: {"category": "카테고리명", "product_name": "세부항목명"}

category 예시: 진료안내, 영업시간, 메뉴, 예약, 위치, 가격, 준비사항, 기타
product_name: 해당 청크의 핵심 주제를 한 줄로 요약
JSON 객체만 출력하고 다른 텍스트는 포함하지 마세요."""


class PDFProcessor:
    def __init__(
        self,
        chunker: BaseChunkingService,
        embedder: BaseEmbeddingService,
        rag: ChromaRAGService,
        llm: BaseLLMService,
    ):
        self._chunker = chunker
        self._embedder = embedder
        self._rag = rag
        self._llm = llm

    async def process(
        self,
        pdf_path: str,
        tenant_id: str,
        file_name: str,
        industry: str,
    ) -> str:
        """PDF → 청킹 → 분류 → 임베딩 → ChromaDB 저장. document_id 반환."""

        # 1. 중복 문서 처리 — 같은 tenant + file_name이 존재하면 기존 청크 삭제
        existing = await self._find_existing_document(tenant_id, file_name)
        if existing:
            logger.info("duplicate detected, replacing doc_id=%s file=%s", existing["id"], file_name)
            await self._rag.delete_by_document(str(existing["id"]), tenant_id)
            await self._delete_rag_document(existing["id"])

        # 2. rag_documents INSERT (status=processing)
        document_id = await self._insert_rag_document(tenant_id, file_name)
        logger.info("rag_document created id=%s file=%s", document_id, file_name)

        try:
            # 2. PDF 텍스트 추출
            text = await self._extract_text(pdf_path)
            logger.info("pdf extracted len=%d file=%s", len(text), file_name)

            # 3. 단락 청킹
            chunks = await self._chunker.chunk(text)
            logger.info("chunked count=%d file=%s", len(chunks), file_name)

            # 4. LLM 주제 분류 (배치)
            classifications = await self._classify_chunks(chunks)

            # 5. 임베딩 (배치)
            embeddings = await self._embedder.embed_batch(chunks)

            # 6. ChromaDB upsert
            collection_name = self._rag._collection_name(tenant_id)
            for i, (chunk, embedding, cls) in enumerate(
                zip(chunks, embeddings, classifications)
            ):
                chunk_doc_id = f"{document_id}_chunk_{i}"
                metadata = {
                    "tenant_id": tenant_id,
                    "document_id": str(document_id),
                    "file_name": file_name,
                    "chunk_index": i,
                    "product_name": cls.get("product_name", ""),
                    "category": cls.get("category", "기타"),
                    "industry": industry,
                }
                await self._rag.upsert(
                    doc_id=chunk_doc_id,
                    content=chunk,
                    embedding=embedding,
                    tenant_id=tenant_id,
                    metadata=metadata,
                )

            # 7. rag_documents UPDATE (status=ready)
            await self._update_rag_document(document_id, len(chunks), collection_name)
            logger.info("pdf_processor done doc_id=%s chunks=%d", document_id, len(chunks))

        except Exception as e:
            await self._fail_rag_document(document_id)
            logger.error("pdf_processor failed doc_id=%s err=%s", document_id, e)
            raise

        return str(document_id)

    async def _extract_text(self, pdf_path: str) -> str:
        def _read():
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read)

    async def _classify_chunks(self, chunks: list[str]) -> list[dict]:
        tasks = [self._classify_single(chunk) for chunk in chunks]
        return list(await asyncio.gather(*tasks))

    async def _classify_single(self, chunk: str) -> dict:
        raw = await self._llm.generate(
            system_prompt=_CLASSIFY_SYSTEM,
            user_message=chunk,
            temperature=0.1,
            max_tokens=150,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0].strip()

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict) and "category" in result:
                return result
        except json.JSONDecodeError:
            pass
        return {"category": "기타", "product_name": ""}

    async def _insert_rag_document(self, tenant_id: str, file_name: str) -> uuid.UUID:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_documents (tenant_id, file_name, file_type, status)
                VALUES ($1::uuid, $2, 'pdf', 'processing')
                RETURNING id
                """,
                tenant_id,
                file_name,
            )
            return row["id"]
        finally:
            await conn.close()

    async def _update_rag_document(
        self, document_id: uuid.UUID, chunk_count: int, collection_name: str
    ) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """
                UPDATE rag_documents
                SET status = 'ready',
                    chunk_count = $2,
                    chroma_collection = $3,
                    indexed_at = now()
                WHERE id = $1
                """,
                document_id,
                chunk_count,
                collection_name,
            )
        finally:
            await conn.close()

    async def _fail_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                "UPDATE rag_documents SET status = 'failed' WHERE id = $1",
                document_id,
            )
        finally:
            await conn.close()

    async def _find_existing_document(self, tenant_id: str, file_name: str) -> dict | None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                SELECT id FROM rag_documents
                WHERE tenant_id = $1::uuid AND file_name = $2 AND status != 'failed'
                ORDER BY uploaded_at DESC LIMIT 1
                """,
                tenant_id,
                file_name,
            )
            return dict(row) if row else None
        finally:
            await conn.close()

    async def _delete_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute("DELETE FROM rag_documents WHERE id = $1", document_id)
        finally:
            await conn.close()

    async def delete_document(self, document_id: str, tenant_id: str) -> None:
        """문서 삭제 — ChromaDB 청크 + rag_documents 동시 삭제."""
        await self._rag.delete_by_document(document_id, tenant_id)
        await self._delete_rag_document(uuid.UUID(document_id))
        logger.info("document deleted doc_id=%s tenant=%s", document_id, tenant_id)
