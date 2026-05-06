from app.services.rag.base import BaseRAGService
from app.utils.config import settings
from app.utils.logger import get_logger

# ChromaDB 컬렉션명: tenant_{tenant_id_without_hyphens}_docs

logger = get_logger(__name__)


class ChromaRAGService(BaseRAGService):
    def __init__(self):
        import chromadb
        self._client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )

    def _collection_name(self, tenant_id: str) -> str:
        return f"tenant_{tenant_id.replace('-', '')}_docs"

    async def search(
        self, query_embedding: list[float], tenant_id: str, top_k: int = 3
    ) -> list[str]:
        """벡터 유사도 검색 — FAQ 브랜치 전용."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _query():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            result = col.query(query_embeddings=[query_embedding], n_results=top_k)
            return result["documents"][0] if result["documents"] else []

        return await loop.run_in_executor(None, _query)

    async def search_with_meta(
        self,
        query_embedding: list[float],
        tenant_id: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """벡터 검색 + id/distance/metadata 동봉 반환 — 진단/로깅용.

        where: ChromaDB metadata 필터. 예) {"doc_type": "model_spec", "model_id": "B1"}.
        None 이면 필터 없이 컬렉션 전체에서 top_k 검색.
        """
        import asyncio

        loop = asyncio.get_event_loop()

        def _query():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            result = col.query(**kwargs)
            docs_outer = result.get("documents") or []
            if not docs_outer:
                return []
            ids = (result.get("ids") or [[]])[0]
            docs = docs_outer[0] or []
            metas = (result.get("metadatas") or [[]])[0]
            dists = (result.get("distances") or [[]])[0]
            out: list[dict] = []
            for i, doc in enumerate(docs):
                out.append({
                    "id": ids[i] if i < len(ids) else "",
                    "document": doc,
                    "distance": dists[i] if i < len(dists) else None,
                    "metadata": metas[i] if i < len(metas) else {},
                })
            return out

        return await loop.run_in_executor(None, _query)

    async def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        tenant_id: str,
        metadata: dict,
    ) -> None:
        """RAG 문서 저장 (소프트 삭제 시 ChromaDB 벡터 동시 삭제 필수)."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _upsert():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            col.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
            )

        await loop.run_in_executor(None, _upsert)
        logger.info("chroma upsert doc_id=%s tenant=%s", doc_id, tenant_id)

    async def delete(self, doc_id: str, tenant_id: str) -> None:
        """소프트 삭제 시 ChromaDB 벡터 동시 삭제 (db_schema.md 규칙)."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _delete():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            col.delete(ids=[doc_id])

        await loop.run_in_executor(None, _delete)
        logger.info("chroma delete doc_id=%s tenant=%s", doc_id, tenant_id)

    async def delete_by_document(self, document_id: str, tenant_id: str) -> None:
        """document_id에 속한 모든 청크 삭제 — 문서 교체/삭제 시 사용."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _delete():
            col = self._client.get_or_create_collection(self._collection_name(tenant_id))
            col.delete(where={"document_id": {"$eq": document_id}})

        await loop.run_in_executor(None, _delete)
        logger.info("chroma delete_by_document document_id=%s tenant=%s", document_id, tenant_id)
