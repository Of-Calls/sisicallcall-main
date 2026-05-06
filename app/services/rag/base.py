from abc import ABC, abstractmethod


class BaseRAGService(ABC):
    @abstractmethod
    async def search(
        self, query_embedding: list[float], tenant_id: str, top_k: int = 3
    ) -> list[str]:
        """벡터 유사도 검색 — FAQ 브랜치 전용."""
        raise NotImplementedError

    @abstractmethod
    async def search_with_meta(
        self,
        query_embedding: list[float],
        tenant_id: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[dict]:
        """벡터 검색 + id/distance/metadata 동봉 반환 — 진단/로깅용.

        각 원소: {"id": str, "document": str, "distance": float|None, "metadata": dict}
        where: ChromaDB metadata 필터 (예: {"doc_type": "model_spec", "model_id": "B1"}).
        None 이면 필터 없음.
        """
        raise NotImplementedError

    @abstractmethod
    async def upsert(
        self,
        doc_id: str,
        content: str,
        embedding: list[float],
        tenant_id: str,
        metadata: dict,
    ) -> None:
        """RAG 문서 저장 (소프트 삭제 시 벡터 동시 삭제 필수)."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, doc_id: str, tenant_id: str) -> None:
        """단일 청크 벡터 삭제."""
        raise NotImplementedError

    @abstractmethod
    async def delete_by_document(self, document_id: str, tenant_id: str) -> None:
        """document_id에 속한 모든 청크 삭제 — 문서 교체/삭제 시 사용."""
        raise NotImplementedError
