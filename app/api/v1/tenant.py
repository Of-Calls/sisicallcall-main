from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.api.v1.admin_auth import get_current_admin_user
from app.repositories.rag_document_repo import (
    get_rag_document_for_tenant,
    list_rag_document_chunks_for_tenant,
    list_rag_documents_for_tenant,
    mark_rag_document_processing_for_tenant,
    mark_rag_document_reindex_failed_for_tenant,
    mark_rag_document_reindex_succeeded_for_tenant,
    soft_delete_rag_document_for_tenant,
    update_rag_document_chunk_for_tenant,
    upsert_rag_document_chunks_for_tenant,
)

router = APIRouter()


class DocumentChunkUpdateRequest(BaseModel):
    content: str = Field(min_length=1)
    metadata: dict[str, Any] | None = None


def _request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def _current_admin_tenant_id(current_admin: dict[str, Any]) -> str:
    user = current_admin.get("user") or {}
    tenant_id = str(user.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin tenant",
        )
    return tenant_id


def _validate_path_tenant_id(path_tenant_id: str, current_admin: dict[str, Any]) -> str:
    jwt_tenant_id = _current_admin_tenant_id(current_admin)
    if path_tenant_id.strip().lower() != jwt_tenant_id.lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="tenant 정보가 일치하지 않습니다.",
        )
    return jwt_tenant_id


async def _process_pdf_document(
    *,
    tenant_id: str,
    pdf_path: str,
    file_name: str,
) -> str:
    from app.services.chunking.pdf_processor import PDFProcessor
    from app.services.embedding import get_embedder
    from app.services.rag.chroma import ChromaRAGService

    processor = PDFProcessor(
        embedder=get_embedder(),
        rag=ChromaRAGService(),
    )
    return await processor.process(
        pdf_path=pdf_path,
        tenant_id=tenant_id,
        file_name=file_name,
        industry="general",
    )


async def _delete_document_vectors(document_id: str, tenant_id: str) -> None:
    from app.services.rag.chroma import ChromaRAGService

    await ChromaRAGService().delete_by_document(document_id, tenant_id)


async def _backfill_document_chunks_from_chroma(document_id: str, tenant_id: str) -> None:
    from app.services.rag.chroma import ChromaRAGService

    chroma_chunks = await ChromaRAGService().list_chunks(tenant_id)
    records: list[dict[str, Any]] = []
    for chunk in chroma_chunks:
        metadata = chunk.get("metadata") or {}
        if str(metadata.get("document_id") or "") != document_id:
            continue
        records.append({
            "chunk_index": int(metadata.get("chunk_index") or chunk.get("chunk_index") or 0),
            "page_number": metadata.get("page_number"),
            "content": chunk.get("document") or "",
            "metadata": metadata,
            "embedding_status": "ready",
            "chroma_id": chunk.get("id"),
        })

    records.sort(key=lambda item: item["chunk_index"])
    await upsert_rag_document_chunks_for_tenant(
        document_id=document_id,
        tenant_id=tenant_id,
        chunks=records,
    )


async def _list_document_chunks_with_backfill(
    document_id: str,
    tenant_id: str,
    *,
    offset: int = 0,
    limit: int = 500,
) -> dict:
    result = await list_rag_document_chunks_for_tenant(
        document_id,
        tenant_id,
        offset=offset,
        limit=limit,
    )
    if result["total"] > 0:
        return result

    await _backfill_document_chunks_from_chroma(document_id, tenant_id)
    return await list_rag_document_chunks_for_tenant(
        document_id,
        tenant_id,
        offset=offset,
        limit=limit,
    )


async def _reindex_document_chunks(document_id: str, tenant_id: str) -> int:
    from app.services.cache.chroma_cache import ChromaCacheService
    from app.services.embedding import get_embedder
    from app.services.rag.chroma import ChromaRAGService

    document = await get_rag_document_for_tenant(document_id, tenant_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {document_id!r}",
        )

    chunks_result = await _list_document_chunks_with_backfill(
        document_id,
        tenant_id,
        limit=1000,
    )
    chunks = chunks_result["items"]
    await mark_rag_document_processing_for_tenant(
        document_id=document_id,
        tenant_id=tenant_id,
    )

    try:
        contents = [chunk["content"] for chunk in chunks]
        embeddings = await get_embedder().embed_passages(contents) if contents else []
        rag = ChromaRAGService()
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {
                **(chunk.get("metadata") or {}),
                "tenant_id": tenant_id,
                "document_id": document_id,
                "file_name": document["file_name"],
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk.get("page"),
            }
            chroma_id = chunk.get("chroma_id") or f"{document_id}_chunk_{chunk['chunk_index']}"
            await rag.upsert(
                doc_id=chroma_id,
                content=chunk["content"],
                embedding=embedding,
                tenant_id=tenant_id,
                metadata=metadata,
            )

        await ChromaCacheService().clear(tenant_id)
        await mark_rag_document_reindex_succeeded_for_tenant(document_id, tenant_id)
        return len(chunks)
    except Exception:
        await mark_rag_document_reindex_failed_for_tenant(document_id, tenant_id)
        raise


@router.get("/{tenant_id}/documents")
async def list_documents(
    tenant_id: str,
    status_filter: str | None = Query(default=None, alias="status"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    result = await list_rag_documents_for_tenant(
        tenant_id=jwt_tenant_id,
        status=status_filter,
        offset=offset,
        limit=limit,
    )
    return {
        "data": result,
        "request_id": _request_id(),
    }


@router.get("/{tenant_id}/documents/{document_id}")
async def get_document(
    tenant_id: str,
    document_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    record = await get_rag_document_for_tenant(document_id, jwt_tenant_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {document_id!r}",
        )
    return {
        "data": record,
        "request_id": _request_id(),
    }


@router.get("/{tenant_id}/documents/{document_id}/chunks")
async def list_document_chunks(
    tenant_id: str,
    document_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=1000),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    record = await get_rag_document_for_tenant(document_id, jwt_tenant_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {document_id!r}",
        )

    result = await _list_document_chunks_with_backfill(
        document_id,
        jwt_tenant_id,
        offset=offset,
        limit=limit,
    )
    return {
        "data": {
            "items": result["items"],
            "total": result["total"],
            "offset": offset,
            "limit": limit,
        },
        "request_id": _request_id(),
    }


@router.patch("/{tenant_id}/documents/{document_id}/chunks/{chunk_id}")
async def update_document_chunk(
    tenant_id: str,
    document_id: str,
    chunk_id: str,
    body: DocumentChunkUpdateRequest,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    record = await get_rag_document_for_tenant(document_id, jwt_tenant_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {document_id!r}",
        )

    updated = await update_rag_document_chunk_for_tenant(
        chunk_id=chunk_id,
        document_id=document_id,
        tenant_id=jwt_tenant_id,
        content=body.content,
        metadata=body.metadata,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"chunk not found: {chunk_id!r}",
        )

    return {
        "data": updated,
        "request_id": _request_id(),
    }


@router.post("/{tenant_id}/documents/{document_id}/reindex")
async def reindex_document(
    tenant_id: str,
    document_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    try:
        chunk_count = await _reindex_document_chunks(document_id, jwt_tenant_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"document reindex failed: {exc}",
        ) from exc

    record = await get_rag_document_for_tenant(document_id, jwt_tenant_id)
    return {
        "data": {
            "document_id": document_id,
            "status": (record or {}).get("status") or "ready",
            "chunk_count": chunk_count,
            "indexed_at": (record or {}).get("indexed_at"),
        },
        "request_id": _request_id(),
    }


@router.post("/{tenant_id}/documents")
async def upload_document(
    tenant_id: str,
    file: UploadFile = File(...),
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    file_name = Path(file.filename or "").name
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    tmp_path = ""
    try:
        suffix = Path(file_name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        document_id = await _process_pdf_document(
            tenant_id=jwt_tenant_id,
            pdf_path=tmp_path,
            file_name=file_name,
        )
        record = await get_rag_document_for_tenant(document_id, jwt_tenant_id)
        return {
            "data": {
                "document_id": document_id,
                "status": (record or {}).get("status") or "ready",
            },
            "request_id": _request_id(),
        }
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass


@router.delete("/{tenant_id}/documents/{document_id}")
async def delete_document(
    tenant_id: str,
    document_id: str,
    current_admin: dict[str, Any] = Depends(get_current_admin_user),
):
    jwt_tenant_id = _validate_path_tenant_id(tenant_id, current_admin)
    record = await get_rag_document_for_tenant(document_id, jwt_tenant_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {document_id!r}",
        )

    try:
        await _delete_document_vectors(document_id, jwt_tenant_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"document vector delete failed: {exc}",
        ) from exc

    deleted = await soft_delete_rag_document_for_tenant(document_id, jwt_tenant_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"document not found: {document_id!r}",
        )

    return {
        "data": {
            "document_id": document_id,
            "deleted": True,
        },
        "request_id": _request_id(),
    }


@router.get("/{tenant_id}")
async def get_tenant(tenant_id: str):
    raise NotImplementedError


@router.post("/")
async def create_tenant():
    raise NotImplementedError
