from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _is_uuid(value: str | None) -> bool:
    try:
        UUID(str(value))
        return True
    except Exception:
        return False


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _json_payload(value: Any) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _row_to_document(row: Any) -> dict:
    return {
        "id": str(row["id"]),
        "tenant_id": str(row["tenant_id"]),
        "file_name": row["file_name"],
        "file_type": row["file_type"],
        "chunk_count": row["chunk_count"],
        "status": row["status"],
        "chroma_collection": row["chroma_collection"],
        "uploaded_at": _iso(row["uploaded_at"]),
        "indexed_at": _iso(row["indexed_at"]),
    }


def _row_to_chunk(row: Any) -> dict:
    return {
        "id": str(row["id"]),
        "document_id": str(row["document_id"]),
        "tenant_id": str(row["tenant_id"]),
        "chunk_index": row["chunk_index"],
        "page": row["page_number"],
        "content": row["content"],
        "metadata": _json_payload(row["metadata"]),
        "embedding_status": row["embedding_status"],
        "chroma_id": row["chroma_id"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


async def list_rag_documents_for_tenant(
    tenant_id: str,
    *,
    status: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    if not _is_uuid(tenant_id):
        return {"items": [], "total": 0, "offset": max(0, offset), "limit": max(1, min(limit, 100))}

    normalized_offset = max(0, int(offset))
    normalized_limit = max(1, min(int(limit), 100))
    normalized_status = status.strip() if status else None

    where = ["tenant_id = $1::uuid", "deleted_at IS NULL"]
    params: list[Any] = [tenant_id]
    if normalized_status:
        params.append(normalized_status)
        where.append(f"status = ${len(params)}")

    where_sql = " AND ".join(where)
    count_sql = f"""
        SELECT COUNT(*)::int AS total
        FROM rag_documents
        WHERE {where_sql}
    """

    list_params = list(params)
    list_params.append(normalized_offset)
    offset_pos = len(list_params)
    list_params.append(normalized_limit)
    limit_pos = len(list_params)
    list_sql = f"""
        SELECT
            id,
            tenant_id,
            file_name,
            file_type,
            chunk_count,
            status,
            chroma_collection,
            uploaded_at,
            indexed_at
        FROM rag_documents
        WHERE {where_sql}
        ORDER BY uploaded_at DESC
        OFFSET ${offset_pos}
        LIMIT ${limit_pos}
    """

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        total_row = await conn.fetchrow(count_sql, *params)
        rows = await conn.fetch(list_sql, *list_params)
        return {
            "items": [_row_to_document(row) for row in rows],
            "total": int((total_row or {})["total"] or 0) if total_row else 0,
            "offset": normalized_offset,
            "limit": normalized_limit,
        }
    except Exception as exc:
        logger.warning("rag document list failed tenant_id=%s err=%s", tenant_id, exc)
        return {"items": [], "total": 0, "offset": normalized_offset, "limit": normalized_limit}
    finally:
        if conn is not None:
            await conn.close()


async def list_rag_document_chunks_for_tenant(
    document_id: str,
    tenant_id: str,
    *,
    offset: int = 0,
    limit: int = 500,
) -> dict:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return {"items": [], "total": 0, "offset": max(0, offset), "limit": max(1, min(limit, 1000))}

    normalized_offset = max(0, int(offset))
    normalized_limit = max(1, min(int(limit), 1000))
    base_where = """
        c.document_id = $1::uuid
        AND c.tenant_id = $2::uuid
        AND c.deleted_at IS NULL
        AND d.deleted_at IS NULL
    """
    count_sql = f"""
        SELECT COUNT(*)::int AS total
        FROM rag_document_chunks c
        JOIN rag_documents d
          ON d.id = c.document_id
         AND d.tenant_id = c.tenant_id
        WHERE {base_where}
    """
    list_sql = f"""
        SELECT
            c.id,
            c.document_id,
            c.tenant_id,
            c.chunk_index,
            c.page_number,
            c.content,
            c.metadata,
            c.embedding_status,
            c.chroma_id,
            c.created_at,
            c.updated_at
        FROM rag_document_chunks c
        JOIN rag_documents d
          ON d.id = c.document_id
         AND d.tenant_id = c.tenant_id
        WHERE {base_where}
        ORDER BY c.chunk_index ASC
        OFFSET $3
        LIMIT $4
    """

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        total_row = await conn.fetchrow(count_sql, document_id, tenant_id)
        rows = await conn.fetch(list_sql, document_id, tenant_id, normalized_offset, normalized_limit)
        return {
            "items": [_row_to_chunk(row) for row in rows],
            "total": int((total_row or {})["total"] or 0) if total_row else 0,
            "offset": normalized_offset,
            "limit": normalized_limit,
        }
    except Exception as exc:
        logger.warning(
            "rag document chunk list failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        return {"items": [], "total": 0, "offset": normalized_offset, "limit": normalized_limit}
    finally:
        if conn is not None:
            await conn.close()


async def upsert_rag_document_chunks_for_tenant(
    document_id: str,
    tenant_id: str,
    chunks: list[dict[str, Any]],
) -> None:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id) or not chunks:
        return

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        for chunk in chunks:
            await conn.execute(
                """
                INSERT INTO rag_document_chunks (
                    document_id,
                    tenant_id,
                    chunk_index,
                    page_number,
                    content,
                    metadata,
                    embedding_status,
                    chroma_id,
                    deleted_at
                )
                VALUES (
                    $1::uuid,
                    $2::uuid,
                    $3,
                    $4,
                    $5,
                    $6::jsonb,
                    $7,
                    $8,
                    NULL
                )
                ON CONFLICT (document_id, chunk_index)
                DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    page_number = EXCLUDED.page_number,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding_status = EXCLUDED.embedding_status,
                    chroma_id = EXCLUDED.chroma_id,
                    deleted_at = NULL,
                    updated_at = now()
                """,
                document_id,
                tenant_id,
                int(chunk["chunk_index"]),
                chunk.get("page_number"),
                chunk["content"],
                _json_dumps(chunk.get("metadata")),
                chunk.get("embedding_status") or "ready",
                chunk.get("chroma_id"),
            )
    except Exception as exc:
        logger.warning(
            "rag document chunk upsert failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        raise
    finally:
        if conn is not None:
            await conn.close()


async def update_rag_document_chunk_for_tenant(
    *,
    chunk_id: str,
    document_id: str,
    tenant_id: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict | None:
    if not _is_uuid(chunk_id) or not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return None

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        row = await conn.fetchrow(
            """
            UPDATE rag_document_chunks c
            SET content = $4,
                metadata = CASE
                    WHEN $5::jsonb IS NULL THEN c.metadata
                    ELSE COALESCE(c.metadata, '{}'::jsonb) || $5::jsonb
                END,
                embedding_status = 'processing',
                updated_at = now()
            FROM rag_documents d
            WHERE c.id = $1::uuid
              AND c.document_id = $2::uuid
              AND c.tenant_id = $3::uuid
              AND c.deleted_at IS NULL
              AND d.id = c.document_id
              AND d.tenant_id = c.tenant_id
              AND d.deleted_at IS NULL
            RETURNING
                c.id,
                c.document_id,
                c.tenant_id,
                c.chunk_index,
                c.page_number,
                c.content,
                c.metadata,
                c.embedding_status,
                c.chroma_id,
                c.created_at,
                c.updated_at
            """,
            chunk_id,
            document_id,
            tenant_id,
            content,
            _json_dumps(metadata) if metadata is not None else None,
        )
        if row is None:
            return None

        await mark_rag_document_processing_for_tenant(
            document_id=document_id,
            tenant_id=tenant_id,
            conn=conn,
        )
        return _row_to_chunk(row)
    except Exception as exc:
        logger.warning(
            "rag document chunk update failed chunk_id=%s document_id=%s tenant_id=%s err=%s",
            chunk_id,
            document_id,
            tenant_id,
            exc,
        )
        return None
    finally:
        if conn is not None:
            await conn.close()


async def mark_rag_document_processing_for_tenant(
    *,
    document_id: str,
    tenant_id: str,
    conn: Any | None = None,
) -> bool:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return False

    own_conn = conn is None
    if own_conn:
        conn = await asyncpg.connect(_database_url())
    try:
        result = await conn.execute(
            """
            UPDATE rag_documents
            SET status = 'processing',
                indexed_at = NULL
            WHERE id = $1::uuid
              AND tenant_id = $2::uuid
              AND deleted_at IS NULL
            """,
            document_id,
            tenant_id,
        )
        return result.upper().startswith("UPDATE 1")
    except Exception as exc:
        logger.warning(
            "rag document processing mark failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        return False
    finally:
        if own_conn and conn is not None:
            await conn.close()


async def mark_rag_document_reindex_succeeded_for_tenant(
    document_id: str,
    tenant_id: str,
) -> bool:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return False

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        await conn.execute(
            """
            UPDATE rag_document_chunks
            SET embedding_status = 'ready',
                updated_at = now()
            WHERE document_id = $1::uuid
              AND tenant_id = $2::uuid
              AND deleted_at IS NULL
            """,
            document_id,
            tenant_id,
        )
        result = await conn.execute(
            """
            UPDATE rag_documents
            SET status = 'ready',
                indexed_at = now(),
                chunk_count = (
                    SELECT COUNT(*)::int
                    FROM rag_document_chunks
                    WHERE document_id = $1::uuid
                      AND tenant_id = $2::uuid
                      AND deleted_at IS NULL
                )
            WHERE id = $1::uuid
              AND tenant_id = $2::uuid
              AND deleted_at IS NULL
            """,
            document_id,
            tenant_id,
        )
        return result.upper().startswith("UPDATE 1")
    except Exception as exc:
        logger.warning(
            "rag document reindex success mark failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        return False
    finally:
        if conn is not None:
            await conn.close()


async def mark_rag_document_reindex_failed_for_tenant(
    document_id: str,
    tenant_id: str,
) -> bool:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return False

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        result = await conn.execute(
            """
            UPDATE rag_documents
            SET status = 'failed'
            WHERE id = $1::uuid
              AND tenant_id = $2::uuid
              AND deleted_at IS NULL
            """,
            document_id,
            tenant_id,
        )
        return result.upper().startswith("UPDATE 1")
    except Exception as exc:
        logger.warning(
            "rag document reindex failure mark failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        return False
    finally:
        if conn is not None:
            await conn.close()


async def get_rag_document_for_tenant(
    document_id: str,
    tenant_id: str,
) -> dict | None:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return None

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        row = await conn.fetchrow(
            """
            SELECT
                id,
                tenant_id,
                file_name,
                file_type,
                chunk_count,
                status,
                chroma_collection,
                uploaded_at,
                indexed_at
            FROM rag_documents
            WHERE id = $1::uuid
              AND tenant_id = $2::uuid
              AND deleted_at IS NULL
            LIMIT 1
            """,
            document_id,
            tenant_id,
        )
        return _row_to_document(row) if row else None
    except Exception as exc:
        logger.warning(
            "rag document lookup failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        return None
    finally:
        if conn is not None:
            await conn.close()


async def soft_delete_rag_document_for_tenant(
    document_id: str,
    tenant_id: str,
) -> bool:
    if not _is_uuid(document_id) or not _is_uuid(tenant_id):
        return False

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        result = await conn.execute(
            """
            UPDATE rag_documents
            SET deleted_at = now()
            WHERE id = $1::uuid
              AND tenant_id = $2::uuid
              AND deleted_at IS NULL
            """,
            document_id,
            tenant_id,
        )
        return result.upper().startswith("UPDATE 1")
    except Exception as exc:
        logger.warning(
            "rag document soft delete failed document_id=%s tenant_id=%s err=%s",
            document_id,
            tenant_id,
            exc,
        )
        return False
    finally:
        if conn is not None:
            await conn.close()
