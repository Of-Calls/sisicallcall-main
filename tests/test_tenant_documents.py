from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import admin_auth, tenant
from app.core.security import create_access_token
from app.repositories import rag_document_repo


USER_ID = "11111111-1111-1111-1111-111111111111"
TENANT_ID = "22222222-2222-2222-2222-222222222222"
OTHER_TENANT_ID = "33333333-3333-3333-3333-333333333333"
DOCUMENT_ID = "44444444-4444-4444-4444-444444444444"
CHUNK_ID = "55555555-5555-5555-5555-555555555555"
EMAIL = "admin@example.test"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(tenant.router, prefix="/tenant")
    return TestClient(app)


def _token(tenant_id: str = TENANT_ID) -> str:
    return create_access_token(
        user_id=USER_ID,
        tenant_id=tenant_id,
        role="owner",
        email=EMAIL,
    )


def _auth_headers(token: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token or _token()}"}


def _admin_context(tenant_id: str = TENANT_ID) -> dict:
    return {
        "user": {
            "id": USER_ID,
            "tenant_id": tenant_id,
            "email": EMAIL,
            "name": "Test Admin",
            "role": "owner",
            "is_active": True,
            "password_hash": "unused",
            "last_login_at": None,
        },
        "tenant": {
            "id": tenant_id,
            "name": "Test Tenant",
            "industry": "test",
            "plan": "basic",
            "twilio_number": "+821000000000",
            "is_active": True,
        },
    }


def _patch_admin_lookup(monkeypatch, tenant_id: str = TENANT_ID) -> None:
    async def fake_find_admin_user_by_id(user_id: str):
        assert user_id == USER_ID
        return _admin_context(tenant_id)

    monkeypatch.setattr(admin_auth, "find_admin_user_by_id", fake_find_admin_user_by_id)


def _document_payload(
    document_id: str = DOCUMENT_ID,
    tenant_id: str = TENANT_ID,
    status: str = "ready",
) -> dict:
    return {
        "id": document_id,
        "tenant_id": tenant_id,
        "file_name": "faq.pdf",
        "file_type": "pdf",
        "chunk_count": 42,
        "status": status,
        "chroma_collection": f"tenant_{tenant_id.replace('-', '')}_docs",
        "uploaded_at": "2026-05-06T01:00:00+00:00",
        "indexed_at": "2026-05-06T01:01:00+00:00",
    }


def _chunk_payload() -> dict:
    return {
        "id": CHUNK_ID,
        "document_id": DOCUMENT_ID,
        "tenant_id": TENANT_ID,
        "chunk_index": 0,
        "page": 1,
        "content": "original chunk content",
        "metadata": {"page_number": 1, "chunk_type": "section"},
        "embedding_status": "ready",
        "chroma_id": f"{DOCUMENT_ID}_chunk_0",
        "created_at": "2026-05-06T01:00:00+00:00",
        "updated_at": "2026-05-06T01:01:00+00:00",
    }


def test_list_documents_without_token_returns_401():
    resp = _client().get(f"/tenant/{TENANT_ID}/documents")

    assert resp.status_code == 401


def test_list_documents_tenant_mismatch_returns_403(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    resp = _client().get(f"/tenant/{OTHER_TENANT_ID}/documents", headers=_auth_headers())

    assert resp.status_code == 403


def test_list_documents_uses_jwt_tenant_and_filters(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_list_rag_documents_for_tenant(**kwargs):
        captured.append(kwargs)
        return {
            "items": [_document_payload()],
            "total": 1,
            "offset": kwargs["offset"],
            "limit": kwargs["limit"],
        }

    monkeypatch.setattr(
        tenant,
        "list_rag_documents_for_tenant",
        fake_list_rag_documents_for_tenant,
    )

    resp = _client().get(
        f"/tenant/{TENANT_ID}/documents?status=ready&offset=2&limit=5",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert captured == [{"tenant_id": TENANT_ID, "status": "ready", "offset": 2, "limit": 5}]
    assert resp.json()["data"]["items"][0]["id"] == DOCUMENT_ID


def test_get_document_returns_404_when_missing(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        assert document_id == DOCUMENT_ID
        assert tenant_id == TENANT_ID
        return None

    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)

    resp = _client().get(f"/tenant/{TENANT_ID}/documents/{DOCUMENT_ID}", headers=_auth_headers())

    assert resp.status_code == 404


def test_get_document_returns_tenant_scoped_document(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        assert document_id == DOCUMENT_ID
        assert tenant_id == TENANT_ID
        return _document_payload()

    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)

    resp = _client().get(f"/tenant/{TENANT_ID}/documents/{DOCUMENT_ID}", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.json()["data"]["tenant_id"] == TENANT_ID


def test_list_document_chunks_returns_chunk_content(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        assert document_id == DOCUMENT_ID
        assert tenant_id == TENANT_ID
        return _document_payload()

    async def fake_list_rag_document_chunks_for_tenant(
        document_id: str,
        tenant_id: str,
        *,
        offset: int = 0,
        limit: int = 500,
    ):
        captured.append({
            "document_id": document_id,
            "tenant_id": tenant_id,
            "offset": offset,
            "limit": limit,
        })
        return {"items": [_chunk_payload()], "total": 1, "offset": offset, "limit": limit}

    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)
    monkeypatch.setattr(
        tenant,
        "list_rag_document_chunks_for_tenant",
        fake_list_rag_document_chunks_for_tenant,
    )

    resp = _client().get(
        f"/tenant/{TENANT_ID}/documents/{DOCUMENT_ID}/chunks?offset=0&limit=10",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert captured == [{"document_id": DOCUMENT_ID, "tenant_id": TENANT_ID, "offset": 0, "limit": 10}]
    assert resp.json()["data"]["items"][0]["content"] == "original chunk content"


def test_update_document_chunk_marks_chunk_processing(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        return _document_payload(document_id=document_id, tenant_id=tenant_id)

    async def fake_update_rag_document_chunk_for_tenant(**kwargs):
        captured.append(kwargs)
        return {
            **_chunk_payload(),
            "content": kwargs["content"],
            "metadata": {"page_number": 1, "reviewed": True},
            "embedding_status": "processing",
        }

    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)
    monkeypatch.setattr(
        tenant,
        "update_rag_document_chunk_for_tenant",
        fake_update_rag_document_chunk_for_tenant,
    )

    resp = _client().patch(
        f"/tenant/{TENANT_ID}/documents/{DOCUMENT_ID}/chunks/{CHUNK_ID}",
        headers=_auth_headers(),
        json={"content": "edited chunk content", "metadata": {"reviewed": True}},
    )

    assert resp.status_code == 200
    assert captured[0]["chunk_id"] == CHUNK_ID
    assert captured[0]["tenant_id"] == TENANT_ID
    assert resp.json()["data"]["content"] == "edited chunk content"
    assert resp.json()["data"]["embedding_status"] == "processing"


def test_reindex_document_returns_ready_status(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    calls: list[tuple[str, str]] = []

    async def fake_reindex_document_chunks(document_id: str, tenant_id: str):
        calls.append((document_id, tenant_id))
        return 1

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        return {
            **_document_payload(document_id=document_id, tenant_id=tenant_id),
            "chunk_count": 1,
            "indexed_at": "2026-05-06T02:00:00+00:00",
        }

    monkeypatch.setattr(tenant, "_reindex_document_chunks", fake_reindex_document_chunks)
    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)

    resp = _client().post(
        f"/tenant/{TENANT_ID}/documents/{DOCUMENT_ID}/reindex",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert calls == [(DOCUMENT_ID, TENANT_ID)]
    assert resp.json()["data"]["status"] == "ready"
    assert resp.json()["data"]["chunk_count"] == 1


def test_upload_document_rejects_non_pdf(monkeypatch):
    _patch_admin_lookup(monkeypatch)

    resp = _client().post(
        f"/tenant/{TENANT_ID}/documents",
        headers=_auth_headers(),
        files={"file": ("faq.txt", b"hello", "text/plain")},
    )

    assert resp.status_code == 400


def test_upload_document_processes_pdf_and_returns_status(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    captured: list[dict] = []

    async def fake_process_pdf_document(**kwargs):
        captured.append(kwargs)
        assert Path(kwargs["pdf_path"]).exists()
        return DOCUMENT_ID

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        assert document_id == DOCUMENT_ID
        assert tenant_id == TENANT_ID
        return _document_payload(status="ready")

    monkeypatch.setattr(tenant, "_process_pdf_document", fake_process_pdf_document)
    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)

    resp = _client().post(
        f"/tenant/{TENANT_ID}/documents",
        headers=_auth_headers(),
        files={"file": ("faq.pdf", b"%PDF-1.4\n", "application/pdf")},
    )

    assert resp.status_code == 200
    assert captured[0]["tenant_id"] == TENANT_ID
    assert captured[0]["file_name"] == "faq.pdf"
    assert resp.json()["data"] == {"document_id": DOCUMENT_ID, "status": "ready"}


def test_delete_document_deletes_vectors_then_soft_deletes(monkeypatch):
    _patch_admin_lookup(monkeypatch)
    calls: list[tuple] = []

    async def fake_get_rag_document_for_tenant(document_id: str, tenant_id: str):
        calls.append(("get", document_id, tenant_id))
        return _document_payload()

    async def fake_delete_document_vectors(document_id: str, tenant_id: str):
        calls.append(("chroma", document_id, tenant_id))

    async def fake_soft_delete_rag_document_for_tenant(document_id: str, tenant_id: str):
        calls.append(("soft_delete", document_id, tenant_id))
        return True

    monkeypatch.setattr(tenant, "get_rag_document_for_tenant", fake_get_rag_document_for_tenant)
    monkeypatch.setattr(tenant, "_delete_document_vectors", fake_delete_document_vectors)
    monkeypatch.setattr(
        tenant,
        "soft_delete_rag_document_for_tenant",
        fake_soft_delete_rag_document_for_tenant,
    )

    resp = _client().delete(
        f"/tenant/{TENANT_ID}/documents/{DOCUMENT_ID}",
        headers=_auth_headers(),
    )

    assert resp.status_code == 200
    assert calls == [
        ("get", DOCUMENT_ID, TENANT_ID),
        ("chroma", DOCUMENT_ID, TENANT_ID),
        ("soft_delete", DOCUMENT_ID, TENANT_ID),
    ]
    assert resp.json()["data"] == {"document_id": DOCUMENT_ID, "deleted": True}


class FakeRow(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class FakeConn:
    def __init__(self):
        self.fetchrow_calls: list[tuple[str, tuple]] = []
        self.fetch_calls: list[tuple[str, tuple]] = []
        self.execute_calls: list[tuple[str, tuple]] = []
        self.closed = False

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "COUNT" in query:
            return FakeRow({"total": 1})
        return _rag_row()

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return [_rag_row()]

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "UPDATE 1"

    async def close(self):
        self.closed = True


def _rag_row() -> FakeRow:
    now = datetime(2026, 5, 6, 1, 0, tzinfo=timezone.utc)
    return FakeRow(
        {
            "id": DOCUMENT_ID,
            "tenant_id": TENANT_ID,
            "file_name": "faq.pdf",
            "file_type": "pdf",
            "chunk_count": 42,
            "status": "ready",
            "chroma_collection": "tenant_docs",
            "uploaded_at": now,
            "indexed_at": now,
        }
    )


def _chunk_row() -> FakeRow:
    now = datetime(2026, 5, 6, 1, 0, tzinfo=timezone.utc)
    return FakeRow(
        {
            "id": CHUNK_ID,
            "document_id": DOCUMENT_ID,
            "tenant_id": TENANT_ID,
            "chunk_index": 0,
            "page_number": 1,
            "content": "original chunk content",
            "metadata": {"page_number": 1},
            "embedding_status": "ready",
            "chroma_id": f"{DOCUMENT_ID}_chunk_0",
            "created_at": now,
            "updated_at": now,
        }
    )


class ChunkFakeConn(FakeConn):
    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "COUNT" in query:
            return FakeRow({"total": 1})
        return _chunk_row()

    async def fetch(self, query, *args):
        self.fetch_calls.append((query, args))
        return [_chunk_row()]


@pytest.mark.asyncio
async def test_list_rag_documents_sql_uses_tenant_deleted_and_status(monkeypatch):
    conn = FakeConn()

    async def fake_connect(url):
        return conn

    monkeypatch.setattr(rag_document_repo.asyncpg, "connect", fake_connect)

    result = await rag_document_repo.list_rag_documents_for_tenant(
        TENANT_ID,
        status="ready",
        offset=3,
        limit=7,
    )

    count_sql, count_args = conn.fetchrow_calls[0]
    list_sql, list_args = conn.fetch_calls[0]
    assert "tenant_id = $1::uuid" in count_sql
    assert "deleted_at IS NULL" in count_sql
    assert "status = $2" in count_sql
    assert "ORDER BY uploaded_at DESC" in list_sql
    assert count_args == (TENANT_ID, "ready")
    assert list_args == (TENANT_ID, "ready", 3, 7)
    assert result["items"][0]["id"] == DOCUMENT_ID


@pytest.mark.asyncio
async def test_get_rag_document_sql_uses_tenant_and_deleted(monkeypatch):
    conn = FakeConn()

    async def fake_connect(url):
        return conn

    monkeypatch.setattr(rag_document_repo.asyncpg, "connect", fake_connect)

    result = await rag_document_repo.get_rag_document_for_tenant(DOCUMENT_ID, TENANT_ID)

    sql, args = conn.fetchrow_calls[0]
    assert "id = $1::uuid" in sql
    assert "tenant_id = $2::uuid" in sql
    assert "deleted_at IS NULL" in sql
    assert args == (DOCUMENT_ID, TENANT_ID)
    assert result is not None
    assert result["id"] == DOCUMENT_ID


@pytest.mark.asyncio
async def test_soft_delete_rag_document_sql_sets_deleted_at(monkeypatch):
    conn = FakeConn()

    async def fake_connect(url):
        return conn

    monkeypatch.setattr(rag_document_repo.asyncpg, "connect", fake_connect)

    deleted = await rag_document_repo.soft_delete_rag_document_for_tenant(
        DOCUMENT_ID,
        TENANT_ID,
    )

    sql, args = conn.execute_calls[0]
    assert "SET deleted_at = now()" in sql
    assert "id = $1::uuid" in sql
    assert "tenant_id = $2::uuid" in sql
    assert "deleted_at IS NULL" in sql
    assert args == (DOCUMENT_ID, TENANT_ID)
    assert deleted is True


@pytest.mark.asyncio
async def test_list_rag_document_chunks_sql_uses_document_tenant_and_deleted(monkeypatch):
    conn = ChunkFakeConn()

    async def fake_connect(url):
        return conn

    monkeypatch.setattr(rag_document_repo.asyncpg, "connect", fake_connect)

    result = await rag_document_repo.list_rag_document_chunks_for_tenant(
        DOCUMENT_ID,
        TENANT_ID,
        offset=4,
        limit=8,
    )

    count_sql, count_args = conn.fetchrow_calls[0]
    list_sql, list_args = conn.fetch_calls[0]
    assert "JOIN rag_documents d" in count_sql
    assert "c.document_id = $1::uuid" in list_sql
    assert "c.tenant_id = $2::uuid" in list_sql
    assert "c.deleted_at IS NULL" in list_sql
    assert "d.deleted_at IS NULL" in list_sql
    assert "ORDER BY c.chunk_index ASC" in list_sql
    assert count_args == (DOCUMENT_ID, TENANT_ID)
    assert list_args == (DOCUMENT_ID, TENANT_ID, 4, 8)
    assert result["items"][0]["id"] == CHUNK_ID
    assert result["items"][0]["content"] == "original chunk content"


@pytest.mark.asyncio
async def test_update_rag_document_chunk_sql_uses_tenant_and_marks_processing(monkeypatch):
    conn = ChunkFakeConn()

    async def fake_connect(url):
        return conn

    monkeypatch.setattr(rag_document_repo.asyncpg, "connect", fake_connect)

    result = await rag_document_repo.update_rag_document_chunk_for_tenant(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        tenant_id=TENANT_ID,
        content="edited chunk",
        metadata={"reviewed": True},
    )

    update_sql, update_args = conn.fetchrow_calls[0]
    doc_sql, doc_args = conn.execute_calls[0]
    assert "UPDATE rag_document_chunks c" in update_sql
    assert "c.id = $1::uuid" in update_sql
    assert "c.document_id = $2::uuid" in update_sql
    assert "c.tenant_id = $3::uuid" in update_sql
    assert "embedding_status = 'processing'" in update_sql
    assert update_args[:4] == (CHUNK_ID, DOCUMENT_ID, TENANT_ID, "edited chunk")
    assert "UPDATE rag_documents" in doc_sql
    assert "status = 'processing'" in doc_sql
    assert doc_args == (DOCUMENT_ID, TENANT_ID)
    assert result is not None
    assert result["id"] == CHUNK_ID
