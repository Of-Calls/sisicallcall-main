CREATE TABLE IF NOT EXISTS rag_document_chunks (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    tenant_id         UUID NOT NULL REFERENCES tenants(id),
    chunk_index       INTEGER NOT NULL,
    page_number       INTEGER,
    content           TEXT NOT NULL,
    metadata          JSONB DEFAULT '{}'::jsonb,
    embedding_status  VARCHAR(20) DEFAULT 'ready'
                      CHECK (embedding_status IN ('processing', 'ready', 'failed')),
    chroma_id         VARCHAR(255),
    created_at        TIMESTAMPTZ DEFAULT now(),
    updated_at        TIMESTAMPTZ DEFAULT now(),
    deleted_at        TIMESTAMPTZ DEFAULT NULL,
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_rag_document_chunks_document_id
    ON rag_document_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_rag_document_chunks_tenant_id
    ON rag_document_chunks(tenant_id);

CREATE INDEX IF NOT EXISTS idx_rag_document_chunks_deleted_at
    ON rag_document_chunks(deleted_at)
    WHERE deleted_at IS NULL;
