# face_embeddings 테이블 — M3+ 기능 — 상세 스키마: docs/db_schema.md 참조
# ORM 사용 금지 — raw SQL + asyncpg 만 사용

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(tenant_id),
    user_id         VARCHAR(64) NOT NULL,
    embedding       FLOAT[] NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, user_id)
);
"""
