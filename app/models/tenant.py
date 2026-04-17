# tenants 테이블 — 상세 스키마: docs/db_schema.md 2.1
# ORM 사용 금지 — raw SQL + asyncpg 만 사용

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(255) NOT NULL,
    phone       VARCHAR(20) UNIQUE NOT NULL,
    settings    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""
