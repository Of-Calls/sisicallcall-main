# calls 테이블 — 상세 스키마: docs/db_schema.md 2.2
# ORM 사용 금지 — raw SQL + asyncpg 만 사용

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    call_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL REFERENCES tenants(tenant_id),
    twilio_call_sid VARCHAR(64) UNIQUE NOT NULL,
    caller_number   VARCHAR(20),
    status          VARCHAR(20) DEFAULT 'active',
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    branch_stats    JSONB DEFAULT '{}'
);
"""
