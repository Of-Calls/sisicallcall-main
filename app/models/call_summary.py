# call_summaries 테이블 — 상세 스키마: docs/db_schema.md 2.4
# ORM 사용 금지 — raw SQL + asyncpg 만 사용

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS call_summaries (
    summary_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id         UUID NOT NULL REFERENCES calls(call_id),
    summary_short   TEXT,
    summary_long    TEXT,
    sync_done       BOOLEAN DEFAULT FALSE,
    async_done      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
"""
