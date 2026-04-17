# voc_analyses 테이블 — M2 기능 — 상세 스키마: docs/db_schema.md 2.5
# ORM 사용 금지 — raw SQL + asyncpg 만 사용

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS voc_analyses (
    voc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id         UUID NOT NULL REFERENCES calls(call_id),
    sentiment_label VARCHAR(20),
    sentiment_score FLOAT,
    intent_label    VARCHAR(50),
    priority_level  VARCHAR(20),
    partial_success BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""
