"""
DB transcript adapter.

calls / transcripts 테이블에서 종료된 통화 context를 조회한다.

── DB 접근 방식 ───────────────────────────────────────────────────────────────
asyncpg per-call connect (call_repo.py / app/services/tenant.py 와 동일 방식).
import 시점에 DB 연결을 만들지 않는다.
함수 호출 시점에만 asyncpg.connect()를 호출한다.

── call_id 조회 기준 ──────────────────────────────────────────────────────────
UUID 형식  → calls.id = $1::uuid OR calls.twilio_call_sid = $1
그 외 형식 → calls.twilio_call_sid = $1  (Twilio SID: CAxxxx...)

── calls 테이블 주요 컬럼 (db/init/02_calls.sql 기준) ─────────────────────────
  id, tenant_id, twilio_call_sid, caller_number,
  status, started_at, ended_at, branch_stats (JSONB)

── transcripts 테이블 주요 컬럼 (db/init/03_transcripts.sql 기준) ──────────────
  call_id, turn_index, speaker ('customer'|'agent'), text, spoken_at
  — tenant_id 컬럼 없음 (calls JOIN으로 tenant 격리)

── 반환 형식 ──────────────────────────────────────────────────────────────────
{
  "metadata": {
    "call_id":       "...",
    "tenant_id":     "...",
    "start_time":    "ISO8601 | None",
    "end_time":      "ISO8601 | None",
    "status":        "completed | ...",
    "customer_phone": "... | None"
  },
  "transcripts": [
    {"role": "customer", "text": "...", "timestamp": "ISO8601 | None"},
    ...
  ],
  "branch_stats": {"faq": int, "task": int, "escalation": int, ...}
}

── 예외 정책 ──────────────────────────────────────────────────────────────────
- call_id 해당 row 없음 → None
- tenant_id 제공 & DB tenant_id 불일치 → None
- DB 연결/쿼리 실패 → logger.warning 후 None (예외 전파 금지)
- sample transcript 절대 반환 안 함
"""
from __future__ import annotations

import json
import re

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger
from app.utils.phone import normalize_korean_phone

logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    return bool(value and _UUID_RE.match(value))


def _parse_jsonb(value) -> dict:
    """asyncpg JSONB 반환값 → dict. 문자열 / dict / None 모두 처리."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


async def get_completed_call_context_from_db(
    call_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    """DB에서 종료된 통화 context를 조회한다.

    Args:
        call_id:   calls.id (UUID) 또는 calls.twilio_call_sid.
        tenant_id: 제공 시 calls.tenant_id 와 일치 여부를 검증한다.

    Returns:
        context dict 또는 None (row 없음 / tenant mismatch / DB 오류).
    """
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            # UUID 여부에 따라 쿼리 분기 — 잘못된 UUID 문자열로 인한 type error 방지
            if _is_uuid(call_id):
                call_row = await conn.fetchrow(
                    """
                    SELECT id, tenant_id, caller_number, status,
                           started_at, ended_at, branch_stats
                    FROM calls
                    WHERE id = $1::uuid OR twilio_call_sid = $2
                    LIMIT 1
                    """,
                    call_id, call_id,
                )
            else:
                call_row = await conn.fetchrow(
                    """
                    SELECT id, tenant_id, caller_number, status,
                           started_at, ended_at, branch_stats
                    FROM calls
                    WHERE twilio_call_sid = $1
                    LIMIT 1
                    """,
                    call_id,
                )

            if call_row is None:
                logger.debug("DB: call_id=%s 해당 row 없음", call_id)
                return None

            # tenant_id 검증 — 제공된 경우에만
            if tenant_id:
                db_tenant_id = str(call_row["tenant_id"])
                if db_tenant_id.lower() != tenant_id.lower():
                    logger.warning(
                        "DB: tenant_id 불일치 call_id=%s expected=%s actual=%s",
                        call_id, tenant_id, db_tenant_id,
                    )
                    return None

            db_call_uuid = str(call_row["id"])

            # transcripts 조회 — row 없으면 빈 리스트 반환 (예외 없음)
            if tenant_id:
                transcript_rows = await conn.fetch(
                    """
                    SELECT t.speaker, t.text, t.spoken_at
                    FROM transcripts t
                    JOIN calls c ON c.id = t.call_id
                    WHERE t.call_id = $1::uuid
                      AND c.tenant_id = $2::uuid
                    ORDER BY t.turn_index ASC, t.spoken_at ASC
                    """,
                    db_call_uuid, tenant_id,
                )
            else:
                # Internal legacy fallback only. Admin/API paths must pass tenant_id.
                transcript_rows = await conn.fetch(
                    """
                    SELECT t.speaker, t.text, t.spoken_at
                    FROM transcripts t
                    JOIN calls c ON c.id = t.call_id
                    WHERE t.call_id = $1::uuid
                    ORDER BY t.turn_index ASC, t.spoken_at ASC
                    """,
                    db_call_uuid,
                )

            transcripts = [
                {
                    "role": row["speaker"],
                    "text": row["text"],
                    "timestamp": (
                        row["spoken_at"].isoformat() if row["spoken_at"] else None
                    ),
                }
                for row in transcript_rows
            ]

            started_at = call_row["started_at"]
            ended_at = call_row["ended_at"]

            metadata: dict = {
                "call_id": call_id,
                "tenant_id": str(call_row["tenant_id"]),
                "start_time": started_at.isoformat() if started_at else None,
                "end_time": ended_at.isoformat() if ended_at else None,
                "status": call_row["status"] or "completed",
            }
            # caller_number → customer_phone (NULL/empty 면 키 자체를 비워 둠)
            normalized_phone = normalize_korean_phone(call_row["caller_number"])
            if normalized_phone:
                metadata["customer_phone"] = normalized_phone

            return {
                "metadata": metadata,
                "transcripts": transcripts,
                "branch_stats": _parse_jsonb(call_row["branch_stats"]),
            }

        finally:
            await conn.close()

    except Exception as exc:
        logger.warning("DB 조회 실패 call_id=%s err=%s — None 반환", call_id, exc)
        return None
