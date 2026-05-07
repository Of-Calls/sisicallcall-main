"""calls 테이블 INSERT / UPDATE — 통화 시작 + 종료 시점 메타 기록.

통화 흐름 차단 방지 정책:
- 모든 함수 best-effort. asyncpg 예외는 흡수 + WARNING 로그만 남기고 None 반환.
- DB 다운 / connection 고갈 / schema 불일치 시에도 통화 응대 자체는 막지 않는다.

connection pool 미사용 — `app/services/tenant.py` 와 동일하게 per-call asyncpg.connect.
풀 도입 시 본 모듈만 수정 (`_OPEN_ISSUES.md` 의 asyncpg pool 항목).
"""
import json
import re
from datetime import datetime

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    return bool(value and _UUID_RE.match(value))


_CALL_SELECT_COLUMNS = """
id,
tenant_id,
twilio_call_sid,
caller_number,
status,
started_at,
ended_at,
duration_sec,
latency_log,
branch_stats,
created_at
"""

_GET_CALL_BY_ID_FOR_TENANT_SQL = f"""
SELECT
    {_CALL_SELECT_COLUMNS}
FROM calls
WHERE id = $1::uuid
  AND tenant_id = $2::uuid
LIMIT 1
"""


def _parse_jsonb(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def _isoformat(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _call_row_to_dict(row) -> dict:
    return {
        "id": str(row["id"]),
        "tenant_id": str(row["tenant_id"]),
        "twilio_call_sid": row["twilio_call_sid"],
        "caller_number": row["caller_number"],
        "status": row["status"],
        "started_at": _isoformat(row["started_at"]),
        "ended_at": _isoformat(row["ended_at"]),
        "duration_sec": row["duration_sec"],
        "latency_log": _parse_jsonb(row["latency_log"]),
        "branch_stats": _parse_jsonb(row["branch_stats"]),
        "created_at": _isoformat(row["created_at"]),
    }


async def insert_call(
    tenant_id: str,
    twilio_call_sid: str,
    caller_number: str | None = None,
) -> str | None:
    """calls 에 새 row INSERT 후 생성된 UUID 반환.

    실패 시 (DB 다운, 미등록 tenant, schema 위반 등) None 반환 — 호출자는 DB 추적 없이 진행.
    """
    if not _is_uuid(tenant_id):
        logger.warning("insert_call skip — invalid tenant_id=%s", tenant_id)
        return None
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO calls (tenant_id, twilio_call_sid, caller_number, status)
                VALUES ($1::uuid, $2, $3, 'in_progress')
                RETURNING id
                """,
                tenant_id, twilio_call_sid, caller_number,
            )
            if row:
                db_call_id = str(row["id"])
                logger.info(
                    "calls INSERT db_call_id=%s twilio_call_sid=%s tenant_id=%s",
                    db_call_id, twilio_call_sid, tenant_id,
                )
                return db_call_id
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(
            "insert_call failed twilio_call_sid=%s tenant_id=%s err=%s",
            twilio_call_sid, tenant_id, e,
        )
    return None


async def finalize_call(
    db_call_id: str,
    status: str,
    duration_sec: int | None = None,
) -> None:
    """통화 종료 시 status / ended_at / duration_sec 업데이트.

    status: 'completed' | 'abandoned' | 'error'  (스키마 CHECK 제약과 일치)
    """
    if not _is_uuid(db_call_id):
        return
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """
                UPDATE calls
                SET status = $1, ended_at = now(), duration_sec = $2
                WHERE id = $3::uuid
                """,
                status, duration_sec, db_call_id,
            )
            logger.info(
                "calls UPDATE db_call_id=%s status=%s duration_sec=%s",
                db_call_id, status, duration_sec,
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(
            "finalize_call failed db_call_id=%s status=%s err=%s",
            db_call_id, status, e,
        )


async def list_calls_for_tenant(
    tenant_id: str,
    status: str | None = None,
    started_from: datetime | None = None,
    started_to: datetime | None = None,
    offset: int = 0,
    limit: int = 20,
) -> dict:
    """관리자용 tenant-scoped calls 목록 조회."""
    offset = max(int(offset or 0), 0)
    limit = min(max(int(limit or 20), 1), 100)

    if not _is_uuid(tenant_id):
        return {"items": [], "total": 0, "offset": offset, "limit": limit}

    where_parts = ["tenant_id = $1::uuid"]
    args: list = [tenant_id]

    if status:
        args.append(status)
        where_parts.append(f"status = ${len(args)}")
    if started_from is not None:
        args.append(started_from)
        where_parts.append(f"started_at >= ${len(args)}")
    if started_to is not None:
        args.append(started_to)
        where_parts.append(f"started_at <= ${len(args)}")

    where_sql = " AND ".join(where_parts)
    count_sql = f"""
        SELECT COUNT(*) AS total
        FROM calls
        WHERE {where_sql}
    """

    list_args = [*args, offset, limit]
    offset_param = len(args) + 1
    limit_param = len(args) + 2
    list_sql = f"""
        SELECT
            {_CALL_SELECT_COLUMNS}
        FROM calls
        WHERE {where_sql}
        ORDER BY started_at DESC
        OFFSET ${offset_param}
        LIMIT ${limit_param}
    """

    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            total = await conn.fetchval(count_sql, *args)
            rows = await conn.fetch(list_sql, *list_args)
            return {
                "items": [_call_row_to_dict(row) for row in rows],
                "total": int(total or 0),
                "offset": offset,
                "limit": limit,
            }
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("list_calls_for_tenant failed tenant_id=%s err=%s", tenant_id, e)
        return {"items": [], "total": 0, "offset": offset, "limit": limit}


async def get_call_by_id_for_tenant(call_id: str, tenant_id: str) -> dict | None:
    """관리자용 tenant-scoped call 상세 조회."""
    if not _is_uuid(call_id) or not _is_uuid(tenant_id):
        return None

    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                _GET_CALL_BY_ID_FOR_TENANT_SQL,
                call_id,
                tenant_id,
            )
            return _call_row_to_dict(row) if row else None
        finally:
            await conn.close()
    except Exception as e:
        logger.warning(
            "get_call_by_id_for_tenant failed call_id=%s tenant_id=%s err=%s",
            call_id, tenant_id, e,
        )
        return None
