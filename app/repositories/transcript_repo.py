"""transcripts 테이블 INSERT — 발화 단위 기록.

CLAUDE.md 규약: transcripts 에는 tenant_id 컬럼 없음 — calls JOIN 으로 격리.

스키마 제약 주의 (db/init/03_transcripts.sql):
- speaker CHECK ('customer','agent')
- response_path CHECK ('cache','faq','task','auth','escalation') — 'clarify','repeat'
  같은 신규 path 는 미허용. 본 모듈은 허용 외 값을 받으면 NULL 로 INSERT 한다.
  (스키마 갱신은 _OPEN_ISSUES.md 의 별건)
- reviewer_verdict CHECK ('pass','revise')

best-effort: asyncpg 예외 흡수 + WARNING. 통화 흐름 차단 안 함.

asyncpg 모듈 레벨 connection pool 사용 — 매 INSERT 마다 connect/close 안 함.
첫 INSERT 시 lazy 초기화. 종료 시 close_pool() 명시 호출 권장(미호출도 OS 정리).
"""
import asyncio
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

# transcripts.response_path CHECK 제약과 동일. 외 값은 NULL 로 INSERT.
_ALLOWED_RESPONSE_PATHS = {"cache", "faq", "task", "auth", "escalation"}

# transcripts.reviewer_verdict CHECK 제약과 동일.
_ALLOWED_REVIEWER_VERDICTS = {"pass", "revise"}

# Pool 설정 — 통화 응대 best-effort. 동시 통화 + 턴 빈도 고려해 작게 유지.
_POOL_MIN_SIZE = 1
_POOL_MAX_SIZE = 5
_POOL_COMMAND_TIMEOUT = 5.0  # seconds — 통화 응대 5초 절대 제약 안 넘김

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def _get_pool() -> asyncpg.Pool:
    """모듈 레벨 풀 lazy 초기화. 동시 첫 호출은 Lock 으로 직렬화."""
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=_POOL_MIN_SIZE,
                max_size=_POOL_MAX_SIZE,
                command_timeout=_POOL_COMMAND_TIMEOUT,
            )
    return _pool


async def close_pool() -> None:
    """프로세스 종료 / 테스트 격리용. 미호출 시에도 OS 가 정리."""
    global _pool
    if _pool is None:
        return
    try:
        await _pool.close()
    finally:
        _pool = None


def _is_uuid(value: str) -> bool:
    return bool(value and _UUID_RE.match(value))


def _normalize_response_path(path: str | None) -> str | None:
    if not path:
        return None
    return path if path in _ALLOWED_RESPONSE_PATHS else None


def _normalize_reviewer_verdict(verdict: str | None) -> str | None:
    if not verdict:
        return None
    return verdict if verdict in _ALLOWED_REVIEWER_VERDICTS else None


async def insert_transcript(
    db_call_id: str,
    turn_index: int,
    speaker: str,
    text: str,
    response_path: str | None = None,
    reviewer_applied: bool = False,
    reviewer_verdict: str | None = None,
    is_barge_in: bool = False,
) -> None:
    """transcripts 에 발화 1건 INSERT.

    speaker: 'customer' | 'agent'
    response_path: agent 발화에만 의미 있음. customer 발화는 None.
    """
    if not text:
        return
    if not _is_uuid(db_call_id):
        logger.warning(
            "insert_transcript skip — invalid db_call_id=%s turn=%s speaker=%s",
            db_call_id, turn_index, speaker,
        )
        return
    if speaker not in ("customer", "agent"):
        logger.warning("insert_transcript skip — invalid speaker=%s", speaker)
        return

    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO transcripts (
                    call_id, turn_index, speaker, text,
                    response_path, reviewer_applied, reviewer_verdict, is_barge_in
                )
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
                """,
                db_call_id,
                turn_index,
                speaker,
                text,
                _normalize_response_path(response_path),
                bool(reviewer_applied),
                _normalize_reviewer_verdict(reviewer_verdict),
                bool(is_barge_in),
            )
    except Exception as e:
        logger.warning(
            "insert_transcript failed db_call_id=%s turn=%s speaker=%s err=%s",
            db_call_id, turn_index, speaker, e,
        )


_TRANSCRIPTS_BY_CALL_ID_SQL = """
SELECT
    t.id,
    t.call_id,
    t.turn_index,
    t.speaker,
    t.text,
    t.response_path,
    t.reviewer_applied,
    t.reviewer_verdict,
    t.is_barge_in,
    t.spoken_at
FROM transcripts t
JOIN calls c ON c.id = t.call_id
WHERE t.call_id = $1::uuid
  AND c.tenant_id = $2::uuid
ORDER BY t.turn_index ASC, t.spoken_at ASC
"""

_CALL_BELONGS_TO_TENANT_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM calls c
    WHERE c.id = $1::uuid
      AND c.tenant_id = $2::uuid
)
"""


def _isoformat(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _transcript_row_to_dict(row) -> dict:
    return {
        "id": str(row["id"]),
        "call_id": str(row["call_id"]),
        "turn_index": row["turn_index"],
        "speaker": row["speaker"],
        "text": row["text"],
        "response_path": row["response_path"],
        "reviewer_applied": row["reviewer_applied"],
        "reviewer_verdict": row["reviewer_verdict"],
        "is_barge_in": row["is_barge_in"],
        "spoken_at": _isoformat(row["spoken_at"]),
    }


async def get_transcripts_by_call_id(
    call_id: str,
    tenant_id: str,
) -> list[dict] | None:
    """관리자/API 외부 응답용 transcript 조회.

    transcripts에는 tenant_id가 없으므로 반드시 calls와 JOIN해서 JWT tenant를
    검증한다. 반환값이 None이면 call이 없거나 tenant가 다른 경우이고, 빈 list는
    해당 tenant의 정상 call이지만 transcript row가 없는 경우다.
    """
    if not _is_uuid(call_id) or not _is_uuid(tenant_id):
        return None

    try:
        pool = await _get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(_TRANSCRIPTS_BY_CALL_ID_SQL, call_id, tenant_id)
            if rows:
                return [_transcript_row_to_dict(row) for row in rows]

            belongs_to_tenant = await conn.fetchval(
                _CALL_BELONGS_TO_TENANT_SQL,
                call_id,
                tenant_id,
            )
            return [] if belongs_to_tenant else None
    except Exception as e:
        logger.warning(
            "get_transcripts_by_call_id failed call_id=%s tenant_id=%s err=%s",
            call_id, tenant_id, e,
        )
        return None
