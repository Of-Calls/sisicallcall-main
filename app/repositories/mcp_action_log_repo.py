"""
MCPActionLogRepository — Postgres `mcp_action_logs` 백엔드.

운영 모드는 db 단일. memory / file 백엔드는 제거됐다 — 운영 데이터의 단일 진실
소스는 Postgres 다 (TenantIntegrationRepository 와 동일 패턴).

연결 정보는 ``settings.database_url`` 사용 (asyncpg). 환경변수 ``MCP_ACTION_LOG_STORE``
``MCP_ACTION_LOG_FILE`` 등은 더 이상 사용되지 않는다.

── DB 스키마 ──────────────────────────────────────────────────────────────────
db/init/09_mcp_action_logs.sql 참조:
  id (UUID PK), call_id, tenant_id, action_type, tool_name,
  request_payload (JSONB), response_payload (JSONB),
  status, external_id, error_message, created_at, updated_at

idempotency 정책:
- application-level: ``find_existing_action(call_id, action_type, tool, idempotency_token?)``
  이 SELECT 로 같은 의도의 row 를 찾고 executor 가 skip 처리. status 무관 매칭 —
  success/skipped/failed 어느 상태든 같은 token row 가 1건이라도 있으면 차단.
  이유: sms_config_missing / oauth_expired 등 환경 이슈로 skipped 된 케이스도
  재시도 의미 없음. 한 통화에서 같은 의도의 액션은 1 row 만.
- ``find_successful_action`` 은 status='success' 만 매칭 — backward compat 유지용.
  새 호출자는 ``find_existing_action`` 사용 권장.
- token 이 주어지면 ``request_payload->>'idempotency_token'`` JSONB 매치도 함께
  적용 — 같은 action_type 이라도 의도가 다르면 별개로 인식됨 (다중 의도 통화).
- DB UNIQUE 제약은 없음. 동시 호출 시 race 가능성은 application-level 의 한계로
  남기고, executor 가 sequential 처리하므로 실제 발생 빈도 낮음.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_STATUSES = frozenset({"success", "failed", "fail", "skipped", "pending"})


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _now_dt()
    return _now_dt()


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _json_dumps(value) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _json_payload(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _iso(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


def _row_get(row, key: str, default=None):
    try:
        return row[key]
    except Exception:
        return default


def _row_to_log_entry(row) -> dict:
    return {
        "id": str(_row_get(row, "id") or ""),
        "call_id": str(_row_get(row, "call_id") or ""),
        "tenant_id": _row_get(row, "tenant_id") or "",
        "action_type": _row_get(row, "action_type") or "",
        "tool_name": _row_get(row, "tool_name") or "",
        "request_payload": _json_payload(_row_get(row, "request_payload")),
        "response_payload": _json_payload(_row_get(row, "response_payload")),
        "status": _row_get(row, "status") or "pending",
        "external_id": _row_get(row, "external_id"),
        "error_message": _row_get(row, "error_message"),
        "created_at": _iso(_row_get(row, "created_at")),
        "updated_at": _iso(_row_get(row, "updated_at")),
    }


def _to_log_entry(action: dict, *, call_id: str, tenant_id: str, now: datetime) -> dict:
    status = action.get("status", "pending")
    if status not in _VALID_STATUSES:
        status = "pending"
    request_payload = copy.deepcopy(action.get("params", {})) or {}
    # idempotency_token 은 action 의 top-level 메타 — request_payload 에 mirror 해서
    # find_successful_action(...token) 의 JSONB 매치가 동작하도록 한다.
    token = action.get("idempotency_token")
    if token and isinstance(request_payload, dict):
        request_payload.setdefault("idempotency_token", str(token))
    return {
        "id": str(uuid.uuid4()),
        "call_id": call_id,
        "tenant_id": tenant_id,
        "action_type": action.get("action_type", ""),
        "tool_name": action.get("tool", ""),
        "request_payload": request_payload,
        "response_payload": copy.deepcopy(action.get("result", {})),
        "status": status,
        "external_id": action.get("external_id"),
        "error_message": action.get("error"),
        "created_at": now,
        "updated_at": now,
    }


def _normalize_tenant_id(tenant_id: str | None, call_id: str) -> str:
    if tenant_id:
        return str(tenant_id)
    logger.warning("action_logs save without tenant_id call_id=%s", call_id)
    return ""


# ── 모듈 레벨 인터페이스 — db only ───────────────────────────────────────────

async def save_action_logs(
    call_id: str,
    tenant_id: str | None = None,
    executed_actions: list[dict] | None = None,
) -> None:
    tenant_id = _normalize_tenant_id(tenant_id, call_id)
    executed_actions = executed_actions or []
    now = _now_dt()
    entries = [
        _to_log_entry(a, call_id=call_id, tenant_id=tenant_id, now=now)
        for a in executed_actions
    ]
    if not entries:
        return

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        for entry in entries:
            created_at = _coerce_datetime(entry.get("created_at"))
            updated_at = _coerce_datetime(entry.get("updated_at"))
            await conn.execute(
                """
                INSERT INTO mcp_action_logs (
                    call_id, tenant_id, action_type, tool_name,
                    request_payload, response_payload, status,
                    external_id, error_message, created_at, updated_at
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5::jsonb, $6::jsonb, $7,
                    $8, $9, $10, $11
                )
                """,
                entry["call_id"],
                entry["tenant_id"],
                entry["action_type"],
                entry["tool_name"],
                _json_dumps(entry["request_payload"]),
                _json_dumps(entry["response_payload"]),
                entry["status"],
                entry["external_id"],
                entry["error_message"],
                created_at,
                updated_at,
            )
        logger.debug("action_logs db saved call_id=%s count=%d", call_id, len(entries))
    except Exception as exc:
        logger.warning("action_logs db save failed call_id=%s err=%s", call_id, exc)
    finally:
        if conn is not None:
            await conn.close()


async def find_successful_action(
    call_id: str,
    action_type: str,
    tool: str,
    idempotency_token: str | None = None,
) -> dict | None:
    """동일 (call_id, action_type, tool) 의 성공 row 조회.

    idempotency_token 이 주어지면 ``request_payload->>'idempotency_token'`` JSONB
    매치도 함께 적용 (다중 의도: 같은 action_type 이라도 token 이 다르면 별개).
    None 이면 (call_id, action_type, tool) 3-tuple 만 매칭.
    """
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        if idempotency_token is None:
            row = await conn.fetchrow(
                """
                SELECT id, call_id, tenant_id, action_type, tool_name,
                       request_payload, response_payload, status,
                       external_id, error_message, created_at, updated_at
                FROM mcp_action_logs
                WHERE call_id = $1
                  AND action_type = $2
                  AND tool_name = $3
                  AND status = 'success'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                call_id,
                action_type,
                tool,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT id, call_id, tenant_id, action_type, tool_name,
                       request_payload, response_payload, status,
                       external_id, error_message, created_at, updated_at
                FROM mcp_action_logs
                WHERE call_id = $1
                  AND action_type = $2
                  AND tool_name = $3
                  AND status = 'success'
                  AND request_payload->>'idempotency_token' = $4
                ORDER BY created_at DESC
                LIMIT 1
                """,
                call_id,
                action_type,
                tool,
                idempotency_token,
            )
        return _row_to_log_entry(row) if row is not None else None
    except Exception as exc:
        logger.warning(
            "action_logs db find_successful failed call_id=%s action_type=%s tool=%s token=%s err=%s",
            call_id,
            action_type,
            tool,
            idempotency_token,
            exc,
        )
        return None
    finally:
        if conn is not None:
            await conn.close()


async def find_existing_action(
    call_id: str,
    action_type: str,
    tool: str,
    idempotency_token: str | None = None,
) -> dict | None:
    """동일 (call_id, action_type, tool) + token 의 row 조회. status 무관.

    한 통화에서 같은 의도의 액션이 이미 DB 에 기록되어 있으면 (success/skipped/failed
    무관) 재시도 의미 없음 — executor 가 skip 처리. ``find_successful_action`` 과의
    차이는 status 필터 부재 — sms_config_missing 같은 환경 이슈로 skipped 된 row
    도 차단 대상.

    idempotency_token 이 주어지면 ``request_payload->>'idempotency_token'`` JSONB
    매치도 함께 적용. None 이면 (call_id, action_type, tool) 3-tuple 만 매칭.
    """
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        if idempotency_token is None:
            row = await conn.fetchrow(
                """
                SELECT id, call_id, tenant_id, action_type, tool_name,
                       request_payload, response_payload, status,
                       external_id, error_message, created_at, updated_at
                FROM mcp_action_logs
                WHERE call_id = $1
                  AND action_type = $2
                  AND tool_name = $3
                ORDER BY created_at DESC
                LIMIT 1
                """,
                call_id,
                action_type,
                tool,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT id, call_id, tenant_id, action_type, tool_name,
                       request_payload, response_payload, status,
                       external_id, error_message, created_at, updated_at
                FROM mcp_action_logs
                WHERE call_id = $1
                  AND action_type = $2
                  AND tool_name = $3
                  AND request_payload->>'idempotency_token' = $4
                ORDER BY created_at DESC
                LIMIT 1
                """,
                call_id,
                action_type,
                tool,
                idempotency_token,
            )
        return _row_to_log_entry(row) if row is not None else None
    except Exception as exc:
        logger.warning(
            "action_logs db find_existing failed call_id=%s action_type=%s tool=%s token=%s err=%s",
            call_id,
            action_type,
            tool,
            idempotency_token,
            exc,
        )
        return None
    finally:
        if conn is not None:
            await conn.close()


async def get_action_logs_by_call_id(call_id: str) -> list[dict]:
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        rows = await conn.fetch(
            """
            SELECT id, call_id, tenant_id, action_type, tool_name,
                   request_payload, response_payload, status,
                   external_id, error_message, created_at, updated_at
            FROM mcp_action_logs
            WHERE call_id = $1
            ORDER BY created_at ASC
            """,
            call_id,
        )
        return [_row_to_log_entry(row) for row in rows]
    except Exception as exc:
        logger.warning("action_logs db list failed call_id=%s err=%s", call_id, exc)
        return []
    finally:
        if conn is not None:
            await conn.close()


async def get_action_logs_by_call_id_for_tenant(
    call_id: str,
    tenant_id: str,
) -> list[dict]:
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        rows = await conn.fetch(
            """
            SELECT id, call_id, tenant_id, action_type, tool_name,
                   request_payload, response_payload, status,
                   external_id, error_message, created_at, updated_at
            FROM mcp_action_logs
            WHERE call_id = $1
              AND lower(COALESCE(tenant_id, '')) = lower($2)
            ORDER BY created_at ASC
            """,
            call_id,
            tenant_id,
        )
        return [_row_to_log_entry(row) for row in rows]
    except Exception as exc:
        logger.warning(
            "action_logs db tenant list failed call_id=%s tenant_id=%s err=%s",
            call_id,
            tenant_id,
            exc,
        )
        return []
    finally:
        if conn is not None:
            await conn.close()


async def get_action_logs(
    tenant_id: str | None = None,
    started_from: str | None = None,
    started_to: str | None = None,
) -> list[dict]:
    clauses: list[str] = []
    values: list[str] = []

    if tenant_id is not None:
        values.append(tenant_id)
        clauses.append(f"tenant_id = ${len(values)}")
    if started_from is not None:
        values.append(started_from)
        clauses.append(f"created_at >= ${len(values)}::timestamptz")
    if started_to is not None:
        values.append(started_to)
        clauses.append(f"created_at <= ${len(values)}::timestamptz")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT id, call_id, tenant_id, action_type, tool_name,
               request_payload, response_payload, status,
               external_id, error_message, created_at, updated_at
        FROM mcp_action_logs
        {where}
        ORDER BY created_at ASC
    """

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        rows = await conn.fetch(sql, *values)
        return [_row_to_log_entry(row) for row in rows]
    except Exception as exc:
        logger.warning("action_logs db list failed err=%s", exc)
        return []
    finally:
        if conn is not None:
            await conn.close()


# ── Backward-compatible class interface (used by save_result_node) ────────────

class MCPActionLogRepository:
    """모듈 함수 wrapper. 인스턴스 생성에 별도 인자가 필요 없다.

    ``settings.database_url`` 만 유효하면 동작한다.
    """

    def __init__(self) -> None:
        logger.info("MCPActionLogRepo db mode")

    async def save_action_log(
        self,
        call_id: str,
        actions: list[dict],
        tenant_id: str | None = None,
    ) -> None:
        await save_action_logs(
            call_id=call_id,
            tenant_id=tenant_id,
            executed_actions=actions,
        )
        logger.debug(
            "action_log saved call_id=%s tenant_id=%s actions=%d",
            call_id,
            tenant_id or "",
            len(actions),
        )

    async def get_action_log(self, call_id: str) -> list[dict]:
        return await get_action_logs_by_call_id(call_id)

    async def find_successful_action(
        self,
        call_id: str,
        action_type: str,
        tool: str,
    ) -> dict | None:
        return await find_successful_action(
            call_id=call_id,
            action_type=action_type,
            tool=tool,
        )

    async def find_existing_action(
        self,
        call_id: str,
        action_type: str,
        tool: str,
        idempotency_token: str | None = None,
    ) -> dict | None:
        return await find_existing_action(
            call_id=call_id,
            action_type=action_type,
            tool=tool,
            idempotency_token=idempotency_token,
        )
