from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────

_DEFAULT_STORE_PATH = Path(".local/mcp_action_logs.json")
_action_store: dict[str, list[dict]] = {}   # call_id → [log_entry, ...]

_VALID_STATUSES = frozenset({"success", "failed", "skipped", "pending"})
_STORE_MODE_FILE = "file"
_STORE_MODE_DB = "db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _get_store_path() -> Path:
    return Path(os.getenv("MCP_ACTION_LOG_FILE", str(_DEFAULT_STORE_PATH)))


def _get_store_mode() -> str:
    mode = os.getenv("MCP_ACTION_LOG_STORE", _STORE_MODE_FILE).strip().lower()
    if mode == _STORE_MODE_DB:
        return _STORE_MODE_DB
    if mode != _STORE_MODE_FILE:
        logger.warning("unknown MCP_ACTION_LOG_STORE=%s; falling back to file", mode)
    return _STORE_MODE_FILE


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


def _row_to_log_entry(row) -> dict:
    return {
        "call_id": str(row["call_id"]),
        "tenant_id": row["tenant_id"] or "",
        "action_type": row["action_type"] or "",
        "tool_name": row["tool_name"] or "",
        "request_payload": _json_payload(row["request_payload"]),
        "response_payload": _json_payload(row["response_payload"]),
        "status": row["status"] or "pending",
        "external_id": row["external_id"],
        "error_message": row["error_message"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


def _load_store_from_file() -> dict[str, list[dict]]:
    path = _get_store_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("mcp_action_logs file load failed path=%s err=%s", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("mcp_action_logs file ignored path=%s reason=not_dict", path)
        return {}
    store: dict[str, list[dict]] = {}
    for call_id, entries in raw.items():
        if isinstance(call_id, str) and isinstance(entries, list):
            store[call_id] = [
                copy.deepcopy(entry)
                for entry in entries
                if isinstance(entry, dict)
            ]
    return store


def _save_store_to_file(store: dict[str, list[dict]]) -> None:
    path = _get_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: replace local file mode with a DB-backed action log before production.
    # Local demo mode intentionally avoids file locking.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(store), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _load_into_memory() -> None:
    for call_id, entries in _load_store_from_file().items():
        _action_store[call_id] = entries


def _reset(remove_file: bool = False) -> None:
    """테스트 격리용."""
    _action_store.clear()
    if remove_file:
        path = _get_store_path()
        if path.exists():
            path.unlink()


def _to_log_entry(action: dict, *, call_id: str, tenant_id: str, now: datetime) -> dict:
    status = action.get("status", "pending")
    if status not in _VALID_STATUSES:
        status = "pending"
    return {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "action_type": action.get("action_type", ""),
        "tool_name": action.get("tool", ""),
        "request_payload": copy.deepcopy(action.get("params", {})),
        "response_payload": copy.deepcopy(action.get("result", {})),
        "status": status,
        "external_id": action.get("external_id"),
        "error_message": action.get("error"),
        "created_at": now,
        "updated_at": now,
    }


# ── Module-level functions (KDT-77 interface) ─────────────────────────────────

async def save_action_logs(
    call_id: str,
    tenant_id: str | None = None,
    executed_actions: list[dict] | None = None,
) -> None:
    tenant_id = _normalize_tenant_id(tenant_id, call_id)
    executed_actions = executed_actions or []
    if _get_store_mode() == _STORE_MODE_DB:
        await _save_action_logs_to_db(call_id, tenant_id, executed_actions)
        return

    _load_into_memory()
    now = _now_dt()
    entries = [
        _to_log_entry(a, call_id=call_id, tenant_id=tenant_id, now=now)
        for a in executed_actions
    ]
    _action_store.setdefault(call_id, []).extend(entries)
    _save_store_to_file(_action_store)
    logger.debug("action_logs saved call_id=%s count=%d", call_id, len(entries))


def _normalize_tenant_id(tenant_id: str | None, call_id: str) -> str:
    if tenant_id:
        return str(tenant_id)
    logger.warning("action_logs save without tenant_id call_id=%s", call_id)
    return ""


async def _save_action_logs_to_db(
    call_id: str,
    tenant_id: str,
    executed_actions: list[dict],
) -> None:
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
) -> dict | None:
    if _get_store_mode() == _STORE_MODE_DB:
        return await _find_successful_action_from_db(call_id, action_type, tool)

    _load_into_memory()
    entries = _action_store.get(call_id, [])
    for entry in reversed(entries):
        if (
            entry.get("action_type") == action_type
            and entry.get("tool_name") == tool
            and entry.get("status") == "success"
        ):
            return copy.deepcopy(entry)
    return None


async def _find_successful_action_from_db(
    call_id: str,
    action_type: str,
    tool: str,
) -> dict | None:
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        row = await conn.fetchrow(
            """
            SELECT call_id, tenant_id, action_type, tool_name,
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
        return _row_to_log_entry(row) if row is not None else None
    except Exception as exc:
        logger.warning(
            "action_logs db find_successful failed call_id=%s action_type=%s tool=%s err=%s",
            call_id,
            action_type,
            tool,
            exc,
        )
        return None
    finally:
        if conn is not None:
            await conn.close()


async def get_action_logs_by_call_id(call_id: str) -> list[dict]:
    if _get_store_mode() == _STORE_MODE_DB:
        return await _get_action_logs_by_call_id_from_db(call_id)

    _load_into_memory()
    entries = _action_store.get(call_id, [])
    return copy.deepcopy(entries)


async def _get_action_logs_by_call_id_from_db(call_id: str) -> list[dict]:
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        rows = await conn.fetch(
            """
            SELECT call_id, tenant_id, action_type, tool_name,
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


async def get_action_logs(
    tenant_id: str | None = None,
    started_from: str | None = None,
    started_to: str | None = None,
) -> list[dict]:
    if _get_store_mode() == _STORE_MODE_DB:
        return await _get_action_logs_from_db(
            tenant_id=tenant_id,
            started_from=started_from,
            started_to=started_to,
        )

    _load_into_memory()
    all_logs: list[dict] = []
    for entries in _action_store.values():
        all_logs.extend(copy.deepcopy(entries))

    if tenant_id is not None:
        all_logs = [e for e in all_logs if e.get("tenant_id") == tenant_id]
    if started_from is not None:
        all_logs = [e for e in all_logs if e.get("created_at", "") >= started_from]
    if started_to is not None:
        all_logs = [e for e in all_logs if e.get("created_at", "") <= started_to]
    return all_logs


async def _get_action_logs_from_db(
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
        SELECT call_id, tenant_id, action_type, tool_name,
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
