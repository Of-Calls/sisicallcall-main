from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from uuid import UUID

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────

_voc_store: dict[str, dict] = {}   # call_id → voc record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _is_uuid(value: str | None) -> bool:
    try:
        UUID(str(value))
        return True
    except Exception:
        return False


def _json_dumps(value) -> str:
    default = [] if isinstance(value, list) else {}
    return json.dumps(value if value is not None else default, ensure_ascii=False)


def _as_dict(value) -> dict:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _voc_db_payload(voc_analysis: dict) -> dict:
    sentiment_result = _as_dict(voc_analysis.get("sentiment_result"))
    if not sentiment_result and voc_analysis.get("sentiment"):
        sentiment_result = {"sentiment": voc_analysis.get("sentiment")}

    intent_result = _as_dict(voc_analysis.get("intent_result"))
    priority_result = _as_dict(voc_analysis.get("priority_result"))

    return {
        "sentiment_result": sentiment_result,
        "intent_result": intent_result,
        "priority_result": priority_result,
    }


def _reset() -> None:
    """테스트 격리용."""
    _voc_store.clear()


# ── Module-level functions (KDT-77 interface) ─────────────────────────────────

async def save_voc_analysis(
    call_id: str,
    tenant_id: str,
    voc_analysis: dict,
    partial_success: bool = False,
    failed_subagents: list | None = None,
) -> None:
    now = _now()
    existing = _voc_store.get(call_id, {})
    _voc_store[call_id] = {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "voc_analysis": copy.deepcopy(voc_analysis),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    logger.debug("voc_analysis saved call_id=%s", call_id)
    await _upsert_voc_analysis_to_db_if_possible(
        call_id=call_id,
        tenant_id=tenant_id,
        voc_analysis=voc_analysis,
        partial_success=partial_success,
        failed_subagents=failed_subagents,
    )


async def _upsert_voc_analysis_to_db_if_possible(
    call_id: str,
    tenant_id: str,
    voc_analysis: dict,
    partial_success: bool = False,
    failed_subagents: list | None = None,
) -> None:
    if not _is_uuid(call_id):
        logger.warning(
            "voc_analysis db save skipped non_uuid_call_id=%s tenant_id=%s",
            call_id,
            tenant_id,
        )
        return

    payload = _voc_db_payload(voc_analysis)
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        call_tenant_id = await _fetch_call_tenant_id(conn, call_id)
        if call_tenant_id is None:
            logger.warning(
                "post_call call not found: skip voc_analysis db save call_id=%s state_tenant_id=%s",
                call_id,
                tenant_id,
            )
            return

        if call_tenant_id.lower() != str(tenant_id).lower():
            logger.warning(
                "post_call tenant mismatch: skip voc_analysis save call_id=%s state_tenant_id=%s call_tenant_id=%s",
                call_id,
                tenant_id,
                call_tenant_id,
            )
            return

        await conn.execute(
            """
            INSERT INTO voc_analyses (
              call_id,
              tenant_id,
              sentiment_result,
              intent_result,
              priority_result,
              partial_success,
              failed_subagents,
              updated_at
            )
            VALUES (
              $1::uuid,
              $2::uuid,
              $3::jsonb,
              $4::jsonb,
              $5::jsonb,
              $6,
              $7::jsonb,
              now()
            )
            ON CONFLICT (call_id)
            DO UPDATE SET
              tenant_id = EXCLUDED.tenant_id,
              sentiment_result = EXCLUDED.sentiment_result,
              intent_result = EXCLUDED.intent_result,
              priority_result = EXCLUDED.priority_result,
              partial_success = EXCLUDED.partial_success,
              failed_subagents = EXCLUDED.failed_subagents,
              updated_at = now()
            """,
            call_id,
            tenant_id,
            _json_dumps(payload["sentiment_result"]),
            _json_dumps(payload["intent_result"]),
            _json_dumps(payload["priority_result"]),
            bool(partial_success),
            _json_dumps(_as_list(failed_subagents)),
        )
        logger.info("voc_analyses db upserted call_id=%s", call_id)
    except Exception as exc:
        logger.warning("voc_analyses db upsert failed call_id=%s err=%s", call_id, exc)
    finally:
        if conn is not None:
            await conn.close()


async def _fetch_call_tenant_id(conn, call_id: str) -> str | None:
    row = await conn.fetchrow(
        """
        SELECT tenant_id
        FROM calls
        WHERE id = $1::uuid
        LIMIT 1
        """,
        call_id,
    )
    if row is None:
        return None
    return str(row["tenant_id"])


async def get_voc_by_call_id(call_id: str) -> dict | None:
    record = _voc_store.get(call_id)
    return copy.deepcopy(record) if record is not None else None


# ── Backward-compatible class interface (used by save_result_node) ────────────

class VOCAnalysisRepository:
    async def save_voc_analysis(
        self,
        call_id: str,
        voc: dict,
        tenant_id: str = "",
        partial_success: bool = False,
        failed_subagents: list | None = None,
    ) -> None:
        await save_voc_analysis(
            call_id=call_id,
            tenant_id=tenant_id,
            voc_analysis=voc,
            partial_success=partial_success,
            failed_subagents=failed_subagents,
        )

    async def get_voc_analysis(self, call_id: str) -> dict | None:
        record = await get_voc_by_call_id(call_id)
        return copy.deepcopy(record["voc_analysis"]) if record else None
