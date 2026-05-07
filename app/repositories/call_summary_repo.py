from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from uuid import UUID

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory stores ──────────────────────────────────────────────────────────

_summary_store: dict[str, dict] = {}   # call_id → summary record
_context_store: dict[str, dict] = {}   # call_id → call context

_SAMPLE_CONTEXT = {
    "metadata": {
        "call_id": "sample-001",
        "tenant_id": "default",
        "start_time": "2026-04-25T10:00:00Z",
        "end_time": "2026-04-25T10:03:00Z",
    },
    "transcripts": [
        {"role": "customer", "text": "요금제 변경하고 싶은데요."},
        {"role": "agent", "text": "네, 도와드리겠습니다. 어떤 요금제로 변경을 원하시나요?"},
        {"role": "customer", "text": "더 저렴한 걸로 바꾸고 싶어요."},
    ],
    "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
}


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
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _isoformat(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _jsonb_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        if isinstance(parsed, list):
            return copy.deepcopy(parsed)
    return []


def _summary_row_to_record(row) -> dict:
    summary = {
        "summary_short": row["summary_short"] or "",
        "summary_detailed": row["summary_detailed"] or "",
        "customer_intent": row["customer_intent"] or "",
        "customer_emotion": row["customer_emotion"] or "",
        "resolution_status": row["resolution_status"] or "",
        "keywords": _jsonb_list(row["keywords"]),
        "handoff_notes": row["handoff_notes"] or "",
        "generation_mode": row["generation_mode"] or "",
        "model_used": row["model_used"] or "",
    }
    return {
        "call_id": str(row["call_id"]),
        "tenant_id": str(row["tenant_id"]),
        "summary": summary,
        "created_at": _isoformat(row["created_at"]),
        "updated_at": _isoformat(row["updated_at"]),
    }


def _summary_db_payload(summary: dict) -> dict:
    emotion = summary.get("customer_emotion") or summary.get("emotion") or "neutral"
    if emotion not in {"positive", "neutral", "negative", "angry"}:
        emotion = "neutral"

    resolution_status = (
        summary.get("resolution_status")
        or summary.get("status")
        or "resolved"
    )
    if resolution_status not in {"resolved", "escalated", "abandoned"}:
        resolution_status = "resolved"

    generation_mode = summary.get("generation_mode") or "async"
    if generation_mode not in {"sync", "async"}:
        generation_mode = "async"

    return {
        "summary_short": summary.get("summary_short") or summary.get("short") or "",
        "summary_detailed": summary.get("summary_detailed") or summary.get("detailed"),
        "customer_intent": summary.get("customer_intent") or summary.get("intent"),
        "customer_emotion": emotion,
        "resolution_status": resolution_status,
        "keywords": _as_list(summary.get("keywords")),
        "handoff_notes": summary.get("handoff_notes"),
        "generation_mode": generation_mode,
        "model_used": summary.get("model_used") or summary.get("model") or "demo-mock-llm",
    }


def _reset() -> None:
    """테스트 격리용 — 모든 store를 초기화한다."""
    _summary_store.clear()
    _context_store.clear()


# ── Module-level functions (KDT-77 interface) ─────────────────────────────────

async def save_summary(
    call_id: str,
    tenant_id: str,
    summary: dict,
) -> None:
    now = _now()
    existing = _summary_store.get(call_id, {})
    _summary_store[call_id] = {
        "call_id": call_id,
        "tenant_id": tenant_id,
        "summary": copy.deepcopy(summary),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
    }
    logger.debug("summary saved call_id=%s", call_id)
    await _upsert_summary_to_db_if_possible(call_id, tenant_id, summary)


async def _upsert_summary_to_db_if_possible(
    call_id: str,
    tenant_id: str,
    summary: dict,
) -> None:
    if not _is_uuid(call_id):
        logger.warning(
            "summary db save skipped non_uuid_call_id=%s tenant_id=%s",
            call_id,
            tenant_id,
        )
        return

    payload = _summary_db_payload(summary)
    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        call_tenant_id = await _fetch_call_tenant_id(conn, call_id)
        if call_tenant_id is None:
            logger.warning(
                "post_call call not found: skip summary db save call_id=%s state_tenant_id=%s",
                call_id,
                tenant_id,
            )
            return

        if call_tenant_id.lower() != str(tenant_id).lower():
            logger.warning(
                "post_call tenant mismatch: skip summary save call_id=%s state_tenant_id=%s call_tenant_id=%s",
                call_id,
                tenant_id,
                call_tenant_id,
            )
            return

        await conn.execute(
            """
            INSERT INTO call_summaries (
              call_id,
              tenant_id,
              summary_short,
              summary_detailed,
              customer_intent,
              customer_emotion,
              resolution_status,
              keywords,
              handoff_notes,
              generation_mode,
              model_used,
              updated_at
            )
            VALUES (
              $1::uuid,
              $2::uuid,
              $3,
              $4,
              $5,
              $6,
              $7,
              $8::jsonb,
              $9,
              $10,
              $11,
              now()
            )
            ON CONFLICT (call_id)
            DO UPDATE SET
              tenant_id = EXCLUDED.tenant_id,
              summary_short = EXCLUDED.summary_short,
              summary_detailed = EXCLUDED.summary_detailed,
              customer_intent = EXCLUDED.customer_intent,
              customer_emotion = EXCLUDED.customer_emotion,
              resolution_status = EXCLUDED.resolution_status,
              keywords = EXCLUDED.keywords,
              handoff_notes = EXCLUDED.handoff_notes,
              generation_mode = EXCLUDED.generation_mode,
              model_used = EXCLUDED.model_used,
              updated_at = now()
            """,
            call_id,
            tenant_id,
            payload["summary_short"],
            payload["summary_detailed"],
            payload["customer_intent"],
            payload["customer_emotion"],
            payload["resolution_status"],
            _json_dumps(payload["keywords"]),
            payload["handoff_notes"],
            payload["generation_mode"],
            payload["model_used"],
        )
        logger.info("call_summaries db upserted call_id=%s", call_id)
    except Exception as exc:
        logger.warning("call_summaries db upsert failed call_id=%s err=%s", call_id, exc)
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


async def _fetch_summary_from_db(call_id: str, tenant_id: str) -> dict | None:
    if not _is_uuid(call_id) or not _is_uuid(tenant_id):
        return None

    conn = None
    try:
        conn = await asyncpg.connect(_database_url())
        row = await conn.fetchrow(
            """
            SELECT
                call_id,
                tenant_id,
                summary_short,
                summary_detailed,
                customer_intent,
                customer_emotion,
                resolution_status,
                keywords,
                handoff_notes,
                generation_mode,
                model_used,
                created_at,
                updated_at
            FROM call_summaries
            WHERE call_id = $1::uuid
              AND tenant_id = $2::uuid
            LIMIT 1
            """,
            call_id,
            tenant_id,
        )
        if row is None:
            return None
        return _summary_row_to_record(row)
    except Exception as exc:
        logger.warning("call summary DB fallback failed call_id=%s err=%s", call_id, exc)
        return None
    finally:
        if conn is not None:
            await conn.close()


async def get_summary_by_call_id(
    call_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    record = _summary_store.get(call_id)
    if record is None:
        if tenant_id is None:
            return None
        return await _fetch_summary_from_db(call_id, tenant_id)
    if tenant_id is not None:
        record_tenant_id = str(record.get("tenant_id") or "").strip().lower()
        expected_tenant_id = str(tenant_id).strip().lower()
        if record_tenant_id != expected_tenant_id:
            return None
    return copy.deepcopy(record)


async def seed_call_context(
    call_id: str,
    tenant_id: str = "default",
    transcripts: list[dict] | None = None,
    call_metadata: dict | None = None,
    branch_stats: dict | None = None,
) -> None:
    _context_store[call_id] = copy.deepcopy({
        "metadata": {
            **(call_metadata or {}),
            "call_id": call_id,
            "tenant_id": tenant_id,
        },
        "transcripts": transcripts or [],
        "branch_stats": branch_stats or {},
    })
    logger.debug("call_context seeded call_id=%s", call_id)


async def get_call_context(call_id: str) -> dict | None:
    record = _context_store.get(call_id)
    if record is not None:
        return copy.deepcopy(record)
    # fallback: sample context with call_id patched in
    ctx = copy.deepcopy(_SAMPLE_CONTEXT)
    ctx["metadata"] = {**_SAMPLE_CONTEXT["metadata"], "call_id": call_id}
    logger.debug("call_context not found call_id=%s — sample 반환", call_id)
    return ctx


async def get_seeded_call_context(call_id: str) -> dict | None:
    """명시적으로 seed된 컨텍스트만 반환한다 — sample fallback 없음.

    context_provider가 '실제로 주입된 데이터가 있는지' 판별할 때 사용한다.
    주입된 데이터가 없으면 None을 반환한다.
    """
    record = _context_store.get(call_id)
    return copy.deepcopy(record) if record is not None else None


# ── Backward-compatible class interface (used by load_context_node / save_result_node) ──

class CallSummaryRepository:
    async def get_call_context(self, call_id: str) -> dict:
        result = await get_call_context(call_id)
        return result or {}

    async def save_summary(
        self,
        call_id: str,
        summary: dict,
        tenant_id: str = "",
    ) -> None:
        await save_summary(call_id=call_id, tenant_id=tenant_id, summary=summary)

    async def seed(self, call_id: str, context: dict) -> None:
        await seed_call_context(
            call_id=call_id,
            tenant_id=context.get("metadata", {}).get("tenant_id", "default"),
            transcripts=context.get("transcripts"),
            call_metadata=context.get("metadata"),
            branch_stats=context.get("branch_stats"),
        )
