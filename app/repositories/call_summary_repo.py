from __future__ import annotations

import copy
from datetime import datetime, timezone

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


async def get_summary_by_call_id(
    call_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    record = _summary_store.get(call_id)
    if record is None:
        return None
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
