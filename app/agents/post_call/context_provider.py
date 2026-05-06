"""
Post-call Context Provider.

통화 종료 후 후처리 에이전트가 사용할 통화 컨텍스트(transcript / metadata / branch_stats)를
여러 소스에서 우선순위 순으로 조회하고 정규화해 반환한다.

── 조회 우선순위 ─────────────────────────────────────────────────────────────
1. DB (calls / transcripts 테이블) — 운영 원본 데이터
   app/services/db/transcripts.py DB ORM 확정 후 실제 데이터 반환
2. in-memory seed — 테스트 / 개발 환경 전용
   seed_test_context / seed_call_context로 사전 주입한 데이터
3. None — 어디서도 찾지 못한 경우

── Redis 미사용 이유 ─────────────────────────────────────────────────────────
Redis는 실시간 통화 세션 캐시(STT 스트림, TTS 상태 등)에 사용한다.
후처리의 원본 데이터 소스는 DB이므로 Redis는 후처리 경로에 포함하지 않는다.

── 반환 형식 (정규화 후) ─────────────────────────────────────────────────────
{
  "metadata":     {"call_id": ..., "tenant_id": ..., ...},
  "transcripts":  [{"role": "customer"|"agent", "text": ...}, ...],
  "branch_stats": {"faq": int, "task": int, "escalation": int}
}
- metadata가 없으면 {}
- metadata.call_id / tenant_id 없으면 인자로 보강
- transcripts가 None이면 []
- branch_stats가 None이면 {}
"""
from __future__ import annotations

import copy

from app.repositories import get_seeded_call_context, seed_call_context
from app.services.db.transcripts import get_completed_call_context_from_db
from app.utils.logger import get_logger
from app.utils.phone import normalize_korean_phone

logger = get_logger(__name__)


def _normalize(ctx: dict, call_id: str, tenant_id: str | None) -> dict:
    """context를 PostCallAgent가 사용할 수 있는 형태로 정규화한다."""
    result = copy.deepcopy(ctx)

    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not metadata.get("call_id"):
        metadata["call_id"] = call_id
    if not metadata.get("tenant_id") and tenant_id:
        metadata["tenant_id"] = tenant_id

    # customer_phone 보존 + 정규화. seed 경로 / DB 경로 모두 일관된 로컬 형식으로
    # 통일한다. 빈 값이면 키 자체를 제거해 action_planner 가 .get(..., "")로
    # 안전하게 빈 문자열을 받게 한다.
    phone_raw = metadata.get("customer_phone")
    if phone_raw:
        normalized = normalize_korean_phone(phone_raw)
        if normalized:
            metadata["customer_phone"] = normalized
        else:
            metadata.pop("customer_phone", None)
    elif "customer_phone" in metadata:
        # None / "" 처럼 falsy 값은 키를 비워 둔다.
        metadata.pop("customer_phone", None)

    result["metadata"] = metadata

    if result.get("transcripts") is None:
        result["transcripts"] = []

    if result.get("branch_stats") is None:
        result["branch_stats"] = {}

    return result


async def get_call_context_for_post_call(
    call_id: str,
    tenant_id: str | None = None,
) -> dict | None:
    """Post-call 에이전트가 사용할 통화 컨텍스트를 반환한다.

    조회 우선순위:
      1) DB (get_completed_call_context_from_db)
      2) in-memory seed (get_seeded_call_context)
      3) None

    반환값은 항상 _normalize를 거쳐 metadata / transcripts / branch_stats가
    None이 아닌 정규화된 dict로 반환된다.
    deepcopy를 사용하므로 반환된 dict를 수정해도 내부 저장소가 오염되지 않는다.
    """
    # ── Step 1: DB 조회 (운영 원본) ───────────────────────────────────────────
    try:
        raw = await get_completed_call_context_from_db(call_id, tenant_id=tenant_id)
        if raw is not None:
            logger.info(
                "context_provider: DB context 사용 call_id=%s transcripts=%d",
                call_id, len(raw.get("transcripts") or []),
            )
            return _normalize(raw, call_id, tenant_id)
    except Exception as exc:
        logger.warning(
            "context_provider: DB 조회 실패 call_id=%s err=%s — fallback",
            call_id, exc,
        )

    # ── Step 2: in-memory seed (테스트 · 개발 환경) ───────────────────────────
    raw = await get_seeded_call_context(call_id)
    if raw is not None:
        logger.debug(
            "context_provider: in-memory seed 사용 call_id=%s",
            call_id,
        )
        return _normalize(raw, call_id, tenant_id)

    # ── Step 3: 찾지 못한 경우 ────────────────────────────────────────────────
    logger.warning(
        "context_provider: call_id=%s 에 대한 컨텍스트를 찾지 못함 — None 반환",
        call_id,
    )
    return None


async def seed_test_context(
    call_id: str,
    tenant_id: str = "default",
    transcripts: list[dict] | None = None,
    call_metadata: dict | None = None,
    branch_stats: dict | None = None,
) -> None:
    """테스트용 컨텍스트를 in-memory repository에 주입한다.

    운영 환경에서는 사용하지 않는다.
    테스트 코드에서 get_call_context_for_post_call을 검증할 때 호출한다.
    """
    await seed_call_context(
        call_id=call_id,
        tenant_id=tenant_id,
        transcripts=transcripts,
        call_metadata=call_metadata,
        branch_stats=branch_stats,
    )
    logger.debug("context_provider: test context seeded call_id=%s", call_id)
