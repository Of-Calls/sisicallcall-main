"""E2E 시나리오용 completed call 3건을 Postgres 에 시드.

사용 예:
    python scripts/seed_e2e_completed_call.py --tenant-id ba2bf499-6fcc-4340-b3dd-9341f8bcc915

각 시나리오:
  e2e-001 angry_unresolved   → Slack + Gmail + Jira 액션 기대
  e2e-002 callback_request   → Calendar event 기대 (customer_phone 포함)
  e2e-003 simple_inquiry     → propose_no_action 기대

idempotent — 같은 call_id 로 재실행 시 ON CONFLICT 가지가 동작.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone

import asyncpg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import settings  # noqa: E402

DEFAULT_TENANT_ID = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"
DEFAULT_CUSTOMER_PHONE = "010-0000-0000"   # e2e — 임의 (사용자 합의)


# ── 시나리오 transcripts ─────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "e2e-retry-001": {
        "scenario": "retry_force_fail_then_pass",
        "caller_number": DEFAULT_CUSTOMER_PHONE,
        "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "지난주 주문한 가방, 사진이랑 색상이 완전히 다른데 환불 처리해주세요."},
            {"role": "agent", "text": "고객님, 단순 변심으로는 환불이 어렵습니다."},
            {"role": "customer", "text": "단순 변심이 아니라 상품 설명이랑 다르다고요. 이거 사기 아닌가요?"},
            {"role": "agent", "text": "정책상 7일 이내라도 사용 흔적이 있으면 환불이 어렵습니다."},
            {"role": "customer", "text": "사용도 안 했어요. 박스 그대로 있는데 사용 흔적이 어디 있어요? 정말 화가 납니다."},
            {"role": "agent", "text": "내부에서 검토 후 다시 연락드리겠습니다."},
            {"role": "customer", "text": "검토 같은 소리 하지 마세요. 소비자보호원 신고하고 민원 넣을 거예요."},
            {"role": "agent", "text": "죄송합니다. 상부에 보고드리겠습니다."},
        ],
    },
    "e2e-001-v2": {
        "scenario": "angry_unresolved",
        "caller_number": DEFAULT_CUSTOMER_PHONE,
        "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "지난주 주문한 가방, 사진이랑 색상이 완전히 다른데 환불 처리해주세요."},
            {"role": "agent", "text": "고객님, 단순 변심으로는 환불이 어렵습니다."},
            {"role": "customer", "text": "단순 변심이 아니라 상품 설명이랑 다르다고요. 이거 사기 아닌가요?"},
            {"role": "agent", "text": "정책상 7일 이내라도 사용 흔적이 있으면 환불이 어렵습니다."},
            {"role": "customer", "text": "사용도 안 했어요. 박스 그대로 있는데 사용 흔적이 어디 있어요? 정말 화가 납니다."},
            {"role": "agent", "text": "내부에서 검토 후 다시 연락드리겠습니다."},
            {"role": "customer", "text": "검토 같은 소리 하지 마세요. 소비자보호원 신고하고 민원 넣을 거예요."},
            {"role": "agent", "text": "죄송합니다. 상부에 보고드리겠습니다."},
        ],
    },
    "e2e-001": {
        "scenario": "angry_unresolved",
        "caller_number": DEFAULT_CUSTOMER_PHONE,
        "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "지난주 주문한 가방, 사진이랑 색상이 완전히 다른데 환불 처리해주세요."},
            {"role": "agent", "text": "고객님, 단순 변심으로는 환불이 어렵습니다."},
            {"role": "customer", "text": "단순 변심이 아니라 상품 설명이랑 다르다고요. 이거 사기 아닌가요?"},
            {"role": "agent", "text": "정책상 7일 이내라도 사용 흔적이 있으면 환불이 어렵습니다."},
            {"role": "customer", "text": "사용도 안 했어요. 박스 그대로 있는데 사용 흔적이 어디 있어요? 정말 화가 납니다."},
            {"role": "agent", "text": "내부에서 검토 후 다시 연락드리겠습니다."},
            {"role": "customer", "text": "검토 같은 소리 하지 마세요. 소비자보호원 신고하고 민원 넣을 거예요."},
            {"role": "agent", "text": "죄송합니다. 상부에 보고드리겠습니다."},
        ],
    },
    "e2e-002": {
        "scenario": "callback_request",
        "caller_number": DEFAULT_CUSTOMER_PHONE,
        "branch_stats": {"faq": 0, "task": 1, "escalation": 0},
        "transcripts": [
            {"role": "customer", "text": "지금 회의 중이라 통화가 어려운데, 내일 오후 3시에 다시 전화 주실 수 있나요?"},
            {"role": "agent", "text": "네, 내일 오후 3시에 콜백 예약 도와드리겠습니다. 연락처 확인 부탁드립니다."},
            {"role": "customer", "text": "010-0000-0000입니다. 잘 부탁드립니다."},
            {"role": "agent", "text": "010-0000-0000으로 내일 15:00 콜백 예약 완료했습니다."},
        ],
    },
    "e2e-003": {
        "scenario": "simple_inquiry",
        "caller_number": None,   # 단순 문의 — 번호 미수집
        "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
        "transcripts": [
            {"role": "customer", "text": "거기 영업시간이 어떻게 되나요?"},
            {"role": "agent", "text": "평일 오전 9시부터 오후 6시까지이고, 주말은 휴무입니다."},
            {"role": "customer", "text": "위치도 알려주실 수 있나요?"},
            {"role": "agent", "text": "강남역 3번 출구에서 도보 5분 거리입니다."},
            {"role": "customer", "text": "감사합니다."},
        ],
    },
}


def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def _upsert_call(
    conn: asyncpg.Connection,
    *,
    db_tenant_id: str,
    call_id: str,
    caller_number: str | None,
    branch_stats: dict,
) -> str:
    now = datetime.now(timezone.utc)
    row = await conn.fetchrow(
        """
        INSERT INTO calls (
            tenant_id, twilio_call_sid, caller_number, status,
            started_at, ended_at, duration_sec, branch_stats
        )
        VALUES ($1::uuid, $2, $3, 'completed', $4, $5, 60, $6::jsonb)
        ON CONFLICT (twilio_call_sid) DO UPDATE
            SET tenant_id = EXCLUDED.tenant_id,
                caller_number = EXCLUDED.caller_number,
                status = EXCLUDED.status,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                duration_sec = EXCLUDED.duration_sec,
                branch_stats = EXCLUDED.branch_stats
        RETURNING id
        """,
        db_tenant_id,
        call_id,
        caller_number,
        now,
        now,
        json.dumps(branch_stats),
    )
    return str(row["id"])


async def _replace_transcripts(
    conn: asyncpg.Connection,
    *,
    db_call_id: str,
    transcripts: list[dict],
) -> int:
    await conn.execute("DELETE FROM transcripts WHERE call_id = $1::uuid", db_call_id)
    base = datetime.now(timezone.utc)
    for index, item in enumerate(transcripts):
        await conn.execute(
            """
            INSERT INTO transcripts (call_id, turn_index, speaker, text, spoken_at)
            VALUES ($1::uuid, $2, $3, $4, $5)
            """,
            db_call_id,
            index,
            item.get("role") or "customer",
            item.get("text") or "",
            base,
        )
    return len(transcripts)


async def seed_one(
    conn: asyncpg.Connection,
    *,
    tenant_id: str,
    call_id: str,
    fixture: dict,
) -> dict:
    db_call_id = await _upsert_call(
        conn,
        db_tenant_id=tenant_id,
        call_id=call_id,
        caller_number=fixture["caller_number"],
        branch_stats=fixture["branch_stats"],
    )
    transcript_count = await _replace_transcripts(
        conn,
        db_call_id=db_call_id,
        transcripts=fixture["transcripts"],
    )
    return {
        "call_id": call_id,
        "scenario": fixture["scenario"],
        "db_call_uuid": db_call_id,
        "transcript_count": transcript_count,
        "caller_number": fixture["caller_number"],
    }


async def seed_all(tenant_id: str) -> list[dict]:
    conn = await asyncpg.connect(_database_url())
    results = []
    try:
        async with conn.transaction():
            for call_id, fixture in SCENARIOS.items():
                r = await seed_one(conn, tenant_id=tenant_id, call_id=call_id, fixture=fixture)
                results.append(r)
                print(f"  seeded {call_id} ({r['scenario']}) — db_uuid={r['db_call_uuid']} transcripts={r['transcript_count']}")
    finally:
        await conn.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed e2e completed calls into Postgres")
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    args = parser.parse_args()

    print(f"Seeding e2e completed calls for tenant_id={args.tenant_id}")
    asyncio.run(seed_all(args.tenant_id))
    print("done.")


if __name__ == "__main__":
    main()
