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
    "e2e-007": {
        # 다중 의도: 환불 요청(불만) + 콜백 + 본인인증 → 5개+ 액션 기대
        "scenario": "multi_intent",
        "caller_number": "010-0000-0000",
        "branch_stats": {"faq": 0, "task": 1, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "지난주 결제한 건이 이중으로 청구됐어요. 환불 처리해주세요. 정말 황당하네요."},
            {"role": "agent", "text": "고객님, 본인 확인부터 도와드리겠습니다. 가입하신 휴대폰 번호 뒷자리 4자리 부탁드립니다."},
            {"role": "customer", "text": "지금 회의 들어가야 해서 본인인증을 못 하겠어요. 내일 오후 3시에 다시 전화 주실 수 있나요?"},
            {"role": "agent", "text": "네, 010-0000-0000 으로 내일 15:00 콜백 예약 도와드릴게요. 본인인증은 그때 진행하시고, 결제 환불 건은 운영팀에 즉시 전달하겠습니다."},
            {"role": "customer", "text": "환불 못 받으면 카드사 분쟁 신청할 거예요. 꼭 처리 부탁합니다."},
            {"role": "agent", "text": "최우선으로 처리하겠습니다. 콜백 시간에 본인인증 + 환불 처리 결과 안내드리겠습니다."},
        ],
    },
    "e2e-008": {
        # 다중 row: VOC 두 종류 (배송 누락 + 가격 오안내) → 같은 action_type 두 번 propose 기대
        "scenario": "multi_voc_row",
        "caller_number": "010-0000-0000",
        "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "주문한 품목 중에 두 개나 누락돼서 왔어요. 박스 안에 영수증만 있고 실물이 없어요."},
            {"role": "agent", "text": "정말 죄송합니다. 누락된 품목을 즉시 재발송 처리하겠습니다."},
            {"role": "customer", "text": "그리고 또 하나, 결제 금액이 광고에서 본 가격이랑 다르게 청구됐어요. 광고는 5만원인데 7만원이 빠졌어요."},
            {"role": "agent", "text": "두 건 모두 별도로 운영팀에 신고드리고 차액 환불 + 재발송 진행하겠습니다."},
            {"role": "customer", "text": "두 건 다 정확히 처리해주세요. 한꺼번에 묶지 말고요."},
            {"role": "agent", "text": "각각 별도 티켓으로 등록해서 처리 진행 상황 따로 안내드리겠습니다."},
        ],
    },
    "e2e-notion-split-001": {
        # call_record / voc_record 데이터 분리 검증용 — angry+high (voc_record 도 주입)
        "scenario": "notion_split_verify",
        "caller_number": "010-7777-7777",
        "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "지난주 결제한 상품이 도착하지도 않았는데 환불도 안 해주시네요. 정말 답답합니다."},
            {"role": "agent", "text": "고객님 죄송합니다. 배송 추적부터 확인해드리겠습니다."},
            {"role": "customer", "text": "이미 두 번이나 전화했어요. 매번 확인하겠다고만 하고 진행이 없어요."},
            {"role": "agent", "text": "이번엔 즉시 운영팀에 에스컬레이션 처리해드리겠습니다."},
            {"role": "customer", "text": "환불 처리 안 되면 카드사 분쟁 신청하고 소비자보호원에 신고할 거예요."},
            {"role": "agent", "text": "팀장에게 즉시 보고드리고 오늘 안에 답변 드리겠습니다."},
        ],
    },
    "e2e-notion-split-002": {
        # call_record 단독 검증용 — neutral (voc_record 미주입)
        "scenario": "notion_split_call_only",
        "caller_number": "010-3333-3333",
        "branch_stats": {"faq": 1, "task": 0, "escalation": 0},
        "transcripts": [
            {"role": "customer", "text": "주말 영업시간 알려주세요."},
            {"role": "agent", "text": "토요일은 오전 10시부터 오후 4시, 일요일은 휴무입니다."},
            {"role": "customer", "text": "감사합니다."},
        ],
    },
    "e2e-verify-001": {
        # 검증용: 명확히 다른 의도 (교환 + 멤버십 문의 + 콜백) — 중복 발송 0 확인
        "scenario": "verify_idempotency",
        "caller_number": "010-9999-8888",
        "branch_stats": {"faq": 1, "task": 1, "escalation": 0},
        "transcripts": [
            {"role": "customer", "text": "안녕하세요. 두 가지 문의가 있어서 전화 드렸어요."},
            {"role": "agent", "text": "네 고객님, 말씀해주세요."},
            {"role": "customer", "text": "첫 번째는 지난주 주문한 운동화가 사이즈가 안 맞아서 교환하고 싶어요. 250mm로 주문했는데 245mm가 왔어요."},
            {"role": "agent", "text": "사이즈 오배송 건 죄송합니다. 교환 처리 도와드리겠습니다."},
            {"role": "customer", "text": "두 번째는 별개 건인데, 멤버십 등급 업그레이드 조건이 어떻게 되는지 궁금해요."},
            {"role": "agent", "text": "멤버십 등급은 누적 구매 금액 50만원 이상이면 실버, 200만원 이상이면 골드입니다."},
            {"role": "customer", "text": "감사합니다. 그리고 교환 건은 지금 회의 중이라 자세히 못 말씀드려서, 내일 오전 10시에 다시 전화 주실 수 있을까요?"},
            {"role": "agent", "text": "010-9999-8888 으로 내일 10:00 콜백 예약 도와드리겠습니다."},
            {"role": "customer", "text": "네 부탁드려요."},
        ],
    },
    "e2e-conf-001": {
        # confidence 강등 검증용 — 의도가 모호한 transcript.
        # 환불 같기도 하고 단순 문의 같기도 함. reviewer 가 confidence<0.6 이면
        # verdict 가 pass → correctable 로 자동 강등되는지 확인.
        "scenario": "ambiguous_intent_low_confidence",
        "caller_number": "010-0000-0000",
        "branch_stats": {"faq": 0, "task": 0, "escalation": 0},
        "transcripts": [
            {"role": "customer", "text": "저번에 산 거 있는데, 그게 좀 그래요."},
            {"role": "agent", "text": "어떤 부분이 불편하셨을까요?"},
            {"role": "customer", "text": "아 음... 정확히는 모르겠는데 그냥 좀."},
            {"role": "agent", "text": "혹시 환불을 원하시나요?"},
            {"role": "customer", "text": "꼭 그런 건 아니고... 일단 알겠습니다."},
        ],
    },
    "e2e-complex-001": {
        # 종합 시나리오: 강한 불만 + 본인인증 + 콜백 + 다중 VOC + 상담원 연결
        # 기대 액션: Notion(call+voc×2), SMS×1, schedule_callback, jira×2, slack, email
        "scenario": "complex_multi_intent",
        "caller_number": "010-1234-5678",
        "branch_stats": {"faq": 0, "task": 1, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "지난주 주문한 가방인데, 일단 두 가지 문제가 있어요. 첫째, 같이 주문한 지갑이 박스에 안 들어있었어요."},
            {"role": "agent", "text": "정말 죄송합니다. 누락된 지갑 즉시 재발송 처리해드리겠습니다."},
            {"role": "customer", "text": "두 번째 문제는, 광고에서 본 가격은 8만원이었는데 결제는 12만원으로 빠졌어요. 4만원 차액 환불해주세요."},
            {"role": "agent", "text": "광고 가격과 다른 청구 건도 별도로 운영팀에 신고하고 차액 환불 처리하겠습니다."},
            {"role": "customer", "text": "그리고 환불 처리 진행 상황 확인하려면 본인 인증을 해야 한다고 하는데, 지금 회의 들어가야 해서 시간이 없어요."},
            {"role": "agent", "text": "괜찮습니다. 본인 인증은 콜백 시간에 진행하시면 됩니다. 언제 다시 통화 가능하실까요?"},
            {"role": "customer", "text": "내일 오후 3시에 다시 전화 주세요. 010-1234-5678입니다."},
            {"role": "agent", "text": "010-1234-5678 으로 내일 15:00 콜백 예약 도와드리겠습니다."},
            {"role": "customer", "text": "그런데 환불 처리 안 되면 카드사 분쟁 신청하고 소비자보호원에도 신고할 거예요. 정말 황당하네요."},
            {"role": "agent", "text": "최우선으로 처리하겠습니다. 팀장에게 즉시 보고하고 콜백 시 결과 안내드리겠습니다."},
            {"role": "customer", "text": "꼭 부탁드려요. 상담원 연결도 가능하면 해주세요."},
            {"role": "agent", "text": "네, 슈퍼바이저에게 인계하고 콜백 시 직접 통화 가능하도록 준비하겠습니다."},
        ],
    },
    "e2e-009": {
        # fail+auto 검증용 — angry+high 시나리오 (force_fail 스크립트로 retry max 도달 유도)
        "scenario": "fail_then_auto_only",
        "caller_number": "010-0000-0000",
        "branch_stats": {"faq": 0, "task": 0, "escalation": 1},
        "transcripts": [
            {"role": "customer", "text": "이게 벌써 세 번째 전화인데 매번 다른 답변을 받네요. 정말 신뢰할 수가 없어요."},
            {"role": "agent", "text": "죄송합니다. 이번에는 확실히 처리해드리겠습니다."},
            {"role": "customer", "text": "지난번 상담에서는 환불된다고 했는데 이번엔 안 된다니, 회사 정책이 도대체 뭐예요?"},
            {"role": "agent", "text": "내부 검토 후 정확한 답변을 다시 드리겠습니다."},
            {"role": "customer", "text": "민원 넣을 거예요. 이거 녹음하고 있어요."},
            {"role": "agent", "text": "팀장에게 즉시 보고드리겠습니다."},
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
