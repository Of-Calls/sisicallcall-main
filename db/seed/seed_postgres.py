"""
PostgreSQL 시드 — 병원 + 음식점 tenant 및 KNN intent 예시

실행: python db/seed/seed_postgres.py

멱등성 보장: ON CONFLICT DO NOTHING — 여러 번 실행해도 중복 INSERT 없음.

중요:
  - KNN intent 임베딩은 **더미 제로 벡터(1024d)** 로 채워집니다.
  - 희영(BGE-M3 연구) 완료 후 별도 재계산 스크립트로 교체 필요.
  - 그 전까지 KNN Router는 정상 동작하지 않습니다 (신용 연구 시 의도된 제약).
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.security import hash_password

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL 환경변수가 없습니다. .env 파일을 확인하세요.", file=sys.stderr)
    sys.exit(1)

# BGE-M3 기본 차원 (희영 연구 완료 후 실제 값으로 재계산됨)
EMBEDDING_DIM = 1024
DUMMY_EMBEDDING = [0.0] * EMBEDDING_DIM

# ==============================================================================
# tenant 2개 (병원 + 음식점)
# ==============================================================================
TENANTS = [
    {
        "name": "서울중앙병원",
        "twilio_number": "+821000000001",
        "industry": "hospital",
        "plan": "vertical",
        "settings": {
            "business_hours": {
                "mon-fri": {"start": "09:00", "end": "18:00"},
                "sat": {"start": "09:00", "end": "13:00"},
                "sun": "closed",
            },
            "cache_threshold": 0.92,
            "knn_threshold": 0.80,
        },
    },
    {
        "name": "한밭식당",
        "twilio_number": "+821000000002",
        "industry": "restaurant",
        "plan": "basic",
        "settings": {
            "business_hours": {
                "mon-sun": {"start": "11:00", "end": "22:00"},
            },
            "cache_threshold": 0.92,
            "knn_threshold": 0.80,
        },
    },
]

# ==============================================================================
# KNN intent 예시 문장 — tenant별로 관리
# intent_label prefix: intent_faq_* / intent_task_* / intent_auth_* / intent_escalation_*
# ==============================================================================
KNN_INTENTS = {
    "서울중앙병원": [
        # FAQ
        ("intent_faq_hours", "영업시간이 어떻게 되나요?"),
        ("intent_faq_hours", "몇 시에 문 여나요?"),
        ("intent_faq_hours", "진료 시간이 궁금해요"),
        ("intent_faq_location", "병원 위치가 어디인가요?"),
        ("intent_faq_location", "주소 알려주세요"),
        ("intent_faq_prep", "MRI 촬영 전에 금식해야 하나요?"),
        ("intent_faq_prep", "검사 전 준비사항 알려주세요"),
        # Task (M2)
        ("intent_task_reservation", "진료 예약하고 싶어요"),
        ("intent_task_reservation", "내일 오후 2시 예약 가능한가요?"),
        ("intent_task_reservation", "예약 변경할 수 있나요?"),
        ("intent_task_reservation", "예약 취소해주세요"),
        # Escalation
        ("intent_escalation", "상담원 바꿔주세요"),
        ("intent_escalation", "사람이랑 통화하고 싶어요"),
    ],
    "한밭식당": [
        # FAQ
        ("intent_faq_hours", "영업시간이 어떻게 되나요?"),
        ("intent_faq_hours", "몇 시까지 하나요?"),
        ("intent_faq_location", "가게 위치가 어디인가요?"),
        ("intent_faq_menu", "대표 메뉴가 뭔가요?"),
        ("intent_faq_menu", "추천 메뉴 알려주세요"),
        ("intent_faq_menu", "비건 메뉴 있나요?"),
        # Task (M2)
        ("intent_task_reservation", "오늘 저녁 7시 예약되나요?"),
        ("intent_task_reservation", "4명 자리 있나요?"),
        ("intent_task_reservation", "단체 예약 가능한가요?"),
        ("intent_task_reservation", "예약 취소해주세요"),
        # Escalation
        ("intent_escalation", "사장님과 통화하고 싶어요"),
        ("intent_escalation", "직원 바꿔주세요"),
    ],
}

# 개발/시드 전용 관리자 계정입니다. 운영 비밀번호로 사용하지 마세요.
ADMIN_USERS = [
    {
        "tenant_name": "서울중앙병원",
        "email": "admin@seoul-hospital.test",
        "password": "password1234",
        "name": "서울중앙병원 관리자",
        "role": "owner",
    },
    {
        "tenant_name": "한밭식당",
        "email": "admin@hanbat.test",
        "password": "password1234",
        "name": "한밭식당 관리자",
        "role": "owner",
    },
]


async def seed():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        print("🌱 PostgreSQL 시드 시작")

        # --- tenants INSERT (멱등) ---
        tenant_ids = {}
        for t in TENANTS:
            row = await conn.fetchrow(
                """
                INSERT INTO tenants (name, twilio_number, industry, plan, settings)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (twilio_number) DO UPDATE
                    SET name = EXCLUDED.name,
                        industry = EXCLUDED.industry,
                        plan = EXCLUDED.plan,
                        settings = EXCLUDED.settings,
                        updated_at = now()
                RETURNING id
                """,
                t["name"],
                t["twilio_number"],
                t["industry"],
                t["plan"],
                __import__("json").dumps(t["settings"]),
            )
            tenant_ids[t["name"]] = row["id"]
            print(f"  ✅ tenant: {t['name']} ({t['twilio_number']}) — id={row['id']}")

        # --- admin_users INSERT (개발/시드 전용 계정, 멱등 upsert) ---
        for admin in ADMIN_USERS:
            tenant_id = tenant_ids[admin["tenant_name"]]
            password_hash = hash_password(admin["password"])
            row = await conn.fetchrow(
                """
                INSERT INTO admin_users (
                    tenant_id, email, password_hash, name, role, is_active
                )
                VALUES ($1::uuid, LOWER($2), $3, $4, $5, TRUE)
                ON CONFLICT (email) DO UPDATE
                    SET tenant_id = EXCLUDED.tenant_id,
                        password_hash = EXCLUDED.password_hash,
                        name = EXCLUDED.name,
                        role = EXCLUDED.role,
                        is_active = TRUE,
                        updated_at = now()
                RETURNING id
                """,
                tenant_id,
                admin["email"],
                password_hash,
                admin["name"],
                admin["role"],
            )
            print(
                f"  ✅ admin_user: {admin['email']} "
                f"({admin['tenant_name']}) — id={row['id']}"
            )

        # --- knn_intents INSERT (멱등: 같은 tenant_id + example_text 중복 체크) ---
        total_inserted = 0
        for tenant_name, intents in KNN_INTENTS.items():
            tenant_id = tenant_ids[tenant_name]
            for label, text in intents:
                result = await conn.execute(
                    """
                    INSERT INTO knn_intents (tenant_id, intent_label, example_text, embedding)
                    SELECT $1, $2, $3, $4::real[]
                    WHERE NOT EXISTS (
                        SELECT 1 FROM knn_intents
                        WHERE tenant_id = $1 AND example_text = $3
                    )
                    """,
                    tenant_id,
                    label,
                    text,
                    DUMMY_EMBEDDING,
                )
                if result.endswith("1"):
                    total_inserted += 1

        print(f"  ✅ knn_intents: 신규 {total_inserted}개 삽입 (더미 임베딩)")

        # --- 최종 통계 ---
        tenant_count = await conn.fetchval("SELECT COUNT(*) FROM tenants")
        intent_count = await conn.fetchval("SELECT COUNT(*) FROM knn_intents")
        admin_count = await conn.fetchval("SELECT COUNT(*) FROM admin_users")
        print(
            f"\n📊 현재 상태: tenants={tenant_count}개, "
            f"admin_users={admin_count}개, knn_intents={intent_count}개"
        )
        print("✅ PostgreSQL 시드 완료\n")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())
