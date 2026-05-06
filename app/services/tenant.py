"""테넌트 정보 헬퍼 — DB 조회 + in-process 캐시.

caller (예정):
- app/api/v1/call.py: Twilio webhook 진입 시 to_number → tenant_id (resolve_tenant_id),
                      통화 시작 인사말 송출 (get_greeting), LangGraph 호출 직전
                      state 에 tenant_name/industry 주입 (get_tenant_meta)
- scripts/graph_test.py: 동일 (resolve 는 생략 — tenant_id 하드코딩)

asyncpg per-call connection (call_repo.py 컨벤션). 통화당 1회 DB hit 후 캐시 재사용.
운영 단계에서 tenant 데이터 변경 시 stale — TTL/invalidation 은 별건.
"""
import json as _json
import re

import asyncpg

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_NAME = "고객센터"
DEFAULT_INDUSTRY = "unknown"
DEFAULT_GREETING = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"
DEFAULT_OFFHOURS_GREETING = (
    "안녕하세요, 고객센터입니다. "
    "현재 상담원 운영 시간이 아니지만 기본적인 문의는 도와드릴 수 있습니다. "
    "무엇을 도와드릴까요?"
)

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# 통화당 1회 DB 조회 후 후속 노드들이 같은 dict 공유.
_tenant_cache: dict[str, dict] = {}


async def _load_tenant(tenant_id: str) -> dict | None:
    """tenants 테이블 단일 row 로드 + 캐시.

    반환 dict: {"name": str, "industry": str, "settings": dict}
    """
    if tenant_id in _tenant_cache:
        return _tenant_cache[tenant_id]
    if not _UUID_RE.match(tenant_id):
        return None
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT name, industry, settings FROM tenants WHERE id = $1::uuid",
                tenant_id,
            )
            if row is None:
                return None
            settings_data = row["settings"] or {}
            if isinstance(settings_data, str):
                settings_data = _json.loads(settings_data)
            data = {
                "name": row["name"],
                "industry": row["industry"] or DEFAULT_INDUSTRY,
                "settings": settings_data,
            }
            _tenant_cache[tenant_id] = data
            return data
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("tenant 로드 실패 tenant_id=%s err=%s", tenant_id, e)
        return None


async def resolve_tenant_id(to_number: str) -> str:
    """Twilio To 번호(또는 SIP URI user part) → tenant UUID.

    1차: raw 값 매칭 → 2차: sip user 매칭. 미등록 시 raw 값 그대로 반환.
    캐시 안 함 (webhook 진입 시 1회만 호출됨).
    """
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                "SELECT id, name FROM tenants WHERE twilio_number = $1", to_number,
            )
            if row:
                logger.info(
                    "tenant 매칭 to=%s → id=%s name=%s",
                    to_number, row["id"], row["name"],
                )
                return str(row["id"])
            if to_number.startswith("sip:"):
                user_part = to_number.split("sip:")[1].split("@")[0]
                row = await conn.fetchrow(
                    "SELECT id, name FROM tenants WHERE twilio_number = $1", user_part,
                )
                if row:
                    logger.info(
                        "tenant 매칭 (sip user=%s) to=%s → id=%s name=%s",
                        user_part, to_number, row["id"], row["name"],
                    )
                    return str(row["id"])
        finally:
            await conn.close()
    except Exception as e:
        logger.warning("tenant 조회 실패 to=%s err=%s", to_number, e)
    logger.warning("미등록 tenant to=%s — raw 값으로 진행", to_number)
    return to_number


async def get_tenant_meta(tenant_id: str) -> tuple[str, str]:
    """(name, industry) 반환. 미등록/무효 시 (DEFAULT_NAME, DEFAULT_INDUSTRY).

    LangGraph 노드의 동적 prompt 주입용.
    """
    data = await _load_tenant(tenant_id)
    if data is None:
        return (DEFAULT_NAME, DEFAULT_INDUSTRY)
    return (data["name"], data["industry"])


async def get_greeting(tenant_id: str, within_hours: bool = True) -> str:
    """통화 시작 인사말. 우선순위:
    1. tenants.settings.greeting (또는 offhours_greeting)
    2. 미설정 시 tenants.name 으로 동적 생성
    3. 그것도 없으면 기본값 상수

    within_hours 판단은 caller 책임 (현재 시간 vs business_hours).
    """
    field = "greeting" if within_hours else "offhours_greeting"
    default = DEFAULT_GREETING if within_hours else DEFAULT_OFFHOURS_GREETING

    data = await _load_tenant(tenant_id)
    if data is None:
        return default

    msg = data["settings"].get(field)
    if msg:
        return msg

    # offhours 미설정 시 평시 greeting 폴백
    if not within_hours:
        fallback = data["settings"].get("greeting")
        if fallback:
            return fallback

    name = data["name"]
    if name:
        if within_hours:
            return f"안녕하세요, {name}입니다. 무엇을 도와드릴까요?"
        return (
            f"안녕하세요, {name}입니다. "
            f"현재 상담원 운영 시간이 아니지만 기본적인 문의는 도와드릴 수 있습니다. "
            f"무엇을 도와드릴까요?"
        )
    return default
