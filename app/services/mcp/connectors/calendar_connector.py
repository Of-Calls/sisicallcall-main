"""
Calendar MCP Connector.

지원 action_type:
  - schedule_callback     (events.insert)
  - check_availability    (events.list + 슬롯 충돌/휴무 판정 + 빈 슬롯 추천)

── tenant OAuth mode (MCP_USE_TENANT_OAUTH=true) ────────────────────────────
  google_calendar integration token을 사용해 Google Calendar API 를 호출.
  tenant_id 기준으로 TenantIntegration을 조회하고 Fernet 복호화한 access_token으로
  Authorization: Bearer 헤더를 구성한다.
  token이 없으면 skipped, API 실패 시 failed, 성공 시 success를 반환한다.
  access_token 원문은 로그에 출력하지 않는다.

── real mode env (.env 계정) ─────────────────────────────────────────────────
  CALENDAR_MCP_REAL=true   real mode 활성화 (현재 미구현 — tenant OAuth 권장)

── mock mode ─────────────────────────────────────────────────────────────────
  status: success
  external_id: calendar-mock-{call_id}
  result (schedule_callback): {scheduled, title, preferred_time, customer_phone, reason, mock}
  result (check_availability): {available=True, status="available", requested_time, suggested_slots=[]}

── Calendar API 관련 env ─────────────────────────────────────────────────────
  GOOGLE_CALENDAR_ID          캘린더 ID (기본: primary)
  CALENDAR_DEFAULT_DURATION_MIN 기본 일정 길이 분 (기본: 30)

── 영업시간 (check_availability 휴무/슬롯 판정 용) ──────────────────────────
  industry 기반 hardcode (DB 도착 전 시연 안전망). params["tenant_industry"]
  로 받아 _INDUSTRY_BUSINESS_HOURS lookup. 미매칭 industry → 평일 9-18 generic.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from app.services.mcp.connectors.base import BaseMCPConnector
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3/calendars"
_DEFAULT_TZ = "Asia/Seoul"
_DEFAULT_DURATION_MIN = 30

# (mon, tue, wed, thu, fri, sat, sun) — None 이면 휴무
_INDUSTRY_BUSINESS_HOURS: dict[str, tuple] = {
    "restaurant": (
        ("11:00", "22:00"),
        ("11:00", "22:00"),
        ("11:00", "22:00"),
        ("11:00", "22:00"),
        ("11:00", "22:00"),
        ("11:00", "22:00"),
        ("11:00", "22:00"),
    ),
    "finance": (
        ("09:00", "16:00"),
        ("09:00", "16:00"),
        ("09:00", "16:00"),
        ("09:00", "16:00"),
        ("09:00", "16:00"),
        None,
        None,
    ),
}
_GENERIC_BUSINESS_HOURS: tuple = (
    ("09:00", "18:00"),
    ("09:00", "18:00"),
    ("09:00", "18:00"),
    ("09:00", "18:00"),
    ("09:00", "18:00"),
    None,
    None,
)

# industry → 캘린더 이벤트 title suffix
_INDUSTRY_TITLE: dict[str, str] = {
    "restaurant": "예약",
    "finance": "방문 예약",
    "hospital": "진료 예약",
    "government": "민원 콜백",
    "appliance": "A/S 콜백",
    "retail": "콜백",
}
_DEFAULT_TITLE = "예약"


def _get_business_hours(tenant_industry: str, weekday: int) -> tuple | None:
    """tenant_industry + 요일 (mon=0) → (open, close) 또는 None (휴무)."""
    hours = _INDUSTRY_BUSINESS_HOURS.get(tenant_industry, _GENERIC_BUSINESS_HOURS)
    return hours[weekday]


def _default_event_title(tenant_industry: str, tenant_name: str) -> str:
    """industry 기반 default + tenant_name prefix.

    예: restaurant + 한밭식당 → "한밭식당 예약"
        finance + 시시콜콜은행 → "시시콜콜은행 방문 예약"
    """
    suffix = _INDUSTRY_TITLE.get(tenant_industry, _DEFAULT_TITLE)
    if tenant_name:
        return f"{tenant_name} {suffix}"
    return suffix


class CalendarConnector(BaseMCPConnector):
    connector_name = "calendar"
    _real_mode_env = "CALENDAR_MCP_REAL"
    _required_config = ()   # tenant OAuth 모드에서는 env config 불필요
    _oauth_provider_name = "google_calendar"

    async def execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str = "",
    ) -> dict:
        logger.info(
            "CalendarConnector call_id=%s action_type=%s real_mode=%s tenant_oauth=%s",
            call_id, action_type, self.is_real_mode(), self._use_tenant_oauth(),
        )

        # tenant OAuth 우선 처리
        if self._use_tenant_oauth() and tenant_id:
            return await self._oauth_execute(action_type, params, call_id=call_id, tenant_id=tenant_id)

        if not self.is_real_mode():
            return self._mock(params, call_id, action_type)

        # .env real mode — 미구현 (tenant OAuth 사용 권장)
        logger.warning("CalendarConnector: .env real mode 미구현 call_id=%s", call_id)
        return self._skipped("calendar_mcp_real_not_implemented")

    # ── tenant OAuth 실행 ─────────────────────────────────────────────────────

    async def _oauth_execute(
        self,
        action_type: str,
        params: dict,
        *,
        call_id: str,
        tenant_id: str,
    ) -> dict:
        from app.models.tenant_integration import IntegrationStatus
        from app.repositories.tenant_integration_repo import get_integration
        from app.services.oauth.token_crypto import decrypt_token

        integration = get_integration(tenant_id, self._oauth_provider_name)

        if integration is None or integration.status == IntegrationStatus.disconnected:
            if self._allow_env_fallback():
                return self._mock(params, call_id, action_type) if not self.is_real_mode() else self._skipped("calendar_mcp_real_not_implemented")
            return self._skipped("tenant_integration_not_connected")

        # 만료 체크 (naive UTC 비교)
        if integration.expires_at and integration.expires_at < datetime.utcnow():
            if integration.refresh_token_encrypted:
                refreshed = await self._refresh_tenant_token(integration)
                if refreshed:
                    integration = refreshed
                else:
                    return self._skipped("tenant_token_expired_refresh_failed")
            else:
                return self._skipped("tenant_token_expired_no_refresh")

        try:
            access_token = decrypt_token(integration.access_token_encrypted or "")
        except Exception:
            logger.error(
                "CalendarConnector: token 복호화 실패 call_id=%s tenant_id=%s",
                call_id, tenant_id,
            )
            return self._failed("tenant_token_decryption_failed")

        if action_type == "check_availability":
            return await self._check_availability(access_token, params, call_id=call_id)
        return await self._insert_google_event(access_token, params, call_id=call_id)

    async def _insert_google_event(
        self,
        access_token: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        """Google Calendar events.insert API 호출.

        access_token은 로그에 출력하지 않는다.
        """
        import httpx

        calendar_id = (
            params.get("calendar_id")
            or os.getenv("GOOGLE_CALENDAR_ID", "primary")
        )
        url = f"{_CALENDAR_API_BASE}/{calendar_id}/events"
        event_body = self._build_event_body(params)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=event_body,
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=15.0,
                )

            if resp.status_code not in (200, 201):
                preview = resp.text[:200]
                logger.error(
                    "CalendarConnector: API 오류 call_id=%s status=%d preview=%s",
                    call_id, resp.status_code, preview,
                )
                return self._failed(f"google_calendar_api_error:{resp.status_code}")

            data = resp.json()
            logger.info(
                "CalendarConnector: 일정 생성 완료 call_id=%s event_id=%s",
                call_id, data.get("id"),
            )
            return self._success(
                external_id=data.get("id"),
                result={
                    "event_id": data.get("id"),
                    "html_link": data.get("htmlLink"),
                    "start": (data.get("start") or {}).get("dateTime"),
                    "end": (data.get("end") or {}).get("dateTime"),
                },
            )

        except Exception as exc:
            logger.error(
                "CalendarConnector: events.insert 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"calendar_insert_exception:{type(exc).__name__}")

    # ── 이벤트 바디 생성 ──────────────────────────────────────────────────────

    def _build_event_body(self, params: dict) -> dict:
        title = params.get("title") or _default_event_title(
            params.get("tenant_industry", ""), params.get("tenant_name", "")
        )

        description = ""
        for key in ("description", "reason", "callback_reason", "summary_short"):
            val = params.get(key)
            if val:
                description = str(val)
                break

        tz = params.get("timezone", _DEFAULT_TZ)
        duration = int(os.getenv("CALENDAR_DEFAULT_DURATION_MIN", str(_DEFAULT_DURATION_MIN)))

        start_str = params.get("start_time") or params.get("preferred_time")
        end_str = params.get("end_time")

        if start_str:
            try:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                start_dt = datetime.utcnow() + timedelta(hours=1)
        else:
            start_dt = datetime.utcnow() + timedelta(hours=1)

        if end_str:
            try:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                end_dt = start_dt + timedelta(minutes=duration)
        else:
            end_dt = start_dt + timedelta(minutes=duration)

        return {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": tz},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": tz},
        }

    # ── check_availability ───────────────────────────────────────────────────

    async def _check_availability(
        self,
        access_token: str,
        params: dict,
        *,
        call_id: str,
    ) -> dict:
        """예약 가능 여부 확인 + 충돌/휴무 시 같은 날 (또는 가까운 영업일) 빈 슬롯 추천.

        params:
          preferred_time   "YYYY-MM-DD HH:MM" 또는 ISO format
          tenant_industry  영업시간 lookup 용 (없으면 generic 평일 9-18)

        result:
          available           True/False
          status              "available" | "conflict" | "closed_day"
          requested_time      입력 그대로 (음성 응답 조립 용)
          suggested_slots     ["YYYY-MM-DD HH:MM", ...] (충돌/휴무 시 빈 슬롯 추천)
        """
        import httpx

        preferred_str = (params.get("preferred_time") or "").strip()
        tenant_industry = params.get("tenant_industry") or ""

        if not preferred_str:
            return self._failed("missing_preferred_time")

        try:
            requested_dt = datetime.fromisoformat(preferred_str.replace(" ", "T"))
        except (ValueError, AttributeError):
            return self._failed(f"invalid_preferred_time:{preferred_str}")
        if requested_dt.tzinfo:
            requested_dt = requested_dt.replace(tzinfo=None)

        weekday = requested_dt.weekday()
        business_hours = _get_business_hours(tenant_industry, weekday)

        # 휴무일 → events.list 생략, 가까운 영업일 추천
        if business_hours is None:
            next_open = self._find_next_open_day(requested_dt, tenant_industry)
            suggestions = [self._format_iso(next_open)] if next_open else []
            logger.info(
                "CalendarConnector: closed_day call_id=%s requested=%s industry=%s suggest=%s",
                call_id, preferred_str, tenant_industry, suggestions,
            )
            return self._success(
                external_id=f"calendar-check-{call_id}",
                result={
                    "available": False,
                    "status": "closed_day",
                    "requested_time": preferred_str,
                    "suggested_slots": suggestions,
                },
            )

        # events.list 로 그날 일정 조회
        day_start = requested_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        calendar_id = (
            params.get("calendar_id")
            or os.getenv("GOOGLE_CALENDAR_ID", "primary")
        )
        url = f"{_CALENDAR_API_BASE}/{calendar_id}/events"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={
                        "timeMin": day_start.isoformat() + "+09:00",
                        "timeMax": day_end.isoformat() + "+09:00",
                        "singleEvents": "true",
                        "orderBy": "startTime",
                        "maxResults": "100",
                    },
                    timeout=15.0,
                )
            if resp.status_code != 200:
                preview = resp.text[:200]
                logger.error(
                    "CalendarConnector: events.list 오류 call_id=%s status=%d preview=%s",
                    call_id, resp.status_code, preview,
                )
                return self._failed(f"google_calendar_api_error:{resp.status_code}")
            events = resp.json().get("items", []) or []
        except Exception as exc:
            logger.error(
                "CalendarConnector: events.list 예외 call_id=%s err=%s",
                call_id, type(exc).__name__,
            )
            return self._failed(f"calendar_list_exception:{type(exc).__name__}")

        # busy 인터벌 추출 (naive 비교)
        busy: list[tuple[datetime, datetime]] = []
        for ev in events:
            start_str = (ev.get("start") or {}).get("dateTime")
            end_str = (ev.get("end") or {}).get("dateTime")
            if not start_str or not end_str:
                continue
            try:
                ev_start = datetime.fromisoformat(start_str)
                ev_end = datetime.fromisoformat(end_str)
            except (ValueError, AttributeError):
                continue
            if ev_start.tzinfo:
                ev_start = ev_start.replace(tzinfo=None)
            if ev_end.tzinfo:
                ev_end = ev_end.replace(tzinfo=None)
            busy.append((ev_start, ev_end))

        duration_min = int(os.getenv("CALENDAR_DEFAULT_DURATION_MIN", str(_DEFAULT_DURATION_MIN)))
        requested_end = requested_dt + timedelta(minutes=duration_min)

        # 요청 슬롯 충돌 = 어떤 busy 와 시간 겹치면
        has_conflict = any(
            not (requested_end <= b_start or requested_dt >= b_end)
            for b_start, b_end in busy
        )

        if not has_conflict:
            logger.info(
                "CalendarConnector: available call_id=%s requested=%s",
                call_id, preferred_str,
            )
            return self._success(
                external_id=f"calendar-check-{call_id}",
                result={
                    "available": True,
                    "status": "available",
                    "requested_time": preferred_str,
                    "suggested_slots": [],
                },
            )

        # 충돌 → 같은 날 빈 슬롯 검색
        open_str, close_str = business_hours
        open_h, open_m = map(int, open_str.split(":"))
        close_h, close_m = map(int, close_str.split(":"))
        open_dt = day_start.replace(hour=open_h, minute=open_m)
        close_dt = day_start.replace(hour=close_h, minute=close_m)
        free_slots = self._find_free_slots(
            open_dt, close_dt, busy, requested_dt, duration_min
        )
        suggestions = [self._format_iso(s) for s in free_slots[:1]]
        logger.info(
            "CalendarConnector: conflict call_id=%s requested=%s suggest=%s",
            call_id, preferred_str, suggestions,
        )
        return self._success(
            external_id=f"calendar-check-{call_id}",
            result={
                "available": False,
                "status": "conflict",
                "requested_time": preferred_str,
                "suggested_slots": suggestions,
            },
        )

    def _find_free_slots(
        self,
        open_dt: datetime,
        close_dt: datetime,
        busy: list,
        requested: datetime,
        duration_min: int,
    ) -> list[datetime]:
        """30분 step 슬롯 중 빈 곳, 요청 시간과의 거리로 정렬."""
        candidates: list[datetime] = []
        cur = open_dt
        step = timedelta(minutes=30)
        dur = timedelta(minutes=duration_min)
        while cur + dur <= close_dt:
            slot_end = cur + dur
            conflict = any(
                not (slot_end <= b_start or cur >= b_end)
                for b_start, b_end in busy
            )
            if not conflict:
                candidates.append(cur)
            cur += step
        candidates.sort(key=lambda dt: abs((dt - requested).total_seconds()))
        return candidates

    def _find_next_open_day(
        self, requested: datetime, tenant_industry: str
    ) -> datetime | None:
        """요청일 휴무 시 가까운 영업일의 영업 시작 시각 반환 (최대 7일 이내)."""
        for offset in range(1, 8):
            cand = requested + timedelta(days=offset)
            bh = _get_business_hours(tenant_industry, cand.weekday())
            if bh is not None:
                open_str, _ = bh
                h, m = map(int, open_str.split(":"))
                return cand.replace(hour=h, minute=m, second=0, microsecond=0)
        return None

    @staticmethod
    def _format_iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M")

    # ── mock ─────────────────────────────────────────────────────────────────

    def _mock(self, params: dict, call_id: str, action_type: str = "schedule_callback") -> dict:
        if action_type == "check_availability":
            return self._success(
                external_id=f"calendar-mock-check-{call_id}",
                result={
                    "available": True,
                    "status": "available",
                    "requested_time": params.get("preferred_time", ""),
                    "suggested_slots": [],
                    "mock": True,
                },
            )
        external_id = f"calendar-mock-{call_id}"
        return self._success(
            external_id=external_id,
            result={
                "scheduled": True,
                "event_id": external_id,
                "title": params.get("title", "콜백 예약"),
                "preferred_time": params.get("preferred_time"),
                "customer_phone": params.get("customer_phone"),
                "reason": params.get("callback_reason", ""),
                "mock": True,
            },
        )
