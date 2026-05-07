"""한국어 상대 시간 표현 → 절대 datetime 변환.

LLM (gpt-4o-mini) 의 한국어 요일/주 산수가 비결정적 (같은 발화가 가끔 정답
가끔 틀림) — 시연 안정성을 위해 핵심 표현은 코드로 deterministic 변환.

지원 패턴 (시연 자주 쓰는 발화 위주):
- 날짜: 오늘 / 내일 / 모레 / 글피
        이번주 X요일 / 다음주 X요일 / 다다음주 X요일
- 시간: (오전|오후|저녁|아침|점심|새벽)?\\s*N시 (M분)?
- 둘 다 매칭되어야 datetime 생성. 매칭 실패 시 None — LLM 결과로 fallback.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_WEEKDAY_NUM = {
    "월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6,
}
_WEEKDAY_KO = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

_RELATIVE_DAY_RE = re.compile(r"(오늘|내일|모레|글피)")
_WEEK_WEEKDAY_RE = re.compile(
    r"(이번\s*주|다음\s*주|다다음\s*주)\s*([월화수목금토일])\s*요일"
)
_TIME_RE = re.compile(
    r"(오전|오후|저녁|아침|점심|새벽)?\s*(\d{1,2})\s*시\s*(?:(\d{1,2})\s*분)?"
)


def _resolve_relative_day(keyword: str, today: date) -> date:
    return today + timedelta(
        days={"오늘": 0, "내일": 1, "모레": 2, "글피": 3}[keyword]
    )


def _resolve_week_weekday(week_kw: str | None, target_wd: int, today: date) -> date:
    """주 키워드 + 요일 → 절대 날짜."""
    today_wd = today.weekday()
    normalized = week_kw.replace(" ", "") if week_kw else "이번주"

    if normalized == "이번주":
        if target_wd >= today_wd:
            return today + timedelta(days=target_wd - today_wd)
        return today + timedelta(days=7 - today_wd + target_wd)
    if normalized == "다음주":
        next_monday = today + timedelta(days=7 - today_wd)
        return next_monday + timedelta(days=target_wd)
    if normalized == "다다음주":
        next_next_monday = today + timedelta(days=14 - today_wd)
        return next_next_monday + timedelta(days=target_wd)
    return today + timedelta(days=target_wd - today_wd)


def _resolve_hour(qual: str | None, hour: int) -> int:
    """오전/오후/저녁/등 + N시 → 24h hour.

    경계 처리:
    - 오전 12시 = 0 (자정), 오후 12시 = 12 (정오)
    - 점심 12시 = 12, 점심 N시 (1~3) = N+12
    - 단순 "N시" (qual 없음): hour 그대로 (사용자 의도 모호하면 LLM 의존)
    """
    if qual is None:
        return hour
    if qual == "오전":
        return 0 if hour == 12 else hour
    if qual == "오후":
        if hour == 12:
            return 12
        return hour + 12 if 1 <= hour <= 11 else hour
    if qual == "저녁":
        return hour + 12 if 5 <= hour <= 11 else hour
    if qual == "아침":
        return hour
    if qual == "점심":
        if hour == 12:
            return 12
        return hour + 12 if 1 <= hour <= 3 else hour
    if qual == "새벽":
        return hour
    return hour


def extract_absolute_datetime(
    text: str, today: date | None = None
) -> datetime | None:
    """한국어 상대 시간 표현에서 절대 datetime 추출.

    날짜 + 시간 둘 다 매칭되어야 datetime 반환. 부분 매칭은 None.
    """
    if today is None:
        today = date.today()

    target_date: date | None = None
    weekday_m = _WEEK_WEEKDAY_RE.search(text)
    if weekday_m:
        target_date = _resolve_week_weekday(
            weekday_m.group(1), _WEEKDAY_NUM[weekday_m.group(2)], today
        )
    else:
        rel_m = _RELATIVE_DAY_RE.search(text)
        if rel_m:
            target_date = _resolve_relative_day(rel_m.group(1), today)

    if target_date is None:
        return None

    time_m = _TIME_RE.search(text)
    if not time_m:
        return None

    qual = time_m.group(1)
    hour = int(time_m.group(2))
    minute = int(time_m.group(3)) if time_m.group(3) else 0
    if not (0 <= hour <= 24 and 0 <= minute <= 59):
        return None
    hour = _resolve_hour(qual, hour)
    if not 0 <= hour <= 23:
        return None

    return datetime(target_date.year, target_date.month, target_date.day, hour, minute)


def _hour_label(hour: int) -> tuple[str, int]:
    """24h → (한국어 라벨, 12h-style hour).

    0 → 오전 12 (자정 표현)
    1~5 → 새벽 N
    6~11 → 오전 N
    12 → 오후 12 (정오)
    13~17 → 오후 (N-12)
    18~23 → 저녁 (N-12)
    """
    if hour == 0:
        return ("오전", 12)
    if 1 <= hour <= 5:
        return ("새벽", hour)
    if 6 <= hour <= 11:
        return ("오전", hour)
    if hour == 12:
        return ("오후", 12)
    if 13 <= hour <= 17:
        return ("오후", hour - 12)
    if 18 <= hour <= 23:
        return ("저녁", hour - 12)
    return ("오전", hour)


def format_korean_friendly(dt: datetime) -> str:
    """datetime → '5월 10일 일요일 오후 4시' / '5월 10일 일요일 오후 4시 30분'."""
    weekday = _WEEKDAY_KO[dt.weekday()]
    label, hour12 = _hour_label(dt.hour)
    minute_part = f" {dt.minute}분" if dt.minute else ""
    return f"{dt.month}월 {dt.day}일 {weekday} {label} {hour12}시{minute_part}"


def format_iso(dt: datetime) -> str:
    """datetime → 'YYYY-MM-DD HH:MM' — task_branch args 형식."""
    return dt.strftime("%Y-%m-%d %H:%M")
