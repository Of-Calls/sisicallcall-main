"""한국어 시간 변환 모듈 단위 테스트.

기준일 = 2026-05-07 (목요일) — 시연 시나리오와 같은 시점.
"""
from datetime import date, datetime

import pytest

from app.utils.korean_time import (
    extract_absolute_datetime,
    format_iso,
    format_korean_friendly,
)

# 기준일: 2026-05-07 (목)
TODAY = date(2026, 5, 7)


@pytest.mark.parametrize(
    "text, expected",
    [
        # --- 상대 일자 ---
        ("오늘 저녁 7시", datetime(2026, 5, 7, 19, 0)),
        ("내일 오후 8시", datetime(2026, 5, 8, 20, 0)),
        ("모레 점심 12시", datetime(2026, 5, 9, 12, 0)),
        ("글피 오전 10시", datetime(2026, 5, 10, 10, 0)),

        # --- 이번주 X요일 ---
        # 오늘=목 → 이번주 목요일 = 오늘
        ("이번주 목요일 오후 5시", datetime(2026, 5, 7, 17, 0)),
        # 오늘 이후
        ("이번주 금요일 오후 3시", datetime(2026, 5, 8, 15, 0)),
        ("이번주 토요일 저녁 7시", datetime(2026, 5, 9, 19, 0)),
        ("이번주 일요일 오후 4시", datetime(2026, 5, 10, 16, 0)),
        # 오늘 이전 → 다음주 X요일 fallback
        ("이번주 화요일 오후 2시", datetime(2026, 5, 12, 14, 0)),
        ("이번주 월요일 오전 9시", datetime(2026, 5, 11, 9, 0)),

        # --- 다음주 X요일 ---
        # 다음주 = 5/11(월) ~ 5/17(일)
        ("다음주 월요일 오후 1시", datetime(2026, 5, 11, 13, 0)),
        ("다음주 화요일 오전 11시", datetime(2026, 5, 12, 11, 0)),
        ("다음주 목요일 저녁 7시", datetime(2026, 5, 14, 19, 0)),
        ("다음주 일요일 오후 4시", datetime(2026, 5, 17, 16, 0)),

        # --- 다다음주 X요일 ---
        # 다다음주 = 5/18(월) ~ 5/24(일)
        ("다다음주 수요일 오후 3시", datetime(2026, 5, 20, 15, 0)),

        # --- 시간 변형 ---
        ("내일 오전 9시 30분", datetime(2026, 5, 8, 9, 30)),
        ("내일 새벽 5시", datetime(2026, 5, 8, 5, 0)),
        ("내일 아침 8시", datetime(2026, 5, 8, 8, 0)),
        ("내일 점심 12시", datetime(2026, 5, 8, 12, 0)),
        ("내일 점심 1시", datetime(2026, 5, 8, 13, 0)),
        ("내일 오후 12시", datetime(2026, 5, 8, 12, 0)),
        ("내일 오전 12시", datetime(2026, 5, 8, 0, 0)),

        # --- 발화 변형 (자연 한국어) ---
        ("이번주토요일저녁7시예약해주세요", datetime(2026, 5, 9, 19, 0)),
        ("내일오후8시4명예약해주세요", datetime(2026, 5, 8, 20, 0)),
        ("다음주 화요일 오전 11시에 2명 예약해주세요", datetime(2026, 5, 12, 11, 0)),
    ],
)
def test_extract_absolute_datetime_success(text, expected):
    assert extract_absolute_datetime(text, TODAY) == expected


@pytest.mark.parametrize(
    "text",
    [
        "예약하고 싶어요",  # 시간 표현 없음
        "다음주에 예약",  # 요일 없음
        "오후 7시",  # 날짜 없음
        "내일 예약",  # 시간 없음
        "5월 14일 화요일 오전 11시",  # 절대 날짜 — 코드 cover X (LLM 의존)
    ],
)
def test_extract_absolute_datetime_no_match(text):
    assert extract_absolute_datetime(text, TODAY) is None


@pytest.mark.parametrize(
    "dt, expected",
    [
        (datetime(2026, 5, 9, 19, 0), "5월 9일 토요일 저녁 7시"),
        (datetime(2026, 5, 10, 16, 0), "5월 10일 일요일 오후 4시"),
        (datetime(2026, 5, 7, 12, 0), "5월 7일 목요일 오후 12시"),
        (datetime(2026, 5, 8, 0, 0), "5월 8일 금요일 오전 12시"),
        (datetime(2026, 5, 8, 9, 30), "5월 8일 금요일 오전 9시 30분"),
        (datetime(2026, 5, 8, 5, 0), "5월 8일 금요일 새벽 5시"),
        (datetime(2026, 5, 14, 19, 0), "5월 14일 목요일 저녁 7시"),
    ],
)
def test_format_korean_friendly(dt, expected):
    assert format_korean_friendly(dt) == expected


def test_format_iso():
    assert format_iso(datetime(2026, 5, 9, 19, 0)) == "2026-05-09 19:00"
    assert format_iso(datetime(2026, 5, 8, 9, 30)) == "2026-05-08 09:30"
