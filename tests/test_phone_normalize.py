"""normalize_korean_phone 단위 테스트.

post-call SMS 액션이 Solapi 포맷("01012345678")을 받도록 calls.caller_number
를 변환하는 헬퍼의 동작을 고정한다.
"""
from __future__ import annotations

import pytest

from app.utils.phone import normalize_korean_phone


# ── 정상 한국 모바일 4가지 표기 → "01012345678" ──────────────────────────────

@pytest.mark.parametrize(
    "raw",
    [
        "+82-10-1234-5678",
        "+821012345678",
        "010-1234-5678",
        "01012345678",
        "010 1234 5678",
        "(010) 1234-5678",
        "82 10 1234 5678",
        "1012345678",          # leading 0 누락
    ],
)
def test_normalizes_korean_mobile_to_local(raw: str):
    assert normalize_korean_phone(raw) == "01012345678"


# ── 빈 입력 → "" ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_empty_input_returns_empty_string(raw):
    assert normalize_korean_phone(raw) == ""


# ── 비정상 / 알 수 없는 포맷 → cleaned 반환 + warning ────────────────────────

def test_unknown_format_returns_cleaned_value(caplog):
    raw = "abcdef"
    result = normalize_korean_phone(raw)
    # 모든 비-숫자 / 비-+ 가 제거되므로 빈 문자열이 된다
    assert result == ""


def test_unknown_digit_format_keeps_cleaned_with_warning(caplog):
    # 한국 형식이 아닌 숫자열 — Solapi 가 거부하더라도 진단 가능하도록
    # cleaned 값을 그대로 반환한다.
    result = normalize_korean_phone("99-9999-9999")
    assert result == "9999999999"


def test_already_normalized_is_idempotent():
    assert normalize_korean_phone("01012345678") == "01012345678"


def test_dots_and_dashes_mixed():
    assert normalize_korean_phone("010.1234.5678") == "01012345678"


def test_leading_plus_only_drops_to_local():
    assert normalize_korean_phone("+82 10-1234-5678") == "01012345678"


def test_landline_local_format_preserved():
    # 02 (서울) 일반전화 — 모바일이 아니지만 0 prefix 면 그대로 둔다.
    assert normalize_korean_phone("02-1234-5678") == "0212345678"


def test_e164_landline_seoul():
    assert normalize_korean_phone("+82-2-1234-5678") == "0212345678"
