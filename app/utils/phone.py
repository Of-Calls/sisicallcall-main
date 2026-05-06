"""한국 전화번호 정규화 헬퍼.

post-call SMS 액션이 Solapi로 발송될 때 일관된 형식("01012345678")을
기대하므로, calls.caller_number 처럼 다양한 포맷으로 들어오는 입력을
하나의 로컬 형식으로 변환한다.

지원 입력:
  +82-10-1234-5678    →  01012345678
  +821012345678       →  01012345678
  010-1234-5678       →  01012345678
  010 1234 5678       →  01012345678
  01012345678         →  01012345678

빈 문자열 / None / 공백 → "" 반환 (호출 측이 매핑 안 함을 결정).
한국 형식이 아니면 cleaned 결과를 그대로 반환하고 warning 을 남긴다 —
Solapi 가 거부하더라도 진단할 수 있도록 raw 값은 가공해 두지 않는다.
"""
from __future__ import annotations

import re

from app.utils.logger import get_logger

logger = get_logger(__name__)

# 숫자와 leading '+' 만 남긴다. ' ', '-', '.', '(', ')' 등은 제거.
_NON_PHONE_CHAR_RE = re.compile(r"[^\d+]")


def normalize_korean_phone(raw: str | None) -> str:
    """다양한 한국 전화번호 표기를 Solapi 권장 로컬 형식으로 변환한다.

    빈 입력은 빈 문자열을 반환한다. 매핑 불가능한 형식이면 cleaned 결과를
    반환하고 warning 을 남긴다.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""

    # 비-숫자/비-+ 문자 제거. leading '+' 만 의미 있고 그 외 '+' 는 무시한다.
    cleaned = _NON_PHONE_CHAR_RE.sub("", s)
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    cleaned = cleaned.replace("+", "")

    if not cleaned:
        return ""

    # +82 / 82 prefix → 0
    if cleaned.startswith("82"):
        return "0" + cleaned[2:]

    # 이미 로컬 형식 (010..., 011..., 02... 등)
    if cleaned.startswith("0"):
        return cleaned

    # 모바일 leading '0' 누락 (1012345678 → 01012345678)
    if cleaned.startswith("1") and len(cleaned) == 10:
        return "0" + cleaned

    logger.warning(
        "normalize_korean_phone: 한국 형식으로 인식 못함 raw=%r cleaned=%s",
        raw, cleaned,
    )
    return cleaned
