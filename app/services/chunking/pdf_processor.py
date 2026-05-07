"""PDF 청킹 파이프라인 — opendataloader-pdf JSON tree 기반.

청킹 전략:
    1. opendataloader-pdf JSON 변환 (heading level / table row-cell / list item 구조 보존)
    2. heading 단위 section 분할 + table 단독 청크 (가운데 split 절대 X)
    3. section 본문 polish (한국어 자연어화 → 임베딩 친화, 숫자 보존)
    4. table 본문 LLM 자연어화 (모든 셀 정보 보존, 친근체)
    5. chunk 별 LLM 메타 (title/summary/keywords/topic)
    6. tenant 가용 카테고리 5~7개 정제 → Redis
"""
import asyncio
import json
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import asyncpg

from app.repositories.rag_document_repo import upsert_rag_document_chunks_for_tenant
from app.services.embedding.base import BaseEmbeddingService
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.services.session.redis_session import RedisSessionService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 짧은 청크 merge 임계 / paragraph 누적 flush 기준.
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 800

_JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)
_NUMBER_RE = re.compile(r'\d+')

# polish 결과 silent 압축/숫자 누락 방어 — 위반 시 원본 chunk fallback.
_MIN_POLISH_RATIO = 0.7


# ── PDF → JSON 변환 ──────────────────────────────────────────────

def _extract_json(pdf_path: str) -> dict:
    """opendataloader-pdf 로 PDF → JSON tree."""
    from opendataloader_pdf import convert
    with tempfile.TemporaryDirectory(prefix="odl_") as tmp:
        convert(
            input_path=pdf_path,
            output_dir=tmp,
            format="json",
            use_struct_tree=True,
            table_method="cluster",
            quiet=True,
        )
        json_files = list(Path(tmp).rglob("*.json"))
        if not json_files:
            raise RuntimeError("opendataloader produced no JSON output")
        raw = json_files[0].read_text(encoding="utf-8")
    data = json.loads(raw)
    logger.info(
        "pdf parsed by opendataloader pages=%d path=%s",
        data.get("number of pages", 0), pdf_path,
    )
    return data


# ── JSON tree → flat element ──────────────────────────────────────

@dataclass
class FlatElement:
    type: str  # heading / paragraph / list / table
    heading_level: Optional[int]
    page: int
    bbox: Optional[list[float]]
    text: str
    raw: dict = field(default_factory=dict)


def _cell_text(cell: dict) -> str:
    """table cell 의 kids paragraph content 평탄화."""
    parts: list[str] = []
    for k in cell.get("kids") or []:
        if isinstance(k, dict):
            content = k.get("content")
            if content:
                parts.append(content.strip())
    return " ".join(parts).strip()


def _table_text_dump(raw_table: dict) -> str:
    """표 row/cell raw text → ` | ` 구분 다중 행 텍스트 (자연어화 LLM 입력용)."""
    lines: list[str] = []
    for row in raw_table.get("rows") or []:
        cells = row.get("cells") or []
        cell_texts = [_cell_text(c) for c in cells]
        lines.append(" | ".join(cell_texts))
    return "\n".join(lines)


def _flatten_json(data: dict) -> list[FlatElement]:
    """opendataloader JSON kids 트리를 평탄한 list 로 (heading/paragraph/list/table 채택)."""
    out: list[FlatElement] = []
    for kid in data.get("kids") or []:
        if not isinstance(kid, dict):
            continue
        kt = kid.get("type")
        page = int(kid.get("page number") or 0)
        bbox = kid.get("bounding box")

        if kt == "heading":
            text = (kid.get("content") or "").strip()
            if not text:
                continue
            out.append(FlatElement(
                type="heading",
                heading_level=kid.get("heading level"),
                page=page, bbox=bbox, text=text, raw=kid,
            ))
        elif kt == "paragraph":
            text = (kid.get("content") or "").strip()
            if not text:
                continue
            out.append(FlatElement(
                type="paragraph",
                heading_level=None,
                page=page, bbox=bbox, text=text, raw=kid,
            ))
        elif kt == "list":
            items = kid.get("list items") or []
            item_texts: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                content = (item.get("content") or "").strip()
                if content:
                    item_texts.append("- " + content)
            if not item_texts:
                continue
            out.append(FlatElement(
                type="list",
                heading_level=None,
                page=page, bbox=bbox,
                text="\n".join(item_texts), raw=kid,
            ))
        elif kt == "table":
            text_dump = _table_text_dump(kid)
            if not text_dump.strip():
                continue
            out.append(FlatElement(
                type="table",
                heading_level=None,
                page=page, bbox=bbox,
                text=text_dump, raw=kid,
            ))
    return out


# ── flat list → Chunk 그룹화 ──────────────────────────────────────

@dataclass
class Chunk:
    heading: str            # 직전 가까운 heading content (없으면 빈)
    chunk_type: str         # section / table
    text: str               # 청크 본문 (그룹화 시점은 raw, polish 후 자연어로 교체)
    page: int               # 시작 페이지
    bbox: Optional[list[float]]
    raw_table: Optional[dict] = None  # table 청크만 — 자연어화 LLM 입력용
    heading_path: list[str] = field(default_factory=list)  # L2 이상 hierarchy (L1 은 거의 모든 청크 공통이라 제외)


def _group_into_chunks(elements: list[FlatElement]) -> list[Chunk]:
    """flat element list → 의미 단위 Chunk list.

    규칙:
      - heading 만나면 누적 section flush + heading stack 갱신 (heading 자체는 본문에 X)
      - table 만나면 누적 flush 후 단독 청크 (절대 split X)
      - paragraph/list 누적, MAX_CHUNK_CHARS 초과 시 자동 flush
      - chunk 마다 현재 heading hierarchy snapshot — L1 제외 L2 이상만 path 로
    """
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text), L1 부터 가장 깊은 level 까지
    current_section: list[FlatElement] = []

    def current_heading() -> str:
        return heading_stack[-1][1] if heading_stack else ""

    def current_path() -> list[str]:
        return [t for lv, t in heading_stack if lv >= 2]

    def flush_section():
        nonlocal current_section
        if not current_section:
            return
        text = "\n\n".join(e.text for e in current_section if e.text).strip()
        if not text:
            current_section = []
            return
        chunks.append(Chunk(
            heading=current_heading(),
            heading_path=current_path(),
            chunk_type="section",
            text=text,
            page=current_section[0].page,
            bbox=current_section[0].bbox,
            raw_table=None,
        ))
        current_section = []

    for elem in elements:
        if elem.type == "heading":
            flush_section()
            level = elem.heading_level if elem.heading_level is not None else 99
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, elem.text))
        elif elem.type == "table":
            flush_section()
            chunks.append(Chunk(
                heading=current_heading(),
                heading_path=current_path(),
                chunk_type="table",
                text=elem.text,
                page=elem.page,
                bbox=elem.bbox,
                raw_table=elem.raw,
            ))
        elif elem.type in ("paragraph", "list"):
            current_section.append(elem)
            running = sum(len(e.text) for e in current_section)
            if running >= MAX_CHUNK_CHARS:
                flush_section()
    flush_section()
    return chunks


def _merge_short(chunks: list[Chunk]) -> list[Chunk]:
    """짧은 section 청크를 직전 section 과 merge. table 은 절대 X."""
    merged: list[Chunk] = []
    for c in chunks:
        if c.chunk_type != "section":
            merged.append(c)
            continue
        if (
            merged
            and merged[-1].chunk_type == "section"
            and len(c.text) < MIN_CHUNK_CHARS
        ):
            prev = merged[-1]
            merged[-1] = Chunk(
                heading=prev.heading,
                heading_path=prev.heading_path,
                chunk_type="section",
                text=prev.text + "\n\n" + c.text,
                page=prev.page,
                bbox=prev.bbox,
                raw_table=None,
            )
        else:
            merged.append(c)
    return merged


# ── 표 자연어화 LLM ──────────────────────────────────────────────

_TABLE_NATURALIZE_SYSTEM_PROMPT = """당신은 PDF 표를 한국어 자연 문장으로 풀어 쓰는 변환기입니다.

입력: heading + 표의 행/셀 (한 행은 ` | ` 로 구분).
출력: heading 맥락에서 모든 셀 정보를 보존한 자연스러운 한국어 단락 (1~3문단).

규칙:
- 표 첫 행을 컬럼명으로 인식, 이후 행을 그 의미에 맞춰 풀어 쓴다.
- 모든 셀 값 그대로 보존 (숫자/시간/금액 변형·축약 금지).
- "~입니다" 대신 "~이에요/있어요" 친근체 (음성 안내용).
- 시간은 "11시 30분" 형식, 콜론 ":" / 물결 "~" / 하이픈 "-" 사용 금지.
- 시간 범위는 "11시 30분부터 22시까지" 형식.
- 금액은 "55,000원" 그대로.
- 출력은 자연어 본문만. JSON / 머릿말 / 따옴표 / 코드블록 금지.

예시 입력:
[heading]
요일별 영업시간

[표 본문]
구분 | 점심 영업 | 브레이크 타임 | 저녁 영업 | 라스트 오더
평일 화 금 | 11:30 ~ 15:00 | 15:00 ~ 17:30 | 17:30 ~ 22:00 | 21:00
토요일 | 11:30 ~ 15:30 | 15:30 ~ 17:30 | 17:30 ~ 22:00 | 21:00

예시 출력:
요일별 영업시간이에요. 평일 화요일부터 금요일까지는 점심 영업이 11시 30분부터 15시까지 운영되고, 브레이크 타임은 15시부터 17시 30분까지예요. 저녁 영업은 17시 30분부터 22시까지 이어지고 라스트 오더는 21시예요. 토요일은 점심 영업이 11시 30분부터 15시 30분까지로 30분 길어지고 나머지 시간대는 평일과 같아요."""


async def _naturalize_table(heading: str, raw_dump: str, llm: BaseLLMService) -> str:
    """표 row/cell raw dump → 친근체 자연어 단락. 실패 시 raw dump 그대로 fallback."""
    user_msg = f"[heading]\n{heading or '(헤더 없음)'}\n\n[표 본문]\n{raw_dump}"
    try:
        text = await llm.generate(
            system_prompt=_TABLE_NATURALIZE_SYSTEM_PROMPT,
            user_message=user_msg,
            temperature=0.1,
            max_tokens=700,
        )
    except Exception as e:
        logger.error("table naturalize failed: %s — fallback to raw dump", e)
        return raw_dump
    text = (text or "").strip().strip('"').strip("'")
    return text or raw_dump


# ── section polish (samsong 시드도 import 해 사용) ───────────────────

def _validate_polish(orig: str, polished: str) -> tuple[bool, str]:
    """polish 결과가 원본 정보를 보존했는지 검증. (ok, reason)."""
    if len(polished) < len(orig) * _MIN_POLISH_RATIO:
        ratio = len(polished) / max(len(orig), 1)
        return False, f"shrink ratio={ratio:.2f}"
    orig_nums = _NUMBER_RE.findall(orig)
    missing = [n for n in orig_nums if n not in polished]
    if missing:
        return False, f"missing digits={missing[:5]}"
    return True, ""


_CHUNK_POLISH_SYSTEM_PROMPT = """당신은 RAG 임베딩용 청크 정제기입니다.
입력: PDF 추출 텍스트 청크 N 개 (헤더/리스트/줄바꿈 raw 포함).
출력: JSON 배열, 원소 N 개 — 각 원소는 정제된 자연어 본문 (string, plain Korean).

목적: 임베딩이 짧은 음성 질문 (예: "메뉴가 뭐가 있어요", "주차 가능?") 와 매칭
정확도를 높이도록 chunk 본문을 자연어 형태로 다듬는 것.

정제 규칙:
- 마크다운 마커 제거: ##, ###, ####, ■, ▶, ☑, ※, **, `, 리스트 -.
- 줄바꿈 \\n 제거 → 자연스러운 단락 / 마침표로 연결.
- "~입니다" 격식체 대신 "~이에요/있어요" 친근체 (음성 안내용).
- 시간은 "11시 30분", 시간 범위는 "11시 30분부터 22시까지" 형식.
- 금액은 "55,000원" 그대로.

★ 절대 규칙 (위반 시 후처리에서 원본으로 자동 폴백):
- 입력에 있는 모든 사실/정보 보존. 재요약·압축 금지. 표현만 자연어로 변환.
- 출력 길이는 입력 길이의 70% 이상 유지 (마크다운 마커 제거로 약간 짧아지는 정도만 허용).
- 숫자 (가격, 시간, 전화번호, 주소, 면적, 인원 수 등) 는 입력에 등장한 모든 숫자열을
  출력에 한 글자도 변형 없이 그대로 포함. 누락 시 원본으로 폴백 처리됨.
- 입력에 없는 정보 추측·추가 금지.
- 같은 의미의 chunk 가 N 개 중 여러 개여도 각각 독립 정제. 합치지 마라.

출력 형식: JSON 배열 [\"정제본1\", \"정제본2\", ...]. 다른 텍스트 절대 금지."""


async def _polish_chunks_for_embedding(
    chunks: list[str], llm: BaseLLMService
) -> list[str]:
    """chunks 를 임베딩 친화 자연어로 정제. 실패 batch 는 원본 fallback."""
    POLISH_BATCH = 5
    results: list[str] = []
    for start in range(0, len(chunks), POLISH_BATCH):
        batch = chunks[start : start + POLISH_BATCH]
        user_msg = "\n\n".join(f"[{j + 1}]\n{c}" for j, c in enumerate(batch))
        try:
            raw = await llm.generate(
                system_prompt=_CHUNK_POLISH_SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=4500,
            )
        except Exception as e:
            logger.error(
                "chunk polish LLM call failed batch=%d: %s",
                start // POLISH_BATCH, e,
            )
            results.extend(batch)
            continue

        match = _JSON_ARRAY_RE.search(raw or "")
        if not match:
            logger.warning(
                "chunk polish JSON not found batch=%d raw=%r",
                start // POLISH_BATCH, (raw or "")[:200],
            )
            results.extend(batch)
            continue
        try:
            parsed = json.loads(match.group(0))
        except Exception as e:
            logger.error(
                "chunk polish JSON parse failed batch=%d: %s",
                start // POLISH_BATCH, e,
            )
            results.extend(batch)
            continue
        if not isinstance(parsed, list):
            results.extend(batch)
            continue

        normalized: list[str] = []
        for j, item in enumerate(parsed[: len(batch)]):
            polished = item.strip() if isinstance(item, str) and item.strip() else ""
            if polished:
                ok, reason = _validate_polish(batch[j], polished)
                if not ok:
                    logger.warning(
                        "polish suspicious batch=%d j=%d %s — fallback",
                        start // POLISH_BATCH, j, reason,
                    )
                    polished = ""
            normalized.append(polished or batch[j])
        while len(normalized) < len(batch):
            normalized.append(batch[len(normalized)])
        results.extend(normalized)
    return results


# ── chunk enrich (메타 LLM) ──────────────────────────────────────

_CHUNK_ENRICH_BATCH = 10


def _default_chunk_meta() -> dict:
    return {"title": "", "summary": "", "keywords": [], "topic": "기타"}


_CHUNK_ENRICH_SYSTEM_PROMPT = """당신은 PDF 청크의 메타데이터 추출기입니다.
입력: 청크 N 개 (인덱스 1~N).
출력: JSON 배열, 원소 N 개. 각 원소 필드:
  - title: 청크 핵심 주제 한 줄 (10~25자, 한국어)
  - summary: 청크 요약 1~2문장 (50자 이내)
  - keywords: 사용자가 음성 전화로 짧게 물어볼 때 쓸 법한 한국어 단어 (배열)
  - topic: 카테고리 한 단어 또는 짧은 구 (예: "위치", "예약", "진료시간", "주차", "응급실")

★ keywords 규칙 (음성 query substring 매칭 + STT keyterm biasing 에 직접 사용 — 운영 핵심):

[1. 개수 — 청크 길이별 차등]
- 청크 길이 ≤ 300자  : 핵심 명사 2~3개
- 301 ~ 500자        : 핵심 명사 3~4개
- > 500자            : 핵심 명사 4~5개 + 음성 변형 1~2개 추가
※ keywords 는 절대 빈 배열 [] 금지. 최소 2개 필수.

[2. 음성 변형 필수 포함]
청크에 명시된 단어가 일상 발화에서 축약·변형되는 경우, 변형도 반드시 포함:
- "주차장" → ["주차장", "주차"]
- "진료시간" → ["진료시간", "진료", "시간"] 중 최소 2개
- "영업시간" → ["영업시간", "영업", "시간"] 중 최소 2개
- "위치" → ["위치"] + ["오시는 길", "어디", "찾는 길"] 중 변형 1개

[3. 단일 명사 우선]
복합어 ("메뉴 종류") 금지. 짧은 단일 명사로 분리.

[4. 메타·일반 단어 금지]
- "정보", "안내", "문의", "주의사항", "참고사항", "기타", "상담", "기본정보", "내용"
- "번호", "방법" (구체 값이 청크마다 다름)
대신 도메인 특정 명사 ("영업시간", "주차", "예약").

[5. 청크에 없는 동의어 추가 (선택)]
일반인이 같은 의미로 실제 사용하는 동의어 0~2개 추가 허용.

일반 규칙:
- 청크 내용에만 의존. 없는 정보 추측 금지 (단, [2] 변형 + [5] 동의어는 예외).
- 청크가 짧으면 모든 필드를 짧게 유지.
- 출력은 JSON 배열만, 다른 설명 절대 금지."""


async def _enrich_chunks_with_llm(
    chunks: list[str], llm: BaseLLMService
) -> list[dict]:
    """chunks → metadata list. 실패 batch 는 default 메타로 채움."""
    results: list[dict] = []
    for start in range(0, len(chunks), _CHUNK_ENRICH_BATCH):
        batch = chunks[start : start + _CHUNK_ENRICH_BATCH]
        user_msg = "\n\n".join(f"[{j + 1}]\n{c}" for j, c in enumerate(batch))
        try:
            raw = await llm.generate(
                system_prompt=_CHUNK_ENRICH_SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error(
                "chunk enrich LLM call failed batch=%d: %s",
                start // _CHUNK_ENRICH_BATCH, e,
            )
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        match = _JSON_ARRAY_RE.search(raw or "")
        if not match:
            results.extend([_default_chunk_meta()] * len(batch))
            continue
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            results.extend([_default_chunk_meta()] * len(batch))
            continue
        if not isinstance(parsed, list):
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        normalized: list[dict] = []
        for item in parsed[: len(batch)]:
            if isinstance(item, dict):
                normalized.append({
                    "title": str(item.get("title", ""))[:100],
                    "summary": str(item.get("summary", ""))[:300],
                    "keywords": [str(k) for k in (item.get("keywords") or []) if k][:5],
                    "topic": str(item.get("topic", "기타"))[:50],
                })
            else:
                normalized.append(_default_chunk_meta())
        while len(normalized) < len(batch):
            normalized.append(_default_chunk_meta())
        results.extend(normalized)
    return results


# ── tenant 가용 카테고리 정제 ─────────────────────────────────────

_CATEGORY_REFINE_SYSTEM_PROMPT = """당신은 음성 안내용 카테고리 정제기입니다.
입력: chunk 별 raw topic 문자열 list.
출력: 자연스러운 음성 안내용 카테고리 5~7개 (JSON array, 한국어).

규칙:
- 비슷한 의미 통합 (예: "주차장 이용", "주차" → "주차 안내").
- 너무 길거나 모호한 topic 제외.
- 음성으로 자연스럽게 들리는 표현.
- 5~7개로 제한.
- JSON array 만 출력."""


async def _refine_categories(topics: list[str], llm: BaseLLMService) -> list[str]:
    """raw topic list → 자연스러운 5~7개 카테고리. LLM 1회 호출."""
    distinct = sorted({t.strip() for t in topics if t and t.strip() and t != "기타"})
    if not distinct:
        return []
    try:
        raw = await llm.generate(
            system_prompt=_CATEGORY_REFINE_SYSTEM_PROMPT,
            user_message=f"raw topics: {distinct}",
            temperature=0.1,
            max_tokens=300,
        )
    except Exception as e:
        logger.error("category refine LLM call failed: %s", e)
        return distinct[:7]
    match = _JSON_ARRAY_RE.search(raw or "")
    if not match:
        return distinct[:7]
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return distinct[:7]
    if not isinstance(parsed, list):
        return distinct[:7]
    return [str(c).strip() for c in parsed if c][:7]


# ── PDFProcessor ─────────────────────────────────────────────────

class PDFProcessor:
    def __init__(
        self,
        embedder: BaseEmbeddingService,
        rag: ChromaRAGService,
        llm: Optional[BaseLLMService] = None,
        session: Optional[RedisSessionService] = None,
    ):
        self._embedder = embedder
        self._rag = rag
        self._llm = llm or GPT4OMiniService()
        self._session = session or RedisSessionService()

    async def process(
        self,
        pdf_path: str,
        tenant_id: str,
        file_name: str,
        industry: str,
        doc_type: str = "general",
    ) -> str:
        """PDF → JSON tree → 의미 단위 chunking → polish/자연어화 → ChromaDB.

        doc_type: ChromaDB 메타 분류. 일반 FAQ 는 "general" (기본값),
        모델 사양 등 vision 관련 청크는 "model_spec" (별도 시드 스크립트에서 사용).
        """
        existing = await self._find_existing_document(tenant_id, file_name)
        if existing:
            logger.info(
                "duplicate detected, replacing doc_id=%s file=%s",
                existing["id"], file_name,
            )
            await self._rag.delete_by_document(str(existing["id"]), tenant_id)
            await self._delete_rag_document(existing["id"])

        document_id = await self._insert_rag_document(tenant_id, file_name)
        logger.info("rag_document created id=%s file=%s", document_id, file_name)

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _extract_json, pdf_path)
            elements = _flatten_json(data)
            logger.info(
                "flattened elements count=%d (heading=%d, para=%d, list=%d, table=%d)",
                len(elements),
                sum(1 for e in elements if e.type == "heading"),
                sum(1 for e in elements if e.type == "paragraph"),
                sum(1 for e in elements if e.type == "list"),
                sum(1 for e in elements if e.type == "table"),
            )

            chunks_obj = _group_into_chunks(elements)
            chunks_obj = _merge_short(chunks_obj)
            logger.info(
                "chunked count=%d (section=%d, table=%d) file=%s",
                len(chunks_obj),
                sum(1 for c in chunks_obj if c.chunk_type == "section"),
                sum(1 for c in chunks_obj if c.chunk_type == "table"),
                file_name,
            )

            # 1. 표 청크 자연어화 (병렬 LLM)
            table_indices = [
                i for i, c in enumerate(chunks_obj) if c.chunk_type == "table"
            ]
            if table_indices:
                tasks = [
                    _naturalize_table(chunks_obj[i].heading, chunks_obj[i].text, self._llm)
                    for i in table_indices
                ]
                naturalized = await asyncio.gather(*tasks)
                for idx, txt in zip(table_indices, naturalized):
                    chunks_obj[idx].text = txt
                logger.info("table naturalize done count=%d", len(table_indices))

            # 2. section 청크 polish (기존 batch 로직)
            section_indices = [
                i for i, c in enumerate(chunks_obj) if c.chunk_type == "section"
            ]
            if section_indices:
                section_texts = [chunks_obj[i].text for i in section_indices]
                polished = await _polish_chunks_for_embedding(section_texts, self._llm)
                for idx, txt in zip(section_indices, polished):
                    chunks_obj[idx].text = txt
                logger.info("section polish done count=%d", len(section_indices))

            # 이 시점부터 모든 chunk.text 가 polished/자연어화 상태
            final_texts = [c.text for c in chunks_obj]

            # 3. 임베딩 (passage 측 — heading_path prepend 으로 큰 청크의 sub-topic 매칭 보강)
            #    chunk.text 자체는 보존 (humanize 단계에 원본 전달). 임베딩에만 path 포함.
            embed_texts = [
                f"[{' > '.join(c.heading_path)}]\n\n{c.text}" if c.heading_path else c.text
                for c in chunks_obj
            ]
            embeddings = await self._embedder.embed_passages(embed_texts)

            # 4. 메타 enrich (chunk 본문 기반 — path 미포함)
            llm_metas = await _enrich_chunks_with_llm(final_texts, self._llm)
            logger.info(
                "llm enrich done doc_id=%s metas=%d",
                document_id, len(llm_metas),
            )

            # 5. ChromaDB upsert
            collection_name = self._rag._collection_name(tenant_id)
            db_chunks: list[dict] = []
            for i, (chunk, embedding, llm_meta) in enumerate(
                zip(chunks_obj, embeddings, llm_metas)
            ):
                # heading 우선, 없으면 LLM title
                title = chunk.heading or llm_meta.get("title", "") or f"chunk #{i}"
                keywords_str = ", ".join(llm_meta.get("keywords") or [])[:200]
                bbox_str = (
                    ",".join(f"{v:.1f}" for v in chunk.bbox) if chunk.bbox else ""
                )
                await self._rag.upsert(
                    doc_id=f"{document_id}_chunk_{i}",
                    content=chunk.text,
                    embedding=embedding,
                    tenant_id=tenant_id,
                    metadata={
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "file_name": file_name,
                        "chunk_index": i,
                        "industry": industry,
                        "chunk_type": chunk.chunk_type,
                        "page_number": int(chunk.page),
                        "bbox": bbox_str,
                        "heading_path": " > ".join(chunk.heading_path),
                        "llm_title": title[:100],
                        "llm_summary": llm_meta.get("summary", ""),
                        "llm_keywords": keywords_str,
                        "llm_topic": llm_meta.get("topic", "기타"),
                        # 권한 게이트 — 시드 시점 항상 False, admin UI 또는 별도 시드 스크립트에서 토글.
                        "is_auth": False,
                        "is_vision": False,
                        "doc_type": doc_type,
                    },
                )
                db_chunks.append({
                    "chunk_index": i,
                    "page_number": int(chunk.page),
                    "content": chunk.text,
                    "metadata": {
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "file_name": file_name,
                        "chunk_index": i,
                        "industry": industry,
                        "chunk_type": chunk.chunk_type,
                        "page_number": int(chunk.page),
                        "bbox": bbox_str,
                        "heading_path": " > ".join(chunk.heading_path),
                        "llm_title": title[:100],
                        "llm_summary": llm_meta.get("summary", ""),
                        "llm_keywords": keywords_str,
                        "llm_topic": llm_meta.get("topic", "湲고?"),
                        "is_auth": False,
                        "is_vision": False,
                        "doc_type": doc_type,
                    },
                    "embedding_status": "ready",
                    "chroma_id": f"{document_id}_chunk_{i}",
                })

            # 6. tenant 가용 카테고리 (Redis) — 부가 산출물. 실패해도 인덱싱은 성공.
            await upsert_rag_document_chunks_for_tenant(
                document_id=str(document_id),
                tenant_id=tenant_id,
                chunks=db_chunks,
            )

            topics = [m.get("topic", "") for m in llm_metas]
            refined_categories = await _refine_categories(topics, self._llm)
            if refined_categories:
                try:
                    await self._session.set_rag_categories(tenant_id, refined_categories)
                    logger.info(
                        "rag_categories refined tenant=%s categories=%s",
                        tenant_id, refined_categories,
                    )
                except Exception as e:
                    logger.warning(
                        "rag_categories save failed (인덱싱은 성공): %s", e,
                    )

            await self._update_rag_document(document_id, len(chunks_obj), collection_name)
            logger.info(
                "pdf_processor done doc_id=%s chunks=%d",
                document_id, len(chunks_obj),
            )

        except Exception as e:
            await self._fail_rag_document(document_id)
            logger.error("pdf_processor failed doc_id=%s err=%s", document_id, e)
            raise

        return str(document_id)

    async def delete_document(self, document_id: str, tenant_id: str) -> None:
        """문서 삭제 — ChromaDB 청크 + rag_documents 동시 삭제."""
        await self._rag.delete_by_document(document_id, tenant_id)
        await self._delete_rag_document(uuid.UUID(document_id))
        logger.info("document deleted doc_id=%s tenant=%s", document_id, tenant_id)

    async def _insert_rag_document(self, tenant_id: str, file_name: str) -> uuid.UUID:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO rag_documents (tenant_id, file_name, file_type, status)
                VALUES ($1::uuid, $2, 'pdf', 'processing')
                RETURNING id
                """,
                tenant_id, file_name,
            )
            return row["id"]
        finally:
            await conn.close()

    async def _update_rag_document(
        self, document_id: uuid.UUID, chunk_count: int, collection_name: str
    ) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                """
                UPDATE rag_documents
                SET status = 'ready',
                    chunk_count = $2,
                    chroma_collection = $3,
                    indexed_at = now()
                WHERE id = $1
                """,
                document_id, chunk_count, collection_name,
            )
        finally:
            await conn.close()

    async def _fail_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                "UPDATE rag_documents SET status = 'failed' WHERE id = $1",
                document_id,
            )
        finally:
            await conn.close()

    async def _find_existing_document(
        self, tenant_id: str, file_name: str
    ) -> dict | None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                SELECT id FROM rag_documents
                WHERE tenant_id = $1::uuid
                  AND file_name = $2
                  AND status != 'failed'
                  AND deleted_at IS NULL
                ORDER BY uploaded_at DESC LIMIT 1
                """,
                tenant_id, file_name,
            )
            return dict(row) if row else None
        finally:
            await conn.close()

    async def _delete_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute(
                "DELETE FROM rag_document_chunks WHERE document_id = $1",
                document_id,
            )
            await conn.execute(
                "DELETE FROM rag_documents WHERE id = $1", document_id,
            )
        finally:
            await conn.close()
