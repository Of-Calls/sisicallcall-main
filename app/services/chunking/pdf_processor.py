"""PDF 청킹 파이프라인 — opendataloader-pdf (Java) + pymupdf4llm fallback.

청킹 전략:
    1차: ## 헤더 기준 섹션 분할 → 짧은 섹션 merge → 700자 초과 시 RCS 분할
    2차: chunk 본문 LLM polish (BGE-M3 임베딩 친화 자연어화)
    3차: chunk 별 LLM 메타데이터 (title/summary/keywords/topic) 추출
    4차: tenant raw topic → 음성 안내용 카테고리 5~7개 정제 → Redis write
"""
import asyncio
import json
import re
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import asyncpg
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.services.embedding.base import BaseEmbeddingService
from app.services.llm.base import BaseLLMService
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.services.session.redis_session import RedisSessionService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SECTION_SPLIT_RE = re.compile(r'(?=\n## )')
_BOLD_RE = re.compile(r'\*\*([^*]+)\*\*')

# MIN=200 / MAX=700: 짧은 음성 query 매칭 정밀도 회복용 (PoC 195111.log).
MAX_SECTION_CHARS = 700
MIN_CHUNK_CHARS = 200

_RCS = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n■", "\n▶", "\n☑", "\n※", "\n", " "],
    chunk_size=MAX_SECTION_CHARS,
    chunk_overlap=100,
)


def _extract_text(pdf_path: str) -> str:
    """PDF → 마크다운. opendataloader-pdf 우선, 실패(JVM 미설치 등) 시 pymupdf4llm fallback."""
    try:
        from opendataloader_pdf import convert
        with tempfile.TemporaryDirectory(prefix="odl_") as tmp:
            convert(input_path=pdf_path, output_dir=tmp, format="markdown", quiet=True)
            md_files = list(Path(tmp).rglob("*.md"))
            if not md_files:
                raise RuntimeError("opendataloader produced no markdown output")
            text = md_files[0].read_text(encoding="utf-8")
        logger.info("pdf parsed by opendataloader len=%d path=%s", len(text), pdf_path)
        return text
    except Exception as e:
        logger.warning("opendataloader failed (%s) — fallback to pymupdf4llm: %s", e, pdf_path)

    import pymupdf4llm
    text = pymupdf4llm.to_markdown(pdf_path)
    logger.info("pdf parsed by pymupdf4llm len=%d path=%s", len(text), pdf_path)
    return text


def _clean(text: str) -> str:
    """pymupdf4llm 마크다운 아티팩트 정리.

    진짜 헤더 vs 가짜 헤더(볼드 문장) 구분 기준:
        ** 마크업 포함 → 볼드 문장이 ## 로 오인식된 것 → ## 제거
        ** 마크업 없음  → 실제 섹션 헤더 → 유지
    예:
        ## 찾아오시는 길           (** 없음) → 유지 ✅
        ## 강남구청은 ... **.** ` (** 있음) → ## 제거 ✅
    """
    lines = []
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('## ') and '**' in s:
            # 볼드 문장 오인식 → ## 제거 후 ** 도 제거해 일반 텍스트로
            lines.append(_BOLD_RE.sub(r'\1', s[3:]))
        else:
            lines.append(line)
    cleaned = '\n'.join(lines)
    return _BOLD_RE.sub(r'\1', cleaned)


def _merge_short_sections(sections: list[str]) -> list[str]:
    """`MIN_CHUNK_CHARS` 미만 섹션을 다음 섹션 머리에 prepend 해 헤더-only
    chunk 양산을 막는다. 마지막에 짧은 섹션이 남으면 직전 섹션 끝에 append.
    """
    if not sections:
        return []
    merged: list[str] = []
    pending = ""
    for section in sections:
        if pending:
            section = pending + "\n\n" + section
            pending = ""
        if len(section) < MIN_CHUNK_CHARS:
            pending = section
            continue
        merged.append(section)
    if pending:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + pending
        else:
            merged.append(pending)
    return merged


def _split_sections(text: str) -> list[str]:
    """헤더 기준 섹션 분할 → 짧은 섹션 merge → 긴 섹션은 RCS 추가 분할."""
    raw = _SECTION_SPLIT_RE.split(text)
    sections = _merge_short_sections([s.strip() for s in raw if s.strip()])

    chunks: list[str] = []
    for section in sections:
        if len(section) <= MAX_SECTION_CHARS:
            chunks.append(section)
        else:
            sub = _RCS.split_text(section)
            chunks.extend(c.strip() for c in sub if c.strip())
    return chunks


# ── LLM 메타데이터 보강 (Phase 2) ─────────────────────────────────

_CHUNK_ENRICH_BATCH = 10  # batch 10 청크/호출 — 비용/시간/정확도 균형
_JSON_ARRAY_RE = re.compile(r'\[.*\]', re.DOTALL)

# polish 결과 silent 압축/숫자 누락 방어 — 위반 시 원본 chunk fallback.
_MIN_POLISH_RATIO = 0.7
_NUMBER_RE = re.compile(r'\d+')


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

_CHUNK_ENRICH_SYSTEM_PROMPT = """당신은 PDF 청크의 메타데이터 추출기입니다.
입력: 청크 N 개 (인덱스 1~N).
출력: JSON 배열, 원소 N 개. 각 원소 필드:
  - title: 청크 핵심 주제 한 줄 (10~25자, 한국어)
  - summary: 청크 요약 1~2문장 (50자 이내)
  - keywords: 사용자가 음성 전화로 짧게 물어볼 때 쓸 법한 한국어 단어 (배열)
  - topic: 카테고리 한 단어 또는 짧은 구 (예: "위치", "예약", "진료시간", "주차", "응급실")

★ keywords 규칙 (음성 query substring 매칭 + STT keyterm biasing 에 직접 사용 — 운영 핵심):

[1. 개수 — 청크 길이별 차등]
- 청크 길이 ≤ 300자  : 핵심 명사 2~3개 (정보 밀도 높음, 적게)
- 301 ~ 500자        : 핵심 명사 3~4개
- > 500자            : 핵심 명사 4~5개 + 음성 변형 1~2개 추가
※ 항상 핵심부터 채우고, 부족하면 변형/동의어로 보강. 빈 슬롯 채우려고 의미 없는 단어 넣지 말 것.
※ **keywords 는 절대 빈 배열 [] 금지. 최소 2개 필수.** 청크가 짧거나 메뉴 목록 / 표 / 리스트 류여도 반드시 2개 이상 추출:
  - 음식 목록 청크 → 음식 카테고리 명사 (예: ["메뉴", "한우", "갈비", "단품"])
  - 위치·교통 안내 → 위치 관련 명사 + 변형 (예: ["대중교통", "교통", "위치", "오시는 길"])
  - 표/리스트만 있는 청크 → 행 헤더의 명사 추출

[2. 음성 변형 필수 포함]
청크에 명시된 단어가 일상 발화에서 축약·변형되는 경우, 변형도 반드시 포함:
- "주차장" 명시 → ["주차장", "주차"] 둘 다
- "진료시간" 명시 → ["진료시간", "진료", "시간"] 중 최소 2개 (3개 셋 선호)
- "영업시간" 명시 → ["영업시간", "영업", "시간"] 중 최소 2개
- "위치" 명시 → ["위치"] + ["오시는 길", "어디", "찾는 길"] 중 변형 1개 추가
이유: STT 사용자가 "주차 가능?", "언제 진료?" 같이 도메인 용어를 축약·변형해 발화 — 원문 형태만으로는 substring 매칭 실패.

[3. 단일 명사 우선]
복합어 ("메뉴 종류", "주차장 이용") 금지. 띄어쓰기 없는 짧은 단일 명사로 분리:
- "메뉴 종류" → ["메뉴", "종류"]
- "주차장 이용" → ["주차장", "주차", "이용"]
복합 의미가 단일 단어로 굳어진 경우 ("진료시간", "영업시간") 는 [2] 규칙에 따라 원형 + 분리형 둘 다 포함 OK.

[4. 메타·일반 단어 금지 (STT/RAG 매칭 효율 해침)]
다음 단어는 keywords 에 절대 넣지 말 것 — 모든 도메인 청크에 혼재되어 차별 신호 0:
- 일반 메타: "정보", "안내", "문의", "주의사항", "참고사항", "기타", "상담", "기본정보", "내용"
- 가변 generic: "번호", "방법" (구체 값이 청크마다 달라 keyterm biasing 의미 없음)
대신 도메인 특정 명사를 구체적으로 사용 ("영업시간", "주차", "예약" 처럼).

[5. 청크에 없는 동의어 추가 (선택)]
일반인이 같은 의미로 실제 사용하는 동의어 0~2개 추가 허용 (강제 아님):
- "구청" 명시 → ["구청", "시청", "청사"] (시청은 동의어로 추가 가능)
- "응급실" 명시 → ["응급실", "응급"] (변형은 [2], 동의어는 별도)
※ 일반 한국인이 실제로 쓰는 표현만. 사전적·학술적 용어 추가 금지.

일반 규칙:
- 청크 내용에만 의존. 없는 정보 추측 절대 금지 (단, [2] 변형 + [5] 동의어는 예외).
- 청크가 모호하거나 짧으면 모든 필드를 짧게 유지.
- 출력은 JSON 배열만, 다른 설명 텍스트 절대 포함하지 않는다."""

_CHUNK_POLISH_SYSTEM_PROMPT = """당신은 RAG 임베딩용 청크 정제기입니다.
입력: PDF 추출 마크다운 chunk N 개 (헤더 / 표 / 리스트 / 줄바꿈 raw 포함).
출력: JSON 배열, 원소 N 개 — 각 원소는 정제된 자연어 본문 (string, plain Korean).

목적: BGE-M3 임베딩이 짧은 음성 질문 (예: "메뉴가 뭐가 있어요", "주차 가능?") 와 매칭
정확도를 높이도록 chunk 본문을 자연어 형태로 다듬는 것.

정제 규칙:
- 마크다운 마커 제거: ##, ###, ####, ■, ▶, ☑, ※, **, `, 표 파이프 |, 리스트 -
- 표 raw → 자연어로 풀어 쓰기 (예: "|코스명|가격|\\n|화담|8만원|\\n|사담|12만원|" → "한정식 코스는 화담 8만원, 사담 12만원으로 구성된다")
- 줄바꿈 \\n 제거 → 자연스러운 단락 / 마침표로 연결
- 헤더 (예: "## 5. 메뉴 구성 및 코스 안내") → 본문 첫 문장에 자연스럽게 흡수 (예: "한밭식당의 메뉴 구성 및 코스 안내. ...")

★ 절대 규칙 (위반 시 후처리에서 원본으로 자동 폴백):
- 입력에 있는 모든 사실/정보 보존. 재요약·압축 금지. 표현만 자연어로 변환.
- 출력 길이는 입력 길이의 70% 이상 유지 (마크다운 마커 제거로 약간 짧아지는 정도만 허용).
- 숫자 (가격, 시간, 전화번호, 주소, 면적, 인원 수 등) 는 입력에 등장한 모든 숫자열을
  출력에 한 글자도 변형 없이 그대로 포함. 누락 시 원본으로 폴백 처리됨.
- 입력에 없는 정보 추측·추가 금지.
- 같은 의미의 chunk 가 N 개 중 여러 개여도 각각 독립 정제. 합치지 마라.

출력 형식: JSON 배열 [\"정제본1\", \"정제본2\", ...]. 다른 텍스트 절대 금지."""

_CATEGORY_REFINE_SYSTEM_PROMPT = """당신은 음성 안내용 카테고리 정제기입니다.
입력: chunk 별 raw topic 문자열 list.
출력: 자연스러운 음성 안내용 카테고리 5~7개 (JSON array, 한국어).

규칙:
- 비슷한 의미의 topic 은 통합 (예: "주차장 이용", "주차" → "주차 안내").
- 너무 길거나 모호한 topic 은 제외.
- 음성으로 자연스럽게 들리는 표현 (예: "찾아오시는 길" → "위치 안내").
- 5~7개로 제한.
- JSON array 만 출력, 다른 텍스트 절대 금지."""


def _default_chunk_meta() -> dict:
    return {"title": "", "summary": "", "keywords": [], "topic": "기타"}


async def _enrich_chunks_with_llm(
    chunks: list[str], llm: BaseLLMService
) -> list[dict]:
    """chunks 를 batch 단위로 LLM 호출해 metadata list 반환.

    실패한 batch 는 default 메타로 채워 길이 보장.
    """
    results: list[dict] = []
    for start in range(0, len(chunks), _CHUNK_ENRICH_BATCH):
        batch = chunks[start : start + _CHUNK_ENRICH_BATCH]
        user_msg = "\n\n".join(
            f"[{j + 1}]\n{c}" for j, c in enumerate(batch)
        )
        try:
            raw = await llm.generate(
                system_prompt=_CHUNK_ENRICH_SYSTEM_PROMPT,
                user_message=user_msg,
                temperature=0.1,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("chunk enrich LLM call failed batch=%d: %s", start // _CHUNK_ENRICH_BATCH, e)
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        match = _JSON_ARRAY_RE.search(raw or "")
        if not match:
            logger.warning(
                "chunk enrich JSON not found batch=%d raw=%r",
                start // _CHUNK_ENRICH_BATCH, (raw or "")[:200],
            )
            results.extend([_default_chunk_meta()] * len(batch))
            continue
        try:
            parsed = json.loads(match.group(0))
        except Exception as e:
            logger.error("chunk enrich JSON parse failed batch=%d: %s", start // _CHUNK_ENRICH_BATCH, e)
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        if not isinstance(parsed, list):
            results.extend([_default_chunk_meta()] * len(batch))
            continue

        # 길이 맞추기 — LLM 이 N 개 안 맞춰 줄 수 있음
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


async def _polish_chunks_for_embedding(
    chunks: list[str], llm: BaseLLMService
) -> list[str]:
    """chunks 를 BGE-M3 임베딩 친화 자연어로 정제. 정보·숫자 보존, 마크다운 마커 제거.

    호출자 책임:
      - 검색 매칭은 정제본 임베딩 사용
      - LLM 응답 input (ChromaDB document) 은 원본 chunk 유지
    실패한 batch 는 원본 chunk 그대로 사용 (안전 fallback).
    """
    POLISH_BATCH = 5  # polish 출력은 메타보다 길어 batch 작게
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
            logger.error("chunk polish LLM call failed batch=%d: %s", start // POLISH_BATCH, e)
            results.extend(batch)  # fallback: 원본
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
            logger.error("chunk polish JSON parse failed batch=%d: %s", start // POLISH_BATCH, e)
            results.extend(batch)
            continue

        if not isinstance(parsed, list):
            results.extend(batch)
            continue

        # 길이 맞추기 + 빈/None 항목 + 검증 실패는 원본 fallback
        normalized: list[str] = []
        for j, item in enumerate(parsed[: len(batch)]):
            polished = item.strip() if isinstance(item, str) and item.strip() else ""
            if polished:
                ok, reason = _validate_polish(batch[j], polished)
                if not ok:
                    logger.warning(
                        "polish suspicious batch=%d j=%d %s — fallback to orig",
                        start // POLISH_BATCH, j, reason,
                    )
                    polished = ""
            normalized.append(polished or batch[j])
        while len(normalized) < len(batch):
            normalized.append(batch[len(normalized)])
        results.extend(normalized)

    return results


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
        logger.warning("category refine JSON not found raw=%r", (raw or "")[:200])
        return distinct[:7]
    try:
        parsed = json.loads(match.group(0))
    except Exception as e:
        logger.error("category refine JSON parse failed: %s", e)
        return distinct[:7]

    if not isinstance(parsed, list):
        return distinct[:7]
    return [str(c).strip() for c in parsed if c][:7]


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
        """PDF → 청킹 → 임베딩 → ChromaDB 저장. document_id 반환.

        doc_type: ChromaDB 메타 분류. 일반 FAQ 는 "general" (기본값),
        모델 사양 등 vision 관련 청크는 "model_spec" (별도 시드 스크립트에서 사용).
        """
        existing = await self._find_existing_document(tenant_id, file_name)
        if existing:
            logger.info("duplicate detected, replacing doc_id=%s file=%s", existing["id"], file_name)
            await self._rag.delete_by_document(str(existing["id"]), tenant_id)
            await self._delete_rag_document(existing["id"])

        document_id = await self._insert_rag_document(tenant_id, file_name)
        logger.info("rag_document created id=%s file=%s", document_id, file_name)

        try:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, _extract_text, pdf_path)
            text = _clean(raw)
            logger.info("pdf extracted+cleaned len=%d file=%s", len(text), file_name)

            chunks = _split_sections(text)
            logger.info("chunked count=%d file=%s", len(chunks), file_name)

            # chunk 본문 임베딩-친화 정제 (Phase 4) — 마크다운 마커/표/줄바꿈 제거 + 자연어화.
            # 검색 매칭은 정제본 임베딩으로 수행, ChromaDB document 는 원본 유지 (LLM 응답 input).
            polished_chunks = await _polish_chunks_for_embedding(chunks, self._llm)
            logger.info(
                "polish done doc_id=%s polished=%d (orig avg=%d → polished avg=%d chars)",
                document_id, len(polished_chunks),
                int(sum(len(c) for c in chunks) / max(len(chunks), 1)),
                int(sum(len(c) for c in polished_chunks) / max(len(polished_chunks), 1)),
            )

            embeddings = await self._embedder.embed_batch(polished_chunks)

            # LLM 메타데이터 보강 — chunk 별 title/summary/keywords/topic (원본 chunk 사용)
            llm_metas = await _enrich_chunks_with_llm(chunks, self._llm)
            logger.info("llm enrich done doc_id=%s metas=%d", document_id, len(llm_metas))

            collection_name = self._rag._collection_name(tenant_id)
            for i, (chunk, polished, embedding, llm_meta) in enumerate(
                zip(chunks, polished_chunks, embeddings, llm_metas)
            ):
                # ChromaDB metadata 는 primitive 만 → keywords list 는 콤마 join
                keywords_str = ", ".join(llm_meta.get("keywords") or [])[:200]
                await self._rag.upsert(
                    doc_id=f"{document_id}_chunk_{i}",
                    content=chunk,            # 원본 (LLM 응답 input)
                    embedding=embedding,      # 정제본 임베딩 (검색 매칭)
                    tenant_id=tenant_id,
                    metadata={
                        "tenant_id": tenant_id,
                        "document_id": str(document_id),
                        "file_name": file_name,
                        "chunk_index": i,
                        "industry": industry,
                        "llm_title": llm_meta.get("title", ""),
                        "llm_summary": llm_meta.get("summary", ""),
                        "llm_keywords": keywords_str,
                        "llm_topic": llm_meta.get("topic", "기타"),
                        # 권한 게이트 플래그 — PDF 임베딩 시점은 항상 False.
                        # 추후 프론트 admin UI 에서 청크별 토글 (title/summary 보고 사람이 결정).
                        "is_auth": False,
                        "is_vision": False,
                        # 청크 분류 — vision 시드 스크립트는 "model_spec" 사용,
                        # 일반 FAQ PDF 는 기본값 "general".
                        "doc_type": doc_type,
                        # 디버그/검증용 — ChromaDB metadata 1KB 제한 회피로 800자 컷.
                        "polished_text": polished[:800],
                    },
                )

            # tenant 가용 카테고리 LLM 정제 + Redis write
            topics = [m.get("topic", "") for m in llm_metas]
            refined_categories = await _refine_categories(topics, self._llm)
            if refined_categories:
                await self._session.set_rag_categories(tenant_id, refined_categories)
                logger.info(
                    "rag_categories refined tenant=%s categories=%s",
                    tenant_id, refined_categories,
                )

            await self._update_rag_document(document_id, len(chunks), collection_name)
            logger.info("pdf_processor done doc_id=%s chunks=%d", document_id, len(chunks))

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
                tenant_id,
                file_name,
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
                document_id,
                chunk_count,
                collection_name,
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

    async def _find_existing_document(self, tenant_id: str, file_name: str) -> dict | None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            row = await conn.fetchrow(
                """
                SELECT id FROM rag_documents
                WHERE tenant_id = $1::uuid AND file_name = $2 AND status != 'failed'
                ORDER BY uploaded_at DESC LIMIT 1
                """,
                tenant_id,
                file_name,
            )
            return dict(row) if row else None
        finally:
            await conn.close()

    async def _delete_rag_document(self, document_id: uuid.UUID) -> None:
        conn = await asyncpg.connect(settings.database_url)
        try:
            await conn.execute("DELETE FROM rag_documents WHERE id = $1", document_id)
        finally:
            await conn.close()
