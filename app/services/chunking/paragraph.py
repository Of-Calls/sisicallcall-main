import re

from app.services.chunking.base import BaseChunkingService

MIN_CHUNK_LEN = 100
MAX_CHUNK_LEN = 800
OVERLAP_LEN = 50


class ParagraphChunkingService(BaseChunkingService):
    async def chunk(self, text: str) -> list[str]:
        paragraphs = self._split_paragraphs(text)
        merged = self._merge_short(paragraphs)
        return self._apply_overlap(merged)

    def _split_paragraphs(self, text: str) -> list[str]:
        parts = re.split(r"\n{2,}", text)
        result = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= MAX_CHUNK_LEN:
                result.append(part)
            else:
                # 최대 길이 초과 단락은 문장 단위로 추가 분할
                result.extend(self._split_by_sentence(part))
        return result

    def _split_by_sentence(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?。])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) <= MAX_CHUNK_LEN:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    def _merge_short(self, paragraphs: list[str]) -> list[str]:
        """MIN_CHUNK_LEN 미만 단락만 인접 단락과 병합. 이미 충분한 단락은 독립 청크."""
        merged, buffer = [], ""
        for para in paragraphs:
            if not buffer:
                buffer = para
            elif len(buffer) < MIN_CHUNK_LEN and len(buffer) + len(para) + 2 <= MAX_CHUNK_LEN:
                buffer = buffer + "\n\n" + para
            else:
                merged.append(buffer)
                buffer = para
        if buffer:
            merged.append(buffer)
        return merged

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """인접 청크 앞부분 OVERLAP_LEN 글자를 다음 청크 앞에 추가."""
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            overlap = chunks[i - 1][-OVERLAP_LEN:]
            result.append(overlap + "\n" + chunks[i])
        return result
