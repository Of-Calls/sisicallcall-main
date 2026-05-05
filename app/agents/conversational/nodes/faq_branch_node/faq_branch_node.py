from app.agents.conversational.state import CallState
from app.services.cache import get_cache
from app.services.embedding import get_embedder
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.utils.config import settings

_llm = GPT4OMiniService()
_rag = ChromaRAGService()
_cache = get_cache()

_TOP_K = 3
# ChromaDB default L2 distance (정규화 벡터) — 작을수록 유사.
# 모델별 분포:
#   - BGE-M3:  정답 0.6~0.85, 무관 1.0+
#   - Qwen3:   정답 0.5~1.20, 무관 1.21+
# settings.faq_distance_threshold 로 조정 가능 (.env: FAQ_DISTANCE_THRESHOLD).
_DIST_THRESHOLD = settings.faq_distance_threshold
# vision 게이트 — model_spec 청크 top_k 진입 자체가 강한 신호.
_VISION_GATE_THRESHOLD = settings.faq_vision_gate_threshold

_POLITE_AUTH = "본인 인증이 필요한 정보예요. 인증 진행해드릴까요?"
_POLITE_VISION = "확인하시려는 게 어떤 건지 사진으로 봐야 정확히 안내드릴 수 있어요. 사진 보내주실 수 있을까요?"
_POLITE_NO_RESULT = "제가 잘 모르는 부분이에요. 상담원 연결해드릴까요?"
_POLITE_DECLINE_FALLBACK = "알겠습니다. 그러면 다른 무엇을 도와드릴까요?"

_FAQ_SYSTEM_PROMPT = """당신은 매장 전화 상담 AI 입니다. 사용자의 질문에 RAG 검색 결과를 바탕으로 친절하게 답변하세요.

[지침]
- 검색 결과 컨텍스트에 있는 사실만 사용. 없는 정보는 추측하지 마세요.
- 한국어 한두 문장으로 자연스럽게 답변. 너무 길면 안 됨 (음성 안내).
- "검색 결과", "문서에 따르면" 같은 메타 표현 금지. 매장 직원처럼 답하세요.
- 컨텍스트에 답이 없으면: 정확히 "NO_RESULT" 만 출력 (다른 텍스트/구두점 추가 금지). 코드가 감지해 폴백 메시지로 대체함.
- 출력은 답변 텍스트만. 따옴표/머릿말 금지.
- 시간은 "11시 30분" 형식으로. ":" 콜론, "~" 물결 등 기호 사용 금지.
- 시간 범위는 "11시 30분부터 22시까지" 형식. "~", "-" 사용 금지.
- 영업시간 같은 다항목 정보는 사용자가 명시적으로 묻지 않은 항목 (예: 휴무일) 은 생략."""


async def faq_branch_node(state: CallState) -> dict:
    query = state.get("rewritten_query") or state["user_text"]
    user_text = state.get("user_text") or ""  # is_vision 게이트의 model_id substring 매칭용
    tenant_id = state["tenant_id"]

    # 거절 패턴 — RAG 없이 generic 안내. query_refine 이 일관되게
    # "사용자가 ... 거절함" 으로 재작성하므로 string 매칭으로 충분.
    if "거절함" in query:
        print("[faq_branch] 거절 패턴 감지 → polite decline")
        return {"response_text": _POLITE_DECLINE_FALLBACK}

    # 1. 임베딩 (캐시 + RAG 공유) — query 측 asymmetric instruction 적용 (Qwen3) / BGE-M3 는 fallthrough.
    embedder = get_embedder()
    embedding = await embedder.embed_query(query)

    # 2. 캐시 조회 — hit 시 LLM/RAG 둘 다 skip
    cache_hit = await _cache.lookup(tenant_id, embedding)
    if cache_hit:
        print(f"[faq_branch] cache hit distance={cache_hit.distance:.4f}")
        return {"response_text": cache_hit.response_text}

    # 3. RAG 검색 (같은 임베딩 재사용)
    results = await _rag.search_with_meta(embedding, tenant_id, top_k=_TOP_K)
    print(f"[faq_branch] query='{query}' results={len(results)}")
    for i, r in enumerate(results):
        meta = r.get("metadata", {}) or {}
        print(
            f"  [{i+1}] distance={r.get('distance')} "
            f"title='{meta.get('llm_title', '')}' "
            f"is_auth={meta.get('is_auth', False)} is_vision={meta.get('is_vision', False)}"
        )

    if not results:
        return {"response_text": _POLITE_NO_RESULT}

    # 3. 메타데이터 게이트 — distance threshold 이내 청크만 검사 (false positive 방지)
    related = [
        r for r in results
        if r.get("distance") is not None and r["distance"] <= _DIST_THRESHOLD
    ]

    if any((r.get("metadata") or {}).get("is_auth", False) for r in related):
        print("[faq_branch] is_auth=True 청크 발견 (threshold 이내) → polite auth")
        return {"response_text": _POLITE_AUTH}

    # is_vision 게이트 — 별도 threshold (0.95) 로 results 전체 검사.
    # 모델 식별 필요 query 는 일반 humanize 보다 느슨하게 잡아야 함 (모델 사양 청크가
    # top_k 안에 들어왔다는 것 자체가 강한 신호).
    # 매칭 검사는 user_text (raw) + rewritten_query (history 기반 추론) 둘 다.
    rewritten = state.get("rewritten_query") or ""
    vision_chunks = [
        r for r in results
        if r.get("distance") is not None
        and r["distance"] <= _VISION_GATE_THRESHOLD
        and (r.get("metadata") or {}).get("is_vision", False)
    ]
    if vision_chunks:
        candidate_ids = {
            (r.get("metadata") or {}).get("model_id", "")
            for r in vision_chunks
        }
        candidate_ids = {mid for mid in candidate_ids if mid}
        search_text = f"{user_text} {rewritten}".upper()
        matched = any(
            mid and mid.upper() in search_text for mid in candidate_ids
        )
        if not matched:
            print(
                f"[faq_branch] is_vision 청크 발견 (threshold {_VISION_GATE_THRESHOLD}) "
                f"candidates={candidate_ids} 매칭 X → polite vision"
            )
            return {"response_text": _POLITE_VISION}
        print(
            f"[faq_branch] is_vision 청크 발견 candidates={candidate_ids} "
            f"발화/재작성에 모델 명시 → 게이트 우회"
        )

    # threshold 통과 청크가 없으면 LLM 환각 차단 — 즉시 NO_RESULT.
    if not related:
        print("[faq_branch] threshold 통과 청크 없음 → polite no_result")
        return {"response_text": _POLITE_NO_RESULT}

    # 4. 게이트 통과 → LLM 응답 (컨텍스트는 threshold 통과 청크만)
    context = "\n\n".join(
        f"[청크 {i+1}]\n{r.get('document', '')}" for i, r in enumerate(related)
    )
    user_message = f"[검색 결과]\n{context}\n\n[사용자 질문]\n{query}"

    try:
        text = await _llm.generate(
            system_prompt=_FAQ_SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.2,
            max_tokens=200,
        )
        text = text.strip().strip('"').strip("'")
        # NO_RESULT 신호 (또는 빈값) → cache 저장 안 함 + 폴백 메시지로 대체.
        # exact 매칭 — prompt 가 "정확히 NO_RESULT 만 출력" 강제.
        if not text or text == "NO_RESULT":
            print("[faq_branch] LLM NO_RESULT → polite_no_result (cache 저장 안 함)")
            return {"response_text": _POLITE_NO_RESULT}
        await _cache.save(tenant_id, query, embedding, text)
        print("[faq_branch] cache save")
    except Exception as exc:
        print(f"[faq_branch] LLM 실패 → fallback: {exc}")
        text = _POLITE_NO_RESULT

    print(f"[faq_branch] response='{text}'")
    return {"response_text": text}
