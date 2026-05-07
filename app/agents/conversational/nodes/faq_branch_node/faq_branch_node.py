from app.agents.conversational.prompts.faq import build_system_prompt
from app.agents.conversational.state import CallState
from app.services.cache import get_cache
from app.services.embedding import get_embedder
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.services.retrieval import get_hybrid_retriever
from app.utils.config import settings

_llm = GPT4OMiniService()
_rag = ChromaRAGService()
_cache = get_cache()
# module-level singleton — main.py 의 prewarm 결과 (BM25 캐시) 와 동일 인스턴스 공유.
# embedder 는 외부 (faq_branch) 에서 주입 — 여기서는 search_with_embedding 만 사용.
_hybrid = get_hybrid_retriever(rag=_rag)

_TOP_K = 5  # Hybrid: dense+bm25 결합이라 single mode 보다 약간 넓게 후보 받음
# Dense distance pass — ChromaDB default L2 (정규화 벡터, 작을수록 유사).
# Qwen3 분포: 정답 0.5~1.20, 무관 1.21+. settings.faq_distance_threshold (.env: FAQ_DISTANCE_THRESHOLD).
_DIST_THRESHOLD = settings.faq_distance_threshold
# vision 게이트 — model_spec 청크 top_k 진입 자체가 강한 신호.
_VISION_GATE_THRESHOLD = settings.faq_vision_gate_threshold

_POLITE_AUTH = "본인 인증이 필요한 정보예요. 인증 진행해드릴까요?"
_POLITE_VISION = "확인하시려는 게 어떤 건지 사진으로 봐야 정확히 안내드릴 수 있어요. 사진 보내주실 수 있을까요?"
_POLITE_NO_RESULT = "제가 잘 모르는 부분이에요. 상담원 연결해드릴까요?"
_POLITE_DECLINE_FALLBACK = "알겠습니다. 그러면 다른 무엇을 도와드릴까요?"


async def faq_branch_node(state: CallState) -> dict:
    query = state.get("rewritten_query") or state["user_text"]
    user_text = state.get("user_text") or ""  # is_vision 게이트의 model_id substring 매칭용
    tenant_id = state["tenant_id"]
    tenant_name = state.get("tenant_name", "고객센터")
    tenant_industry = state.get("tenant_industry", "unknown")

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

    # 3. Hybrid 검색 (dense + BM25 RRF) — 같은 임베딩 재사용.
    #    dense 의 짧은 단어 query 약점을 BM25 (Kiwi 형태소) 가 보강.
    results = await _hybrid.search_with_embedding(
        query=query, query_embedding=embedding, tenant_id=tenant_id, top_k=_TOP_K,
    )
    print(f"[faq_branch] query='{query}' hybrid_results={len(results)}")
    for i, r in enumerate(results):
        meta = r.get("metadata", {}) or {}
        dist = r.get("distance")
        bm25 = r.get("bm25_score") or 0
        dist_str = f"{dist:.3f}" if dist is not None else "-"
        print(
            f"  [{i+1}] dense={dist_str} bm25={bm25:.2f} "
            f"title='{meta.get('llm_title', '')}' "
            f"is_auth={meta.get('is_auth', False)} is_vision={meta.get('is_vision', False)}"
        )

    if not results:
        return {"response_text": _POLITE_NO_RESULT}

    # 4. Pass 판정 — dense_pass (distance ≤ threshold) OR bm25_pass (score > 0).
    #    어느 한쪽이라도 hit 한 청크는 LLM 컨텍스트 후보. 두 retriever 의 OR 로 recall ↑.
    def _is_passed(r: dict) -> bool:
        dist = r.get("distance")
        dense_pass = dist is not None and dist <= _DIST_THRESHOLD
        bm25_pass = (r.get("bm25_score") or 0) > 0
        return dense_pass or bm25_pass

    related = [r for r in results if _is_passed(r)]

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
            system_prompt=build_system_prompt(tenant_name, tenant_industry),
            user_message=user_message,
            temperature=0.2,
            max_tokens=150,  # prompt 의 "150자 이내" 와 일치 (한국어 ~1.5자/token, 약간 여유)
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
