from app.agents.conversational.state import CallState
from app.services.embedding import get_embedder
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService

_llm = GPT4OMiniService()
_rag = ChromaRAGService()

_TOP_K = 3
# ChromaDB default L2 distance (BGE-M3 normalized) — 작을수록 유사.
# 분포: 매우 관련 0.6~0.8, 약 관련 0.8~1.0, 무관 1.0+ (max √2 ≈ 1.41).
# 0.85 — 한밭식당 검증 결과 정확히 매칭되는 청크는 0.6~0.85 범위에 분포.
_DIST_THRESHOLD = 0.85
# vision 게이트는 모델 식별 필요 신호만 잡으면 됨 — 일반 humanize 보다 느슨한 기준.
# 0.95 — model_spec 청크가 top_k 안에 들어왔다는 사실 자체가 강한 신호.
_VISION_GATE_THRESHOLD = 0.95

_POLITE_AUTH = "본인 인증이 필요한 정보예요. 인증 진행해드릴까요?"
_POLITE_VISION = "확인하시려는 게 어떤 건지 사진으로 봐야 정확히 안내드릴 수 있어요. 사진 보내주실 수 있을까요?"
_POLITE_NO_RESULT = "제가 잘 모르는 부분이에요. 상담원 연결해드릴까요?"
_POLITE_DECLINE_FALLBACK = "알겠습니다. 그러면 다른 무엇을 도와드릴까요?"

_FAQ_SYSTEM_PROMPT = """당신은 매장 전화 상담 AI 입니다. 사용자의 질문에 RAG 검색 결과를 바탕으로 친절하게 답변하세요.

[지침]
- 검색 결과 컨텍스트에 있는 사실만 사용. 없는 정보는 추측하지 마세요.
- 한국어 한두 문장으로 자연스럽게 답변. 너무 길면 안 됨 (음성 안내).
- "검색 결과", "문서에 따르면" 같은 메타 표현 금지. 매장 직원처럼 답하세요.
- 컨텍스트에 답이 없으면: "그 부분은 제가 정확히 모르는데, 상담원 연결해드릴까요?"
- 출력은 답변 텍스트만. 따옴표/머릿말 금지."""


async def faq_branch_node(state: CallState) -> dict:
    query = state.get("rewritten_query") or state["user_text"]
    user_text = state.get("user_text") or ""  # is_vision 게이트의 model_id substring 매칭용
    tenant_id = state["tenant_id"]

    # 거절 패턴 — RAG 없이 generic 안내. query_refine 이 일관되게
    # "사용자가 ... 거절함" 으로 재작성하므로 string 매칭으로 충분.
    if "거절함" in query:
        print("[faq_branch] 거절 패턴 감지 → polite decline")
        return {"response_text": _POLITE_DECLINE_FALLBACK}

    # 1. 임베딩
    embedder = get_embedder()
    embedding = await embedder.embed(query)

    # 2. RAG 검색
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
        if not text:
            text = _POLITE_NO_RESULT
    except Exception as exc:
        print(f"[faq_branch] LLM 실패 → fallback: {exc}")
        text = _POLITE_NO_RESULT

    print(f"[faq_branch] response='{text}'")
    return {"response_text": text}
