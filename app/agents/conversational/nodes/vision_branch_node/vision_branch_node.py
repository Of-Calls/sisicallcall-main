import os

from app.agents.conversational.prompts.fallback_phrases import get_inquiry_phrase
from app.agents.conversational.state import CallState
from app.services.embedding import get_embedder
from app.services.llm.gpt4o_mini import GPT4OMiniService
from app.services.rag.chroma import ChromaRAGService
from app.services.session.redis_session import RedisSessionService
from app.services.sms import get_sms_service
from app.services.vision.session import VisionSessionService
from app.utils.config import settings

_llm = GPT4OMiniService()
_rag = ChromaRAGService()
_vision_session_svc = VisionSessionService()
_call_session_svc = RedisSessionService()
_sms_svc = get_sms_service()

_TOP_K = 3
# faq_branch 와 동일 — ChromaDB default L2 distance (정규화 임베딩, 작을수록 유사).
_DIST_THRESHOLD = 0.85

_POLITE_SMS_FAILED = "사진 업로드 링크 발송에 문제가 생겼어요. 잠시 후 다시 시도해주세요."
_POLITE_SMS_SENT = (
    "어떤 정수기인지 확인할 수 있도록 휴대폰으로 사진 업로드 링크를 보내드렸어요. "
    "사진 업로드가 끝나면 저에게 말씀해주세요. 어떤 모델인지 바로 안내드릴게요."
)
_POLITE_NOT_RECEIVED = "아직 사진이 도착하지 않았어요. 업로드 후 다시 알려주세요."
_POLITE_ANALYZING = "사진을 분석 중이에요. 잠시만 기다려주세요."
_POLITE_FAILED = "죄송해요, 사진 분석에 문제가 생겼어요. 상담원으로 연결해드릴게요."


def _polite_no_phone(industry: str) -> str:
    return f"사진 안내를 위한 정보가 부족해요. {get_inquiry_phrase(industry)}."


def _polite_no_result(industry: str) -> str:
    return f"해당 모델 정보를 찾기 어려워요. {get_inquiry_phrase(industry)}."

_VISION_HUMANIZE_PROMPT = """당신은 매장 전화 상담 AI 입니다.
사용자가 사진을 보내 인식된 제품 모델과 그 모델의 설명 데이터(RAG)가 주어집니다.
이를 바탕으로 친절한 음성 안내로 한두 문장 응답하세요.

[지침]
- 첫 문장에서 인식된 모델명을 자연스럽게 언급. (예: "샘솟 B1 정수기네요")
- 이어서 검색 결과의 핵심 사양을 짧게 요약.
- 검색 결과에 없는 정보는 추측하지 마세요.
- "검색 결과", "RAG", "메타", "도구" 같은 메타 표현 금지. 매장 직원처럼 답하세요.
- 출력은 응답 텍스트만. 따옴표/머릿말 금지."""


async def _create_new_vision(call_id: str, tenant_id: str, customer_phone: str) -> dict:
    """새 vision 세션 생성 → SMS 발송 → 통화 세션에 vision_id 저장."""
    vision_id = await _vision_session_svc.create_session(
        tenant_id=tenant_id,
        customer_phone=customer_phone,
        call_id=call_id,
    )
    print(f"[vision_branch] 세션 생성 vision_id={vision_id}")

    upload_url = f"{settings.auth_web_base_url}/vision/{vision_id}"
    sms_body = f"[시시콜콜] 정수기 사진 업로드 링크입니다.\n{upload_url}"
    sent = await _sms_svc.send_sms(to=customer_phone, body=sms_body)
    print(f"[vision_branch] SMS 발송 sent={sent} to={customer_phone}")

    if not sent:
        return {"response_text": _POLITE_SMS_FAILED}

    await _call_session_svc.set_vision_id(call_id, vision_id)
    print(f"[vision_branch] 통화 세션에 vision_id 저장")
    return {"response_text": _POLITE_SMS_SENT}


async def _humanize_with_rag(label: str, tenant_id: str, tenant_industry: str = "") -> str:
    """analyzed 결과 라벨로 ChromaDB 에서 model_spec 청크 검색 → LLM humanize.

    where 필터: doc_type=model_spec AND model_id=label.
    fallback (NO_RESULT) 메시지는 tenant_industry 기반.
    """
    query = f"{label} 정수기 사양"
    embedder = get_embedder()
    embedding = await embedder.embed_query(query)

    where = {
        "$and": [
            {"doc_type": {"$eq": "model_spec"}},
            {"model_id": {"$eq": label}},
        ]
    }
    results = await _rag.search_with_meta(embedding, tenant_id, top_k=_TOP_K, where=where)
    print(f"[vision_branch] RAG label={label} results={len(results)}")
    for i, r in enumerate(results):
        meta = r.get("metadata", {}) or {}
        print(
            f"  [{i+1}] distance={r.get('distance')} "
            f"model_id='{meta.get('model_id', '')}' "
            f"title='{meta.get('llm_title', '')}'"
        )

    related = [
        r for r in results
        if r.get("distance") is not None and r["distance"] <= _DIST_THRESHOLD
    ]
    if not related:
        print(f"[vision_branch] threshold 통과 청크 없음 → no_result")
        return _polite_no_result(tenant_industry)

    context = "\n\n".join(
        f"[청크 {i+1}]\n{r.get('document', '')}" for i, r in enumerate(related)
    )
    user_message = (
        f"[인식된 모델명]\n{label}\n\n"
        f"[모델 설명 검색 결과]\n{context}"
    )
    try:
        text = await _llm.generate(
            system_prompt=_VISION_HUMANIZE_PROMPT,
            user_message=user_message,
            temperature=0.2,
            max_tokens=200,
        )
        text = text.strip().strip('"').strip("'")
        return text or _polite_no_result(tenant_industry)
    except Exception as exc:
        print(f"[vision_branch] LLM 실패 → fallback: {exc}")
        return _polite_no_result(tenant_industry)


async def vision_branch_node(state: CallState) -> dict:
    tenant_id = state["tenant_id"]
    tenant_industry = state.get("tenant_industry", "")
    call_id = state["call_id"]

    customer_phone = os.getenv("SMS_TEST_RECIPIENT", "")
    if not customer_phone:
        print("[vision_branch] SMS_TEST_RECIPIENT 미설정 → polite_no_phone")
        return {"response_text": _polite_no_phone(tenant_industry)}

    existing_vision_id = await _call_session_svc.get_vision_id(call_id)
    if existing_vision_id:
        session = await _vision_session_svc.get_session(existing_vision_id)
        if session is None:
            print(f"[vision_branch] 기존 vision_id={existing_vision_id} TTL 만료 → 재발급")
            return await _create_new_vision(call_id, tenant_id, customer_phone)

        status = session.get("status", "")
        print(f"[vision_branch] 기존 vision_id={existing_vision_id} status={status}")

        if status == "analyzed":
            label = session.get("label", "")
            await _call_session_svc.clear_vision_id(call_id)
            if not label:
                print(f"[vision_branch] analyzed 인데 label 비어있음 → no_result")
                return {"response_text": _polite_no_result(tenant_industry)}
            text = await _humanize_with_rag(label, tenant_id, tenant_industry)
            return {"response_text": text}
        if status == "failed":
            # 새 사이클 가능하도록 정리. 실제 escalation 라우팅은 escalation 노드 구현 시점.
            await _call_session_svc.clear_vision_id(call_id)
            return {"response_text": _POLITE_FAILED}
        if status == "analyzing":
            return {"response_text": _POLITE_ANALYZING}
        # pending / 그 외 → 사진 미수신 안내
        return {"response_text": _POLITE_NOT_RECEIVED}

    return await _create_new_vision(call_id, tenant_id, customer_phone)
