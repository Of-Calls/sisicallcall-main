"""터미널에서 텍스트 입력 → 그래프 실행 → 결과 출력.

매 턴마다:
    1. Redis 에서 session_view 로드
    2. state 에 주입 후 그래프 실행
    3. 응답을 Redis 에 append (다음 턴이 history 로 사용)

사용법:
    python -m scripts.graph_test
"""
import asyncio
import sys
import uuid
from pathlib import Path

# 프로젝트 루트를 sys.path 에 추가 — `python scripts/graph_test.py` 직접 실행 지원
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from app.agents.conversational.graph import build_graph
from app.services.session.redis_session import RedisSessionService
from app.services.tenant import get_tenant_meta


async def main():
    graph = build_graph()
    session = RedisSessionService()

    call_id = str(uuid.uuid4())
    tenant_id = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"  # 한밭식당 (검증용 임시값)
    tenant_name, tenant_industry = await get_tenant_meta(tenant_id)

    print(f"call_id={call_id} tenant_id={tenant_id} name={tenant_name} industry={tenant_industry}")
    print("종료: 'exit' 또는 Ctrl+C")
    print("세션 초기화: 'clear'\n")

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료")
            await session.clear(call_id)
            break

        if not user_text:
            continue
        if user_text.lower() == "exit":
            await session.clear(call_id)
            break
        if user_text.lower() == "clear":
            await session.clear(call_id)
            print("세션 초기화됨\n")
            continue

        # 1. Redis 에서 session_view 로드
        session_view = await session.load(call_id)

        # 2. state 구성 + 그래프 실행
        state = {
            "call_id": call_id,
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "tenant_industry": tenant_industry,
            "user_text": user_text,
            "intent": "",
            "response_text": "",
            "session_view": session_view,
            "rewritten_query": "",
            "is_clear": False,
            "missing_info": "",
        }
        result = await graph.ainvoke(state)
        response_text = result["response_text"]

        # 3. 이번 턴을 Redis 에 저장
        await session.append_turn(call_id, user_text, response_text)

        print(f"응답: {response_text}\n")


if __name__ == "__main__":
    asyncio.run(main())
