"""
시연용 Post-call 컨텍스트 픽스처.

demo context 특성:
- call_id = demo-call-critical
- customer_phone 포함
- angry + escalated → Slack, SMS, Notion, Gmail, CompanyDB 액션 유도
- callback 필요 문구 포함 → Calendar, SMS 액션 유도
- critical priority → Slack, SMS, Notion 추가 유도
- JIRA_MCP_REAL → Jira 이슈 생성 유도
- POST_CALL_ENABLE_NOTION_RECORD → Notion call record 유도

소비처:
- scripts/run_post_call_demo.py
- scripts/seed_demo_completed_call_db.py

레거시 DEMO_LLM_* 상수들은 v2 리팩터(2-에이전트 그래프) 이후 모두 삭제됨 —
신규 에이전트는 mock LLM 또는 monkeypatch 로 직접 응답을 주입한다.
"""
from __future__ import annotations

DEMO_POST_CALL_CONTEXT: dict = {
    "metadata": {
        "call_id": "demo-call-critical",
        "tenant_id": "demo-tenant",
        "customer_phone": "+821049460829",
        "start_time": "2026-04-28T14:00:00Z",
        "end_time": "2026-04-28T14:14:00Z",
        "status": "completed",
    },
    "transcripts": [
        {
            "role": "customer",
            "text": "환불 요청한 지 2주가 지났는데 아직 입금이 안 됐습니다. 지난주에도 전화했는데 또 처음부터 설명해야 하나요?",
            "timestamp": "2026-04-28T14:01:00Z",
        },
        {
            "role": "agent",
            "text": "반복해서 불편을 드려 죄송합니다. 결제/환불 담당팀에 긴급 건으로 바로 에스컬레이션하겠습니다.",
            "timestamp": "2026-04-28T14:03:00Z",
        },
        {
            "role": "customer",
            "text": "오늘 안에 담당자가 직접 전화 주세요. 처리 상황도 문자로 꼭 안내해 주세요. 더는 기다리기 어렵습니다.",
            "timestamp": "2026-04-28T14:06:00Z",
        },
        {
            "role": "agent",
            "text": "네, 오늘 중 담당자 콜백 요청을 남기고 환불 지연 건으로 긴급 후속 안내 문자를 발송하겠습니다.",
            "timestamp": "2026-04-28T14:08:00Z",
        },
    ],
    "branch_stats": {"faq": 0, "task": 1, "escalation": 1},
}
