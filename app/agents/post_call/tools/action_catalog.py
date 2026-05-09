"""analysis_planner_agent 가 자율 선택하는 propose_* 도구 카탈로그.

LLM 은 이 카탈로그의 도구를 호출해 액션 후보(proposed_actions) 만 만들고,
실제 실행은 reviewer 통과 후 action_executor 가 담당한다.

각 entry 는:
  - name              : OpenAI Function Calling tool name (LLM 노출)
  - description       : LLM 이 도구 선택에 참조
  - parameters        : JSON Schema (Function Calling 표준)
  - action_type       : ActionType enum value (executor 호환)
  - tool              : Tool enum value (executor 호환)
  - required_oauth    : 이 provider 가 tenant 에 connected 일 때만 카탈로그 노출.
                        None 이면 OAuth 게이트 없음.
  - requires_notion_env : True 면 NOTION_API_TOKEN + NOTION_DATABASE_ID 둘 다
                          있을 때만 카탈로그 노출 (Notion env-based readiness).
"""
from __future__ import annotations

import os
from typing import Any

from app.models.tenant_integration import IntegrationStatus
from app.repositories.tenant_integration_repo import list_integrations
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _notion_env_ready() -> bool:
    """Notion env-based readiness — token + database id 둘 다 있을 때만 True."""
    token = (os.environ.get("NOTION_API_TOKEN") or "").strip()
    db_id = (os.environ.get("NOTION_DATABASE_ID") or "").strip()
    return bool(token) and bool(db_id)


_PROPOSE_CATALOG: list[dict[str, Any]] = [
    {
        "name": "propose_send_slack_alert",
        "description": (
            "긴급/중대 통화에 대해 Slack 알림 전송을 제안합니다. "
            "고객이 강한 불만을 표시하거나 즉각적인 팀 공유가 필요할 때 사용. "
            "urgency / channel_type 은 priority_result.priority 에서 자동 매핑되므로 "
            "별도 인자 없음 (low→info, medium→warning, high/critical→critical)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Slack 채널에 보낼 메시지 본문 (한국어)",
                },
            },
            "required": ["message"],
        },
        "action_type": "send_slack_alert",
        "tool": "slack",
        "required_oauth": "slack",
    },
    {
        "name": "propose_schedule_callback",
        "description": (
            "고객 콜백 예약을 제안합니다. 통화 중 고객이 다시 연락 요청을 했거나 "
            "에스컬레이션 후 follow-up 이 필요할 때 사용."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "preferred_time": {
                    "type": "string",
                    "description": (
                        "콜백 희망 일시 — ISO 8601 형식 'YYYY-MM-DD HH:MM' 필수. "
                        "자연어('내일 오후 3시', 'tomorrow 3pm') 절대 금지. "
                        "현재 시각 기준으로 절대 시각으로 계산해서 채워라. "
                        "시각이 transcript에 명시되어 있지 않거나 모호하면 빈 문자열로."
                    ),
                },
                "phone": {
                    "type": "string",
                    "description": "콜백할 고객 전화번호. 없으면 빈 문자열.",
                },
                "reason": {
                    "type": "string",
                    "description": "콜백 사유",
                },
            },
            "required": ["reason"],
        },
        "action_type": "schedule_callback",
        "tool": "calendar",
        "required_oauth": "google_calendar",
    },
    {
        "name": "propose_create_jira_ticket",
        "description": (
            "Jira 이슈 생성을 제안합니다. 운영팀의 후속 처리가 필요한 "
            "VOC/장애/요구사항이 있을 때 사용. priority 는 분석의 priority_result 를 "
            "자동 사용하므로 별도 인자 없음."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "이슈 한 줄 제목",
                },
                "description": {
                    "type": "string",
                    "description": "이슈 본문 (한국어)",
                },
            },
            "required": ["summary", "description"],
        },
        "action_type": "create_jira_issue",
        "tool": "jira",
        "required_oauth": "jira",
    },
    {
        "name": "propose_send_sms_followup",
        "description": (
            "고객에게 follow-up SMS 발송을 제안합니다. VOC 접수 안내, "
            "처리 진행 안내 등 고객 직접 알림이 필요할 때 사용."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "수신자 전화번호. 모르면 빈 문자열.",
                },
                "message": {
                    "type": "string",
                    "description": "SMS 본문",
                },
            },
            "required": ["message"],
        },
        "action_type": "send_voc_receipt_sms",
        "tool": "sms",
        # SMS 는 OAuth 가 아닌 자체 게이트웨이 — 항상 노출
        "required_oauth": None,
    },
    {
        "name": "propose_send_email_supervisor",
        "description": (
            "팀장/관리자에게 이메일 보고를 제안합니다. critical 우선순위, "
            "법적/정책적 이슈, 반복 에스컬레이션 등 임원 보고가 필요할 때 사용."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "이메일 제목",
                },
                "body": {
                    "type": "string",
                    "description": "이메일 본문 (한국어)",
                },
            },
            "required": ["subject", "body"],
        },
        "action_type": "send_manager_email",
        "tool": "gmail",
        "required_oauth": "gmail",
    },
    {
        "name": "propose_create_notion_call_record",
        "description": (
            "모든 통화의 기본 기록을 Notion DB 에 생성합니다. 통화 요약 / 감정 / "
            "결과 상태 등 기본 메타데이터 row 1건. **모든 통화에 대해 기본 호출** — "
            "단, 명백히 잡음 / 무음 / 잘못 걸린 전화 등 분석 가치가 없는 통화면 생략. "
            "sentiment / priority 는 분석 결과를 그대로 복사해서 넣으세요."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notion row 의 Name 컬럼 — 한 줄 요약 (50자 이내)",
                },
                "summary": {
                    "type": "string",
                    "description": "통화 요약 본문 (Notion summary 컬럼)",
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative", "angry"],
                    "description": "분석 결과의 customer_emotion 그대로",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "분석 결과의 priority_result.priority 그대로",
                },
            },
            "required": ["title", "summary", "sentiment", "priority"],
        },
        "action_type": "create_notion_call_record",
        "tool": "notion",
        "required_oauth": None,
        "requires_notion_env": True,
    },
    {
        "name": "propose_create_notion_voc_record",
        "description": (
            "VOC 후속 처리 추적용 Notion DB row 별도 생성. customer_emotion 이 "
            "angry / negative 이거나 priority 가 high / critical 인 경우에만 호출. "
            "call_record 와 별도로 VOC 만 모아 보는 db 운영을 위함. "
            "단순 inquiry / resolved 통화에는 호출 금지."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notion row 의 Name 컬럼 — VOC 한 줄 요약",
                },
                "voc_content": {
                    "type": "string",
                    "description": "VOC 본문 (고객 불만 / 요청 사항 상세)",
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative", "angry"],
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "suggested_action": {
                    "type": "string",
                    "description": "후속 권장 조치 (예: '환불 처리 후 24h 내 콜백')",
                },
            },
            "required": ["title", "voc_content", "sentiment", "priority", "suggested_action"],
        },
        "action_type": "create_notion_voc_record",
        "tool": "notion",
        "required_oauth": None,
        "requires_notion_env": True,
    },
    {
        "name": "propose_no_action",
        "description": (
            "어떤 외부 액션도 필요 없다고 판단할 때 호출하세요. "
            "단순 정보 문의, 이미 해결된 통화, 후속 처리 불필요 케이스."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "외부 액션이 필요 없는 사유",
                },
            },
            "required": ["reason"],
        },
        "action_type": "_no_action",
        "tool": "_none",
        "required_oauth": None,
    },
    {
        "name": "record_analysis",
        "description": (
            "통화 분석 결과를 기록합니다. 반드시 한 번만 호출하세요 — "
            "propose_* 도구를 호출하기 전이든 후든 이 도구 호출이 누락되면 분석이 저장되지 않습니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary_short": {
                    "type": "string",
                    "description": "한 줄 요약 (50자 이내)",
                },
                "summary_detailed": {
                    "type": "string",
                    "description": "상세 요약 (200자 이내)",
                },
                "customer_intent": {
                    "type": "string",
                    "description": "고객의 핵심 문의/요청 의도",
                },
                "customer_emotion": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative", "angry"],
                },
                "resolution_status": {
                    "type": "string",
                    "enum": ["resolved", "escalated", "abandoned"],
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "action_required": {
                    "type": "boolean",
                    "description": "후속 외부 액션 필요 여부",
                },
                "primary_category": {
                    "type": "string",
                    "description": "주요 문의 카테고리 (예: 예약/일정, 환불/결제, 민원/불만)",
                },
                "is_repeat_topic": {
                    "type": "boolean",
                    "description": "반복 문의 여부",
                },
                "faq_candidate": {
                    "type": "boolean",
                    "description": "FAQ 후보 여부",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "핵심 키워드 (최대 5개)",
                },
                "handoff_notes": {
                    "type": "string",
                    "description": "상담원 인수인계 메모. 없으면 빈 문자열.",
                },
            },
            "required": [
                "summary_short",
                "customer_intent",
                "customer_emotion",
                "resolution_status",
                "priority",
                "primary_category",
            ],
        },
        # propose_* 와 달리 record_analysis 는 액션이 아님 — 분석 기록 전용
        "action_type": None,
        "tool": None,
        "required_oauth": None,
    },
]


def get_action_catalog(tenant_id: str) -> list[dict[str, Any]]:
    """tenant 의 OAuth 통합 상태에 따라 propose_* 도구를 필터링한다.

    SMS 처럼 OAuth 무관 도구와 record_analysis / propose_no_action 은 항상 포함.
    """
    try:
        integrations = list_integrations(tenant_id)
        connected = {
            i.provider for i in integrations if i.status == IntegrationStatus.connected
        }
    except Exception as exc:
        logger.warning(
            "action_catalog: tenant_integration 조회 실패 tenant_id=%s err=%s — OAuth 비활성 카탈로그만 노출",
            tenant_id, exc,
        )
        connected = set()

    notion_ready = _notion_env_ready()
    catalog: list[dict[str, Any]] = []
    for entry in _PROPOSE_CATALOG:
        oauth = entry.get("required_oauth")
        # Notion env-based gate: 토큰/DB id 둘 다 있을 때만 카탈로그 노출
        if entry.get("requires_notion_env") and not notion_ready:
            continue
        if oauth is None or oauth in connected:
            catalog.append(entry)
    return catalog


def to_openai_tools(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """카탈로그 → OpenAI Function Calling tools 형식."""
    return [
        {
            "type": "function",
            "function": {
                "name": entry["name"],
                "description": entry["description"],
                "parameters": entry["parameters"],
            },
        }
        for entry in catalog
    ]


def find_entry(catalog: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for entry in catalog:
        if entry["name"] == name:
            return entry
    return None
