"""reviewer_agent 가 사용하는 검증 도구 카탈로그.

ReAct 루프에서 LLM 이 호출 → 노드가 결과를 messages 에 append (observe).
도구 자체는 transcript / proposed_actions 를 읽어 결과를 반환할 뿐,
실제 외부 시스템에는 영향 없음.

finalize_review 호출 시 루프 종료.
"""
from __future__ import annotations

REVIEW_TOOLS_OPENAI: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "re_read_transcript",
            "description": (
                "통화 녹취에서 특정 키워드/주제와 관련된 부분을 다시 읽습니다. "
                "분석 필드의 근거를 확인하거나 액션 후보의 정당성을 검증할 때 사용."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "찾을 키워드/주제 (한국어 가능)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_field_grounding",
            "description": (
                "분석 결과의 특정 필드 (예: customer_emotion, priority) 값이 transcript 에 "
                "근거하는지 검증합니다. 결과는 {ok: bool, reason: str}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "검증할 필드 (예: 'summary.customer_emotion', 'priority_result.priority')",
                    },
                    "value": {
                        "type": "string",
                        "description": "현재 값",
                    },
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_action",
            "description": "특정 액션 후보(action_id)를 승인합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                },
                "required": ["action_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_action",
            "description": "특정 액션 후보를 거부합니다 — 외부 호출되지 않습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["action_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_action",
            "description": (
                "액션 후보의 params 를 보정합니다. 예: 잘못된 메시지 본문 수정, "
                "잘못된 시각 수정 등. priority 는 분석의 priority_result 가 단일 source "
                "이므로 correct_analysis(priority_result.priority) 를 사용하세요. "
                "보정 후 그 액션은 자동 승인됩니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {"type": "string"},
                    "new_params": {
                        "type": "object",
                        "description": (
                            "기존 params 에 deep-merge 할 새 값. value 는 string / "
                            "number / boolean / null 만. JSON 'null' 문자열 사용 금지."
                        ),
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "null"},
                            ]
                        },
                    },
                },
                "required": ["action_id", "new_params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "correct_analysis",
            "description": (
                "분석 결과의 특정 필드를 새 값으로 교정합니다. "
                "neutral / low / resolved 같은 '기본값으로의 변경'은 금지 — "
                "transcript 에 다른 값을 직접 뒷받침하는 발화가 있을 때만 변경. "
                "강한 반대 증거 없으면 호출하지 마세요. transcript_evidence 에 "
                "transcript 원문을 인용해야 합니다 (substring 으로 검증됨)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "필드 경로 (예: 'summary.handoff_notes')",
                    },
                    "new_value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ],
                        "description": (
                            "새 값. JSON null 사용 (문자열 'null' 금지)."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "교정 사유 — 분석 측면 설명",
                    },
                    "transcript_evidence": {
                        "type": "string",
                        "description": (
                            "transcript 원문에서 인용한 발화 (한 문장 이상). "
                            "이 문자열이 transcript 에 substring 으로 실제로 존재해야 "
                            "교정이 적용됩니다. 인용할 게 없으면 보정 호출 자체 금지."
                        ),
                    },
                },
                "required": ["field", "new_value", "reason", "transcript_evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "분석 또는 액션 후보가 위험하여 자동 처리 불가일 때 호출합니다. "
                "verdict=fail 로 강제하고 모든 액션을 차단합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_review",
            "description": (
                "검토를 종료합니다. 모든 액션 결정과 분석 교정 후 반드시 호출하세요. "
                "verdict 는 결정에 따라 자동 산출됩니다. "
                "confidence (0.0~1.0) 는 reviewer 자체 판정 신뢰도 — R4 자동 강등에 사용. "
                "0.6 미만 + verdict=pass → 자동 correctable 강등, 0.4 미만 → 자동 fail 강등."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["pass", "correctable", "fail"],
                        "description": "최종 verdict (없으면 자동 산출)",
                    },
                    "summary_reason": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": (
                            "reviewer 자체 판정 신뢰도 (0.0~1.0). "
                            "1.0=완벽 일치, 0.8=명확하나 일부 모호, 0.6=핵심 맞으나 보정 필요, "
                            "0.4=신뢰 부족, 0.2=심각한 grounding 의심, 0.0=명확히 어긋남. "
                            "생략 시 1.0 으로 간주 (강등 없음, 기존 호환)."
                        ),
                    },
                },
                "required": ["summary_reason"],
            },
        },
    },
]
