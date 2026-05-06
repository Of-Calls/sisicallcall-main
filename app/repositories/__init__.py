from app.repositories.call_repo import (
    insert_call,
    finalize_call,
    list_calls_for_tenant,
    get_call_by_id_for_tenant,
)
from app.repositories.transcript_repo import (
    insert_transcript,
    get_transcripts_by_call_id,
)
from app.repositories.call_summary_repo import (
    CallSummaryRepository,
    save_summary,
    get_summary_by_call_id,
    seed_call_context,
    get_call_context,
    get_seeded_call_context,
)
from app.repositories.voc_analysis_repo import (
    VOCAnalysisRepository,
    save_voc_analysis,
    get_voc_by_call_id,
)
from app.repositories.mcp_action_log_repo import (
    MCPActionLogRepository,
    save_action_logs,
    find_successful_action,
    get_action_logs_by_call_id,
    get_action_logs_by_call_id_for_tenant,
    get_action_logs,
)
from app.repositories.dashboard_repo import (
    DashboardRepository,
    upsert_dashboard_payload,
    get_dashboard_payload,
    get_post_call_detail,
    get_dashboard_overview,
    get_emotion_distribution,
    get_priority_queue,
)
from app.repositories.admin_user_repo import (
    create_admin_user,
    find_admin_user_by_email,
    find_admin_user_by_id,
    update_last_login,
)

__all__ = [
    # classes
    "CallSummaryRepository",
    "VOCAnalysisRepository",
    "MCPActionLogRepository",
    "DashboardRepository",
    # call (calls 테이블 — 통화 메타)
    "insert_call",
    "finalize_call",
    "list_calls_for_tenant",
    "get_call_by_id_for_tenant",
    # transcript (transcripts 테이블 — 발화 단위)
    "insert_transcript",
    "get_transcripts_by_call_id",
    # call_summary
    "save_summary",
    "get_summary_by_call_id",
    "seed_call_context",
    "get_call_context",
    "get_seeded_call_context",
    # voc_analysis
    "save_voc_analysis",
    "get_voc_by_call_id",
    # mcp_action_log
    "save_action_logs",
    "find_successful_action",
    "get_action_logs_by_call_id",
    "get_action_logs_by_call_id_for_tenant",
    "get_action_logs",
    # dashboard
    "upsert_dashboard_payload",
    "get_dashboard_payload",
    "get_post_call_detail",
    "get_dashboard_overview",
    "get_emotion_distribution",
    "get_priority_queue",
    # admin_user
    "create_admin_user",
    "find_admin_user_by_email",
    "find_admin_user_by_id",
    "update_last_login",
]
