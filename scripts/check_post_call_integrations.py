"""Post-call Integration Readiness check.

특정 tenant 또는 전체 tenant의 외부 integration 연결 상태를 조회하고,
Post-call real action 실행 가능 여부를 사람이 읽기 좋게 출력한다.

사용 예:
    python scripts/check_post_call_integrations.py --tenant-id <uuid>
    python scripts/check_post_call_integrations.py --tenant-id <uuid> --show-actions
    python scripts/check_post_call_integrations.py --tenant-id <uuid> --json
    python scripts/check_post_call_integrations.py --all-tenants
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# 프로젝트 루트의 .env 명시적 로드. override=False — OS env / secret manager 우선.
# .env 가 없어도 죽지 않는다 (load_dotenv 가 missing-ok).
load_dotenv(_PROJECT_ROOT / ".env", override=False)

import asyncpg  # noqa: E402

from app.repositories.tenant_integration_repo import (  # noqa: E402
    TenantIntegrationRepository,
    tenant_integration_repo,
)
from app.utils.config import settings  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

_SEP = "─" * 56


# ── ANSI color helpers (lightweight; demo 색상 모듈에 의존하지 않음) ────────────

_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


# ── Provider classifications ──────────────────────────────────────────────────

# Providers that don't require tenant OAuth — handled by internal services.
INTERNAL_PROVIDERS = {"company_db", "internal_dashboard"}

# Providers configured at environment/runtime level rather than per-tenant OAuth.
# - sms uses Solapi env config + customer phone (action-level dependency).
# - notion uses NOTION_API_TOKEN + NOTION_DATABASE_ID; OAuth는 미사용.
ENV_CONFIGURED_PROVIDERS = {"sms", "notion"}


def _notion_env_ready() -> tuple[bool, str | None]:
    """Notion은 env 기반이므로 NOTION_API_TOKEN + NOTION_DATABASE_ID가 모두 있어야 ready."""
    token = (os.getenv("NOTION_API_TOKEN") or "").strip()
    db_id = (os.getenv("NOTION_DATABASE_ID") or "").strip()
    if token and db_id:
        return True, None
    missing: list[str] = []
    if not token:
        missing.append("NOTION_API_TOKEN")
    if not db_id:
        missing.append("NOTION_DATABASE_ID")
    return False, f"missing env: {', '.join(missing)}"


# ── SMS / Solapi readiness ────────────────────────────────────────────────────

_SMS_SENDER_ENV_ALIASES: tuple[str, ...] = (
    "SOLAPI_FROM",
    "SOLAPI_SENDER_NUMBER",
    "SOLAPI_FROM_NUMBER",
    "SMS_FROM",
)


def _resolve_sms_sender() -> str:
    """발신번호를 _SMS_SENDER_ENV_ALIASES 순서대로 조회한다. 없으면 빈 문자열."""
    for name in _SMS_SENDER_ENV_ALIASES:
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return ""


def _sms_env_state() -> dict:
    """Solapi 서버 공통 env + customer_phone/SMS_TEST_TO 상태를 분해해서 보고한다.

    Returns:
        {
          "solapi_installed":     bool,
          "credentials_present":  bool,   # SOLAPI_API_KEY + SOLAPI_API_SECRET 모두 있음
          "sender_present":       bool,   # 아래 발신번호 alias 중 하나
          "test_to_present":      bool,   # SMS_TEST_TO 있음 (로컬/시연용 fallback)
          "missing":              list[str],
        }

    발신번호 env 는 다음 alias 를 호환성 있게 처리한다 (순서 = 우선순위):
        SOLAPI_FROM, SOLAPI_SENDER_NUMBER, SOLAPI_FROM_NUMBER, SMS_FROM

    SMS 는 OAuth provider 가 아니므로 tenant_integrations DB 에 row 가 없다.
    customer_phone 은 readiness 단계에서 알 수 없고 action 실행 시점에 결정된다.
    여기서는 서버 공통 설정만 본다.
    """
    try:
        import solapi  # noqa: F401
        installed = True
    except Exception:
        installed = False

    api_key = (os.getenv("SOLAPI_API_KEY") or "").strip()
    api_secret = (os.getenv("SOLAPI_API_SECRET") or "").strip()
    sender = _resolve_sms_sender()
    test_to = (os.getenv("SMS_TEST_TO") or "").strip()

    missing: list[str] = []
    if not installed:
        missing.append("solapi_package")
    if not api_key:
        missing.append("SOLAPI_API_KEY")
    if not api_secret:
        missing.append("SOLAPI_API_SECRET")
    if not sender:
        # 어떤 alias 라도 채우면 sender_present 가 된다는 사실을 미싱 메시지에 노출.
        missing.append("SOLAPI_FROM(or " + "/".join(_SMS_SENDER_ENV_ALIASES[1:]) + ")")

    return {
        "solapi_installed":    installed,
        "credentials_present": bool(api_key and api_secret),
        "sender_present":      bool(sender),
        "test_to_present":     bool(test_to),
        "missing":             missing,
    }


def _sms_provider_status() -> dict:
    """SMS provider 단계의 readiness (action_type 단계가 아님).

    구분:
      - solapi_not_installed       → not_ready
      - solapi_credentials_missing → not_ready
      - solapi_sender_missing      → not_ready
      - 그 외                       → configured (server-common env OK)
    """
    state = _sms_env_state()
    if not state["solapi_installed"]:
        return {
            "status": "missing",
            "ready": False,
            "reason": "solapi_not_installed",
            "details": state,
        }
    if not state["credentials_present"]:
        return {
            "status": "missing",
            "ready": False,
            "reason": "solapi_credentials_missing",
            "details": state,
        }
    if not state["sender_present"]:
        return {
            "status": "missing",
            "ready": False,
            "reason": "solapi_sender_missing",
            "details": state,
        }
    return {
        "status": "configured",
        "ready": True,
        "reason": None,
        "details": state,
    }


# ── Jira readiness ────────────────────────────────────────────────────────────

_JIRA_ENV_REQUIRED = (
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "JIRA_PROJECT_KEY",
)


def _jira_env_fallback_allowed() -> bool:
    """env fallback 허용 여부 — 로컬/시연/legacy 경로용."""
    return (os.getenv("MCP_ALLOW_ENV_FALLBACK") or "").strip().lower() in ("1", "true")


def _jira_env_state() -> dict:
    """Jira env fallback (local/legacy) 상태."""
    values = {k: (os.getenv(k) or "").strip() for k in _JIRA_ENV_REQUIRED}
    missing = [k for k, v in values.items() if not v]
    return {
        "fallback_allowed": _jira_env_fallback_allowed(),
        "configured":       not missing,
        "missing":          missing,
    }


def _jira_db_status(integration: object | None) -> dict:
    """tenant_integrations 의 Jira row 를 분해해서 readiness 를 결정한다.

    DB row 가 connected 라도 workspace_id (cloud_id) / project_key 가
    빠지면 실제 issue create 가 실패하므로 별도 reason 으로 구분한다.

    Returns dict with: status / ready / reason / scopes
    """
    if integration is None:
        return {
            "status": "missing",
            "ready": False,
            "reason": "tenant_integration_not_connected",
            "scopes": [],
        }

    raw_status = getattr(integration, "status", None)
    status_value = raw_status.value if hasattr(raw_status, "value") else str(raw_status or "unknown")
    scopes = list(getattr(integration, "scopes", []) or [])

    if status_value != "connected":
        # 만료/disconnect/error 는 일반 OAuth 분기와 동일하게 처리
        return {
            "status": status_value if status_value in ("expired", "disconnected") else "invalid",
            "ready": False,
            "reason": {
                "expired":      "token expired",
                "disconnected": "manually disconnected",
            }.get(status_value, "provider returned error"),
            "scopes": scopes,
        }

    metadata = getattr(integration, "metadata", None) or {}
    cloud_id = (
        metadata.get("cloud_id")
        or metadata.get("cloudId")
        or getattr(integration, "external_workspace_id", None)
        or ""
    )
    if not cloud_id or metadata.get("workspace_selection_required"):
        return {
            "status": "incomplete",
            "ready": False,
            "reason": "jira_workspace_not_selected",
            "scopes": scopes,
        }

    project_key = (
        metadata.get("project_key")
        or os.getenv("JIRA_PROJECT_KEY", "").strip()
    )
    if not project_key:
        return {
            "status": "incomplete",
            "ready": False,
            "reason": "jira_project_not_configured",
            "scopes": scopes,
        }

    return {"status": "connected", "ready": True, "reason": None, "scopes": scopes}


def _jira_readiness(integration: object | None) -> dict:
    """Jira readiness — DB 우선, env fallback 은 MCP_ALLOW_ENV_FALLBACK=true 일 때만.

    우선순위:
      1) DB row 가 충분 (connected + workspace + project) → connected/ready
      2) DB row 가 있지만 workspace/project 부족 → incomplete + 명확한 reason
      3) DB row 없음 + fallback 허용 + env 충분 → configured_env / ready_env
      4) DB row 없음 + fallback 허용 + env 부족 → not_ready jira_env_config_missing
      5) DB row 없음 + fallback 비허용 → not_ready tenant_integration_not_connected
    """
    db = _jira_db_status(integration)
    if db["ready"]:
        return db
    if integration is not None:
        # DB row 는 있지만 workspace/project 등 일부 부족. env fallback 보다
        # DB 보강이 우선이라 env 로 우회하지 않는다 — reason 그대로 노출.
        return db

    env = _jira_env_state()
    if not env["fallback_allowed"]:
        return {
            "status": "missing",
            "ready": False,
            "reason": "tenant_integration_not_connected",
            "scopes": [],
        }
    if env["configured"]:
        return {
            "status": "configured_env",
            "ready": True,
            "reason": "env_fallback",
            "scopes": [],
        }
    return {
        "status": "missing",
        "ready": False,
        "reason": f"jira_env_config_missing: {','.join(env['missing'])}",
        "scopes": [],
    }

# All providers we surface in the report. Order matters for console display.
ALL_PROVIDERS = [
    "slack",
    "calendar",
    "notion",
    "gmail",
    "sms",
    "jira",
    "company_db",
    "internal_dashboard",
]

# action_type → required provider/tool. Source: action_planner_node.py rules.
ACTION_PROVIDER_MAP: dict[str, str] = {
    "create_voc_issue":          "company_db",
    "send_manager_email":        "gmail",
    "add_priority_queue":        "internal_dashboard",
    "send_slack_alert":          "slack",
    "send_voc_receipt_sms":      "sms",
    "schedule_callback":         "calendar",
    "send_callback_sms":         "sms",
    "mark_faq_candidate":        "internal_dashboard",
    "create_jira_issue":         "jira",
    "create_notion_call_record": "notion",
}

# Canonical provider name → tenant_integrations.provider candidates.
#
# OAuth callback (`app/api/v1/oauth.py`) stores Google integrations under
# `google_gmail` / `google_calendar`, but action/tool layer references them as
# `gmail` / `calendar`. This map lets readiness recognize a successful OAuth
# even when the stored row uses the OAuth-route name.
#
# The canonical (action-layer) name is listed first so it wins when both rows
# happen to exist.
PROVIDER_ALIASES: dict[str, list[str]] = {
    "slack":              ["slack"],
    "calendar":           ["calendar", "google_calendar"],
    "notion":             ["notion"],
    "gmail":              ["gmail", "google_gmail"],
    "sms":                ["sms"],
    "jira":               ["jira"],
    "company_db":         ["company_db"],
    "internal_dashboard": ["internal_dashboard"],
}


def _resolve_integration_for_canonical(
    canonical: str,
    integrations_by_provider: dict[str, object],
) -> tuple[object | None, str | None]:
    """Pick the best integration row for a canonical provider.

    Priority:
      1. First candidate whose row is connected.
      2. Otherwise, first candidate that has any row.
      3. Otherwise, (None, None).

    The candidate order in PROVIDER_ALIASES decides ties — connected canonical
    names beat connected OAuth-route names. (Future: use updated_at as tie
    breaker if conflicts become common.)
    """
    candidates = PROVIDER_ALIASES.get(canonical, [canonical])

    connected = None
    connected_source = None
    fallback = None
    fallback_source = None

    for cand in candidates:
        integration = integrations_by_provider.get(cand)
        if integration is None:
            continue
        raw_status = getattr(integration, "status", None)
        status_value = raw_status.value if hasattr(raw_status, "value") else str(raw_status or "")
        if status_value == "connected" and connected is None:
            connected = integration
            connected_source = cand
        elif fallback is None:
            fallback = integration
            fallback_source = cand

    if connected is not None:
        return connected, connected_source
    if fallback is not None:
        return fallback, fallback_source
    return None, None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _database_url() -> str:
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


# ── Provider status normalization ─────────────────────────────────────────────

def normalize_provider_status(
    provider: str,
    integration: object | None,
    *,
    source_provider: str | None = None,
) -> dict:
    """Return a normalized status dict for a single canonical provider.

    Output schema:
        {
          "status": "connected"|"missing"|"disconnected"|"expired"
                    |"invalid"|"configured"|"internal"|"unknown",
          "ready":  bool,
          "reason": str|None,
          "scopes": list[str],
          "source_provider":     str|None,   # tenant_integrations row provider name
          "provider_candidates": list[str],  # canonical + aliases tried
        }

    ``source_provider`` is the actual ``tenant_integrations.provider`` value the
    integration row used (e.g. ``google_gmail``). It differs from ``provider``
    (the canonical action-layer name, e.g. ``gmail``) when an alias matched.
    """
    candidates = PROVIDER_ALIASES.get(provider, [provider])
    base_meta = {
        "source_provider": source_provider,
        "provider_candidates": list(candidates),
    }

    if provider in INTERNAL_PROVIDERS:
        return {
            "status": "internal",
            "ready": True,
            "reason": "no OAuth required",
            "scopes": [],
            **base_meta,
        }
    if provider in ENV_CONFIGURED_PROVIDERS:
        if provider == "notion":
            ready, missing_reason = _notion_env_ready()
            return {
                "status": "configured" if ready else "missing",
                "ready": ready,
                "reason": missing_reason or "env/provider level config required",
                "scopes": [],
                **base_meta,
            }
        if provider == "sms":
            sms = _sms_provider_status()
            return {
                "status": sms["status"],
                "ready": sms["ready"],
                "reason": sms["reason"],
                "scopes": [],
                **base_meta,
                "sms_env": sms["details"],
            }
        return {
            "status": "configured",
            "ready": True,
            "reason": "env/provider level config required",
            "scopes": [],
            **base_meta,
        }
    if provider == "jira":
        jira = _jira_readiness(integration)
        return {
            "status": jira["status"],
            "ready": jira["ready"],
            "reason": jira["reason"],
            "scopes": jira["scopes"],
            **base_meta,
            "jira_env": _jira_env_state(),
        }
    if integration is None:
        return {
            "status": "missing",
            "ready": False,
            "reason": "no tenant integration row",
            "scopes": [],
            **base_meta,
        }

    raw_status = getattr(integration, "status", None)
    status_value = raw_status.value if hasattr(raw_status, "value") else str(raw_status or "unknown")
    scopes = list(getattr(integration, "scopes", []) or [])

    if status_value == "connected":
        return {"status": "connected", "ready": True, "reason": None, "scopes": scopes, **base_meta}
    if status_value == "expired":
        return {"status": "expired", "ready": False, "reason": "token expired", "scopes": scopes, **base_meta}
    if status_value == "disconnected":
        return {
            "status": "disconnected",
            "ready": False,
            "reason": "manually disconnected",
            "scopes": scopes,
            **base_meta,
        }
    if status_value == "error":
        return {
            "status": "invalid",
            "ready": False,
            "reason": "provider returned error",
            "scopes": scopes,
            **base_meta,
        }

    return {
        "status": "unknown",
        "ready": False,
        "reason": f"status={status_value}",
        "scopes": scopes,
        **base_meta,
    }


# ── Action readiness mapping ──────────────────────────────────────────────────

def build_action_readiness(provider_statuses: dict[str, dict]) -> dict[str, dict]:
    """Map each post-call action_type to its provider-level readiness."""
    actions: dict[str, dict] = {}
    for action_type, provider in ACTION_PROVIDER_MAP.items():
        if provider in INTERNAL_PROVIDERS:
            actions[action_type] = {
                "provider": provider,
                "ready": True,
                "ready_label": "ready_internal",
                "reason": None,
            }
        elif provider in ENV_CONFIGURED_PROVIDERS:
            if provider == "notion":
                # Notion은 env만 충족되면 action 단계에서도 ready.
                info = provider_statuses.get(provider, {})
                ready = bool(info.get("ready"))
                actions[action_type] = {
                    "provider": provider,
                    "ready": ready,
                    "ready_label": "ready" if ready else "not_ready",
                    "reason": None if ready else (info.get("reason") or "notion_not_configured"),
                }
            else:
                # SMS — server-common Solapi env 가 부족하면 customer_phone 단계까지
                # 가지도 못한다. env 부족 reason 을 그대로 surface 하고, env 가
                # 충족되어 있더라도 customer_phone 은 readiness 단계에서는 알 수
                # 없으므로 needs_customer_phone_or_sms_config 라벨을 유지한다.
                info = provider_statuses.get(provider, {})
                if info.get("ready"):
                    # SOLAPI env 충분. SMS_TEST_TO 가 있으면 시연 fallback 으로 ready.
                    test_to_present = bool((info.get("sms_env") or {}).get("test_to_present"))
                    actions[action_type] = {
                        "provider": provider,
                        "ready": test_to_present,
                        "ready_label": (
                            "ready_test_fallback" if test_to_present
                            else "needs_customer_phone_or_sms_config"
                        ),
                        "reason": None if test_to_present else (
                            "sms tool depends on env config + customer_phone"
                        ),
                    }
                else:
                    actions[action_type] = {
                        "provider": provider,
                        "ready": False,
                        "ready_label": "not_ready",
                        "reason": info.get("reason") or "sms_not_configured",
                    }
        elif provider == "jira":
            # Jira 는 DB row → workspace → project 단계별 reason 을 잃지 않게
            # provider 의 reason 을 그대로 전달한다.
            info = provider_statuses.get(provider, {})
            ready = bool(info.get("ready"))
            actions[action_type] = {
                "provider": provider,
                "ready": ready,
                "ready_label": "ready" if ready else "not_ready",
                "reason": None if ready else (info.get("reason") or "tenant_integration_not_connected"),
            }
        elif provider_statuses.get(provider, {}).get("ready"):
            actions[action_type] = {
                "provider": provider,
                "ready": True,
                "ready_label": "ready",
                "reason": None,
            }
        else:
            actions[action_type] = {
                "provider": provider,
                "ready": False,
                "ready_label": "not_ready",
                "reason": "tenant_integration_not_connected",
            }
    return actions


# ── Per-tenant readiness ──────────────────────────────────────────────────────

def check_tenant_readiness(
    tenant_id: str,
    repo: TenantIntegrationRepository,
) -> dict:
    """Build provider statuses + action readiness for a single tenant.

    Resolves canonical providers (`gmail`, `calendar`) against alias rows
    (`google_gmail`, `google_calendar`) so a successful OAuth surfaces as
    `connected` regardless of which name the row was stored under.
    """
    integrations_by_provider: dict[str, object] = {}
    for integration in repo.list_integrations(tenant_id):
        integrations_by_provider[integration.provider] = integration

    provider_statuses: dict[str, dict] = {}
    for provider in ALL_PROVIDERS:
        integration, source = _resolve_integration_for_canonical(
            provider, integrations_by_provider,
        )
        provider_statuses[provider] = normalize_provider_status(
            provider, integration, source_provider=source,
        )

    return {
        "tenant_id": tenant_id,
        "providers": provider_statuses,
        "actions": build_action_readiness(provider_statuses),
    }


# ── mcp_action_logs query ─────────────────────────────────────────────────────

def build_action_log_summary_sql() -> str:
    """SQL for recent mcp_action_logs distribution per (action_type, tool, status, error).

    Falls back to the calls table for legacy rows where tenant_id is null on
    the log itself. Uses ``c.id::text = ml.call_id`` to bridge the
    UUID/TEXT type difference.

    All column references are qualified with ``ml.`` because both
    ``mcp_action_logs`` and ``calls`` define a ``status`` column — leaving any
    reference unqualified raises ``AmbiguousColumnError`` at execution time.
    """
    return (
        "SELECT\n"
        "  ml.action_type,\n"
        "  ml.tool_name,\n"
        "  ml.status,\n"
        "  ml.error_message,\n"
        "  COUNT(*)::int AS cnt\n"
        "FROM mcp_action_logs ml\n"
        "LEFT JOIN calls c ON c.id::text = ml.call_id\n"
        "WHERE ml.tenant_id = $1::text\n"
        "   OR (ml.tenant_id IS NULL AND c.tenant_id = $1::uuid)\n"
        "GROUP BY\n"
        "  ml.action_type,\n"
        "  ml.tool_name,\n"
        "  ml.status,\n"
        "  ml.error_message\n"
        "ORDER BY cnt DESC\n"
        "LIMIT $2"
    )


async def fetch_action_log_summary(
    conn,
    tenant_id: str,
    limit: int = 20,
) -> list[dict]:
    rows = await conn.fetch(build_action_log_summary_sql(), tenant_id, limit)
    return [
        {
            "action_type":   r["action_type"],
            "tool_name":     r["tool_name"],
            "status":        r["status"],
            "error_message": r["error_message"],
            "count":         int(r["cnt"]),
        }
        for r in rows
    ]


async def fetch_distinct_tenant_ids(conn) -> list[str]:
    rows = await conn.fetch(
        "SELECT DISTINCT tenant_id::text AS tid FROM calls WHERE tenant_id IS NOT NULL ORDER BY 1",
    )
    return [r["tid"] for r in rows]


# ── Console output ────────────────────────────────────────────────────────────

def _status_color(status: str) -> str:
    if status == "connected":
        return _GREEN
    if status == "internal":
        return _CYAN
    if status == "configured":
        return _CYAN
    if status in ("missing", "disconnected", "expired", "invalid", "unknown"):
        return _YELLOW
    return ""


def print_readiness(readiness: dict) -> None:
    print()
    print(_BOLD + "Post-call Integration Readiness" + _RESET)
    print(f"\ntenant_id = {readiness['tenant_id']}")

    print()
    print("Provider Status")
    print(_SEP)
    for provider in ALL_PROVIDERS:
        info = readiness["providers"].get(provider, {})
        status = info.get("status", "unknown")
        scopes = info.get("scopes") or []
        reason = info.get("reason")
        source = info.get("source_provider")
        candidates = info.get("provider_candidates") or []

        suffix_parts: list[str] = []
        # Show source only when it differs from the canonical provider — e.g.
        # gmail row stored as google_gmail. Same-name source is implicit.
        if source and source != provider:
            suffix_parts.append(f"source={source}")
        if scopes:
            suffix_parts.append(f"scopes={','.join(scopes)}")
        if reason:
            suffix_parts.append(f"reason={reason}")
        # When missing AND there were aliases beyond the canonical name,
        # surface them so the operator knows what the lookup tried.
        if status == "missing" and len(candidates) > 1:
            suffix_parts.append(f"candidates={','.join(candidates)}")
        suffix = "  ".join(suffix_parts) if suffix_parts else ""

        color = _status_color(status)
        print(f"  {provider:<18} {_c(color, status):<22} {suffix}")

    print()
    print("Action Readiness")
    print(_SEP)
    for action_type, info in readiness["actions"].items():
        provider = info.get("provider", "—")
        label = info.get("ready_label", "—")
        reason = info.get("reason") or ""
        ready = bool(info.get("ready"))
        color = _GREEN if ready else _YELLOW
        suffix = f"  {reason}" if reason else ""
        print(f"  {action_type:<28} {provider:<20} {_c(color, label)}{suffix}")


def print_action_log_summary(rows: list[dict]) -> None:
    print()
    print("Recent mcp_action_logs Summary")
    print(_SEP)
    if not rows:
        print("  (no recent action logs for this tenant)")
        return
    for r in rows:
        status = r["status"] or "—"
        color = _GREEN if status == "success" else (_YELLOW if status == "skipped" else _RED)
        err = r["error_message"] or ""
        if err and len(err) > 60:
            err = err[:57] + "..."
        print(
            f"  {r['action_type']:<28} {r['tool_name']:<14} "
            f"{_c(color, status):<10}  count={r['count']:<3}  {err}"
        )


# ── Main async flow ───────────────────────────────────────────────────────────

async def _main(
    *,
    tenant_id: str | None,
    all_tenants: bool,
    json_output: bool,
    show_actions: bool,
    limit: int,
    repo: TenantIntegrationRepository | None = None,
) -> None:
    repo = repo or tenant_integration_repo

    # ── Resolve tenant list ──────────────────────────────────────────────────
    if all_tenants:
        try:
            conn = await asyncpg.connect(_database_url())
        except Exception as exc:
            print(f"\n{_c(_RED, 'DB connection failed:')} {exc}")
            sys.exit(1)
        try:
            tenant_ids = await fetch_distinct_tenant_ids(conn)
        finally:
            await conn.close()
    else:
        tenant_ids = [tenant_id]  # type: ignore[list-item]

    # ── Build readiness payloads ─────────────────────────────────────────────
    payloads: list[dict] = []
    for tid in tenant_ids:
        if tid is None:
            continue
        readiness = check_tenant_readiness(tid, repo)
        readiness["recent_action_summary"] = []
        payloads.append(readiness)

    # ── Optional: action log summary ─────────────────────────────────────────
    if show_actions and payloads:
        try:
            conn = await asyncpg.connect(_database_url())
        except Exception as exc:
            print(f"\n{_c(_RED, 'DB connection failed for action logs:')} {exc}")
            sys.exit(1)
        try:
            for payload in payloads:
                payload["recent_action_summary"] = await fetch_action_log_summary(
                    conn, payload["tenant_id"], limit=limit,
                )
        finally:
            await conn.close()

    # ── Output ───────────────────────────────────────────────────────────────
    if json_output:
        out = payloads[0] if (not all_tenants and len(payloads) == 1) else {"tenants": payloads}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    for payload in payloads:
        print_readiness(payload)
        if show_actions:
            print_action_log_summary(payload["recent_action_summary"])
        if all_tenants:
            print()
            print(_SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tenant 별 외부 integration 연결 상태와 Post-call action readiness를 "
            "확인한다. 실제 OAuth flow는 수정하지 않으며, 현재 상태만 진단한다."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tenant-id",
        help="특정 tenant의 integration 상태 확인",
    )
    group.add_argument(
        "--all-tenants",
        action="store_true",
        help="모든 tenant의 integration 상태 확인 (calls 테이블에서 distinct)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="JSON 형태로 출력 (단일 tenant는 객체, all-tenants는 {tenants: [...]})",
    )
    parser.add_argument(
        "--show-actions",
        action="store_true",
        help="mcp_action_logs 기반 최근 action 결과 분포도 같이 출력",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="최근 action log group 수 제한 (기본값 20)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(
        _main(
            tenant_id=args.tenant_id,
            all_tenants=args.all_tenants,
            json_output=args.json_output,
            show_actions=args.show_actions,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
