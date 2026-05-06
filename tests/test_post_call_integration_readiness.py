"""Tests for scripts/check_post_call_integrations.py"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.tenant_integration import IntegrationStatus, TenantIntegration  # noqa: E402
from app.repositories.tenant_integration_repo import (  # noqa: E402
    TenantIntegrationRepository,
)


# ── Provider status normalization ─────────────────────────────────────────────

class TestNormalizeProviderStatus:
    def test_internal_provider_is_ready_internal(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        result = normalize_provider_status("company_db", None)
        assert result["status"] == "internal"
        assert result["ready"] is True

        result = normalize_provider_status("internal_dashboard", None)
        assert result["status"] == "internal"
        assert result["ready"] is True

    def test_env_configured_provider(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        result = normalize_provider_status("sms", None)
        assert result["status"] == "configured"
        # SMS is "ready" at provider level; per-action it requires customer_phone too.
        assert result["ready"] is True

    def test_notion_provider_missing_when_env_unset(self, monkeypatch):
        """Notion은 env 기반 — token/db_id 미설정이면 missing."""
        from scripts.check_post_call_integrations import normalize_provider_status

        monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
        monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

        result = normalize_provider_status("notion", None)
        assert result["status"] == "missing"
        assert result["ready"] is False
        assert "NOTION_API_TOKEN" in result["reason"]
        assert "NOTION_DATABASE_ID" in result["reason"]

    def test_notion_provider_configured_when_env_set(self, monkeypatch):
        """Notion env (token + db_id) 모두 채워지면 configured + ready."""
        from scripts.check_post_call_integrations import normalize_provider_status

        monkeypatch.setenv("NOTION_API_TOKEN", "secret_dummy")
        monkeypatch.setenv("NOTION_DATABASE_ID", "db_dummy")

        result = normalize_provider_status("notion", None)
        assert result["status"] == "configured"
        assert result["ready"] is True

    def test_oauth_provider_no_row_is_missing(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        result = normalize_provider_status("slack", None)
        assert result["status"] == "missing"
        assert result["ready"] is False
        assert result["reason"] == "no tenant integration row"

    def test_connected_integration(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        integration = TenantIntegration(
            tenant_id="t-1",
            provider="slack",
            status=IntegrationStatus.connected,
            scopes=["chat:write", "channels:read"],
        )
        result = normalize_provider_status("slack", integration)
        assert result["status"] == "connected"
        assert result["ready"] is True
        assert result["scopes"] == ["chat:write", "channels:read"]

    def test_disconnected_integration(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        integration = TenantIntegration(
            tenant_id="t-1",
            provider="slack",
            status=IntegrationStatus.disconnected,
        )
        result = normalize_provider_status("slack", integration)
        assert result["status"] == "disconnected"
        assert result["ready"] is False

    def test_expired_integration(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        integration = TenantIntegration(
            tenant_id="t-1",
            provider="gmail",
            status=IntegrationStatus.expired,
        )
        result = normalize_provider_status("gmail", integration)
        assert result["status"] == "expired"
        assert result["ready"] is False

    def test_error_integration_normalized_to_invalid(self):
        from scripts.check_post_call_integrations import normalize_provider_status

        integration = TenantIntegration(
            tenant_id="t-1",
            provider="calendar",
            status=IntegrationStatus.error,
        )
        result = normalize_provider_status("calendar", integration)
        assert result["status"] == "invalid"
        assert result["ready"] is False


# ── Action readiness mapping ──────────────────────────────────────────────────

class TestBuildActionReadiness:
    def test_action_readiness_reflects_provider_status(self):
        from scripts.check_post_call_integrations import build_action_readiness

        provider_statuses = {
            "slack":              {"status": "connected", "ready": True},
            "calendar":           {"status": "missing",   "ready": False},
            "notion":             {"status": "connected", "ready": True},
            "gmail":              {"status": "missing",   "ready": False},
            "sms":                {"status": "configured", "ready": True},
            "jira":               {"status": "missing",   "ready": False},
            "company_db":         {"status": "internal",  "ready": True},
            "internal_dashboard": {"status": "internal",  "ready": True},
        }
        actions = build_action_readiness(provider_statuses)

        # Slack connected → ready
        assert actions["send_slack_alert"]["ready"] is True
        assert actions["send_slack_alert"]["ready_label"] == "ready"

        # Calendar missing → not_ready
        assert actions["schedule_callback"]["ready"] is False
        assert actions["schedule_callback"]["reason"] == "tenant_integration_not_connected"

        # Notion connected → ready
        assert actions["create_notion_call_record"]["ready"] is True

        # Gmail missing → not_ready
        assert actions["send_manager_email"]["ready"] is False

        # Internal providers always ready_internal
        assert actions["create_voc_issue"]["ready_label"] == "ready_internal"
        assert actions["add_priority_queue"]["ready_label"] == "ready_internal"
        assert actions["mark_faq_candidate"]["ready_label"] == "ready_internal"

    def test_sms_action_marked_with_customer_phone_caveat(self):
        from scripts.check_post_call_integrations import build_action_readiness

        actions = build_action_readiness(
            {"sms": {"status": "configured", "ready": True}},
        )
        sms_action = actions["send_callback_sms"]
        assert sms_action["provider"] == "sms"
        assert sms_action["ready_label"] == "needs_customer_phone_or_sms_config"


# ── check_tenant_readiness with in-memory repo ────────────────────────────────

class TestCheckTenantReadiness:
    def _repo(self) -> TenantIntegrationRepository:
        # Storage="memory" + manual clear keeps tests isolated from other suites.
        repo = TenantIntegrationRepository(storage="memory")
        return repo

    def test_no_integrations_all_oauth_providers_missing(self, monkeypatch):
        from scripts.check_post_call_integrations import check_tenant_readiness

        # Notion은 env 기반이라 실제 환경의 .env 값 영향을 받지 않도록 격리한다.
        monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
        monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

        result = check_tenant_readiness("tid-empty", self._repo())

        assert result["providers"]["slack"]["status"] == "missing"
        assert result["providers"]["gmail"]["status"] == "missing"
        assert result["providers"]["notion"]["status"] == "missing"
        # Internal providers always present
        assert result["providers"]["company_db"]["status"] == "internal"
        assert result["providers"]["internal_dashboard"]["status"] == "internal"

    def test_connected_provider_marks_action_ready(self):
        from scripts.check_post_call_integrations import check_tenant_readiness

        repo = self._repo()
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-a",
            provider="slack",
            status=IntegrationStatus.connected,
            scopes=["chat:write"],
        ))

        result = check_tenant_readiness("tid-a", repo)

        assert result["providers"]["slack"]["status"] == "connected"
        assert result["providers"]["slack"]["ready"] is True
        assert result["actions"]["send_slack_alert"]["ready"] is True

        # Other providers still missing
        assert result["providers"]["gmail"]["status"] == "missing"
        assert result["actions"]["send_manager_email"]["ready"] is False

    def test_internal_provider_ready_without_integration_row(self):
        from scripts.check_post_call_integrations import check_tenant_readiness

        result = check_tenant_readiness("tid-empty", self._repo())

        assert result["providers"]["company_db"]["ready"] is True
        assert result["actions"]["create_voc_issue"]["ready_label"] == "ready_internal"


# ── mcp_action_logs SQL ───────────────────────────────────────────────────────

class TestActionLogSummarySql:
    def test_sql_filters_by_tenant_and_uses_text_join(self):
        from scripts.check_post_call_integrations import build_action_log_summary_sql

        sql = build_action_log_summary_sql()

        # tenant filter on log itself
        assert "ml.tenant_id = $1::text" in sql
        # legacy fallback through calls
        assert "c.id::text = ml.call_id" in sql
        assert "c.tenant_id = $1::uuid" in sql
        # group by + limit
        assert "GROUP BY" in sql
        assert "LIMIT $2" in sql

    def test_sql_qualifies_all_columns_with_ml_alias(self):
        """status exists on both mcp_action_logs and calls — every column
        reference must be qualified with ml. to avoid AmbiguousColumnError."""
        from scripts.check_post_call_integrations import build_action_log_summary_sql

        sql = build_action_log_summary_sql()

        # ml-qualified references must be present
        assert "ml.action_type" in sql
        assert "ml.tool_name" in sql
        assert "ml.status" in sql
        assert "ml.error_message" in sql

        # No bare GROUP BY on unqualified columns (would trigger ambiguity)
        assert "GROUP BY status" not in sql
        assert "GROUP BY action_type" not in sql

        # Defensive: no c.status leaking in (we want the action log status,
        # not the call status)
        assert "c.status" not in sql

    @pytest.mark.asyncio
    async def test_fetch_action_log_summary_passes_tenant_and_limit(self):
        from scripts.check_post_call_integrations import fetch_action_log_summary

        captured: list = []

        async def mock_fetch(sql, *params):
            captured.append((sql, params))
            return [
                {
                    "action_type":   "send_slack_alert",
                    "tool_name":     "slack",
                    "status":        "success",
                    "error_message": None,
                    "cnt":           3,
                },
                {
                    "action_type":   "send_manager_email",
                    "tool_name":     "gmail",
                    "status":        "failed",
                    "error_message": "tenant_integration_not_connected",
                    "cnt":           1,
                },
            ]

        mock_conn = MagicMock()
        mock_conn.fetch = mock_fetch

        rows = await fetch_action_log_summary(mock_conn, "tid-a", limit=10)

        assert len(captured) == 1
        assert captured[0][1] == ("tid-a", 10)
        assert rows[0]["action_type"] == "send_slack_alert"
        assert rows[0]["count"] == 3
        assert rows[1]["error_message"] == "tenant_integration_not_connected"


# ── Provider alias resolution ─────────────────────────────────────────────────

class TestProviderAliases:
    def _repo(self) -> TenantIntegrationRepository:
        return TenantIntegrationRepository(storage="memory")

    def test_alias_map_contains_google_pairings(self):
        from scripts.check_post_call_integrations import PROVIDER_ALIASES

        assert "google_gmail" in PROVIDER_ALIASES["gmail"]
        assert "google_calendar" in PROVIDER_ALIASES["calendar"]
        # canonical name must come first so it wins ties
        assert PROVIDER_ALIASES["gmail"][0] == "gmail"
        assert PROVIDER_ALIASES["calendar"][0] == "calendar"

    def test_google_gmail_row_resolves_to_gmail_connected(self):
        from scripts.check_post_call_integrations import check_tenant_readiness

        repo = self._repo()
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-a",
            provider="google_gmail",
            status=IntegrationStatus.connected,
            scopes=["https://www.googleapis.com/auth/gmail.send"],
        ))

        result = check_tenant_readiness("tid-a", repo)

        assert result["providers"]["gmail"]["status"] == "connected"
        assert result["providers"]["gmail"]["ready"] is True
        assert result["providers"]["gmail"]["source_provider"] == "google_gmail"
        # send_manager_email action should reflect alias result
        assert result["actions"]["send_manager_email"]["ready"] is True

    def test_google_calendar_row_resolves_to_calendar_connected(self):
        from scripts.check_post_call_integrations import check_tenant_readiness

        repo = self._repo()
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-b",
            provider="google_calendar",
            status=IntegrationStatus.connected,
            scopes=["https://www.googleapis.com/auth/calendar.events"],
        ))

        result = check_tenant_readiness("tid-b", repo)

        assert result["providers"]["calendar"]["status"] == "connected"
        assert result["providers"]["calendar"]["source_provider"] == "google_calendar"
        # schedule_callback action should reflect alias result
        assert result["actions"]["schedule_callback"]["ready"] is True

    def test_provider_candidates_in_payload(self):
        from scripts.check_post_call_integrations import check_tenant_readiness

        result = check_tenant_readiness("tid-empty", self._repo())

        assert result["providers"]["gmail"]["provider_candidates"] == ["gmail", "google_gmail"]
        assert result["providers"]["calendar"]["provider_candidates"] == ["calendar", "google_calendar"]
        # non-aliased canonical providers still expose a candidates list
        assert result["providers"]["slack"]["provider_candidates"] == ["slack"]

    def test_no_alias_row_keeps_missing_status(self):
        from scripts.check_post_call_integrations import check_tenant_readiness

        result = check_tenant_readiness("tid-empty", self._repo())

        # No row at all → still missing for OAuth providers
        assert result["providers"]["gmail"]["status"] == "missing"
        assert result["providers"]["gmail"]["source_provider"] is None
        assert result["actions"]["send_manager_email"]["ready"] is False

    def test_canonical_row_wins_when_both_connected(self):
        """If both `gmail` and `google_gmail` are connected, the canonical
        name comes first in PROVIDER_ALIASES and therefore wins."""
        from scripts.check_post_call_integrations import check_tenant_readiness

        repo = self._repo()
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-c", provider="gmail",
            status=IntegrationStatus.connected,
        ))
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-c", provider="google_gmail",
            status=IntegrationStatus.connected,
        ))

        result = check_tenant_readiness("tid-c", repo)
        assert result["providers"]["gmail"]["source_provider"] == "gmail"

    def test_connected_alias_beats_disconnected_canonical(self):
        """`gmail` disconnected but `google_gmail` connected → use the alias."""
        from scripts.check_post_call_integrations import check_tenant_readiness

        repo = self._repo()
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-d", provider="gmail",
            status=IntegrationStatus.disconnected,
        ))
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-d", provider="google_gmail",
            status=IntegrationStatus.connected,
        ))

        result = check_tenant_readiness("tid-d", repo)
        assert result["providers"]["gmail"]["status"] == "connected"
        assert result["providers"]["gmail"]["source_provider"] == "google_gmail"


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCli:
    def test_no_tenant_option_raises_system_exit(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["check_integrations"])

        from scripts.check_post_call_integrations import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_both_tenant_and_all_tenants_raises_system_exit(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["check_integrations", "--tenant-id", "t-1", "--all-tenants"],
        )

        from scripts.check_post_call_integrations import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_tenant_id_only_is_accepted(self, monkeypatch):
        """argparse accepts --tenant-id without DB roundtrip when _main is patched."""
        from scripts import check_post_call_integrations as mod

        monkeypatch.setattr(
            sys, "argv",
            ["check_integrations", "--tenant-id", "t-1"],
        )

        called = {"value": False}

        async def fake_main(**_kwargs):
            called["value"] = True

        monkeypatch.setattr(mod, "_main", fake_main)

        mod.main()
        assert called["value"] is True


# ── _main JSON output structure ───────────────────────────────────────────────

class TestMainJsonOutput:
    @pytest.mark.asyncio
    async def test_json_payload_has_expected_keys(self, monkeypatch, capsys):
        """--json output should include providers, actions, recent_action_summary."""
        from scripts import check_post_call_integrations as mod

        repo = TenantIntegrationRepository(storage="memory")
        repo.upsert_integration(TenantIntegration(
            tenant_id="tid-a",
            provider="slack",
            status=IntegrationStatus.connected,
            scopes=["chat:write"],
        ))

        await mod._main(
            tenant_id="tid-a",
            all_tenants=False,
            json_output=True,
            show_actions=False,
            limit=20,
            repo=repo,
        )

        captured = capsys.readouterr()
        import json as _json
        payload = _json.loads(captured.out)

        assert "providers" in payload
        assert "actions" in payload
        assert "recent_action_summary" in payload
        assert payload["providers"]["slack"]["status"] == "connected"
        assert payload["actions"]["send_slack_alert"]["ready"] is True

        # Alias metadata must round-trip through the JSON path so external
        # consumers can tell which row backed each canonical provider.
        gmail = payload["providers"]["gmail"]
        assert "source_provider" in gmail
        assert gmail["provider_candidates"] == ["gmail", "google_gmail"]
        calendar = payload["providers"]["calendar"]
        assert calendar["provider_candidates"] == ["calendar", "google_calendar"]


# ── run_post_call_from_db.py guard message ────────────────────────────────────

class TestRealActionsGuardMessage:
    def test_guard_message_present_on_real_actions(self):
        """Loosely verify the guard text exists in the runner module."""
        import scripts.run_post_call_from_db as runner_mod
        import inspect

        source = inspect.getsource(runner_mod)
        assert "Real actions enabled" in source
        assert "check_post_call_integrations.py" in source
