"""Tests for scripts/run_post_call_batch_from_db.py"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── build_target_calls_sql ────────────────────────────────────────────────────

class TestBuildTargetCallsSql:
    def test_with_tenant_id_includes_filter(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id="ba2bf499-6fcc-4340-b3dd-9341f8bcc915",
            limit=5,
            offset=0,
            only_missing=False,
        )

        assert "c.tenant_id = $1::uuid" in sql
        assert "ba2bf499-6fcc-4340-b3dd-9341f8bcc915" in params
        assert 5 in params

    def test_without_tenant_id_no_tenant_filter(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=10,
            offset=0,
            only_missing=False,
        )

        # c.tenant_id appears in SELECT; the WHERE filter must NOT appear
        assert "AND c.tenant_id" not in sql
        # params should only contain limit and offset
        assert params == [10, 0]

    def test_only_missing_adds_having_clause(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=5,
            offset=0,
            only_missing=True,
        )

        assert "HAVING" in sql
        assert "MAX(cs.call_id::text) IS NULL" in sql
        assert "MAX(va.call_id::text) IS NULL" in sql

    def test_without_only_missing_no_having(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=5,
            offset=0,
            only_missing=False,
        )

        assert "HAVING" not in sql

    def test_tenant_id_and_only_missing_combined(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id="tenant-uuid",
            limit=3,
            offset=2,
            only_missing=True,
        )

        assert "c.tenant_id = $1::uuid" in sql
        assert "HAVING" in sql
        assert params[0] == "tenant-uuid"
        assert 3 in params
        assert 2 in params

    def test_limit_and_offset_are_parameterized(self):
        from scripts.run_post_call_batch_from_db import build_target_calls_sql

        sql, params = build_target_calls_sql(
            tenant_id=None,
            limit=7,
            offset=14,
            only_missing=False,
        )

        assert "LIMIT" in sql
        assert "OFFSET" in sql
        assert 7 in params
        assert 14 in params


# ── run_batch ─────────────────────────────────────────────────────────────────

class TestRunBatch:
    @pytest.mark.asyncio
    async def test_transcript_count_zero_is_skipped(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 0},
        ]

        with patch.object(runner_mod, "run_post_call_for_completed_call") as mock_runner:
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert len(results) == 1
        assert results[0]["status"] == "skip"
        assert results[0]["skip_reason"] == "transcripts_missing"
        mock_runner.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_runner(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 10},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 5},
        ]

        with patch.object(runner_mod, "run_post_call_for_completed_call") as mock_runner:
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=True)

        mock_runner.assert_not_called()
        assert all(r["status"] == "dry_run" for r in results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_single_call_failure_continues_batch(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 5},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 3},
        ]

        async def _mock_runner(call_id, tenant_id, trigger):
            if call_id == "call-001":
                return {"ok": False, "result": None, "error": "call_context_not_found"}
            return _ok_outcome()

        with patch.object(runner_mod, "run_post_call_for_completed_call", side_effect=_mock_runner):
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert len(results) == 2
        assert results[0]["status"] == "fail"
        assert results[0]["error"] == "call_context_not_found"
        assert results[1]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_exception_in_call_continues_batch(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 5},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 3},
        ]

        async def _mock_runner(call_id, tenant_id, trigger):
            if call_id == "call-001":
                raise RuntimeError("unexpected internal error")
            return _ok_outcome()

        with patch.object(runner_mod, "run_post_call_for_completed_call", side_effect=_mock_runner):
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert len(results) == 2
        assert results[0]["status"] == "fail"
        assert "unexpected internal error" in results[0]["error"]
        assert results[1]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_transcript_skip_and_ok_mixed(self):
        from scripts.run_post_call_batch_from_db import run_batch
        import app.agents.post_call.completed_call_runner as runner_mod

        calls = [
            {"call_id": "call-001", "tenant_id": "tenant-a", "transcript_count": 0},
            {"call_id": "call-002", "tenant_id": "tenant-a", "transcript_count": 5},
        ]

        async def _mock_runner(call_id, tenant_id, trigger):
            return _ok_outcome()

        with patch.object(runner_mod, "run_post_call_for_completed_call", side_effect=_mock_runner):
            results = await run_batch(calls=calls, trigger="call_ended", dry_run=False)

        assert results[0]["status"] == "skip"
        assert results[1]["status"] == "ok"


# ── extract_call_result ───────────────────────────────────────────────────────

class TestExtractCallResult:
    def test_failed_outcome(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        result = extract_call_result(
            call_id="call-001",
            tenant_id="tenant-a",
            transcript_count=5,
            outcome={"ok": False, "result": None, "error": "call_context_not_found"},
        )

        assert result["status"] == "fail"
        assert result["error"] == "call_context_not_found"
        assert result["call_id"] == "call-001"

    def test_ok_outcome_extracts_all_fields(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        result = extract_call_result("call-001", "tenant-a", 10, _ok_outcome())

        assert result["status"] == "ok"
        assert result["review_verdict"] == "pass"
        assert result["review_confidence"] == 0.92
        assert result["review_confidence_source"] == "llm"
        assert result["primary_category"] == "예약/일정"
        assert result["customer_emotion"] == "neutral"
        assert result["resolution_status"] == "resolved"
        assert result["priority"] == "medium"
        assert result["sentiment"] == "neutral"
        assert result["action_success"] == 1
        assert result["action_skipped"] == 1
        assert result["action_failed"] == 0
        assert result["action_plan_count"] == 1
        assert result["executed_count"] == 2

    def test_empty_result_uses_fallback_values(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        outcome = {"ok": True, "result": {}, "error": None}
        result = extract_call_result("call-001", "tenant-a", 3, outcome)

        assert result["status"] == "ok"
        assert result["review_verdict"] == "—"
        assert result["primary_category"] == "—"
        assert result["customer_emotion"] == "—"
        assert result["action_plan_count"] == 0
        assert result["action_success"] == 0


# ── fetch_tenant_report SQL ───────────────────────────────────────────────────

class TestFetchTenantReportSql:
    @pytest.mark.asyncio
    async def test_all_queries_use_tenant_id(self):
        from scripts.run_post_call_batch_from_db import fetch_tenant_report

        tenant_id = "ba2bf499-6fcc-4340-b3dd-9341f8bcc915"
        captured: list[tuple] = []

        async def mock_fetchrow(sql, *params):
            captured.append(("fetchrow", params))
            return (0,)

        async def mock_fetch(sql, *params):
            captured.append(("fetch", params))
            return []

        mock_conn = MagicMock()
        mock_conn.fetchrow = mock_fetchrow
        mock_conn.fetch = mock_fetch

        await fetch_tenant_report(mock_conn, tenant_id)

        assert len(captured) > 0
        for entry in captured:
            params = entry[1]
            assert tenant_id in params, f"tenant_id not in params for entry: {entry}"

    @pytest.mark.asyncio
    async def test_action_log_mismatch_uses_text_cast(self):
        from scripts.run_post_call_batch_from_db import fetch_tenant_report

        captured_sqls: list[str] = []

        async def mock_fetchrow(sql, *params):
            captured_sqls.append(sql)
            return (0,)

        async def mock_fetch(sql, *params):
            captured_sqls.append(sql)
            return []

        mock_conn = MagicMock()
        mock_conn.fetchrow = mock_fetchrow
        mock_conn.fetch = mock_fetch

        await fetch_tenant_report(mock_conn, "test-tenant-id")

        action_log_sqls = [s for s in captured_sqls if "mcp_action_logs" in s]
        assert len(action_log_sqls) == 1, "mcp_action_logs query should appear exactly once"
        assert "c.tenant_id::text" in action_log_sqls[0]
        assert "$1::text" in action_log_sqls[0]

    @pytest.mark.asyncio
    async def test_report_returns_expected_keys(self):
        from scripts.run_post_call_batch_from_db import fetch_tenant_report

        mock_conn = MagicMock()
        mock_conn.fetchrow = AsyncMock(return_value=(0,))
        mock_conn.fetch = AsyncMock(return_value=[])

        report = await fetch_tenant_report(mock_conn, "tenant-a")

        expected_keys = {
            "call_type", "emotion", "priority", "resolution",
            "missing_primary_category",
            "tenant_mismatch_summary",
            "tenant_mismatch_voc",
            "tenant_mismatch_action_logs",
        }
        assert expected_keys == set(report.keys())


# ── CLI — missing required group ─────────────────────────────────────────────

class TestCli:
    def test_no_tenant_option_raises_system_exit(self, monkeypatch):
        """Neither --tenant-id nor --all-tenants should produce an error."""
        monkeypatch.setattr(sys, "argv", ["batch_runner", "--limit", "5", "--llm-mode", "mock"])

        from scripts.run_post_call_batch_from_db import main

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code != 0

    def test_all_tenants_flag_accepted(self, monkeypatch):
        """--all-tenants should be accepted by argparse without a parse error.

        _main is patched to a no-op async function so the test never touches
        the DB and produces no RuntimeWarning about unawaited coroutines.
        """
        monkeypatch.setattr(
            sys, "argv",
            ["batch_runner", "--all-tenants", "--limit", "1", "--llm-mode", "mock", "--dry-run"],
        )

        from scripts import run_post_call_batch_from_db as batch_runner

        called = {"value": False}

        async def fake_main(**_kwargs):
            called["value"] = True

        monkeypatch.setattr(batch_runner, "_main", fake_main)

        # Must not raise SystemExit (which argparse raises on parse errors).
        batch_runner.main()

        assert called["value"] is True


# ── LLM mode override ────────────────────────────────────────────────────────

class TestLlmModeOverride:
    def test_cli_llm_mode_real_overrides_mock_env(self, monkeypatch):
        import scripts.run_post_call_from_db as db_runner
        import app.agents.post_call.llm_caller as llm_mod

        monkeypatch.setenv("POST_CALL_LLM_MODE", "mock")
        result = db_runner._apply_llm_mode("real")
        assert result == "real"
        assert llm_mod.get_post_call_llm_mode() == "real"

    def test_cli_llm_mode_mock_overrides_real_env(self, monkeypatch):
        import scripts.run_post_call_from_db as db_runner
        import app.agents.post_call.llm_caller as llm_mod

        monkeypatch.setenv("POST_CALL_LLM_MODE", "real")
        result = db_runner._apply_llm_mode("mock")
        assert result == "mock"
        assert llm_mod.get_post_call_llm_mode() == "mock"


# ── _collect_tenant_ids ───────────────────────────────────────────────────────

class TestCollectTenantIds:
    def test_single_tenant_returns_that_tenant(self):
        from scripts.run_post_call_batch_from_db import _collect_tenant_ids

        results = [
            {"tenant_id": "tid-a", "status": "ok"},
            {"tenant_id": "tid-a", "status": "ok"},
        ]
        ids = _collect_tenant_ids(tenant_id="tid-a", all_tenants=False, results=results)
        assert ids == ["tid-a"]

    def test_all_tenants_collects_unique_ordered(self):
        from scripts.run_post_call_batch_from_db import _collect_tenant_ids

        results = [
            {"tenant_id": "tid-b", "status": "ok"},
            {"tenant_id": "tid-a", "status": "ok"},
            {"tenant_id": "tid-b", "status": "fail"},
        ]
        ids = _collect_tenant_ids(tenant_id=None, all_tenants=True, results=results)
        assert ids == ["tid-b", "tid-a"]

    def test_empty_results_returns_empty(self):
        from scripts.run_post_call_batch_from_db import _collect_tenant_ids

        ids = _collect_tenant_ids(tenant_id=None, all_tenants=True, results=[])
        assert ids == []


# ── Report export ────────────────────────────────────────────────────────────

class TestReportExport:
    # format inference
    def test_infer_format_json_extension(self):
        from scripts.run_post_call_batch_from_db import _infer_output_format
        assert _infer_output_format("report.json", None) == "json"

    def test_infer_format_csv_extension(self):
        from scripts.run_post_call_batch_from_db import _infer_output_format
        assert _infer_output_format("report.csv", None) == "csv"

    def test_infer_format_md_extension(self):
        from scripts.run_post_call_batch_from_db import _infer_output_format
        assert _infer_output_format("report.md", None) == "md"

    def test_infer_format_markdown_extension(self):
        from scripts.run_post_call_batch_from_db import _infer_output_format
        assert _infer_output_format("report.markdown", None) == "md"

    def test_infer_format_unknown_extension_defaults_to_json(self):
        from scripts.run_post_call_batch_from_db import _infer_output_format
        assert _infer_output_format("report.txt", None) == "json"
        assert _infer_output_format("report", None) == "json"

    def test_explicit_format_overrides_extension(self):
        from scripts.run_post_call_batch_from_db import _infer_output_format
        assert _infer_output_format("report.json", "csv") == "csv"
        assert _infer_output_format("report.csv", "md") == "md"

    # JSON export
    def test_export_json_contains_required_keys(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_json

        output = str(tmp_path / "report.json")
        export_json(
            path=output,
            metadata={"llm_mode": "mock", "tenant_id": "tid-a"},
            targets=[{"call_id": "c1", "tenant_id": "tid-a", "transcript_count": 5}],
            records=[_ok_record()],
            tenant_reports={"tid-a": _ok_tenant_report()},
        )

        data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert "metadata" in data
        assert "targets" in data
        assert "records" in data
        assert "tenant_reports" in data
        assert data["metadata"]["llm_mode"] == "mock"
        assert len(data["records"]) == 1
        assert "tid-a" in data["tenant_reports"]

    def test_export_json_tenant_report_has_data_quality(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_json

        output = str(tmp_path / "report.json")
        export_json(
            path=output,
            metadata={},
            targets=[],
            records=[],
            tenant_reports={"tid-a": _ok_tenant_report()},
        )

        data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        tid_report = data["tenant_reports"]["tid-a"]
        assert "data_quality" in tid_report
        assert "missing_primary_category" in tid_report["data_quality"]
        assert "tenant_mismatch_action_logs" in tid_report["data_quality"]

    # CSV export
    def test_export_csv_contains_required_columns(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_csv

        output = str(tmp_path / "report.csv")
        export_csv(path=output, records=[_ok_record()])

        with open(output, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []

        required = [
            "status", "call_id", "tenant_id", "transcript_count",
            "review_verdict", "review_confidence", "primary_category",
            "customer_emotion", "resolution_status", "priority",
            "action_plan_count", "executed_count",
            "action_success", "action_skipped", "action_failed",
            "error",
        ]
        for col in required:
            assert col in columns, f"Missing CSV column: {col}"

    def test_export_csv_utf8_sig_encoding(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_csv

        output = str(tmp_path / "report.csv")
        export_csv(path=output, records=[_ok_record()])

        raw = (tmp_path / "report.csv").read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf", "CSV must start with UTF-8 BOM (utf-8-sig)"

    def test_export_csv_dry_run_records(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_csv

        records = [
            {"call_id": "c1", "tenant_id": "tid-a", "transcript_count": 5, "status": "dry_run"},
        ]
        output = str(tmp_path / "dry_run.csv")
        export_csv(path=output, records=records)

        with open(output, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["status"] == "dry_run"
        assert rows[0]["call_id"] == "c1"

    def test_export_csv_none_values_become_empty_string(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_csv

        record = {**_ok_record(), "error": None, "review_confidence": None}
        output = str(tmp_path / "report.csv")
        export_csv(path=output, records=[record])

        with open(output, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["error"] == ""
        assert rows[0]["review_confidence"] == ""

    # Markdown export
    def test_export_markdown_contains_required_sections(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "report.md")
        export_markdown(
            path=output,
            metadata={"llm_mode": "mock"},
            targets=[],
            records=[_ok_record()],
            tenant_reports={"tid-a": _ok_tenant_report()},
        )

        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "# Post-call Batch Report" in content
        assert "## Metadata" in content
        assert "## Call Results" in content
        assert "## Tenant Reports" in content
        assert "#### Data Quality" in content

    def test_export_markdown_utf8_sig_encoding(self, tmp_path):
        """Markdown export must start with UTF-8 BOM for Windows tooling."""
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "report.md")
        export_markdown(
            path=output,
            metadata={"llm_mode": "mock"},
            targets=[],
            records=[_ok_record()],
            tenant_reports={"tid-a": _ok_tenant_report()},
        )

        raw = (tmp_path / "report.md").read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf", "Markdown must start with UTF-8 BOM (utf-8-sig)"

    def test_export_markdown_preserves_korean_strings(self, tmp_path):
        """Korean characters in records and tenant reports must round-trip intact."""
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "report.md")
        export_markdown(
            path=output,
            metadata={},
            targets=[],
            records=[_ok_record()],
            tenant_reports={"tid-a": _ok_tenant_report()},
        )

        # utf-8-sig decoding strips the BOM transparently
        content = (tmp_path / "report.md").read_text(encoding="utf-8-sig")
        assert "예약/일정" in content
        assert "neutral" in content

    def test_export_markdown_empty_tenant_reports_omits_section(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "report.md")
        export_markdown(
            path=output,
            metadata={},
            targets=[],
            records=[],
            tenant_reports={},
        )

        content = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "## Tenant Reports" not in content

    # Parent directory creation
    def test_output_parent_dir_created_for_json(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_json

        output = str(tmp_path / "nested" / "deep" / "report.json")
        export_json(path=output, metadata={}, targets=[], records=[], tenant_reports={})

        assert os.path.isfile(output)

    def test_output_parent_dir_created_for_csv(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_csv

        output = str(tmp_path / "sub" / "report.csv")
        export_csv(path=output, records=[])

        assert os.path.isfile(output)

    def test_output_parent_dir_created_for_markdown(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "sub" / "report.md")
        export_markdown(path=output, metadata={}, targets=[], records=[], tenant_reports={})

        assert os.path.isfile(output)

    # write_report dispatcher
    def test_write_report_dispatches_json_by_extension(self, tmp_path):
        from scripts.run_post_call_batch_from_db import write_report

        output = str(tmp_path / "out.json")
        write_report(output, None, {}, [], [], {})

        data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert "metadata" in data

    def test_write_report_dispatches_csv_by_explicit_format(self, tmp_path):
        from scripts.run_post_call_batch_from_db import write_report

        output = str(tmp_path / "out.csv")
        write_report(output, "csv", {}, [], [_ok_record()], {})

        raw = (tmp_path / "out.csv").read_bytes()
        assert raw[:3] == b"\xef\xbb\xbf"

    def test_write_report_dispatches_md_by_extension(self, tmp_path):
        from scripts.run_post_call_batch_from_db import write_report

        output = str(tmp_path / "out.md")
        write_report(output, None, {"llm_mode": "mock"}, [], [], {})

        content = (tmp_path / "out.md").read_text(encoding="utf-8")
        assert "# Post-call Batch Report" in content


# ── LLM usage in batch records ────────────────────────────────────────────────

class TestUsageInRecords:
    def test_extract_call_result_includes_usage_fields(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        outcome = _outcome_with_usage(
            analysis_usage={
                "purpose": "analysis", "model": "gpt-4o-mini",
                "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                "source": "openai", "fallback": False,
            },
            review_usage={
                "purpose": "review", "model": "gpt-4o-mini",
                "prompt_tokens": 800, "completion_tokens": 150, "total_tokens": 950,
                "source": "openai", "fallback": False,
            },
        )
        r = extract_call_result("c1", "t1", 5, outcome)

        assert r["llm_model"] == "gpt-4o-mini"
        assert r["llm_fallback"] is False
        assert r["analysis_total_tokens"] == 1200
        assert r["review_total_tokens"] == 950
        assert r["total_tokens"] == 2150
        assert r["total_prompt_tokens"] == 1800
        assert r["total_completion_tokens"] == 350
        assert isinstance(r["estimated_cost_usd"], float)

    def test_extract_call_result_mock_mode_usage_none(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        outcome = _outcome_with_usage(analysis_usage=None, review_usage=None)
        r = extract_call_result("c1", "t1", 5, outcome)

        assert r["llm_model"] is None
        assert r["total_tokens"] is None
        assert r["estimated_cost_usd"] is None

    def test_extract_call_result_fallback_marked_from_usage(self):
        from scripts.run_post_call_batch_from_db import extract_call_result

        outcome = _outcome_with_usage(
            analysis_usage={
                "purpose": "analysis", "model": "gpt-4o-mini",
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                "source": "fallback", "fallback": True,
            },
            review_usage=None,
        )
        r = extract_call_result("c1", "t1", 5, outcome)

        assert r["llm_fallback"] is True


class TestComputeUsageSummary:
    def test_aggregates_analysis_and_review_tokens(self):
        from scripts.run_post_call_batch_from_db import compute_usage_summary

        records = [
            _record_with_usage(a_pt=100, a_ct=20, a_tt=120, r_pt=80, r_ct=15, r_tt=95, model="gpt-4o-mini"),
            _record_with_usage(a_pt=200, a_ct=40, a_tt=240, r_pt=160, r_ct=30, r_tt=190, model="gpt-4o-mini"),
        ]
        summary = compute_usage_summary(records)

        assert summary["model"] == "gpt-4o-mini"
        assert summary["calls_with_usage"] == 2
        assert summary["analysis"]["prompt_tokens"] == 300
        assert summary["analysis"]["total_tokens"] == 360
        assert summary["review"]["total_tokens"] == 285
        assert summary["total"]["total_tokens"] == 645
        assert summary["total"]["estimated_cost_usd"] is not None

    def test_fallback_calls_counted(self):
        from scripts.run_post_call_batch_from_db import compute_usage_summary

        ok_record = _record_with_usage(model="gpt-4o-mini")
        fallback_record = {**ok_record, "llm_fallback": True}
        summary = compute_usage_summary([ok_record, fallback_record])

        assert summary["fallback_calls"] == 1

    def test_mock_mode_summary_zero_usage(self):
        from scripts.run_post_call_batch_from_db import compute_usage_summary

        mock_record = {
            "status": "ok", "call_id": "c1", "tenant_id": "t1",
            "llm_fallback": False, "llm_model": None, "total_tokens": None,
            "analysis_prompt_tokens": None, "analysis_completion_tokens": None,
            "analysis_total_tokens": None,
            "review_prompt_tokens": None, "review_completion_tokens": None,
            "review_total_tokens": None,
        }
        summary = compute_usage_summary([mock_record])

        assert summary["calls_with_usage"] == 0
        assert summary["model"] is None
        assert summary["total"]["total_tokens"] == 0
        assert summary["total"]["estimated_cost_usd"] is None

    def test_skipped_records_not_counted(self):
        from scripts.run_post_call_batch_from_db import compute_usage_summary

        records = [
            _record_with_usage(a_tt=100, r_tt=50, model="gpt-4o-mini"),
            {"status": "skip", "call_id": "c2"},
            {"status": "fail", "call_id": "c3", "error": "x"},
        ]
        summary = compute_usage_summary(records)

        assert summary["calls_with_usage"] == 1
        assert summary["analysis"]["total_tokens"] == 100


class TestUsageInExports:
    def test_csv_export_includes_usage_columns(self, tmp_path):
        from scripts.run_post_call_batch_from_db import _CSV_COLUMNS, export_csv

        for col in [
            "llm_model", "llm_mode", "llm_fallback",
            "analysis_prompt_tokens", "analysis_completion_tokens", "analysis_total_tokens",
            "review_prompt_tokens", "review_completion_tokens", "review_total_tokens",
            "total_prompt_tokens", "total_completion_tokens", "total_tokens",
            "estimated_cost_usd",
        ]:
            assert col in _CSV_COLUMNS, f"Missing CSV usage column: {col}"

        record = _record_with_usage(model="gpt-4o-mini", a_tt=100, r_tt=50)
        record["llm_mode"] = "real"
        record["estimated_cost_usd"] = 0.0001
        record["total_tokens"] = 150

        output = str(tmp_path / "report.csv")
        export_csv(path=output, records=[record])

        with open(output, encoding="utf-8-sig") as f:
            row = list(csv.DictReader(f))[0]

        assert row["llm_model"] == "gpt-4o-mini"
        assert row["llm_mode"] == "real"
        assert row["total_tokens"] == "150"

    def test_json_export_includes_usage_summary(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_json

        output = str(tmp_path / "report.json")
        records = [_record_with_usage(model="gpt-4o-mini", a_tt=100, r_tt=50)]
        export_json(
            path=output, metadata={"llm_mode": "real"},
            targets=[], records=records, tenant_reports={},
        )

        data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert "usage_summary" in data
        usage = data["usage_summary"]
        assert usage["model"] == "gpt-4o-mini"
        assert usage["calls_with_usage"] == 1
        assert usage["total"]["total_tokens"] == 150

    def test_markdown_export_real_mode_includes_usage_summary(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "report.md")
        records = [_record_with_usage(model="gpt-4o-mini", a_tt=100, r_tt=50)]
        export_markdown(
            path=output, metadata={"llm_mode": "real"},
            targets=[], records=records, tenant_reports={},
        )

        content = (tmp_path / "report.md").read_text(encoding="utf-8-sig")
        assert "## LLM Usage Summary" in content
        assert "gpt-4o-mini" in content
        assert "estimated_cost_usd" in content

    def test_markdown_export_mock_mode_shows_unavailable(self, tmp_path):
        from scripts.run_post_call_batch_from_db import export_markdown

        output = str(tmp_path / "report.md")
        # mock mode record (no usage)
        record = {
            "status": "ok", "call_id": "c1", "tenant_id": "t1",
            "llm_fallback": False, "llm_model": None, "total_tokens": None,
            "analysis_total_tokens": None, "review_total_tokens": None,
            "estimated_cost_usd": None,
        }
        export_markdown(
            path=output, metadata={"llm_mode": "mock"},
            targets=[], records=[record], tenant_reports={},
        )

        content = (tmp_path / "report.md").read_text(encoding="utf-8-sig")
        assert "## LLM Usage Summary" in content
        assert "usage unavailable in mock mode" in content


# ── helpers ───────────────────────────────────────────────────────────────────

def _outcome_with_usage(analysis_usage: dict | None, review_usage: dict | None) -> dict:
    return {
        "ok": True,
        "result": {
            "review_verdict": "pass",
            "review_result": {"confidence": 0.9, "confidence_source": "llm", "corrected_keys": []},
            "human_review_required": False,
            "executed_actions": [],
            "action_plan": {"actions": []},
            "summary": {"summary_short": "ok", "customer_emotion": "neutral", "resolution_status": "resolved"},
            "priority_result": {"priority": "low"},
            "voc_analysis": {
                "intent_result": {"primary_category": "category"},
                "sentiment_result": {"sentiment": "neutral"},
            },
            "partial_success": False,
            "analysis_llm_usage": analysis_usage,
            "review_llm_usage": review_usage,
        },
        "error": None,
    }


def _record_with_usage(
    *,
    a_pt: int = 100, a_ct: int = 20, a_tt: int = 120,
    r_pt: int = 80,  r_ct: int = 15, r_tt: int = 95,
    model: str = "gpt-4o-mini",
    fallback: bool = False,
) -> dict:
    return {
        "status": "ok",
        "call_id": "c1",
        "tenant_id": "tid-a",
        "transcript_count": 5,
        "llm_model": model,
        "llm_fallback": fallback,
        "analysis_prompt_tokens": a_pt,
        "analysis_completion_tokens": a_ct,
        "analysis_total_tokens": a_tt,
        "review_prompt_tokens": r_pt,
        "review_completion_tokens": r_ct,
        "review_total_tokens": r_tt,
        "total_prompt_tokens": a_pt + r_pt,
        "total_completion_tokens": a_ct + r_ct,
        "total_tokens": a_tt + r_tt,
    }


def _ok_outcome() -> dict:
    return {
        "ok": True,
        "result": {
            "review_verdict": "pass",
            "review_result": {
                "confidence": 0.92,
                "confidence_source": "llm",
                "corrected_keys": [],
            },
            "human_review_required": False,
            "executed_actions": [
                {"status": "success"},
                {"status": "skipped"},
            ],
            "action_plan": {"actions": [{"action_type": "create_task"}]},
            "summary": {
                "summary_short": "예약 문의",
                "customer_emotion": "neutral",
                "resolution_status": "resolved",
            },
            "priority_result": {"priority": "medium"},
            "voc_analysis": {
                "intent_result": {"primary_category": "예약/일정"},
                "sentiment_result": {"sentiment": "neutral"},
            },
            "partial_success": False,
        },
        "error": None,
    }


def _ok_record() -> dict:
    """Flat result record as returned by extract_call_result (status=ok)."""
    return {
        "status": "ok",
        "call_id": "call-001",
        "tenant_id": "tid-a",
        "transcript_count": 10,
        "review_verdict": "pass",
        "review_confidence": 0.92,
        "review_confidence_source": "llm",
        "human_review_required": False,
        "primary_category": "예약/일정",
        "customer_emotion": "neutral",
        "resolution_status": "resolved",
        "priority": "medium",
        "sentiment": "neutral",
        "action_plan_count": 1,
        "executed_count": 2,
        "action_success": 1,
        "action_skipped": 1,
        "action_failed": 0,
        "error": None,
    }


def _ok_tenant_report() -> dict:
    """Tenant report dict as returned by fetch_tenant_report."""
    return {
        "call_type": {"예약/일정": 1},
        "emotion": {"neutral": 1},
        "priority": {"medium": 1},
        "resolution": {"resolved": 1},
        "missing_primary_category": 0,
        "tenant_mismatch_summary": 0,
        "tenant_mismatch_voc": 0,
        "tenant_mismatch_action_logs": 0,
    }
