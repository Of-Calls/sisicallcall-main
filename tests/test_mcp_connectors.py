"""
MCP Connector 계층 테스트.

검증 범위:
  1.  Gmail connector mock success
  2.  Calendar connector mock success
  3.  Jira connector mock success
  4.  Slack connector mock success
  5.  CompanyDB connector mock success
  6.  real mode env 켰지만 설정 부족 → skipped 또는 failed 반환
  7.  MCPClient가 tool_name으로 connector를 찾아 실행
  8.  MCPClient unknown tool → failed 반환
  9.  connector 예외 발생 시 → failed 반환
  10. connector 결과가 status/external_id/result/error 4개 키를 모두 포함
  11. MCPClient.registered_tools()로 등록된 tool 목록 확인
  12. CompanyDB: MCP_COMPANY_DB_REAL env도 real mode로 인식
  13. real mode env + config OK 이어도 connector 미구현이면 skipped
  14. tenant OAuth: 연동된 테넌트 → tenant_token_found_but_real_execute_not_implemented (Gmail)
  15. tenant OAuth: 미연동 테넌트 → tenant_integration_not_connected
  16. tenant OAuth: 연동 후 폴백 없음 → env fallback 없이 skipped 반환
  17. tenant OAuth: MCP_ALLOW_ENV_FALLBACK=true + 미연동 → mock 결과 반환
  18. tenant OAuth: _oauth_provider_name 설정 확인
  19. Calendar: tenant token 있을 때 Google Calendar API events.insert 성공
  20. Calendar: Google Calendar API HTTP 오류 → failed 반환
  21. Calendar: tenant token 없음 → skipped("tenant_integration_not_connected")
  22. Calendar: params calendar_id > GOOGLE_CALENDAR_ID env > primary 우선순위
  23. Calendar: start_time/end_time 직접 지정 시 이벤트 바디에 반영
  24. Calendar: preferred_time만 있을 때 end_time 자동 생성 (default duration)
  25. Calendar: 시간 정보 없을 때 현재+1시간 기본값 사용
  26. Calendar: 결과에 access_token 평문 미포함
"""
from __future__ import annotations

import pytest

from app.services.mcp.connectors.gmail_connector import GmailConnector
from app.services.mcp.connectors.calendar_connector import CalendarConnector
from app.services.mcp.connectors.jira_connector import JiraConnector
from app.services.mcp.connectors.slack_connector import SlackConnector
from app.services.mcp.connectors.company_db_connector import CompanyDBConnector
from app.services.mcp.client import MCPClient


# ── 1. Gmail connector mock success ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_gmail_connector_mock_success(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email",
        {"to": "manager@example.com", "subject": "테스트", "body": "본문"},
        call_id="conn-001",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "gmail-mock-conn-001"
    assert result["result"]["sent"] is True
    assert result["result"]["to"] == "manager@example.com"
    assert result["result"]["subject"] == "테스트"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 2. Calendar connector mock success ───────────────────────────────────────

@pytest.mark.asyncio
async def test_calendar_connector_mock_success(monkeypatch):
    monkeypatch.delenv("CALENDAR_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback",
        {"title": "콜백 예약", "customer_phone": "010-1234-5678"},
        call_id="conn-002",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "calendar-mock-conn-002"
    assert result["result"]["scheduled"] is True
    assert result["result"]["title"] == "콜백 예약"
    assert result["result"]["customer_phone"] == "010-1234-5678"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 3. Jira connector mock success ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_jira_connector_mock_success(monkeypatch):
    monkeypatch.setenv("JIRA_MCP_REAL", "false")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue",
        {"summary_short": "요금 오류", "reason": "청구서 금액 불일치", "labels": ["billing"]},
        call_id="conn-003",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "jira-mock-conn-003"
    assert result["result"]["mock"] is True
    assert result["result"]["summary"] == "요금 오류"
    assert result["result"]["description"] == "청구서 금액 불일치"
    assert result["result"]["labels"] == ["billing"]
    assert result["error"] is None


# ── 4. Slack connector mock success ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_connector_mock_success(monkeypatch):
    monkeypatch.setenv("SLACK_MCP_REAL", "false")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert",
        {"channel": "#alerts", "message": "[CRITICAL] 테스트"},
        call_id="conn-004",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "slack-mock-conn-004"
    assert result["result"]["channel"] == "#alerts"
    assert result["result"]["message"] == "[CRITICAL] 테스트"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 5. CompanyDB connector mock success ──────────────────────────────────────

@pytest.mark.asyncio
async def test_company_db_connector_mock_success():
    connector = CompanyDBConnector()
    result = await connector.execute(
        "create_voc_issue",
        {"tier": "high", "priority": "urgent", "primary_category": "billing", "reason": "요금 오류"},
        call_id="conn-005",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "VOC-MOCK-conn-005"
    assert result["result"]["created"] is True
    assert result["result"]["tier"] == "high"
    assert result["result"]["priority"] == "urgent"
    assert result["result"]["primary_category"] == "billing"
    assert result["result"]["mock"] is True
    assert result["error"] is None


# ── 6. real mode env 켰지만 설정 부족 → skipped 반환 ─────────────────────────

@pytest.mark.asyncio
async def test_gmail_connector_real_mode_config_missing_returns_skipped(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.delenv("GMAIL_MANAGER_TO", raising=False)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {}, call_id="conn-006",
    )

    assert result["status"] in ("skipped", "failed")
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_jira_connector_real_mode_config_missing_returns_skipped(monkeypatch):
    # real mode + tenant OAuth 미설정 → tenant_oauth_required
    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue", {}, call_id="conn-006b",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"


@pytest.mark.asyncio
async def test_slack_connector_real_mode_config_missing_returns_skipped(monkeypatch):
    # real mode + tenant OAuth 미설정 → tenant_oauth_required
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert", {}, call_id="conn-006c",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"


# ── 7. MCPClient가 tool_name으로 connector를 찾아 실행 ────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_routes_to_correct_connector(monkeypatch):
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    client = MCPClient()
    client.register_connector("gmail", GmailConnector())

    result = await client.call_tool(
        "gmail", "send_manager_email",
        {"subject": "라우팅 테스트"},
        call_id="client-001",
    )

    assert result["status"] == "success"
    assert "gmail-mock-client-001" == result["external_id"]


# ── 8. MCPClient unknown tool → failed 반환 ──────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_unknown_tool_returns_failed():
    client = MCPClient()
    result = await client.call_tool(
        "nonexistent_tool", "some_action", {},
        call_id="client-002",
    )

    assert result["status"] == "failed"
    assert result["error"] is not None
    assert "unknown tool" in result["error"]


# ── 9. connector 예외 발생 시 → failed 반환 ──────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_connector_exception_returns_failed():
    from app.services.mcp.connectors.base import BaseMCPConnector

    class ExplodingConnector(BaseMCPConnector):
        connector_name = "exploding"

        async def execute(self, action_type, params, *, call_id, tenant_id=""):
            raise RuntimeError("의도적 폭발")

    client = MCPClient()
    client.register_connector("exploding", ExplodingConnector())

    result = await client.call_tool(
        "exploding", "boom", {},
        call_id="client-003",
    )

    assert result["status"] == "failed"
    assert "의도적 폭발" in result["error"]


# ── 10. connector 결과가 4개 표준 키를 포함 ──────────────────────────────────

def _force_all_connectors_mock(monkeypatch):
    for key in (
        "GMAIL_MCP_REAL",
        "CALENDAR_MCP_REAL",
        "JIRA_MCP_REAL",
        "SLACK_MCP_REAL",
        "SMS_MCP_REAL",
        "NOTION_MCP_REAL",
        "COMPANY_DB_MCP_REAL",
        "MCP_COMPANY_DB_REAL",
        "MCP_USE_TENANT_OAUTH",
    ):
        monkeypatch.setenv(key, "false")


@pytest.mark.asyncio
async def test_connector_result_has_standard_keys(monkeypatch):
    _force_all_connectors_mock(monkeypatch)
    connectors = [
        ("gmail", GmailConnector(), "send_manager_email", {}),
        ("calendar", CalendarConnector(), "schedule_callback", {}),
        ("jira", JiraConnector(), "create_jira_issue", {}),
        ("slack", SlackConnector(), "send_slack_alert", {}),
        ("company_db", CompanyDBConnector(), "create_voc_issue", {}),
    ]
    for name, connector, action_type, params in connectors:
        result = await connector.execute(action_type, params, call_id=f"key-test-{name}")
        for key in ("status", "external_id", "result", "error"):
            assert key in result, f"{name} connector 결과에 {key!r} 키가 없음"


# ── 11. MCPClient.registered_tools() ─────────────────────────────────────────

def test_mcp_client_registered_tools():
    from app.services.mcp.client import mcp_client

    tools = mcp_client.registered_tools()
    for expected in ("gmail", "calendar", "company_db", "jira", "slack", "sms", "notion"):
        assert expected in tools, f"기본 등록 tool {expected!r} 이 없음"


# ── 12. CompanyDB: MCP_COMPANY_DB_REAL도 real mode로 인식 ─────────────────────

def test_company_db_connector_legacy_env_var(monkeypatch):
    monkeypatch.setenv("MCP_COMPANY_DB_REAL", "true")
    monkeypatch.delenv("COMPANY_DB_MCP_REAL", raising=False)

    connector = CompanyDBConnector()
    assert connector.is_real_mode() is True


# ── 13. real mode + config OK → connector 미구현이면 skipped ─────────────────

@pytest.mark.asyncio
async def test_calendar_connector_real_mode_config_ok_returns_skipped(monkeypatch):
    monkeypatch.setenv("CALENDAR_MCP_REAL", "true")
    monkeypatch.setenv("CALENDAR_DEFAULT_OWNER", "owner@example.com")

    connector = CalendarConnector()
    assert connector.is_real_mode() is True
    ok, err = connector.validate_config()
    assert ok is True

    result = await connector.execute(
        "schedule_callback", {}, call_id="conn-013",
    )

    assert result["status"] in ("skipped", "failed")


# ── 14. Gmail: tenant token 있을 때 Gmail API messages.send 성공 ───────────────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_connected(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import (
        tenant_integration_repo, upsert_integration,
    )
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    plaintext_token = "ya29.fake-google-access-token"
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("GMAIL_MANAGER_TO", "manager@example.com")
    reset_fernet_cache()
    tenant_integration_repo.clear_integrations()

    enc_token = encrypt_token(plaintext_token)
    upsert_integration(TenantIntegration(
        tenant_id="gmail-tenant-001",
        provider="google_gmail",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
    ))

    api_response = {"id": "msg-abc123", "threadId": "thread-001", "labelIds": ["SENT"]}
    mock_client = _make_mock_gmail_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email",
        {"to": "manager@example.com", "subject": "VOC 알림", "body": "내용입니다."},
        call_id="gmail-001",
        tenant_id="gmail-tenant-001",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "msg-abc123"
    assert result["result"]["message_id"] == "msg-abc123"
    assert result["result"]["to"] == "manager@example.com"
    assert plaintext_token not in str(result)

    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 15. Gmail real mode: tenant integration 없음 → not_connected skipped ─────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_not_connected(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    tenant_integration_repo.clear_integrations()

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="oauth-002",
        tenant_id="no-such-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"

    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 16. Gmail real mode: MCP_USE_TENANT_OAUTH=false → tenant_oauth_required ──

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_no_fallback(monkeypatch):
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="oauth-003",
        tenant_id="no-such-tenant",
    )

    # tenant OAuth 미설정 → tenant_oauth_required
    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"

    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)


# ── 17. Gmail GMAIL_MCP_REAL=false → mock 반환 (MCP_USE_TENANT_OAUTH 무관) ────

@pytest.mark.asyncio
async def test_gmail_connector_tenant_oauth_with_env_fallback(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    tenant_integration_repo.clear_integrations()

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email",
        {"to": "manager@example.com", "subject": "mock test"},
        call_id="oauth-004",
        tenant_id="no-such-tenant",
    )

    # GMAIL_MCP_REAL=false → mock 반환 (real_mode 판단이 먼저)
    assert result["status"] == "success"
    assert result["result"]["mock"] is True

    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 18. tenant OAuth: _oauth_provider_name 설정 확인 ─────────────────────────

def test_connectors_have_oauth_provider_name():
    assert GmailConnector._oauth_provider_name == "google_gmail"
    assert CalendarConnector._oauth_provider_name == "google_calendar"
    assert SlackConnector._oauth_provider_name == "slack"
    assert JiraConnector._oauth_provider_name == "jira"
    assert CompanyDBConnector._oauth_provider_name == ""  # OAuth 불필요


# ── Gmail/Jira 실제 API 테스트 공통 fixture ──────────────────────────────────

def _make_mock_gmail_client(status_code: int, json_data: dict):
    from unittest.mock import MagicMock, AsyncMock

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _setup_gmail_tenant(monkeypatch, enc_token: str, tenant_id: str = "gmail-tenant") -> None:
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import tenant_integration_repo, upsert_integration
    tenant_integration_repo.clear_integrations()
    upsert_integration(TenantIntegration(
        tenant_id=tenant_id,
        provider="google_gmail",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
    ))


def _make_mock_jira_client(status_code: int, json_data: dict):
    from unittest.mock import MagicMock, AsyncMock

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _setup_jira_tenant(
    monkeypatch,
    enc_token: str,
    tenant_id: str = "jira-tenant",
    cloud_id: str = "",
    external_workspace_id: str = "",
) -> None:
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import tenant_integration_repo, upsert_integration
    tenant_integration_repo.clear_integrations()
    metadata = {}
    if cloud_id:
        metadata["cloud_id"] = cloud_id
    upsert_integration(TenantIntegration(
        tenant_id=tenant_id,
        provider="jira",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
        metadata=metadata,
        external_workspace_id=external_workspace_id or None,
    ))


# ── Gmail real API 테스트 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gmail_real_api_success(monkeypatch):
    """Gmail real mode + tenant token → Gmail messages.send 성공."""
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    plaintext_token = "ya29.fake-google-token"
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("GMAIL_MANAGER_TO", "manager@example.com")
    reset_fernet_cache()

    enc = encrypt_token(plaintext_token)
    _setup_gmail_tenant(monkeypatch, enc)

    api_response = {"id": "msg-xyz999", "threadId": "t1", "labelIds": ["SENT"]}
    mock_client = _make_mock_gmail_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email",
        {"subject": "VOC 알림", "body": "처리 요청입니다."},
        call_id="gmail-real-001",
        tenant_id="gmail-tenant",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "msg-xyz999"
    assert result["result"]["message_id"] == "msg-xyz999"
    assert result["result"]["to"] == "manager@example.com"
    assert plaintext_token not in str(result)

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


@pytest.mark.asyncio
async def test_gmail_real_api_http_error(monkeypatch):
    """Gmail real mode + tenant token + 401 → failed."""
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("GMAIL_MANAGER_TO", "manager@example.com")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake-token")
    _setup_gmail_tenant(monkeypatch, enc)

    mock_client = _make_mock_gmail_client(401, {"error": "invalid_credentials"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="gmail-real-002",
        tenant_id="gmail-tenant",
    )

    assert result["status"] == "failed"
    assert "401" in result["error"]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


@pytest.mark.asyncio
async def test_gmail_real_mode_no_tenant_oauth_skipped(monkeypatch):
    """Gmail real mode + MCP_USE_TENANT_OAUTH=false → tenant_oauth_required."""
    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="gmail-real-003",
        tenant_id="some-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)


@pytest.mark.asyncio
async def test_gmail_real_mode_no_integration_skipped(monkeypatch):
    """Gmail real mode + tenant integration 없음 → tenant_integration_not_connected."""
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("GMAIL_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    tenant_integration_repo.clear_integrations()

    connector = GmailConnector()
    result = await connector.execute(
        "send_manager_email", {},
        call_id="gmail-real-004",
        tenant_id="no-such-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"
    monkeypatch.delenv("GMAIL_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── Jira real API 테스트 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jira_real_api_success_with_cloud_id(monkeypatch):
    """Jira real mode + tenant token + cloud_id → Jira issue create 성공."""
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    plaintext_token = "atlassian-fake-oauth-token"
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "VOC")
    monkeypatch.setenv("JIRA_ISSUE_TYPE", "Task")
    reset_fernet_cache()

    enc = encrypt_token(plaintext_token)
    _setup_jira_tenant(monkeypatch, enc, cloud_id="fake-cloud-id-123")

    api_response = {
        "id": "10001",
        "key": "VOC-42",
        "self": "https://api.atlassian.com/ex/jira/fake-cloud-id-123/rest/api/3/issue/10001",
    }
    mock_client = _make_mock_jira_client(201, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue",
        {"summary_short": "VOC 건", "reason": "청구 오류"},
        call_id="jira-real-001",
        tenant_id="jira-tenant",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "VOC-42"
    assert result["result"]["issue_key"] == "VOC-42"
    assert plaintext_token not in str(result)

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("JIRA_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


@pytest.mark.asyncio
async def test_jira_real_api_no_site_skipped(monkeypatch):
    """Jira real mode + tenant token + cloud_id/base_url 없음 → jira_site_not_configured."""
    from cryptography.fernet import Fernet
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.delenv("JIRA_BASE_URL", raising=False)
    reset_fernet_cache()

    enc = encrypt_token("atlassian-fake-oauth-token")
    _setup_jira_tenant(monkeypatch, enc, cloud_id="")  # cloud_id 없음, metadata에도 없음

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue", {},
        call_id="jira-real-002",
        tenant_id="jira-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "jira_site_not_configured"

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("JIRA_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


@pytest.mark.asyncio
async def test_jira_real_api_success_with_external_workspace_id(monkeypatch):
    """metadata가 비어 있고 external_workspace_id만 있어도 cloud_id로 사용하여 Jira API 호출 성공."""
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    plaintext_token = "atlassian-fake-oauth-token"
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("JIRA_PROJECT_KEY", "VOC")
    monkeypatch.setenv("JIRA_ISSUE_TYPE", "Task")
    monkeypatch.delenv("JIRA_BASE_URL", raising=False)
    reset_fernet_cache()

    enc = encrypt_token(plaintext_token)
    # metadata는 비어 있고, external_workspace_id에 Atlassian cloud_id가 저장된 실제 케이스
    _setup_jira_tenant(
        monkeypatch, enc,
        cloud_id="",                                    # metadata에 cloud_id 없음
        external_workspace_id="fake-workspace-uuid",   # external_workspace_id가 cloud_id 역할
    )

    api_response = {
        "id": "10002",
        "key": "VOC-99",
        "self": "https://api.atlassian.com/ex/jira/fake-workspace-uuid/rest/api/3/issue/10002",
    }
    mock_client = _make_mock_jira_client(201, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue",
        {"summary_short": "결제 오류", "reason": "중복 청구"},
        call_id="jira-workspace-001",
        tenant_id="jira-tenant",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "VOC-99"
    assert result["result"]["issue_key"] == "VOC-99"
    assert plaintext_token not in str(result)

    # httpx.AsyncClient.post 호출 URL에 external_workspace_id가 cloud_id로 사용됐는지 확인
    call_args = mock_client.post.call_args
    called_url = call_args[0][0] if call_args.args else call_args[1].get("url", "") or str(call_args)
    assert "fake-workspace-uuid" in called_url or "fake-workspace-uuid" in str(call_args)

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("JIRA_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


@pytest.mark.asyncio
async def test_jira_real_mode_no_tenant_oauth_skipped(monkeypatch):
    """Jira real mode + MCP_USE_TENANT_OAUTH=false → tenant_oauth_required."""
    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue", {},
        call_id="jira-real-003",
        tenant_id="some-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"
    monkeypatch.delenv("JIRA_MCP_REAL", raising=False)


@pytest.mark.asyncio
async def test_jira_real_mode_no_integration_skipped(monkeypatch):
    """Jira real mode + tenant integration 없음 → tenant_integration_not_connected."""
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("JIRA_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    tenant_integration_repo.clear_integrations()

    connector = JiraConnector()
    result = await connector.execute(
        "create_jira_issue", {},
        call_id="jira-real-004",
        tenant_id="no-such-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"
    monkeypatch.delenv("JIRA_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── Calendar 실제 API 테스트 공통 fixture ─────────────────────────────────────

def _setup_calendar_tenant(monkeypatch, enc_token: str, tenant_id: str = "cal-tenant") -> None:
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import tenant_integration_repo, upsert_integration
    tenant_integration_repo.clear_integrations()
    upsert_integration(TenantIntegration(
        tenant_id=tenant_id,
        provider="google_calendar",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
    ))


def _make_mock_http_client(status_code: int, json_data: dict):
    """httpx.AsyncClient를 대체하는 동기-호환 mock 팩토리."""
    from unittest.mock import MagicMock, AsyncMock

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ── 19. Calendar: tenant token → events.insert 성공 ──────────────────────────

@pytest.mark.asyncio
async def test_calendar_real_api_success(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake_access_token")
    _setup_calendar_tenant(monkeypatch, enc)

    api_response = {
        "id": "event-id-abc123",
        "htmlLink": "https://calendar.google.com/event/abc123",
        "start": {"dateTime": "2026-04-29T10:00:00+09:00"},
        "end": {"dateTime": "2026-04-29T10:30:00+09:00"},
    }
    mock_client = _make_mock_http_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback",
        {"title": "고객 콜백", "reason": "요금 문의 후속"},
        call_id="cal-001",
        tenant_id="cal-tenant",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "event-id-abc123"
    assert result["result"]["event_id"] == "event-id-abc123"
    assert result["result"]["html_link"] == "https://calendar.google.com/event/abc123"
    assert result["error"] is None

    # 4개 표준 키 확인
    for key_ in ("status", "external_id", "result", "error"):
        assert key_ in result

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 20. Calendar: Google Calendar API HTTP 오류 → failed ─────────────────────

@pytest.mark.asyncio
async def test_calendar_real_api_http_failure(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake_access_token")
    _setup_calendar_tenant(monkeypatch, enc)

    mock_client = _make_mock_http_client(401, {"error": "Unauthorized"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback", {},
        call_id="cal-002",
        tenant_id="cal-tenant",
    )

    assert result["status"] == "failed"
    assert "401" in result["error"]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 21. Calendar: tenant token 없음 → skipped ────────────────────────────────

@pytest.mark.asyncio
async def test_calendar_no_tenant_integration_skipped(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.delenv("MCP_ALLOW_ENV_FALLBACK", raising=False)
    tenant_integration_repo.clear_integrations()

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback", {},
        call_id="cal-003",
        tenant_id="no-calendar-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"

    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 22. Calendar: calendar_id 우선순위 (params > env > primary) ───────────────

@pytest.mark.asyncio
async def test_calendar_calendar_id_priority(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from unittest.mock import AsyncMock, MagicMock

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("GOOGLE_CALENDAR_ID", "env-calendar@group.calendar.google.com")
    reset_fernet_cache()

    enc = encrypt_token("ya29.fake_token")
    _setup_calendar_tenant(monkeypatch, enc)

    captured_urls: list[str] = []

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"id": "ev1", "htmlLink": "", "start": {}, "end": {}}
    mock_resp.text = ""

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def _post(url, **kwargs):
        captured_urls.append(url)
        return mock_resp

    mock_client.post = _post
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()

    # params calendar_id 우선
    captured_urls.clear()
    await connector.execute(
        "schedule_callback",
        {"calendar_id": "params-calendar@group.calendar.google.com"},
        call_id="cal-004a", tenant_id="cal-tenant",
    )
    assert "params-calendar" in captured_urls[0]

    # env GOOGLE_CALENDAR_ID (params 없음)
    captured_urls.clear()
    await connector.execute(
        "schedule_callback", {},
        call_id="cal-004b", tenant_id="cal-tenant",
    )
    assert "env-calendar" in captured_urls[0]

    # primary (params 없음, env 없음)
    monkeypatch.delenv("GOOGLE_CALENDAR_ID", raising=False)
    captured_urls.clear()
    await connector.execute(
        "schedule_callback", {},
        call_id="cal-004c", tenant_id="cal-tenant",
    )
    assert "/primary/" in captured_urls[0]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 23. Calendar: start_time/end_time 직접 지정 ──────────────────────────────

def test_calendar_event_body_explicit_start_end():
    connector = CalendarConnector()
    body = connector._build_event_body({
        "title": "명시 시간 테스트",
        "start_time": "2026-05-01T09:00:00",
        "end_time": "2026-05-01T10:00:00",
    })

    assert body["summary"] == "명시 시간 테스트"
    assert "2026-05-01T09:00:00" in body["start"]["dateTime"]
    assert "2026-05-01T10:00:00" in body["end"]["dateTime"]


# ── 24. Calendar: preferred_time → end_time 자동 생성 ────────────────────────

def test_calendar_event_body_preferred_time_auto_end():
    connector = CalendarConnector()
    body = connector._build_event_body({
        "preferred_time": "2026-05-01T14:00:00",
    })

    from datetime import datetime, timedelta
    start_dt = datetime.fromisoformat(body["start"]["dateTime"])
    end_dt = datetime.fromisoformat(body["end"]["dateTime"])

    assert start_dt.hour == 14
    diff = end_dt - start_dt
    assert diff == timedelta(minutes=30)  # CALENDAR_DEFAULT_DURATION_MIN 기본값


# ── 25. Calendar: 시간 없을 때 현재+1시간 ────────────────────────────────────

def test_calendar_event_body_default_time():
    from datetime import datetime, timedelta

    before = datetime.utcnow() + timedelta(minutes=55)  # 약간의 여유
    connector = CalendarConnector()
    body = connector._build_event_body({"title": "시간 없음"})
    after = datetime.utcnow() + timedelta(hours=1, minutes=5)

    start_dt = datetime.fromisoformat(body["start"]["dateTime"])
    assert before <= start_dt <= after


# ── 26. Calendar: 결과에 access_token 평문 미포함 ────────────────────────────

@pytest.mark.asyncio
async def test_calendar_access_token_not_in_result(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token

    plaintext_token = "ya29.super_secret_access_token_do_not_leak"

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    reset_fernet_cache()

    enc = encrypt_token(plaintext_token)
    _setup_calendar_tenant(monkeypatch, enc)

    api_response = {"id": "ev-safe", "htmlLink": "", "start": {}, "end": {}}
    mock_client = _make_mock_http_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = CalendarConnector()
    result = await connector.execute(
        "schedule_callback", {},
        call_id="cal-safe",
        tenant_id="cal-tenant",
    )

    # result dict를 str로 변환해도 평문 토큰이 없어야 함
    result_str = str(result)
    assert plaintext_token not in result_str
    assert result["status"] == "success"

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── Slack tenant OAuth helper ────────────────────────────────────────────────

def _setup_slack_tenant(monkeypatch, enc_token: str, tenant_id: str = "slack-tenant") -> None:
    from app.models.tenant_integration import TenantIntegration, IntegrationStatus
    from app.repositories.tenant_integration_repo import tenant_integration_repo, upsert_integration
    tenant_integration_repo.clear_integrations()
    upsert_integration(TenantIntegration(
        tenant_id=tenant_id,
        provider="slack",
        status=IntegrationStatus.connected,
        access_token_encrypted=enc_token,
    ))


def _make_mock_slack_client(status_code: int, json_data: dict):
    from unittest.mock import MagicMock, AsyncMock

    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = str(json_data)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client



# ── 27. Slack: tenant token 있을 때 chat.postMessage 성공 ────────────────────

@pytest.mark.asyncio
async def test_slack_real_api_success(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from app.services.mcp.connectors.slack_connector import SlackConnector

    plaintext_token = "xoxb-fake-slack-bot-token"
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#alerts")
    reset_fernet_cache()
    enc = encrypt_token(plaintext_token)
    _setup_slack_tenant(monkeypatch, enc)

    api_response = {"ok": True, "channel": "C123456", "ts": "1745900000.123456", "message": {"text": "[CRITICAL]"}}
    mock_client = _make_mock_slack_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert",
        {"channel": "#alerts", "message": "[CRITICAL] demo-call"},
        call_id="slack-001",
        tenant_id="slack-tenant",
    )

    assert result["status"] == "success"
    assert result["error"] is None
    assert result["result"]["channel"] == "C123456"
    assert result["result"]["ts"] == "1745900000.123456"
    assert plaintext_token not in str(result)
    for key_ in ("status", "external_id", "result", "error"):
        assert key_ in result

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 28. Slack: API ok=false → failed ────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_real_api_ok_false(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from app.services.mcp.connectors.slack_connector import SlackConnector

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#alerts")
    reset_fernet_cache()
    enc = encrypt_token("xoxb-fake-token")
    _setup_slack_tenant(monkeypatch, enc)

    mock_client = _make_mock_slack_client(200, {"ok": False, "error": "not_in_channel"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = SlackConnector()
    result = await connector.execute("send_slack_alert", {}, call_id="slack-002", tenant_id="slack-tenant")

    assert result["status"] == "failed"
    assert "not_in_channel" in result["error"]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 29. Slack: HTTP 오류 → failed ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_real_api_http_failure(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from app.services.mcp.connectors.slack_connector import SlackConnector

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#alerts")
    reset_fernet_cache()
    enc = encrypt_token("xoxb-fake-token")
    _setup_slack_tenant(monkeypatch, enc)

    mock_client = _make_mock_slack_client(500, {})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = SlackConnector()
    result = await connector.execute("send_slack_alert", {}, call_id="slack-003", tenant_id="slack-tenant")

    assert result["status"] == "failed"
    assert "500" in result["error"]

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 30. Slack: channel 우선순위 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_channel_priority(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from unittest.mock import AsyncMock, MagicMock

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#env-alerts")
    monkeypatch.setenv("SLACK_CRITICAL_CHANNEL", "#critical-channel")
    reset_fernet_cache()
    enc = encrypt_token("xoxb-fake")
    _setup_slack_tenant(monkeypatch, enc)

    captured_bodies = []
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "channel": "C1", "ts": "1.0", "message": {}}
    mock_resp.text = ""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def _post(url, *, json=None, headers=None, timeout=None):
        captured_bodies.append(json or {})
        return mock_resp

    mock_client.post = _post
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = SlackConnector()

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {"channel": "#params-channel"}, call_id="ch-1", tenant_id="slack-tenant")
    assert captured_bodies[0]["channel"] == "#params-channel"

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {"channel_type": "critical"}, call_id="ch-2", tenant_id="slack-tenant")
    assert captured_bodies[0]["channel"] == "#critical-channel"

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {}, call_id="ch-3", tenant_id="slack-tenant")
    assert captured_bodies[0]["channel"] == "#env-alerts"

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 31. Slack: message 우선순위 ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_message_priority(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from unittest.mock import AsyncMock, MagicMock

    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#alerts")
    reset_fernet_cache()
    enc = encrypt_token("xoxb-fake")
    _setup_slack_tenant(monkeypatch, enc)

    captured_bodies = []
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"ok": True, "channel": "C1", "ts": "1.0", "message": {}}
    mock_resp.text = ""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def _post(url, *, json=None, headers=None, timeout=None):
        captured_bodies.append(json or {})
        return mock_resp

    mock_client.post = _post
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = SlackConnector()

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {"message": "msg-A"}, call_id="msg-1", tenant_id="slack-tenant")
    assert captured_bodies[0]["text"] == "msg-A"

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {"text": "text-B"}, call_id="msg-2", tenant_id="slack-tenant")
    assert captured_bodies[0]["text"] == "text-B"

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {"summary_short": "summary-C"}, call_id="msg-3", tenant_id="slack-tenant")
    assert captured_bodies[0]["text"] == "summary-C"

    captured_bodies.clear()
    await connector.execute("send_slack_alert", {}, call_id="msg-4", tenant_id="slack-tenant")
    assert captured_bodies[0]["text"] == "Post-call alert"

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 32. Slack: tenant token 없음 → skipped ──────────────────────────────────

@pytest.mark.asyncio
async def test_slack_no_tenant_integration_skipped(monkeypatch):
    from app.repositories.tenant_integration_repo import tenant_integration_repo
    from app.services.mcp.connectors.slack_connector import SlackConnector

    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.delenv("MCP_ALLOW_ENV_FALLBACK", raising=False)
    tenant_integration_repo.clear_integrations()

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert", {},
        call_id="slack-notoken",
        tenant_id="no-slack-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)


# ── 33. Slack: access_token 평문 미포함 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_slack_access_token_not_in_result(monkeypatch):
    from cryptography.fernet import Fernet
    import httpx
    from app.services.oauth.token_crypto import reset_fernet_cache, encrypt_token
    from app.services.mcp.connectors.slack_connector import SlackConnector

    plaintext_token = "xoxb-super-secret-do-not-leak-slack-token"
    key = Fernet.generate_key()
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key.decode())
    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#alerts")
    reset_fernet_cache()
    enc = encrypt_token(plaintext_token)
    _setup_slack_tenant(monkeypatch, enc)

    mock_client = _make_mock_slack_client(200, {"ok": True, "channel": "C1", "ts": "1.0", "message": {}})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = SlackConnector()
    result = await connector.execute("send_slack_alert", {}, call_id="slack-safe", tenant_id="slack-tenant")

    assert plaintext_token not in str(result)
    assert result["status"] == "success"

    from app.repositories.tenant_integration_repo import tenant_integration_repo
    tenant_integration_repo.clear_integrations()
    reset_fernet_cache()
    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)


# ── 34-pre. Slack: _extract_bot_token 단위 테스트 ────────────────────────────

def test_extract_bot_token_v1_priority():
    """metadata["bot"]["bot_access_token"]이 있으면 최우선으로 반환한다."""
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from app.models.tenant_integration import TenantIntegration

    connector = SlackConnector()
    integration = TenantIntegration(
        tenant_id="t", provider="slack",
        metadata={"bot": {"bot_access_token": "xoxb-v1-bot"}},
        access_token_encrypted="placeholder",
    )
    assert connector._extract_bot_token(integration, "xoxb-regular") == "xoxb-v1-bot"


def test_extract_bot_token_v2_metadata_access_token():
    """metadata["access_token"]이 xoxb-로 시작하면 v1 없을 때 사용한다."""
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from app.models.tenant_integration import TenantIntegration

    connector = SlackConnector()
    integration = TenantIntegration(
        tenant_id="t", provider="slack",
        metadata={"access_token": "xoxb-v2-meta"},
        access_token_encrypted="placeholder",
    )
    assert connector._extract_bot_token(integration, "xoxb-regular") == "xoxb-v2-meta"


def test_extract_bot_token_non_bot_meta_access_token_ignored():
    """metadata["access_token"]이 xoxp-로 시작하면 (user token) fallback 사용."""
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from app.models.tenant_integration import TenantIntegration

    connector = SlackConnector()
    integration = TenantIntegration(
        tenant_id="t", provider="slack",
        metadata={"access_token": "xoxp-user-token"},
        access_token_encrypted="placeholder",
    )
    assert connector._extract_bot_token(integration, "xoxb-decrypted") == "xoxb-decrypted"


def test_extract_bot_token_empty_metadata_fallback():
    """metadata가 없으면 decrypted access_token을 반환한다."""
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from app.models.tenant_integration import TenantIntegration

    connector = SlackConnector()
    integration = TenantIntegration(
        tenant_id="t", provider="slack",
        metadata={},
        access_token_encrypted="placeholder",
    )
    assert connector._extract_bot_token(integration, "xoxb-from-decrypt") == "xoxb-from-decrypt"


# ── 34-pre-b. Slack: env fallback (MCP_ALLOW_ENV_FALLBACK + SLACK_BOT_TOKEN) ──

@pytest.mark.asyncio
async def test_slack_env_fallback_bot_token_success(monkeypatch):
    """Slack env fallback 제거 — MCP_ALLOW_ENV_FALLBACK=true여도 tenant integration 없으면 skipped."""
    from app.services.mcp.connectors.slack_connector import SlackConnector
    from app.repositories.tenant_integration_repo import tenant_integration_repo

    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.setenv("MCP_USE_TENANT_OAUTH", "true")
    monkeypatch.setenv("MCP_ALLOW_ENV_FALLBACK", "true")
    monkeypatch.setenv("SLACK_ALERT_CHANNEL", "#alerts")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake-token")
    tenant_integration_repo.clear_integrations()

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert",
        {"channel": "#alerts", "message": "fallback test"},
        call_id="fallback-001",
        tenant_id="no-such-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_integration_not_connected"

    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("MCP_ALLOW_ENV_FALLBACK", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)


@pytest.mark.asyncio
async def test_slack_env_fallback_no_bot_token_skipped(monkeypatch):
    """Slack real mode + tenant OAuth 필수 — MCP_USE_TENANT_OAUTH=false이면 tenant_oauth_required."""
    from app.services.mcp.connectors.slack_connector import SlackConnector

    monkeypatch.setenv("SLACK_MCP_REAL", "true")
    monkeypatch.delenv("MCP_USE_TENANT_OAUTH", raising=False)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    connector = SlackConnector()
    result = await connector.execute(
        "send_slack_alert", {},
        call_id="fallback-002",
        tenant_id="no-such-tenant",
    )

    assert result["status"] == "skipped"
    assert result["error"] == "tenant_oauth_required"

    monkeypatch.delenv("SLACK_MCP_REAL", raising=False)


# ── 34. SMSConnector: mock success ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_sms_connector_mock_success(monkeypatch):
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)

    connector = SMSConnector()
    result = await connector.execute("send_callback_sms", {"customer_phone": "01012345678"}, call_id="sms-001")

    assert result["status"] == "success"
    assert result["external_id"] == "sms-mock-sms-001"
    assert result["result"]["to"] == "01012345678"
    assert result["result"]["mock"] is True
    assert result["error"] is None
    for key_ in ("status", "external_id", "result", "error"):
        assert key_ in result


# ── 35. SMSConnector: real mode Solapi monkeypatched ─────────────────────────

@pytest.mark.asyncio
async def test_sms_connector_real_mode_monkeypatched(monkeypatch):
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.setenv("SMS_MCP_REAL", "true")
    call_log = []

    async def fake_send_real(self, to, message, call_id):
        call_log.append((to, message))
        return self._success(external_id=f"sms-solapi-{call_id}", result={"to": to, "sent": True})

    monkeypatch.setattr(SMSConnector, "_send_real", fake_send_real)

    connector = SMSConnector()
    result = await connector.execute("send_callback_sms", {"customer_phone": "01099998888"}, call_id="sms-real-001")

    assert result["status"] == "success"
    assert result["result"]["sent"] is True
    assert len(call_log) == 1
    assert call_log[0][0] == "01099998888"


# ── 36. SMSConnector: customer_phone 없음 → skipped ─────────────────────────

@pytest.mark.asyncio
async def test_sms_connector_missing_phone_skipped(monkeypatch):
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)
    # SMS_TEST_TO fallback 이 .env 에서 새지 않도록 격리.
    monkeypatch.delenv("SMS_TEST_TO", raising=False)

    connector = SMSConnector()
    result = await connector.execute("send_callback_sms", {}, call_id="sms-noPhone")

    assert result["status"] == "skipped"
    assert result["error"] == "customer_phone_missing"


# ── 36-b. SMSConnector: SMS_TEST_TO fallback ────────────────────────────────

@pytest.mark.asyncio
async def test_sms_connector_uses_sms_test_to_fallback(monkeypatch):
    """customer_phone 부재 + SMS_TEST_TO 설정 → fallback 으로 발송된다."""
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)  # mock mode
    monkeypatch.setenv("SMS_TEST_TO", "01088887777")

    connector = SMSConnector()
    result = await connector.execute("send_callback_sms", {}, call_id="sms-fallback-001")

    assert result["status"] == "success"
    assert result["external_id"] == "sms-mock-sms-fallback-001"
    assert result["result"]["to"] == "01088887777"


@pytest.mark.asyncio
async def test_sms_connector_customer_phone_wins_over_sms_test_to(monkeypatch):
    """params.customer_phone 가 있으면 SMS_TEST_TO 보다 우선된다."""
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)
    monkeypatch.setenv("SMS_TEST_TO", "01088887777")  # 사용되면 안 됨

    connector = SMSConnector()
    result = await connector.execute(
        "send_callback_sms",
        {"customer_phone": "01012345678"},
        call_id="sms-priority-001",
    )

    assert result["status"] == "success"
    assert result["result"]["to"] == "01012345678"


@pytest.mark.asyncio
async def test_sms_connector_sms_test_to_normalized(monkeypatch):
    """SMS_TEST_TO 가 하이픈/국가코드 포함 형식이어도 normalize_korean_phone 로 통일된다."""
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)
    monkeypatch.setenv("SMS_TEST_TO", "+82-10-8888-7777")

    connector = SMSConnector()
    result = await connector.execute("send_callback_sms", {}, call_id="sms-norm-001")

    assert result["status"] == "success"
    assert result["result"]["to"] == "01088887777"


@pytest.mark.asyncio
async def test_sms_connector_empty_sms_test_to_still_skipped(monkeypatch):
    """SMS_TEST_TO 가 빈 문자열/공백이면 fallback 적용 안 되고 skipped 된다."""
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)
    monkeypatch.setenv("SMS_TEST_TO", "   ")

    connector = SMSConnector()
    result = await connector.execute("send_callback_sms", {}, call_id="sms-empty-fallback")

    assert result["status"] == "skipped"
    assert result["error"] == "customer_phone_missing"


@pytest.mark.asyncio
async def test_sms_connector_fallback_logs_warning(monkeypatch):
    """SMS_TEST_TO fallback 사용 시 운영 가시성을 위해 warning 로그가 남는다.

    프로젝트 logger 가 propagate=False 이므로 caplog 대신 핸들러를 직접 부착해
    레코드를 수집한다.
    """
    import logging
    from app.services.mcp.connectors import sms_connector as sms_mod
    from app.services.mcp.connectors.sms_connector import SMSConnector

    monkeypatch.delenv("SMS_MCP_REAL", raising=False)
    monkeypatch.setenv("SMS_TEST_TO", "01088887777")

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    sms_mod.logger.addHandler(handler)
    try:
        connector = SMSConnector()
        await connector.execute("send_callback_sms", {}, call_id="sms-warn-001")
    finally:
        sms_mod.logger.removeHandler(handler)

    assert any(
        "SMS_TEST_TO fallback" in rec.getMessage() and "sms-warn-001" in rec.getMessage()
        for rec in records
    )


# ── 37. SMSConnector: send_voc_receipt_sms 템플릿 ────────────────────────────

@pytest.mark.asyncio
async def test_sms_connector_voc_receipt_template(monkeypatch):
    from app.services.mcp.connectors.sms_connector import SMSConnector
    monkeypatch.delenv("SMS_MCP_REAL", raising=False)

    connector = SMSConnector()
    result = await connector.execute("send_voc_receipt_sms", {"customer_phone": "01011112222"}, call_id="sms-voc-001")

    assert result["status"] == "success"
    assert "sms-voc-001" in result["result"]["message"]
    assert "접수번호" in result["result"]["message"]


# ── 38. NotionConnector: mock success ────────────────────────────────────────

@pytest.mark.asyncio
async def test_notion_connector_mock_success(monkeypatch):
    from app.services.mcp.connectors.notion_connector import NotionConnector
    monkeypatch.delenv("NOTION_MCP_REAL", raising=False)

    connector = NotionConnector()
    result = await connector.execute("create_notion_call_record", {"summary_short": "테스트", "priority": "high"}, call_id="notion-001")

    assert result["status"] == "success"
    assert result["external_id"] == "notion-mock-notion-001"
    assert result["result"]["mock"] is True
    assert result["error"] is None
    for key_ in ("status", "external_id", "result", "error"):
        assert key_ in result


# ── 39. NotionConnector: real mode API 성공 monkeypatch ─────────────────────

@pytest.mark.asyncio
async def test_notion_connector_real_api_success(monkeypatch):
    import httpx
    from app.services.mcp.connectors.notion_connector import NotionConnector

    monkeypatch.setenv("NOTION_MCP_REAL", "true")
    monkeypatch.setenv("NOTION_API_TOKEN", "secret_fake_notion_token")
    monkeypatch.setenv("NOTION_DATABASE_ID", "fake-db-id-12345")

    api_response = {"id": "page-id-abc123", "url": "https://www.notion.so/page-id-abc123"}
    mock_client = _make_mock_http_client(200, api_response)
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = NotionConnector()
    result = await connector.execute(
        "create_notion_call_record",
        {"summary_short": "VOC 요약", "priority": "critical", "customer_emotion": "angry"},
        call_id="notion-real-001",
    )

    assert result["status"] == "success"
    assert result["external_id"] == "page-id-abc123"
    assert result["result"]["page_id"] == "page-id-abc123"
    assert result["error"] is None
    assert "secret_fake_notion_token" not in str(result)


# ── 40. NotionConnector: real mode API 실패 ──────────────────────────────────

@pytest.mark.asyncio
async def test_notion_connector_real_api_failure(monkeypatch):
    import httpx
    from app.services.mcp.connectors.notion_connector import NotionConnector

    monkeypatch.setenv("NOTION_MCP_REAL", "true")
    monkeypatch.setenv("NOTION_API_TOKEN", "secret_fake")
    monkeypatch.setenv("NOTION_DATABASE_ID", "fake-db-id")

    mock_client = _make_mock_http_client(400, {"message": "validation_error"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

    connector = NotionConnector()
    result = await connector.execute("create_notion_voc_record", {}, call_id="notion-fail-001")

    assert result["status"] == "failed"
    assert "400" in result["error"]


# ── 41. NotionConnector: token/db_id 누락 → skipped ─────────────────────────

@pytest.mark.asyncio
async def test_notion_connector_missing_config_skipped(monkeypatch):
    from app.services.mcp.connectors.notion_connector import NotionConnector

    monkeypatch.setenv("NOTION_MCP_REAL", "true")
    monkeypatch.delenv("NOTION_API_TOKEN", raising=False)
    monkeypatch.delenv("NOTION_DATABASE_ID", raising=False)

    connector = NotionConnector()
    result = await connector.execute("create_notion_call_record", {}, call_id="notion-noconfig")

    assert result["status"] == "skipped"
    assert result["error"] == "notion_not_configured"
