# Post-call Real Actions 연결 및 검증 체크리스트

> **이 문서는 통화 프로세스 검증 문서가 아니다.**
> Twilio / Zoiper / STT / TTS / 화자검증 / barge-in 등 통화 중 흐름은 본 문서 범위 밖이다.

---

## 1. 목적

이 문서는 Post-call Agent가 생성한 외부 action을 실제 서비스(Slack /
Google Calendar / Gmail / Notion / SMS / Jira / company_db)와 연결해
검증하기 전에, tenant별 OAuth/integration 상태와 안전한 실행 절차를
확인하기 위한 운영 체크리스트다.

이 문서는 새로운 OAuth flow를 구현하기 위한 설계 문서가 아니라,
**현재 코드 기준으로 재현 가능한 절차를 정리한 운영용 안내**다.

---

## 2. 현재 구조 요약

```text
completed call (calls + transcripts in Postgres)
  ↓
scripts/run_post_call_from_db.py  /  scripts/run_post_call_batch_from_db.py
  ↓
app/agents/post_call/completed_call_runner.run_post_call_for_completed_call()
  ↓
PostCallAgent (LangGraph)
  ├─ post_call_analysis_node      (real / mock LLM 통합 분석)
  ├─ review_node                  (Review Gate)
  ├─ apply_review_corrections     (correctable 시 보정)
  ├─ action_planner_node          (10개 action 규칙)
  ├─ action_router_node
  ├─ action_executor_node
  │   ├─ ActionRegistry           (gmail/calendar/notion/slack/sms/jira/company_db/internal_dashboard)
  │   └─ MCPClient                (각 connector로 라우팅)
  │       └─ BaseMCPConnector     ← real_mode / tenant OAuth / env fallback 분기
  └─ save_result_node             (mcp_action_logs / call_summaries / voc_analyses 저장)
```

---

## 3. Integration 저장 방식

### 3-1. 런타임 저장소 — `TenantIntegrationRepository`

`app/repositories/tenant_integration_repo.py`는 **in-memory + 선택적
JSON 파일** 저장소다. PostgreSQL은 현재 런타임에서 사용하지 않는다.

| `TENANT_INTEGRATION_STORAGE` | 동작 |
|---|---|
| `memory` (기본값) | 프로세스 메모리에만 보관. 서버 재시작 시 소실 |
| `file` | `TENANT_INTEGRATION_FILE_PATH` (기본 `.local/tenant_integrations.json`) JSON 파일에 persist |

기동 시 다음과 같은 로그가 남는다.

```text
TenantIntegrationRepo file mode path=.local\tenant_integrations.json loaded=0
```

`loaded=0`은 **현재 tenant integration row가 0개**라는 뜻이다.
OAuth를 한 번도 완료하지 않았거나, 다른 환경의 파일을 로드했을 때
나타난다. → `missing` 상태로 표시되는 원인.

### 3-2. DB 스키마 (`db/init/11_tenant_integrations.sql`)

테이블 정의는 존재하지만 **현재 코드는 이 테이블을 직접 사용하지 않는다.**
PostgreSQL 전환 시 같은 인터페이스로 repository만 교체하기 위한 사전
스키마다 (`tenant_integration_repo.py` 모듈 docstring의 "PostgreSQL
전환 TODO" 참고).

### 3-3. Token 암호화

OAuth callback (`app/api/v1/oauth.py`)에서 `app.services.oauth.token_crypto.encrypt_token`으로
**Fernet 암호화** 후 저장한다. `TOKEN_ENCRYPTION_KEY` env가 필요하다.

```bash
# Fernet 키 생성 예시
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## 4. provider별 연결 조건

### 4-1. 등록된 connector 매트릭스

소스: `app/services/mcp/connectors/*.py`의 클래스 속성.

| Connector | `_real_mode_env` | `_oauth_provider_name` | `_required_config` (env) |
|---|---|---|---|
| `calendar` | `CALENDAR_MCP_REAL` | `google_calendar` | (없음 — tenant OAuth 전용) |
| `gmail` | `GMAIL_MCP_REAL` | `google_gmail` | `GMAIL_MANAGER_TO` |
| `slack` | `SLACK_MCP_REAL` | `slack` | `SLACK_ALERT_CHANNEL` |
| `jira` | `JIRA_MCP_REAL` | `jira` | `JIRA_PROJECT_KEY`, `JIRA_ISSUE_TYPE` |
| `notion` | `NOTION_MCP_REAL` | (없음 — env 기반) | `NOTION_API_TOKEN`, `NOTION_DATABASE_ID` |
| `sms` | `SMS_MCP_REAL` | (없음 — env 기반) | `SOLAPI_API_KEY`, `SOLAPI_API_SECRET`, `SOLAPI_SENDER_NUMBER` |
| `company_db` | `COMPANY_DB_MCP_REAL` 또는 `MCP_COMPANY_DB_REAL` | (없음) | (구현체에 따라) |
| `internal_dashboard` | (없음) | (없음) | (없음 — 항상 internal) |

### 4-2. `_oauth_provider_name`과 `check_post_call_integrations.py`의 명명 차이

OAuth callback이 저장하는 row의 `provider` 컬럼과 action/tool 레이어의
canonical 이름이 다르다.

| OAuth 라우터 (`app/api/v1/oauth.py`) | tenant_integrations.provider | readiness canonical |
|---|---|---|
| `/api/v1/oauth/google_gmail/...` | `google_gmail` | `gmail` |
| `/api/v1/oauth/google_calendar/...` | `google_calendar` | `calendar` |
| `/api/v1/oauth/slack/...` | `slack` | `slack` |
| `/api/v1/oauth/jira/...` | `jira` | `jira` |

`scripts/check_post_call_integrations.py`는 `PROVIDER_ALIASES` 맵으로
이 차이를 흡수한다. canonical 이름이 우선이고, 동일 canonical 안에서는
**connected → 그 외** 순으로 row를 선택한다.

```python
PROVIDER_ALIASES = {
    "gmail":    ["gmail", "google_gmail"],
    "calendar": ["calendar", "google_calendar"],
    # 그 외는 동일 이름만
}
```

콘솔 출력은 canonical 이름(`gmail` / `calendar`)을 그대로 유지하되
실제 매칭된 row가 alias라면 `source=google_gmail` 같은 suffix를 붙인다.
JSON 출력은 항상 `source_provider`와 `provider_candidates` 필드를 포함한다.

```text
gmail              connected             source=google_gmail  scopes=...
calendar           connected             source=google_calendar  scopes=...
gmail              missing               reason=no tenant integration row  candidates=gmail,google_gmail
```

**교차 검증** (alias 적용 외에도 확인하고 싶을 때):

```bash
curl "http://localhost:8000/api/v1/oauth/google_gmail/status?tenant_id=<uuid>"
curl "http://localhost:8000/api/v1/oauth/google_calendar/status?tenant_id=<uuid>"
curl "http://localhost:8000/api/v1/oauth/slack/status?tenant_id=<uuid>"
curl "http://localhost:8000/api/v1/oauth/jira/status?tenant_id=<uuid>"
```

또는 `.local/tenant_integrations.json`을 직접 열어 `tenant_id::google_gmail`
키가 있는지 확인한다.

### 4-3. provider별 요약

| Provider | OAuth 필요 | 저장 위치 | Ready 조건 | 비고 |
|---|:-:|---|---|---|
| Slack | O | tenant_integrations | OAuth status=connected | real chat.postMessage 구현 (`slack_connector.py`) |
| Google Calendar | O | tenant_integrations | OAuth status=connected | tenant OAuth real execute는 **TODO** (skipped 반환, 후술) |
| Gmail | O | tenant_integrations | OAuth status=connected | tenant OAuth real execute는 **TODO** |
| Notion | X | env | NOTION_API_TOKEN + NOTION_DATABASE_ID | 직접 API 호출 |
| SMS | X | env | SOLAPI_* env + customer_phone | Solapi 사용 |
| Jira | O | tenant_integrations | OAuth status=connected | `JIRA_EMAIL`/`JIRA_API_TOKEN` Basic Auth는 사용 금지 (코드 주석) |
| company_db | X | env / internal | 구현 정책에 따름 | |
| internal_dashboard | X | internal | 항상 ready | OAuth 불필요 |

### 4-4. tenant OAuth 분기 정책

`BaseMCPConnector._try_tenant_token()` (`app/services/mcp/connectors/base.py`)는
`MCP_USE_TENANT_OAUTH=true`일 때 다음 결과를 반환한다.

| 상황 | 결과 |
|---|---|
| integration 없음 / `disconnected` | `skipped("tenant_integration_not_connected")` |
| `expires_at` 지남 + `refresh_token` 없음 | `skipped("tenant_token_expired_no_refresh")` |
| 만료 + refresh 시도 실패 | `skipped("tenant_token_expired_refresh_failed")` |
| Fernet 복호화 실패 | `failed("tenant_token_decryption_failed")` |
| 정상 + 실행 미구현 | `skipped("tenant_token_found_but_real_execute_not_implemented")` ⚠️ |

**중요**: 마지막 항목은 **OAuth는 정상적으로 연결됐어도 실제 외부 호출
실행이 connector에 구현되지 않았음**을 의미한다. 현재 시점에서 tenant
OAuth 기반 real 실행이 명시적으로 구현된 connector는 **Slack
(`chat.postMessage`)** 뿐이다 (`slack_connector.py` 모듈 docstring).
나머지 (Calendar / Gmail / Jira)는 `skipped` 가 정상 동작이다.

---

## 5. 환경변수 체크리스트

`.env.example`과 코드에서 확인된 변수만 나열한다.

### 5-1. 공통

```text
# 토큰 암호화 (필수)
TOKEN_ENCRYPTION_KEY=<Fernet key>

# tenant_integrations 저장소
TENANT_INTEGRATION_STORAGE=file
TENANT_INTEGRATION_FILE_PATH=.local/tenant_integrations.json

# Post-call 연동 정책 (확인됨)
MCP_USE_TENANT_OAUTH=true            # tenant OAuth 우선
MCP_ALLOW_ENV_FALLBACK=false         # tenant 미연결 시 .env로 폴백 허용 여부
MCP_ACTION_LOG_STORE=db              # mcp_action_logs Postgres 저장

# Post-call LLM
POST_CALL_LLM_MODE=mock              # mock | real
POST_CALL_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<sk-...>
```

### 5-2. provider별 (확인됨)

```text
# Slack (OAuth)
SLACK_CLIENT_ID=
SLACK_CLIENT_SECRET=
SLACK_REDIRECT_URI=
SLACK_MCP_REAL=true
SLACK_ALERT_CHANNEL=#voc-alerts
SLACK_CRITICAL_CHANNEL=#urgent-cs

# Google (Gmail + Calendar 공통 OAuth)
GOOGLE_OAUTH_CLIENT_ID=
GOOGLE_OAUTH_CLIENT_SECRET=
GOOGLE_OAUTH_REDIRECT_URI=                # 둘 다 안 두면 fallback
GOOGLE_GMAIL_REDIRECT_URI=
GOOGLE_CALENDAR_REDIRECT_URI=

# Gmail
GMAIL_MCP_REAL=true
GMAIL_MANAGER_TO=manager@example.com
GMAIL_SENDER=your-email@gmail.com

# Calendar
CALENDAR_MCP_REAL=true
CALENDAR_DEFAULT_OWNER=
GOOGLE_CALENDAR_ID=primary

# Jira (OAuth)
ATLASSIAN_REDIRECT_URI=
JIRA_MCP_REAL=true
JIRA_PROJECT_KEY=KDT
JIRA_ISSUE_TYPE=Task
JIRA_VOC_LABEL=post-call-voc
# JIRA_EMAIL / JIRA_API_TOKEN Basic Auth 방식은 connector에서 사용 금지

# Notion (env-only — OAuth 미사용)
NOTION_MCP_REAL=true
NOTION_API_TOKEN=
NOTION_DATABASE_ID=
POST_CALL_ENABLE_NOTION_RECORD=true   # planner에서 Notion action 생성 여부

# SMS / Solapi (env-only)
SMS_MCP_REAL=true
SOLAPI_API_KEY=
SOLAPI_API_SECRET=
SOLAPI_SENDER_NUMBER=

# Company DB (env-only)
COMPANY_DB_MCP_REAL=true
MCP_COMPANY_DB_REAL=true
COMPANY_DB_BASE_URL=
COMPANY_DB_INTERNAL_TOKEN=
```

> `.env.example`에는 현재 `SOLAPI_*`, `NOTION_*`, `TOKEN_ENCRYPTION_KEY`,
> `TENANT_INTEGRATION_STORAGE`, `MCP_USE_TENANT_OAUTH`, `MCP_ALLOW_ENV_FALLBACK`
> 같은 일부 변수가 누락되어 있다. 필요 시 후속 작업으로 보강 권장.

---

## 6. OAuth 실행 순서

### 6-1. OAuth 전용 서버 띄우기 (가장 가벼운 경로)

```bash
# NeMo / STT / TTS 의존 없이 OAuth만 테스트
uvicorn scripts.run_oauth_only:app --reload --port 8000
```

`scripts/run_oauth_only.py`는 `app.api.v1.oauth.router`만 등록한 FastAPI
인스턴스다. health: `GET /health`.

### 6-2. authorize 진입

브라우저에서:

```text
http://localhost:8000/api/v1/oauth/{provider}/authorize?tenant_id=<uuid>
```

지원 provider (4개): `google_gmail`, `google_calendar`, `slack`, `jira`.

### 6-3. callback

provider 측이 `redirect_uri`로 돌려주면 자동 처리된다. 콜백 경로:

```text
http://localhost:8000/api/v1/oauth/{provider}/callback?code=...&state=...
```

`redirect_uri`는 provider별 env에서 결정된다 (`oauth.py:_redirect_uri_for`):

| Provider | env (우선순위) |
|---|---|
| `google_gmail` | `GOOGLE_GMAIL_REDIRECT_URI` → `GOOGLE_OAUTH_REDIRECT_URI` |
| `google_calendar` | `GOOGLE_CALENDAR_REDIRECT_URI` → `GOOGLE_OAUTH_REDIRECT_URI` |
| `slack` | `SLACK_REDIRECT_URI` |
| `jira` | `ATLASSIAN_REDIRECT_URI` |

> Google Cloud Console / Slack App / Atlassian OAuth app에 동일한
> redirect URI를 사전에 등록해야 한다.

### 6-4. status 확인

```text
GET http://localhost:8000/api/v1/oauth/{provider}/status?tenant_id=<uuid>
```

응답 예: `{"status": "connected", "account_email": "...", "scopes": [...]}`.

### 6-5. disconnect

```text
DELETE http://localhost:8000/api/v1/oauth/{provider}/disconnect?tenant_id=<uuid>
```

---

## 7. readiness 확인 방법

```bash
# tenant 단일
python scripts/check_post_call_integrations.py --tenant-id <uuid>

# action log 분포 포함
python scripts/check_post_call_integrations.py --tenant-id <uuid> --show-actions

# JSON
python scripts/check_post_call_integrations.py --tenant-id <uuid> --json

# 전체 tenant (개발/운영자 진단용)
python scripts/check_post_call_integrations.py --all-tenants
```

### 상태 해석

| 상태 | 의미 |
|---|---|
| `connected` | tenant_integration_repo에 row 존재, status=connected |
| `missing` | row 없음 (OAuth 미수행 또는 §4-2의 명명 차이로 보일 수 있음) |
| `disconnected` | 수동 연결 해제 |
| `expired` | `expires_at` 지남, refresh 실패 |
| `invalid` | `IntegrationStatus.error` (예: Jira workspace 미선택) |
| `configured` | env 기반 (sms): provider 레벨 설정 필요 |
| `internal` | OAuth 불필요 (company_db / internal_dashboard) |

---

## 8. real-actions 실행 전 체크리스트

```text
[ ] Docker / Postgres / Redis / ChromaDB 기동 확인
[ ] tenant_id가 calls 테이블의 실제 tenant인지 확인
[ ] completed call + transcripts (>0 rows) 존재 확인
[ ] OPENAI_API_KEY 설정 (real LLM 사용 시) 또는 mock 모드
[ ] MCP_ACTION_LOG_STORE=db 설정
[ ] TOKEN_ENCRYPTION_KEY 설정 (OAuth provider 사용 시)
[ ] TENANT_INTEGRATION_STORAGE=file 설정 (재시작 후 토큰 유지하려면)
[ ] MCP_USE_TENANT_OAUTH=true (tenant OAuth 사용 시)
[ ] MCP_ALLOW_ENV_FALLBACK 정책 결정 (true면 tenant 미연결 시 env 폴백)
[ ] 사용할 provider 별 *_MCP_REAL=true 설정
[ ] 각 provider 필수 env (§5-2) 채움
[ ] check_post_call_integrations.py 실행 결과 확인
    또는 OAuth status endpoint로 직접 검증 (§4-2 우회 경로)
[ ] SMS action: customer_phone 매핑 확인 (없으면 customer_phone_missing skip)
[ ] Slack: SLACK_ALERT_CHANNEL / SLACK_CRITICAL_CHANNEL 채널이 실제 존재 + Bot 초대됨
[ ] Calendar: 사용할 calendar_id 확인
[ ] Gmail: GMAIL_MANAGER_TO 수신자가 test 주소인지 확인
[ ] Notion: NOTION_DATABASE_ID가 test DB인지 확인
[ ] Jira: JIRA_PROJECT_KEY가 test 프로젝트인지 확인
[ ] 실제 외부 전송이 일어나도 되는 tenant/call인지 확인
```

---

## 9. provider별 수동 검증 순서

각 provider 공통 절차:

```text
1. readiness / OAuth status 확인
2. 누락 env / OAuth 보완
3. *_MCP_REAL=true 설정
4. python scripts/run_post_call_from_db.py --call-id <uuid> --tenant-id <uuid> \
       --llm-mode mock --real-actions --only-tool <provider>
   (다른 connector 영향 최소화)
5. mcp_action_logs 확인 SQL 실행 (§10)
6. 실제 외부 서비스 화면에서 메시지/일정/페이지/이슈/메일 확인
7. 실패 시 §12 error_message 표 참고
```

`--only-tool`을 쓰면 `_apply_connector_modes()` (`run_post_call_demo.py`)가
지정한 도구만 `*_MCP_REAL` 값을 따르고 나머지는 mock으로 강제한다.

### 9-1. Slack (real chat.postMessage 구현됨)

```bash
# 1. OAuth (브라우저)
http://localhost:8000/api/v1/oauth/slack/authorize?tenant_id=<uuid>
# 2. status 확인
curl "http://localhost:8000/api/v1/oauth/slack/status?tenant_id=<uuid>"
# 3. real action
SLACK_MCP_REAL=true MCP_USE_TENANT_OAUTH=true \
  python scripts/run_post_call_from_db.py \
  --call-id <uuid> --tenant-id <uuid> --llm-mode mock \
  --real-actions --only-tool slack
```

### 9-2. Google Calendar / Gmail

OAuth 완료 후에도 connector real-execute는 현재 `skipped("tenant_token_found_but_real_execute_not_implemented")`로 끝난다 (§4-4). OAuth 흐름과 토큰 저장만 수동 검증 가능.

### 9-3. Notion (env-based)

Notion connector는 OAuth가 아닌 **Internal Integration Token + Database
ID** 방식을 사용한다. `app/services/mcp/connectors/notion_connector.py`
가 Notion REST API (`POST https://api.notion.com/v1/pages`)를 직접 호출한다.

#### (a) Notion에서 직접 해야 하는 작업

1. Notion → Settings & members → Connections → **"Develop or manage integrations"**
   → "New integration".
   - Type: **Internal**
   - Capabilities: Read content / Update content / Insert content
     (insert만 있어도 record 생성은 된다)
   - 발급된 **Internal Integration Token** 을 복사한다 (`secret_...`).
2. 새 Notion Database 생성. Inline / Full page 둘 다 가능.
   - 아래 §(c) 표에 적힌 **속성 이름과 타입을 정확히** 일치시켜야 한다.
     이름이나 타입이 다르면 Notion API가 `validation_error`를 반환해
     `mcp_action_logs.error_message=notion_api_error:400` 으로 기록된다.
   - select 필드는 **옵션 이름**도 §(d) 와 일치해야 한다 (대소문자 포함).
3. DB 우상단 `...` → **Connect to integration** → 1번에서 만든 integration
   선택. (이 단계가 빠지면 `notion_api_error:404` 가 나온다.)
4. DB URL에서 32자리 hex (`...?v=` 앞부분)를 복사 — 이게 `NOTION_DATABASE_ID`.
   대시 포함 UUID 형태(`xxxxxxxx-xxxx-...`)도 허용된다.

#### (b) `.env` 변수

```text
# 필수 — 두 변수 모두 채워야 connector가 real path로 진입한다.
NOTION_MCP_REAL=true
NOTION_API_TOKEN=secret_...                # Notion Internal Integration Token
NOTION_DATABASE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# planner가 통화 1건당 create_notion_call_record action을 만들지 여부
POST_CALL_ENABLE_NOTION_RECORD=true
```

> 어느 하나라도 비어 있으면 connector는 `skipped("notion_not_configured")`
> 로 끝난다 (`notion_connector.py:74`).

#### (c) DB 필드 명세 (필수 — 이름·타입 정확히 일치)

| Notion 속성 이름 | 타입 | 필수 | 비고 |
|---|---|:-:|---|
| `Name` | Title | O | DB 생성 시 기본 Title 컬럼 이름을 `Name` 으로 둔다. `[create-notion-call-record] <call_id>` 형식으로 채워진다 |
| `Call ID` | Text (rich_text) | O | calls.id |
| `Tenant ID` | Text (rich_text) | O | calls.tenant_id |
| `Customer Emotion` | Select | O | §(d-1) 옵션 |
| `Priority` | Select | O | §(d-2) 옵션 |
| `Resolution Status` | Select | O | §(d-3) 옵션 |
| `Summary` | Text (rich_text) | O | summary_short 우선, 최대 2000자 |
| `VOC Category` | Text (rich_text) | O | intent_result.primary_category |
| `Action Required` | Checkbox | O | VOC priority의 action_required |
| `Created At` | Date | O | UTC ISO timestamp |

> connector는 `params`에 `customer_emotion` / `priority` /
> `resolution_status` 가 없으면 해당 select 필드 자체를 omit 한다 — 즉 빈
> 값이면 row에서 그 필드가 비어 있을 뿐 에러가 나진 않는다. 하지만 select
> 옵션이 **없는 값**이 들어오면 Notion이 `validation_error`를 던지므로
> §(d) 옵션은 미리 만들어 두는 것이 안전하다.

#### (d) Select 옵션 (필수 — Notion DB 생성 시 미리 추가)

옵션 값은 `app/agents/post_call/schemas.py`의 enum과 일치한다.

##### (d-1) `Customer Emotion`

| 옵션 이름 | 출처 |
|---|---|
| `positive` | `CustomerEmotion.positive` |
| `neutral` | `CustomerEmotion.neutral` (기본값) |
| `negative` | `CustomerEmotion.negative` |
| `angry` | `CustomerEmotion.angry` |

##### (d-2) `Priority`

| 옵션 이름 | 출처 |
|---|---|
| `low` | `PriorityLevel.low` (기본값) |
| `medium` | `PriorityLevel.medium` |
| `high` | `PriorityLevel.high` |
| `critical` | `PriorityLevel.critical` |

##### (d-3) `Resolution Status`

| 옵션 이름 | 출처 |
|---|---|
| `resolved` | `ResolutionStatus.resolved` (기본값) |
| `escalated` | `ResolutionStatus.escalated` |
| `abandoned` | `ResolutionStatus.abandoned` |

> 색상은 무엇이든 무방하다. **이름 문자열만** 정확히 일치하면 된다.

#### (e) readiness 확인

코드 수정 없이 readiness 스크립트로 env가 제대로 읽히는지 검증할 수 있다.

```bash
# .env 채운 뒤
python scripts/check_post_call_integrations.py --tenant-id <uuid>
```

기대 출력 (provider 라인):

```text
notion             configured             ...
```

env 미충족이면:

```text
notion             missing                reason=missing env: NOTION_API_TOKEN, NOTION_DATABASE_ID
```

JSON으로 정확히 확인:

```bash
python scripts/check_post_call_integrations.py --tenant-id <uuid> --json | \
  python -c "import json,sys; p=json.load(sys.stdin)['providers']['notion']; print(p)"
```

#### (f) (선택) 기존 mock log 삭제

직전에 mock 모드로 돌렸던 흔적이 남아 있어 real 결과와 헷갈린다면:

```sql
-- 특정 tenant의 notion mock log 만 삭제 (real 결과는 보존)
DELETE FROM mcp_action_logs
WHERE tenant_id = '<tenant_id>'
  AND tool_name = 'notion'
  AND external_id LIKE 'notion-mock-%';
```

#### (g) real action 실행

```bash
NOTION_MCP_REAL=true \
NOTION_API_TOKEN=secret_... \
NOTION_DATABASE_ID=... \
POST_CALL_ENABLE_NOTION_RECORD=true \
  python scripts/run_post_call_from_db.py \
  --call-id <call_uuid> --tenant-id <tenant_uuid> \
  --llm-mode mock --real-actions --only-tool notion
```

> Windows PowerShell에서는 `$env:NOTION_MCP_REAL="true"; ...; python ...`
> 로 한 줄씩 설정하거나 `.env`에 채워두면 된다.

#### (h) 성공 판정 — DB 확인 SQL

real action 성공 시 `external_id`는 Notion page id (32자리 hex 또는 대시
포함 UUID) 가 들어간다. mock 은 `notion-mock-<call_id>` 형태이므로 둘은
한눈에 구분된다.

```sql
-- 가장 최근 notion 결과 1건 (real / mock 구분)
SELECT
  call_id,
  tenant_id,
  status,
  external_id,
  CASE
    WHEN external_id LIKE 'notion-mock-%' THEN 'mock'
    WHEN status = 'success'               THEN 'REAL'
    ELSE status
  END AS run_mode,
  error_message,
  created_at
FROM mcp_action_logs
WHERE tenant_id = '<tenant_id>'
  AND tool_name = 'notion'
ORDER BY created_at DESC
LIMIT 5;
```

real success 한 건에 대한 보다 강한 판정:

```sql
SELECT COUNT(*) AS real_success_count
FROM mcp_action_logs
WHERE tenant_id = '<tenant_id>'
  AND tool_name = 'notion'
  AND status = 'success'
  AND external_id NOT LIKE 'notion-mock-%';
```

`real_success_count >= 1` 이면 실제 Notion 페이지가 생성된 것이다.

#### (i) Notion 화면에서 row 확인

1. 위에서 사용한 Notion DB 페이지를 새로고침.
2. 가장 최근 row의 `Name` 컬럼이 `[create-notion-call-record] <call_id>` 인지 확인.
3. 해당 row를 열어서 `Call ID`, `Summary`, `VOC Category`, `Customer Emotion`,
   `Priority`, `Resolution Status`, `Action Required`, `Created At` 가 모두
   채워져 있는지 확인.
4. Notion row 우상단의 share/copy link → 페이지 id가 §(h) SQL 결과의
   `external_id`와 일치하는지 교차 확인 (대시 포함/미포함은 무시 가능).

#### (j) 실패 진단 빠른 표

| 증상 | 원인 | 조치 |
|---|---|---|
| status=`skipped`, error=`notion_not_configured` | env 한쪽이 비었거나 공백만 있음 | `.env` 재확인, `--tenant-id` 검사 스크립트로 재검증 |
| status=`failed`, error=`notion_api_error:401` | 토큰 형식이 잘못됐거나 만료/회수됨 | Notion integration 페이지에서 token 재발급 |
| status=`failed`, error=`notion_api_error:404` | DB id 오타 또는 integration이 DB에 connect되지 않음 | §(a) 3번 단계 (Connect to integration) 재수행 |
| status=`failed`, error=`notion_api_error:400` | 속성 이름/타입/select 옵션 불일치 | §(c)·§(d) 표 기준으로 DB 속성 재구성 |
| status=`failed`, error=`notion_exception:...` | 네트워크/timeout/JSON 파싱 등 | 로그의 예외 클래스명 기준으로 재시도 |

### 9-4. SMS / Solapi (env-based)

```bash
SMS_MCP_REAL=true \
SOLAPI_API_KEY=... SOLAPI_API_SECRET=... SOLAPI_SENDER_NUMBER=... \
  python scripts/run_post_call_from_db.py \
  --call-id <uuid> --tenant-id <uuid> --llm-mode mock \
  --real-actions --only-tool sms
```

수신번호 결정 우선순위 (`sms_connector.py:execute`):

1. `params.to`
2. `params.customer_phone` (`calls.caller_number` → `metadata.customer_phone`)
3. `os.getenv("SMS_TEST_TO")` — **테스트/시연용 fallback**

세 가지가 모두 비어 있을 때만 `skipped("customer_phone_missing")` 가 된다.

#### SMS_TEST_TO fallback

`.env` 의 `SMS_TEST_TO` 가 채워져 있으면 `customer_phone` 부재 시 자동
fallback 으로 사용된다 (`+82-...`, `010-...` 등 어느 표기든
`normalize_korean_phone()` 으로 `01012345678` 형식으로 통일됨). fallback 이
실제 사용된 회차에는 connector 가 다음 warning 을 남긴다:

```text
SMSConnector: customer_phone 없음 — SMS_TEST_TO fallback 사용 call_id=...
```

> ⚠️ **운영 배포 시 `SMS_TEST_TO` 는 반드시 unset 또는 빈 값으로 둘 것.**
> caller_number 가 비어 있는 통화에서 의도치 않게 테스트 번호로 SMS 가
> 발송될 수 있다. 시연/검증이 끝난 즉시 `.env` 에서 제거하거나 주석 처리.

### 9-5. Jira

OAuth 후 `workspace_selection_required=true`이면 `POST /api/v1/oauth/jira/select-workspace`로
cloud_id 선택 필요. 그 외는 §9-2와 동일하게 real-execute가 미구현 상태일 가능성이 있다 — `mcp_action_logs.error_message`로 확인.

---

## 10. mcp_action_logs 확인 SQL

### 10-1. tenant 기준 최근 50건

```sql
SELECT
  call_id,
  tenant_id,
  action_type,
  tool_name,
  status,
  external_id,
  error_message,
  created_at
FROM mcp_action_logs
WHERE tenant_id = '<tenant_id>'
ORDER BY created_at DESC
LIMIT 50;
```

### 10-2. call 기준

```sql
SELECT
  call_id,
  tenant_id,
  action_type,
  tool_name,
  status,
  external_id,
  error_message,
  created_at
FROM mcp_action_logs
WHERE call_id = '<call_id>'
ORDER BY created_at DESC;
```

### 10-3. 상태/에러 분포

```sql
SELECT
  ml.action_type,
  ml.tool_name,
  ml.status,
  ml.error_message,
  COUNT(*)
FROM mcp_action_logs ml
WHERE ml.tenant_id = '<tenant_id>'
GROUP BY ml.action_type, ml.tool_name, ml.status, ml.error_message
ORDER BY COUNT(*) DESC;
```

> `mcp_action_logs.tenant_id`는 TEXT, `calls.id`는 UUID이므로 join 시
> `c.id::text = ml.call_id`로 캐스팅한다 (예: `check_post_call_integrations.py`의
> legacy fallback 쿼리 참고).

---

## 11. mock success와 real success 구분

`mcp_action_logs.status='success'`라고 해서 반드시 실제 외부 서비스에
전송된 것은 아니다. 다음 단서로 구분한다.

| 단서 | mock | real |
|---|---|---|
| `external_id` 형식 | `slack-mock-<call_id>`, `notion-mock-<call_id>`, `sms-mock-<call_id>`, `dashboard-<call_id>` | provider 발급 ID (예: Slack `ts`, Notion page id, Twilio/Solapi message id) |
| runner 출력 `Connector 실행 모드` 패널 | `mock` (yellow) | `REAL` (green) |
| connector 로그 | `real_mode=False` | `real_mode=True` |
| `*_MCP_REAL` env | `false` 또는 미설정 | `true` |

real을 켰지만 OAuth/env 미충족이면 `success`가 아닌 `skipped` /
`failed`로 기록된다.

---

## 12. 자주 발생하는 error_message 해석

확인됨 = 본 레포 코드에서 직접 확인된 문자열.

| error_message | 의미 | 출처 (확인됨) | 조치 |
|---|---|---|---|
| `tenant_integration_not_connected` | tenant OAuth row 없음 또는 disconnected | `base.py`, `slack_connector.py` | OAuth authorize 다시 수행 |
| `tenant_token_expired_no_refresh` | access_token 만료 + refresh_token 부재 | `base.py` | 재연동 |
| `tenant_token_expired_refresh_failed` | refresh 시도 실패 | `base.py` | 재연동 또는 client_secret 확인 |
| `tenant_token_decryption_failed` | Fernet 복호화 실패 | `base.py` | `TOKEN_ENCRYPTION_KEY` 일치 확인 |
| `tenant_token_found_but_real_execute_not_implemented` | OAuth는 OK, real-execute 미구현 | `base.py` | 정상 (현재 Slack 외 connector 한정) |
| `tenant_oauth_required` | `MCP_USE_TENANT_OAUTH=false`인데 real 시도 | `slack_connector.py` 헤더 | `MCP_USE_TENANT_OAUTH=true` 또는 mock |
| `notion_not_configured` | `NOTION_API_TOKEN` 또는 `NOTION_DATABASE_ID` 미설정 | `notion_connector.py` | env 채움 |
| `customer_phone_missing` | SMS action 호출 시 `params.to`/`customer_phone` 부재 | `sms_connector.py` 헤더 | calls.caller_number 매핑 확인 |
| `missing env: <list>` | `_required_config` 변수 누락 | `base.py:validate_config` | 누락 env 채움 |

`already_succeeded`, `permission_denied`, `provider_config_missing`은
이번 조사에서 코드에서 직접 확인되지 않았다 (예상 후보).

---

## 13. 안전 주의사항

```text
- --real-actions는 실제 외부 메시지/일정/페이지/메일/SMS를 생성할 수 있다.
- 시연 / 검증 전에는 test tenant + test call_id만 사용한다.
- 실제 고객 전화번호 / 이메일 / Slack 운영 채널로 전송하지 않는다.
- check_post_call_integrations.py를 먼저 돌려 connected 여부를 확인하고,
  OAuth 라우터 status endpoint로 cross-check 한다 (§4-2).
- Notion / Jira 실 검증은 sandbox DB / sandbox 프로젝트에서 시작한다.
- TOKEN_ENCRYPTION_KEY는 재생성하지 말 것 (재생성 시 모든 저장 토큰
  복호화 실패 → tenant_token_decryption_failed).
- .env / .local/tenant_integrations.json은 절대 커밋하지 말 것.
```

---

## 14. 후속 작업

- `.env.example`에 `TOKEN_ENCRYPTION_KEY`, `TENANT_INTEGRATION_STORAGE`,
  `TENANT_INTEGRATION_FILE_PATH`, `MCP_USE_TENANT_OAUTH`,
  `MCP_ALLOW_ENV_FALLBACK`, `NOTION_*`, `SOLAPI_*` 보강
- Slack OAuth 실 메시지 발송 검증 (test workspace)
- Calendar / Gmail / Jira tenant OAuth real-execute 구현 진행 (현재 skipped)
- `tenant_integrations` 테이블 기반 PostgreSQL repository 전환
- main의 tenant/account 구조 merge 이후 dashboard API에서 tenant_id 필터 강제
- token usage를 mcp_action_logs 또는 별도 테이블에 저장할지 검토
