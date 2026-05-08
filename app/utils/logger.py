import datetime
import glob
import logging
import logging.handlers
import os
import re
import shutil
import sys

# 크로스플랫폼 ANSI 색상 지원 — Windows cmd/PowerShell 모두에서 색상 출력.
# colorama 가 없을 때를 대비한 fallback 으로 os.system("") 도 실행.
try:
    import colorama
    colorama.just_fix_windows_console()
except ImportError:
    if os.name == "nt":
        os.system("")

# Windows 콘솔 한글 깨짐 방지 — stdout 을 UTF-8 로 강제 재구성 (Python 3.7+).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


class _ColorFormatter(logging.Formatter):
    """터미널 가독성 향상 — 레벨별 색상 + 모듈명 축약 + 시각만 표시."""

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",      # cyan
        "INFO":     "\033[32m",      # green
        "WARNING":  "\033[33m",      # yellow
        "ERROR":    "\033[31m",      # red
        "CRITICAL": "\033[1;31m",    # bold red
    }
    DIM = "\033[2m"
    RESET = "\033[0m"

    # 자주 쓰는 모듈은 짧은 별칭으로 — 가독성 + 정렬 폭 절약.
    SHORT_ALIASES = {
        "app.api.v1.call":                                 "api.call",
        "app.services.tts.twilio_channel":                 "tts.twilio",
        "app.services.tts.azure":                          "tts.azure",
        "app.services.stt.deepgram_streaming":             "stt.stream",
        "app.services.stt.deepgram_prerecorded":           "stt.prerec",
        "app.services.speaker_verify.titanet":             "verify.titanet",
        "app.services.cache.semantic_cache":               "cache.sem",
        "app.services.embedding.local":                    "embed.local",
        "app.services.rag.chroma":                         "rag.chroma",
        "app.services.session.redis_session":              "session.redis",
        "app.agents.conversational.graph":                 "graph",
    }

    @classmethod
    def _short_name(cls, full: str) -> str:
        if full in cls.SHORT_ALIASES:
            return cls.SHORT_ALIASES[full]
        # 노드 모듈은 'node:name' 형태로 — graph 단계 식별 용이
        if ".nodes." in full:
            tail = full.rsplit(".", 1)[-1].removesuffix("_node")
            return f"node.{tail}"
        # 그 외에는 마지막 segment 만
        return full.rsplit(".", 1)[-1]

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        record.colored_level = f"{color}{record.levelname:<7}{self.RESET}"
        record.short_name = f"{self._short_name(record.name):<16}"
        return super().format(record)


class _PlainFormatter(logging.Formatter):
    """파일 출력용 — ANSI 색상 제거, 날짜 포함, 모듈명 축약은 유지."""

    def format(self, record: logging.LogRecord) -> str:
        record.short_name = f"{_ColorFormatter._short_name(record.name):<16}"
        return super().format(record)


# 루트 로거 파일 핸들러 1회 설정 — 여러 get_logger() 호출에서도 중복 방지.
_root_file_handler_added = False

# 서버 실행 단위로 로그 파일을 분리할 때 보관할 최대 일수 (이보다 오래된 파일 자동 삭제).
_LOG_RETENTION_DAYS = 7


_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _expand_log_path(raw_path: str, started_at: datetime.datetime) -> str:
    """LOG_FILE 경로에 날짜 폴더 + 시간 timestamp 를 부착해 실행 단위 파일명 생성.

    예시:
        logs/server.log         → logs/2026-04-28/server_233458.log
        logs/foo/bar.txt        → logs/foo/2026-04-28/bar_233458.txt
        logs/app                → logs/2026-04-28/app_233458.log
    """
    date_str = started_at.strftime("%Y-%m-%d")
    time_str = started_at.strftime("%H%M%S")
    base, ext = os.path.splitext(raw_path)
    parent_dir, file_base = os.path.split(base)
    target_dir = os.path.join(parent_dir, date_str) if parent_dir else date_str
    return os.path.join(target_dir, f"{file_base}_{time_str}{ext or '.log'}")


def _cleanup_old_logs(raw_path: str, retention_days: int) -> None:
    """`{LOG_FILE 부모}/{YYYY-MM-DD}` 형태 폴더 중 retention 초과면 통째 삭제.

    YYYY-MM-DD 패턴인 폴더만 대상이라 다른 시스템 디렉토리를 실수로 지우지 않는다.
    하위 호환: 옛날 평면 패턴(`logs/server_YYYYMMDD_HHMMSS.log`) 파일도 retention 초과 시 정리.
    """
    parent_dir = os.path.dirname(raw_path) or "."
    cutoff = datetime.datetime.now().timestamp() - retention_days * 86400

    if os.path.isdir(parent_dir):
        for entry in os.listdir(parent_dir):
            full = os.path.join(parent_dir, entry)
            # 1) 새 구조 — YYYY-MM-DD 폴더 통째 삭제
            if _DATE_DIR_RE.match(entry) and os.path.isdir(full):
                try:
                    if os.path.getmtime(full) < cutoff:
                        shutil.rmtree(full, ignore_errors=True)
                except OSError:
                    pass

    # 2) 옛 구조 — `{base}_*.{ext}` 평면 파일도 mtime 기반 정리
    base, ext = os.path.splitext(raw_path)
    pattern = f"{base}_*{ext or '.log'}"
    for path in glob.glob(pattern):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _ensure_root_file_handler() -> None:
    """LOG_FILE 환경변수가 설정되어 있으면 루트 로거에 파일 핸들러 부착.

    각 named logger 는 자체 스트림 핸들러(컬러) 를 가지고, 레코드를 루트로
    propagate 하여 파일에도 함께 기록한다.

    파일명에 서버 시작 timestamp 를 부착하여 실행 단위로 파일이 분리된다:
        logs/server.log → logs/server_20260423_233458.log
    7일 이상 된 파일은 자동 정리.

    활성화 방법:
        LOG_FILE=logs/server.log uvicorn app.main:app --reload
        (PowerShell)  $env:LOG_FILE="logs/server.log"; uvicorn app.main:app --reload
    """
    global _root_file_handler_added
    if _root_file_handler_added:
        return

    raw_log_file = os.getenv("LOG_FILE")
    if not raw_log_file:
        _root_file_handler_added = True  # 비활성 상태도 1회만 결정
        return

    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        _root_file_handler_added = True
        return

    started_at = datetime.datetime.now()
    actual_path = _expand_log_path(raw_log_file, started_at)

    # 대상 디렉토리 (날짜 폴더 포함) 자동 생성
    target_dir = os.path.dirname(actual_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    # 오래된 로그 정리 (best-effort) — YYYY-MM-DD 폴더 통째 또는 옛 평면 파일
    _cleanup_old_logs(raw_log_file, _LOG_RETENTION_DAYS)

    file_handler = logging.FileHandler(actual_path, encoding="utf-8")
    file_handler.setFormatter(
        _PlainFormatter(
            "%(asctime)s %(levelname)-7s %(short_name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))

    _root_file_handler_added = True
    root.info(
        "로그 파일 기록 시작 — path=%s (실행 단위 timestamp, %d일 보관)",
        actual_path, _LOG_RETENTION_DAYS,
    )


def _is_mcp_stdio_mode() -> bool:
    """MCP Server 가 stdio transport 로 실행 중인지 표시.

    MCP_STDIO_MODE=true 일 때 stdout 은 JSON-RPC 전용이므로 일반 로그를
    stderr 로 보낸다 (기본 level 도 WARNING 으로 상향). MCP Server entrypoint
    (scripts/run_mcp_server.py, app/services/mcp/server/__main__.py 등) 가
    project import 보다 먼저 이 env 를 set 한다.
    """
    return os.getenv("MCP_STDIO_MODE", "").lower() in ("1", "true")


def get_logger(name: str) -> logging.Logger:
    _ensure_root_file_handler()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # stdio MCP mode 에서는 stdout 출력 금지 — JSON-RPC 프레이밍이 깨진다.
    mcp_stdio = _is_mcp_stdio_mode()
    stream = sys.stderr if mcp_stdio else sys.stdout

    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        _ColorFormatter(
            "%(asctime)s %(colored_level)s %(short_name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)

    if mcp_stdio:
        # stdio mode 의 기본 level 은 WARNING — 너무 verbose 한 INFO 를 끈다.
        # MCP_STDIO_LOG_LEVEL 로 override 가능 (디버그용 INFO/DEBUG).
        level = os.getenv("MCP_STDIO_LOG_LEVEL", "WARNING").upper()
    else:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    # LOG_FILE 설정 시 루트의 파일 핸들러로 전달하기 위해 propagate=True,
    # 아니면 False 로 차단해 외부 라이브러리 핸들러와의 중복 출력을 방지.
    logger.propagate = bool(os.getenv("LOG_FILE"))
    return logger


def reroute_existing_loggers_to_stderr() -> None:
    """이미 만들어진 logger 들의 StreamHandler stream 을 stderr 로 옮긴다.

    MCP_STDIO_MODE 가 import 시점보다 늦게 set 됐을 때 (예: 첫 import 가
    이미 stdout 핸들러를 만든 상태에서 main() 진입) 안전망. 실제로는
    scripts/run_mcp_server.py 에서 import 전에 set 하므로 거의 호출될 일이
    없지만, 이중 안전.
    """
    seen: set[int] = set()
    loggers = [logging.getLogger()]  # root
    loggers.extend(logging.Logger.manager.loggerDict.values())  # type: ignore[arg-type]
    for lg in loggers:
        if not isinstance(lg, logging.Logger):
            continue
        if id(lg) in seen:
            continue
        seen.add(id(lg))
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout:
                h.stream = sys.stderr
