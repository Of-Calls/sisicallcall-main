from app.utils.logger import get_logger

# M2 기능 — Task 브랜치 MCP 도구

logger = get_logger(__name__)


class CalendarMCPService:
    async def create_event(self, title: str, start: str, end: str) -> dict:
        # TODO(M2): Google Calendar MCP 연동 구현
        raise NotImplementedError
