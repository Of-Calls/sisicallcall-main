from app.utils.logger import get_logger

# M2 기능 — Task 브랜치 MCP 도구

logger = get_logger(__name__)


class GmailMCPService:
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        # TODO(M2): Gmail MCP 연동 구현
        raise NotImplementedError
