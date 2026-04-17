from app.utils.logger import get_logger

# M2 기능 — Task 브랜치 MCP 도구

logger = get_logger(__name__)


class CompanyDBMCPService:
    async def query(self, sql: str, params: dict) -> list[dict]:
        # TODO(M2): 임시 회사 DB MCP 연동 구현
        raise NotImplementedError
