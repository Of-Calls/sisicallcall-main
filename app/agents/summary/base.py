from abc import ABC, abstractmethod


class BaseSummaryAgent(ABC):
    @abstractmethod
    async def run(self, call_id: str, tenant_id: str) -> dict:
        """
        통화 요약 실행.
        Returns:
            {
                "summary_short": str,   # 200자 이내 (동기 모드)
                "summary_long": str,    # 500~1000자 구조화 JSON (비동기 모드)
            }
        """
        raise NotImplementedError
