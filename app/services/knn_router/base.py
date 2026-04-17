from abc import ABC, abstractmethod


class BaseKNNRouterService(ABC):
    @abstractmethod
    async def classify(
        self, embedding: list[float], tenant_id: str
    ) -> tuple[str | None, float]:
        """
        KNN 분류 수행.
        Returns:
            (intent_label, confidence_score)
            intent_label: "intent_faq" | "intent_task" | "intent_auth" | "intent_escalation" | None
        """
        raise NotImplementedError
