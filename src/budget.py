from dataclasses import dataclass


@dataclass
class InvestigationBudget:
    max_queries: int = 10
    queries_used: int = 0

    max_cost: float = 0.10
    cost_used: float = 0.0

    @property
    def queries_remaining(self) -> int:
        return max(0, self.max_queries - self.queries_used)

    @property
    def cost_remaining(self) -> float:
        return max(0.0, self.max_cost - self.cost_used)

    def can_continue(self) -> bool:
        return (
            self.queries_remaining > 0
            and self.cost_remaining > 0
        )