from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SteeringConfig:
    component_ids: list[str]
    alpha: float = 1.0


class SteeringEngine:
    """
    Placeholder steering interface.
    Real implementation should patch cached component deltas during forward pass.
    """

    def __init__(self, config: SteeringConfig) -> None:
        self.config = config

    def plan(self) -> dict:
        return {"component_ids": self.config.component_ids, "alpha": self.config.alpha}

