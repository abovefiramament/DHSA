from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CompetitionDelta:
    component_id: str
    component_type: str
    c_t: float
    i_t: float


@dataclass(slots=True)
class CompetitionProfile:
    sample_id: str
    deltas: list[CompetitionDelta]

