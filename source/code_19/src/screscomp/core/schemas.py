from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class FactTriple:
    fact_id: str
    subject_id: str
    subject: str
    relation: str
    question: str
    true_answer: str
    answer_type: str
    domain: str = "general"
    source: str = "seed"
    source_date: str = ""
    true_answer_id: str = ""
    counter_answer: str = ""
    counter_answer_id: str = ""
    counter_difficulty: str = ""
    counter_rationale: str = ""
    split_group_id: str = ""
    single_answer_certainty: str = ""
    time_sensitive: bool = False
    multi_answer: bool = False
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FactPairCandidate:
    sample_id: str
    fact_id: str
    subject_id: str
    subject: str
    relation: str
    question: str
    answer_type: str
    domain: str
    true_answer_id: str
    split_group_id: str
    candidate_1: str
    candidate_2: str
    true_answer: str
    prior_only_prompt: str
    swapped_prior_only_prompt: str
    counter_answer_id: str = ""
    counter_difficulty: str = ""
    counter_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetainedPair:
    sample_id: str
    fact_id: str
    subject_id: str
    subject: str
    relation: str
    question: str
    answer_type: str
    domain: str
    true_answer: str
    true_answer_id: str
    split_group_id: str
    option_A: str
    option_B: str
    prior_answer: str
    counter_prior_answer: str
    counter_answer_id: str
    counter_difficulty: str
    counter_rationale: str
    prior_label: str
    counter_prior_label: str
    B_prior: float
    B_prior_swapped: float
    prior_only_prompt: str
    swapped_prior_only_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExcludedPair:
    sample_id: str
    fact_id: str
    exclusion_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExcludedFact:
    fact_id: str
    subject_id: str
    exclusion_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
