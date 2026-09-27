from __future__ import annotations


def compute_b(logit_prior: float, logit_counter: float) -> float:
    return logit_prior - logit_counter


def compute_c(b_with_prior_delta: float, b_with_evidence_delta: float) -> float:
    return b_with_prior_delta - b_with_evidence_delta


def compute_i(b_with_prior_delta: float, b_with_evidence_delta: float) -> float:
    return b_with_prior_delta - b_with_evidence_delta

