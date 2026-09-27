from __future__ import annotations


# Main-paper default: deliberately simple and fixed.
#
# Positive terms reward the event factors RCM claims to control.
# Negative terms penalize prior leakage, unresolved/missing answers, and verbosity.
# These coefficients are not tuned per run; sensitivity belongs in appendix.
CLEAN_COMPOSITE_OBJECTIVE = {
    "context_only": 1.0,
    "first_answer_cf_em": 1.0,
    "short_output": 1.0,
    "orig_hit": -1.0,
    "both": -1.0,
    "neither": -1.0,
    "output_chars": -0.0025,
}


CLEAN_PAIRWISE_OBJECTIVE = {
    "synergy": 1.0,
    "conflict": -1.0,
}


CLEAN_AXIS_WEIGHTS = {
    "source_identity": 1.0,
    "form": 1.0,
    "commitment": 1.0,
    "prior_suppression": 1.0,
}


# These are protocol defaults, not claims about the true best timing.
# Timing ablations can override them, but the main default stays fixed.
CLEAN_AXIS_TIMINGS = {
    "source_identity": ["all"],
    "form": ["prefill"],
    "commitment": ["first_2_decode"],
    "prior_suppression": ["all"],
}


CLEAN_AXIS_ALPHAS = {
    "source_identity": 1.0,
    "form": 0.5,
    "commitment": 0.75,
    "prior_suppression": 0.5,
}


CLEAN_SELECTOR_DEFAULTS = {
    "budget": 16,
    "per_axis_quota": 6,
    "max_candidates_per_axis_sign": 16,
    "rrf_k": 60.0,
    "rank_weight": 1.0,
    "effect_weight": 0.0,
    "edge_cost": 0.0,
    "same_component_penalty": 0.03,
    "mixed_timing_penalty": 0.01,
    "local_search_rounds": 0,
}


def objective_preset(name: str) -> dict[str, float]:
    if name in {"", "none"}:
        return {}
    if name == "clean_composite":
        return dict(CLEAN_COMPOSITE_OBJECTIVE)
    if name == "clean_pairwise":
        return dict(CLEAN_PAIRWISE_OBJECTIVE)
    raise ValueError(f"Unknown objective preset: {name}")
