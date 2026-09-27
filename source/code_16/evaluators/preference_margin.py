"""Aggregate registered full-continuation preference scores for candidate selection."""

from .callback import align_prediction_references, write_evaluation


def evaluate_preference_margin(request):
    scored = []
    for prediction, reference in align_prediction_references(request):
        if prediction.get("prediction_kind") != "paired_completion_scores":
            raise ValueError("preference selection requires paired completion predictions")
        for role in ("chosen", "rejected"):
            if prediction.get(f"scored_{role}") != reference.get(role):
                raise ValueError(f"preference prediction differs from registered {role}")
        chosen = prediction["chosen_avglogp"]
        rejected = prediction["rejected_avglogp"]
        scored.append((prediction, {
            "chosen_avglogp": chosen, "rejected_avglogp": rejected,
            "pair_avglogp_margin": chosen - rejected,
        }))
    return write_evaluation(request, scored)
