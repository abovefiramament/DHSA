"""Formal Site evaluator registrations.

Task scorers are separate from CAST. Each evaluator owns only its registered
measurement protocol; it cannot select positions or run CAST.
"""

from __future__ import annotations

from experiments.shared.component_registry import (
    COMPONENT_REGISTRY,
    ComponentRegistry,
    ComponentRegistryError,
)

from . import confiqa, imdb, tldr, tldr_deepseek


def _register(
    registry: ComponentRegistry,
    *,
    component_id: str,
    implementation: object,
) -> None:
    try:
        registry.register(
            kind="evaluator",
            component_id=component_id,
            version=1,
            implementation=implementation,
            operations=("evaluate",),
            input_contract="EvaluationRequest/v1",
            output_contract="EvaluationResult/v1",
        )
    except ComponentRegistryError as exc:
        if "duplicate component registration" not in str(exc):
            raise


def register_site_evaluators(
    *,
    device: str,
    registry: ComponentRegistry = COMPONENT_REGISTRY,
) -> None:
    """Register every formal scorer without model/data/machine-path discovery."""

    _register(
        registry,
        component_id="site_evaluator_confiqa",
        implementation=confiqa.ConfiQAFormalEvaluator(),
    )
    _register(
        registry,
        component_id="site_evaluator_imdb",
        implementation=imdb.IMDbSentimentEvaluator(device=device),
    )
    _register(
        registry,
        component_id="site_evaluator_tldr",
        implementation=tldr.TLDRHumanReferenceEvaluator(
            tldr_deepseek.DeepSeekV4ProHumanReferenceJudge()
        ),
    )
