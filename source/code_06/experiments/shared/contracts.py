"""Typed execution contracts shared by controllers and registered backends.

Contracts contain no scientific defaults. They define the complete request
and result boundary that lets a controller call a method backend without
knowing that backend's training or inference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class ComponentContractError(ValueError):
    """Raised when a component request or result is incomplete."""


@dataclass(frozen=True, slots=True)
class BankTrainRequest:
    """One independently trainable bank with all external choices resolved."""

    bank_id: str
    ordered_positions: tuple[Mapping[str, Any], ...]
    method_plan: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    training_data_manifest: Path
    validation_data_manifest: Path | None
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class ComponentPayloadResult:
    """One component-owned payload extracted from a trained bank."""

    component_id: str
    payload_path: Path
    extraction_manifest_path: Path


@dataclass(frozen=True, slots=True)
class BankTrainResult:
    """Files and execution facts returned by one bank training operation."""

    payload_path: Path
    training_manifest_path: Path
    optimizer_steps: int
    component_payloads: tuple[ComponentPayloadResult, ...] = ()


@dataclass(frozen=True, slots=True)
class BankComposeRequest:
    """A zero-training request to compose retained component payloads."""

    bank_id: str
    ordered_component_ids: tuple[str, ...]
    component_manifests: tuple[Mapping[str, Any], ...]
    method_plan: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Task-evaluator request independent of the intervention method."""

    predictions: Path
    references: Path
    evaluation_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Evaluator-owned files returned to the experiment controller."""

    per_sample_scores: Path
    summary_metrics: Path
    auxiliary_artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One controller application with every external choice resolved."""

    controller_manifest: Path | None
    data_role_manifest: Path
    model_config: Mapping[str, Any]
    generation_config: Mapping[str, Any]
    trajectory_config: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    selected_alpha: Any
    selected_alpha_grid: tuple[Any, ...] | None
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Backend-owned generation files returned to the shared controller."""

    predictions: Path
    execution_manifest: Path
    trajectory_path: Path | None = None
    component_scores_path: Path | None = None


@dataclass(frozen=True, slots=True)
class PositionSearchRequest:
    """One position method over one protocol-selected data role."""

    method_plan: Mapping[str, Any]
    trajectory_config: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    selector_data_manifest: Path
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class PositionSearchResult:
    """Backend-owned candidate and execution manifests."""

    candidate_manifest: Path
    execution_manifest: Path
    trajectory_path: Path | None = None
    component_scores_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DataMaterializationRequest:
    """One protocol-owned dataset preparation with machine paths resolved."""

    dataset: str
    data_spec: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class DataMaterializationResult:
    """The backend-owned frozen dataset manifest."""

    dataset_manifest: Path


@dataclass(frozen=True, slots=True)
class AlphaSelectionRequest:
    """Fully declared curve and rule passed to an alpha selector."""

    curve_rows: tuple[Mapping[str, Any], ...]
    selection_config: Mapping[str, Any]


@runtime_checkable
class AlphaSelectorBackend(Protocol):
    """Deterministically chooses alpha from already evaluated calibration rows."""

    def select(self, request: AlphaSelectionRequest) -> Mapping[str, Any]:
        ...


@runtime_checkable
class BankBackend(Protocol):
    """Backend surface consumed by the shared bank execution controller."""

    def train_bank(self, request: BankTrainRequest) -> BankTrainResult:
        ...


@runtime_checkable
class ComponentComposingBankBackend(BankBackend, Protocol):
    """Optional extension used by no-retraining post-audit composition."""

    def compose_bank(self, request: BankComposeRequest) -> BankTrainResult:
        ...


@runtime_checkable
class EvaluatorBackend(Protocol):
    """Task evaluator; it cannot select positions, train, or choose alpha."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        ...


@runtime_checkable
class GenerationBackend(Protocol):
    """Method application backend; it cannot select data roles or alpha."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


@runtime_checkable
class PositionBackend(Protocol):
    """Candidate generator; selection policy remains in the position controller."""

    def select_positions(self, request: PositionSearchRequest) -> PositionSearchResult:
        ...


@runtime_checkable
class DataAdapterBackend(Protocol):
    """Prepares protocol roles; it cannot choose methods or consume test outputs."""

    def materialize(self, request: DataMaterializationRequest) -> DataMaterializationResult:
        ...


def require_operations(implementation: Any, operations: Sequence[str]) -> None:
    missing = [
        name
        for name in operations
        if not callable(getattr(implementation, name, None))
    ]
    if missing:
        raise ComponentContractError(
            f"registered implementation lacks operations: {missing}"
        )
