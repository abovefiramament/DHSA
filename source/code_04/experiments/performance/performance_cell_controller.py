"""Compile one formal Performance cell through the shared experiment compiler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from experiments.performance.performance_protocol_binding import (
    DEFAULT_PROTOCOL,
    iter_declared_performance_cells,
    iter_performance_cells,
    resolve_performance_cell,
)
from experiments.shared.experiment_controller import compile_experiment_cell
from experiments.site.site_cell_controller import component_registration_status


class PerformanceCellError(ValueError):
    """Raised when a compiled cell drifts from its declared Performance row."""


DIRECT_FLOW = (
    "dataset_controller",
    "test_gate",
    "generation",
    "evaluation",
)
CAST_FLOW = {
    "confiqa": (
        "dataset_controller", "position_search", "position_controller",
        "position_freeze", "bank_execute", "generation", "evaluation",
        "alpha_freeze", "test_gate", "generation", "evaluation",
    ),
    "imdb": (
        "dataset_controller", "position_search", "position_controller",
        "position_freeze", "bank_execute", "test_gate", "generation", "evaluation",
    ),
    "tldr": (
        "dataset_controller", "position_search", "position_controller",
        "position_freeze", "bank_execute", "post_training_audit_prepare",
        "post_training_audit_execute", "manual_audit",
        "post_training_audit_freeze", "bank_finalize_audit", "generation",
        "evaluation", "alpha_freeze", "test_gate", "generation", "evaluation",
    ),
}
LOREFT_PREFIX_FLOW = (
    "dataset_controller",
    "native_baseline_train", "generation", "evaluation",
    "native_baseline_train", "generation", "evaluation",
    "native_baseline_train", "generation", "evaluation",
    "native_baseline_train", "generation", "evaluation",
    "controller_select",
)
LOREFT_FLOW = {
    "confiqa": LOREFT_PREFIX_FLOW + (
        "generation", "evaluation", "alpha_freeze", "test_gate",
        "generation", "evaluation",
    ),
    "imdb": LOREFT_PREFIX_FLOW + ("test_gate", "generation", "evaluation"),
    "tldr": LOREFT_PREFIX_FLOW + (
        "generation", "evaluation", "alpha_freeze", "test_gate",
        "generation", "evaluation",
    ),
}
BIPO_PREFIX_FLOW = (
    "dataset_controller",
    "position_search", "position_controller", "position_freeze",
    "native_baseline_train", "generation", "evaluation",
    "native_baseline_train", "generation", "evaluation",
    "native_baseline_train", "generation", "evaluation",
    "controller_select",
)
BIPO_FLOW = {
    "confiqa": BIPO_PREFIX_FLOW + (
        "generation", "evaluation", "alpha_freeze", "test_gate",
        "generation", "evaluation",
    ),
    "imdb": BIPO_PREFIX_FLOW + ("test_gate", "generation", "evaluation"),
    "tldr": BIPO_PREFIX_FLOW + (
        "generation", "evaluation", "alpha_freeze", "test_gate",
        "generation", "evaluation",
    ),
}

def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PerformanceCellError(f"expected object in {path}")
    return value


def _validate_bundle_flow(bundle_dir: Path, row: Mapping[str, Any]) -> None:
    bundle = _load(bundle_dir / "bundle.json")
    flow = bundle.get("shared", {}).get("flow")
    if not isinstance(flow, list):
        raise PerformanceCellError("Performance bundle flow is missing")
    handlers = tuple(stage.get("handler") for stage in flow)
    if row["baseline"]["kind"] == "direct_policy":
        expected = DIRECT_FLOW
    elif row["baseline"]["kind"] == "native_external":
        from experiments.performance.performance_bundle_generator import _bipo_protocol, _loreft_protocol
        method = row["baseline"]["method"]
        protocol = _loreft_protocol(row) if method == "loreft" else _bipo_protocol(row)
        prefix = ("dataset_controller",) + (
            ("position_search", "position_controller", "position_freeze") if method == "bipo" else ()
        )
        candidates = ("native_baseline_train", "generation", "evaluation") * len(protocol["search"]["candidate_order"])
        calibration = () if row["dataset"] == "imdb" else ("generation", "evaluation", "alpha_freeze")
        expected = prefix + candidates + ("controller_select",) + calibration + ("test_gate", "generation", "evaluation")
    else:
        expected = CAST_FLOW[row["dataset"]]
    if handlers != expected:
        raise PerformanceCellError(
            f"Performance flow drift: expected={expected} got={handlers}"
        )


def compile_performance_cell(
    bundle_dir: Path,
    *,
    cell_id: str,
    runtime_registry: Path | None = None,
    matrix_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    row = resolve_performance_cell(cell_id, protocol_path=matrix_path)
    bundle_dir = bundle_dir.expanduser().resolve()
    _validate_bundle_flow(bundle_dir, row)
    manifest = compile_experiment_cell(
        bundle_dir, cell_id=cell_id, runtime_registry=runtime_registry
    )
    cell_root = Path(manifest["runtime_cell_root"])
    resolved = _load(cell_root / "config" / (
        "runtime_config.local.json" if runtime_registry is not None else "resolved_config.json"
    ))
    identity = resolved.get("identity", {})
    if (
        identity.get("dataset") != row["dataset"]
        or identity.get("model_family") != row["model_family"]
    ):
        raise PerformanceCellError("compiled Performance identity drift")
    baseline = resolved.get("baseline", {})
    expected_method = {
        "direct_policy": "direct_policy",
        "cast": "cast",
        "native_external": row["baseline"].get("method"),
    }[row["baseline"]["kind"]]
    if baseline.get("method") != expected_method:
        raise PerformanceCellError("compiled Performance method drift")
    locked = resolved.get("cell", {}).get("locked_job", {})
    if locked.get("baseline_id") != row["baseline_id"]:
        raise PerformanceCellError("compiled Performance baseline ID drift")
    if expected_method == "cast":
        parameters = baseline.get("parameters", {})
        controller = parameters.get("controller", {})
        timing = row["timing"]
        if controller.get("training_timings") != [timing["training"]]:
            raise PerformanceCellError("CAST training timing drift")
        if controller.get("inference_timings") != [timing["inference"]]:
            raise PerformanceCellError("CAST inference timing drift")
        if controller.get("family") != row["baseline"]["family"]:
            raise PerformanceCellError("CAST operator family drift")
        if parameters.get("transfer", {}).get("mode") != row["baseline"]["transfer"]:
            raise PerformanceCellError("CAST transfer mode drift")
    status = (
        component_registration_status(cell_root) if runtime_registry is not None
        else {"status": "pending_machine_paths", "execution_checked": False}
    )
    status.update({"cell_id": cell_id, "baseline_id": row["baseline_id"]})
    (cell_root / "config" / "component_registration_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"row": row, "flow": manifest, "component_status": status}


__all__ = [
    "DEFAULT_PROTOCOL",
    "compile_performance_cell",
    "iter_declared_performance_cells",
    "iter_performance_cells",
]
