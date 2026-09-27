"""Validate and register inference-time text and component-score traces.

The generation backend performs inference and writes the trace files.  This
controller owns the shared evidence contract: every enabled trace is JSONL,
uses stable sample/component identifiers, is rooted under the cell evidence
directory, and is included in the immutable generation artifact closure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

class TrajectoryControllerError(ValueError):
    """Raised when inference trace configuration or artifacts are invalid."""


ALLOWED_COVERAGE = frozenset({"all_samples", "declared_subset"})
ALLOWED_TRACE_KINDS = frozenset({"rcm", "audit", "generation_test"})
DEFAULT_TRAJECTORY_CONFIG = {"enabled": True, "coverage": "all_samples"}


def normalize_trajectory_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the evidence-capture default or validate an explicit override.

    Capturing evidence is an operational default rather than a scientific
    parameter.  A caller may explicitly set ``enabled`` to ``False`` when the
    artifact policy allows a lightweight run.
    """

    if config is None:
        return dict(DEFAULT_TRAJECTORY_CONFIG)
    return validate_trajectory_config(config)


def validate_trace_kind(trace_kind: str) -> str:
    if trace_kind not in ALLOWED_TRACE_KINDS:
        raise TrajectoryControllerError(
            f"trace_kind must be one of {sorted(ALLOWED_TRACE_KINDS)}"
        )
    return trace_kind


def validate_trajectory_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit, method-neutral inference trace switch."""

    if not isinstance(config, Mapping):
        raise TrajectoryControllerError("trajectory_config must be an object")
    required = {"enabled", "coverage"}
    if set(config) != required:
        raise TrajectoryControllerError(
            "trajectory_config must contain exactly enabled and coverage"
        )
    enabled = config.get("enabled")
    coverage = config.get("coverage")
    if not isinstance(enabled, bool):
        raise TrajectoryControllerError("trajectory_config.enabled must be boolean")
    if coverage not in ALLOWED_COVERAGE:
        raise TrajectoryControllerError(
            "trajectory_config.coverage must be all_samples or declared_subset"
        )
    return {"enabled": enabled, "coverage": coverage}


def _artifact(path: Path, root: Path, *, field: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise TrajectoryControllerError(f"{field} escapes evidence root") from exc
    if not resolved.is_file():
        raise TrajectoryControllerError(f"{field} does not exist: {resolved}")
    return {
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
    }


def _jsonl(path: Path, *, field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrajectoryControllerError(f"cannot read {field}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise TrajectoryControllerError(f"{field} contains a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryControllerError(
                f"{field} line {line_number} is not JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise TrajectoryControllerError(f"{field} line {line_number} must be an object")
        rows.append(dict(value))
    return rows


def register_inference_trajectory(
    *,
    config: Mapping[str, Any] | None,
    trajectory_path: Path | None,
    component_scores_path: Path | None,
    evidence_root: Path,
    trace_kind: str = "generation_test",
) -> dict[str, Any]:
    """Validate backend trace files and return portable artifact records.

    Enabled traces require generated text for every sample. A controlled
    generation additionally requires per-component scores. A controller-free
    base generation may instead explicitly mark every trace row as
    ``not_applicable_no_active_controller``; it retains the complete text trace
    and an empty component-score artifact rather than inventing a component.
    """

    normalized = normalize_trajectory_config(config)
    trace_kind = validate_trace_kind(trace_kind)
    if not normalized["enabled"]:
        if trajectory_path is not None or component_scores_path is not None:
            raise TrajectoryControllerError(
                "disabled trajectory_config cannot return trace artifacts"
            )
        return {
            "config": normalized,
            "trajectory": None,
            "component_scores": None,
            "trajectory_rows": 0,
            "component_score_rows": 0,
            "trace_kind": trace_kind,
        }
    if trajectory_path is None or component_scores_path is None:
        raise TrajectoryControllerError(
            "enabled trajectory_config requires trajectory and component_scores artifacts"
        )

    trajectory_artifact = _artifact(
        trajectory_path, evidence_root, field="trajectory"
    )
    score_artifact = _artifact(
        component_scores_path, evidence_root, field="component_scores"
    )
    trajectory_rows = _jsonl(trajectory_path, field="trajectory")
    score_rows = _jsonl(component_scores_path, field="component_scores")
    if not trajectory_rows:
        raise TrajectoryControllerError("trajectory must contain at least one row")

    sample_ids: set[str] = set()
    no_active_component_samples: set[str] = set()
    for index, row in enumerate(trajectory_rows, start=1):
        sample_id = row.get("sample_id")
        generated_text = row.get("generated_text")
        if not isinstance(sample_id, str) or not sample_id:
            raise TrajectoryControllerError(
                f"trajectory row {index} needs a non-empty sample_id"
            )
        if sample_id in sample_ids:
            raise TrajectoryControllerError(
                f"trajectory contains duplicate sample_id: {sample_id}"
            )
        if not isinstance(generated_text, str):
            raise TrajectoryControllerError(
                f"trajectory row {index} needs generated_text"
            )
        component_status = row.get("component_scores_status", "available")
        if component_status not in {"available", "not_applicable_no_active_controller"}:
            raise TrajectoryControllerError(
                f"trajectory row {index} has invalid component_scores_status"
            )
        if component_status == "not_applicable_no_active_controller":
            no_active_component_samples.add(sample_id)
        sample_ids.add(sample_id)

    score_keys: set[tuple[str, str]] = set()
    score_sample_ids: set[str] = set()
    for index, row in enumerate(score_rows, start=1):
        sample_id = row.get("sample_id")
        component_id = row.get("component_id")
        score = row.get("score")
        if not isinstance(sample_id, str) or not sample_id:
            raise TrajectoryControllerError(
                f"component_scores row {index} needs a non-empty sample_id"
            )
        if not isinstance(component_id, str) or not component_id:
            raise TrajectoryControllerError(
                f"component_scores row {index} needs a non-empty component_id"
            )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TrajectoryControllerError(
                f"component_scores row {index} score must be numeric"
            )
        if not math.isfinite(float(score)):
            raise TrajectoryControllerError(
                f"component_scores row {index} score must be finite"
            )
        key = (sample_id, component_id)
        if key in score_keys:
            raise TrajectoryControllerError(
                f"component_scores contains duplicate sample/component pair: {key}"
            )
        score_keys.add(key)
        score_sample_ids.add(sample_id)

    if not score_sample_ids.issubset(sample_ids):
        missing = sorted(score_sample_ids - sample_ids)
        raise TrajectoryControllerError(
            f"component_scores references samples absent from trajectory: {missing}"
        )
    invalid = sorted(score_sample_ids & no_active_component_samples)
    if invalid:
        raise TrajectoryControllerError(
            "controller-free trajectory samples cannot have component scores: "
            f"{invalid}"
        )
    required_score_samples = sample_ids - no_active_component_samples
    if normalized["coverage"] == "all_samples" and score_sample_ids != required_score_samples:
        missing = sorted(required_score_samples - score_sample_ids)
        raise TrajectoryControllerError(
            f"all_samples trajectory is missing component scores for: {missing}"
        )

    return {
        "config": normalized,
        "trajectory": trajectory_artifact,
        "component_scores": score_artifact,
        "trajectory_rows": len(trajectory_rows),
        "component_score_rows": len(score_rows),
        "trace_kind": trace_kind,
    }
