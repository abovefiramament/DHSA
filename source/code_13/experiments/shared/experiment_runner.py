"""Thin shared execution entrypoints for compiled experiment cells."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .flow_executor import FlowAwaitingAudit, execute_flow
from .runtime_environment import validate_and_record_runtime_environment


class ExperimentRunnerError(RuntimeError):
    """Raised when an experiment adapter cannot produce one executable cell."""


@dataclass(frozen=True)
class ExperimentOperations:
    """Experiment-specific functions consumed by the method-neutral runner."""

    iter_cells: Callable[[Path], Iterable[Mapping[str, Any]]]
    build_bundle: Callable[..., Any]
    compile_cell: Callable[..., Mapping[str, Any]]


def _cell_root(compiled: Mapping[str, Any]) -> Path:
    flow = compiled.get("flow")
    if not isinstance(flow, Mapping):
        raise ExperimentRunnerError("compiled cell lacks a flow manifest")
    value = flow.get("runtime_cell_root")
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ExperimentRunnerError("compiled cell lacks an absolute runtime_cell_root")
    return Path(value)


def select_cells(
    cells: Iterable[Mapping[str, Any]],
    *,
    filters: Mapping[str, str] | None = None,
    max_jobs: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select declared matrix rows without adding scientific conditions."""

    selected = [dict(cell) for cell in cells]
    for field, expected in (filters or {}).items():
        selected = [cell for cell in selected if str(cell.get(field)) == expected]
    if max_jobs is not None:
        if max_jobs < 0:
            raise ExperimentRunnerError("max_jobs must be non-negative")
        selected = selected[:max_jobs]
    return tuple(selected)


def run_cell(
    operations: ExperimentOperations,
    *,
    cell_id: str,
    workspace: Path,
    matrix_path: Path,
    runtime_registry: Path,
    execution_profile: str,
    bind_runtime: Callable[[Mapping[str, Any]], None] | None = None,
    freeze: bool = True,
) -> dict[str, Any]:
    """Build, compile, bind paths, and execute one formal matrix cell."""

    if not cell_id:
        raise ExperimentRunnerError("cell_id must be non-empty")
    bundle_dir = workspace.resolve() / "bundles" / cell_id
    bundle_path = bundle_dir / "bundle.json"
    if bundle_path.is_file():
        try:
            existing = json.loads(bundle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentRunnerError(
                f"existing bundle is unreadable: {bundle_path}"
            ) from exc
        profile = existing.get("shared", {}).get("execution", {}).get("profile")
        bundle_id = existing.get("bundle_id")
        if profile != execution_profile or not (
            isinstance(bundle_id, str) and bundle_id.endswith(f"__{cell_id}")
        ):
            raise ExperimentRunnerError(
                "existing bundle identity differs from the requested cell/profile; "
                "use a new workspace"
            )
    else:
        operations.build_bundle(
            cell_id,
            output_dir=bundle_dir,
            matrix_path=matrix_path,
            execution_profile=execution_profile,
        )
    compiled = operations.compile_cell(
        bundle_dir,
        cell_id=cell_id,
        runtime_registry=runtime_registry,
        matrix_path=matrix_path,
    )
    status = compiled.get("component_status", {}).get("status")
    if status != "ready":
        return {
            "cell_id": cell_id,
            "status": "pending_external_registration",
            "compiled": dict(compiled),
        }
    locked = compiled.get("locked_job", {})
    if isinstance(locked, Mapping) and locked.get("status") == "final_frozen":
        return {
            "cell_id": cell_id,
            "status": "final_frozen",
            "frozen_result": {
                "execution_policy": locked.get("execution_policy"),
                "source_registration_id": locked.get("source_registration_id"),
                "source_protocol_id": locked.get("source_protocol_id"),
                "reuse_through": locked.get("reuse_through"),
                "conditions": locked.get("conditions"),
            },
            "message": (
                "cell result is final_frozen; the registered frozen result is "
                "reused and no stages are executed"
            ),
            "compiled": dict(compiled),
        }
    validate_and_record_runtime_environment(
        cell_id=cell_id,
        runtime_registry=runtime_registry,
        cell_root=_cell_root(compiled),
    )
    if bind_runtime is not None:
        bind_runtime(compiled)
    try:
        result = execute_flow(_cell_root(compiled), freeze=freeze)
    except FlowAwaitingAudit as exc:
        return {
            "cell_id": cell_id,
            "status": "awaiting_audit",
            "message": str(exc),
            "compiled": dict(compiled),
        }
    return {
        "cell_id": cell_id,
        "status": str(result.get("status", "unknown")),
        "run": result,
        "compiled": dict(compiled),
    }


def run_matrix(
    operations: ExperimentOperations,
    *,
    workspace: Path,
    matrix_path: Path,
    runtime_registry: Path,
    execution_profile: str,
    filters: Mapping[str, str] | None = None,
    max_jobs: int | None = None,
    bind_runtime: Callable[[Mapping[str, Any]], None] | None = None,
    freeze: bool = True,
) -> list[dict[str, Any]]:
    """Run declared cells serially through the same single-cell entrypoint."""

    rows = select_cells(
        operations.iter_cells(matrix_path),
        filters=filters,
        max_jobs=max_jobs,
    )
    return [
        run_cell(
            operations,
            cell_id=str(row["job_id"]),
            workspace=workspace,
            matrix_path=matrix_path,
            runtime_registry=runtime_registry,
            execution_profile=execution_profile,
            bind_runtime=bind_runtime,
            freeze=freeze,
        )
        for row in rows
    ]
