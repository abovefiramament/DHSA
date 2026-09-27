"""Formal Site entrypoint built on the shared experiment runner.

Site declares its matrix and fixed scientific flow only. Model loading, data
construction, position methods, CAST, generation, and evaluation remain in
their own registered component modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from baseline.implementations.model_runtime import (
    HuggingFaceModelProvider,
    register_model_paths_from_runtime_config,
)
from baseline.implementations.runtime_bindings import (
    register_bipo_runtime_components,
    register_cast_runtime_components,
    register_loreft_runtime_components,
)
from data.adapter_bindings import configure_data_runtime
from evaluators.site import register_site_evaluators
from experiments.shared.component_registry import COMPONENT_REGISTRY
from experiments.shared.experiment_runner import (
    ExperimentOperations,
    run_cell,
    run_matrix,
    select_cells,
)
from experiments.shared.flow_executor import execute_flow
from experiments.shared.runtime_registry import load_runtime_registry
from experiments.site.site_bundle_generator import build_site_job_bundle
from experiments.site.site_cell_controller import (
    DEFAULT_MATRIX,
    compile_site_cell,
    iter_site_cells,
)


class SiteReproductionError(RuntimeError):
    """Raised when the Site entrypoint cannot build a declared matrix cell."""


SITE_OPERATIONS = ExperimentOperations(
    iter_cells=iter_site_cells,
    build_bundle=build_site_job_bundle,
    compile_cell=compile_site_cell,
)


def _runtime_config(compiled: Mapping[str, Any]) -> dict[str, Any]:
    flow = compiled.get("flow")
    root = flow.get("runtime_cell_root") if isinstance(flow, Mapping) else None
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise SiteReproductionError("compiled Site cell lacks a runtime root")
    path = Path(root) / "config" / "runtime_config.local.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SiteReproductionError("compiled Site runtime config must be an object")
    return value


def prepare_site_runtime(
    *,
    device: str,
) -> Callable[[Mapping[str, Any]], None]:
    """Register reusable methods/evaluators, then bind compiled model paths.

    No absolute path is supplied here. The returned callback receives a compiled
    cell, whose portable model reference has already been resolved through the
    machine registry.
    """

    provider = HuggingFaceModelProvider(device=device)
    configure_data_runtime(model_provider=provider)
    register_cast_runtime_components(provider, registry=COMPONENT_REGISTRY)
    register_loreft_runtime_components(provider, registry=COMPONENT_REGISTRY)
    register_bipo_runtime_components(provider, registry=COMPONENT_REGISTRY)
    register_site_evaluators(device=device, registry=COMPONENT_REGISTRY)

    def bind_compiled_cell(compiled: Mapping[str, Any]) -> None:
        register_model_paths_from_runtime_config(provider, _runtime_config(compiled))

    return bind_compiled_cell


def compile_formal_matrix(
    *,
    workspace: Path,
    matrix_path: Path,
    execution_profile: str,
    runtime_registry: Path,
    device: str = "cuda",
    filters: Mapping[str, str] | None = None,
    max_jobs: int | None = None,
) -> list[dict[str, Any]]:
    """Compile every declared Site row without executing a stage."""

    load_runtime_registry(runtime_registry)
    prepare_site_runtime(device=device)
    rows = select_cells(
        iter_site_cells(matrix_path), filters=filters, max_jobs=max_jobs
    )
    compiled: list[dict[str, Any]] = []
    for row in rows:
        bundle_dir = workspace / "bundles" / str(row["job_id"])
        build_site_job_bundle(
            str(row["job_id"]),
            output_dir=bundle_dir,
            matrix_path=matrix_path,
            execution_profile=execution_profile,
        )
        result = compile_site_cell(
            bundle_dir,
            cell_id=str(row["job_id"]),
            runtime_registry=runtime_registry,
            matrix_path=matrix_path,
        )
        compiled.append(
            {
                "job_id": row["job_id"],
                "dataset": row["dataset"],
                "model_family": row["model_family"],
                "position_method": row["position_method"],
                "component_status": result["component_status"]["status"],
                "runtime_cell_root": result["flow"]["runtime_cell_root"],
            }
        )
    return compiled


def run_site_cell(
    *,
    cell_id: str,
    workspace: Path,
    matrix_path: Path = DEFAULT_MATRIX,
    runtime_registry: Path,
    execution_profile: str = "formal",
    device: str = "cuda",
    freeze: bool = True,
) -> dict[str, Any]:
    """Run one declared Site condition through the shared cell entrypoint."""

    load_runtime_registry(runtime_registry)
    return run_cell(
        SITE_OPERATIONS,
        cell_id=cell_id,
        workspace=workspace,
        matrix_path=matrix_path,
        runtime_registry=runtime_registry,
        execution_profile=execution_profile,
        bind_runtime=prepare_site_runtime(device=device),
        freeze=freeze,
    )


def run_site_matrix(
    *,
    workspace: Path,
    matrix_path: Path = DEFAULT_MATRIX,
    runtime_registry: Path,
    execution_profile: str = "formal",
    device: str = "cuda",
    filters: Mapping[str, str] | None = None,
    max_jobs: int | None = None,
    freeze: bool = True,
) -> list[dict[str, Any]]:
    """Run selected declared Site conditions through the shared matrix entrypoint."""

    load_runtime_registry(runtime_registry)
    return run_matrix(
        SITE_OPERATIONS,
        workspace=workspace,
        matrix_path=matrix_path,
        runtime_registry=runtime_registry,
        execution_profile=execution_profile,
        filters=filters,
        max_jobs=max_jobs,
        bind_runtime=prepare_site_runtime(device=device),
        freeze=freeze,
    )



def resume_site_cell(
    *,
    cell_root: Path,
    runtime_registry: Path,
    device: str = "cuda",
    freeze: bool = True,
) -> dict[str, Any]:
    """Resume one paused or failed Site cell through the same runtime bindings."""

    load_runtime_registry(runtime_registry)
    root = cell_root.expanduser().resolve()
    bind_runtime = prepare_site_runtime(device=device)
    bind_runtime({"flow": {"runtime_cell_root": str(root)}})
    return execute_flow(root, freeze=freeze)

def main() -> int:
    parser = argparse.ArgumentParser(description="Run one formal Site matrix cell or a filtered Site matrix.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--cell-id")
    parser.add_argument("--resume-cell-root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--model-family")
    parser.add_argument("--position-method")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profile",
        choices=("formal", "formal_scaled_gpu"),
        default="formal",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if args.resume_cell_root:
        if args.compile_only or args.cell_id:
            parser.error("--resume-cell-root cannot be combined with --compile-only or --cell-id")
        result: Any = resume_site_cell(
            cell_root=args.resume_cell_root,
            runtime_registry=args.runtime_registry,
            device=args.device,
            freeze=not args.no_freeze,
        )
    elif args.compile_only:
        filters = {
            key: value
            for key, value in {
                "job_id": args.cell_id,
                "dataset": args.dataset,
                "model_family": args.model_family,
                "position_method": args.position_method,
            }.items()
            if value is not None
        }
        result = compile_formal_matrix(
            workspace=workspace,
            matrix_path=args.matrix,
            execution_profile=args.profile,
            runtime_registry=args.runtime_registry,
            device=args.device,
            filters=filters,
            max_jobs=args.max_jobs,
        )
    elif args.cell_id:
        result = run_site_cell(
            cell_id=args.cell_id,
            workspace=workspace,
            matrix_path=args.matrix,
            runtime_registry=args.runtime_registry,
            execution_profile=args.profile,
            device=args.device,
            freeze=not args.no_freeze,
        )
    else:
        filters = {
            key: value
            for key, value in {
                "dataset": args.dataset,
                "model_family": args.model_family,
                "position_method": args.position_method,
            }.items()
            if value is not None
        }
        result = run_site_matrix(
            workspace=workspace,
            matrix_path=args.matrix,
            runtime_registry=args.runtime_registry,
            execution_profile=args.profile,
            device=args.device,
            filters=filters,
            max_jobs=args.max_jobs,
            freeze=not args.no_freeze,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
