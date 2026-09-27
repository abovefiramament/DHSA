"""Single formal entrypoint for the non-Tulu Performance matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.performance.performance_bundle_generator import (
    build_performance_job_bundle,
)
from experiments.performance.performance_cell_controller import (
    DEFAULT_PROTOCOL,
    compile_performance_cell,
    iter_performance_cells,
)
from experiments.shared.experiment_runner import (
    ExperimentOperations,
    run_cell,
    run_matrix,
    select_cells,
)
from experiments.shared.flow_executor import execute_flow
from experiments.shared.runtime_registry import load_runtime_registry
from experiments.site.run_site_reproduction import prepare_site_runtime


PERFORMANCE_OPERATIONS = ExperimentOperations(
    iter_cells=iter_performance_cells,
    build_bundle=build_performance_job_bundle,
    compile_cell=compile_performance_cell,
)


def compile_formal_matrix(
    *,
    workspace: Path,
    protocol_path: Path,
    execution_profile: str,
    runtime_registry: Path | None = None,
    device: str = "cuda",
    filters: Mapping[str, str] | None = None,
    max_jobs: int | None = None,
    cell_id: str | None = None,
) -> list[dict[str, Any]]:
    if runtime_registry is not None:
        load_runtime_registry(runtime_registry)
        prepare_site_runtime(device=device)
    compiled: list[dict[str, Any]] = []
    rows = select_cells(
        iter_performance_cells(protocol_path), filters=filters, max_jobs=max_jobs
    )
    if cell_id is not None:
        rows = tuple(row for row in rows if row["job_id"] == cell_id)
        if not rows:
            raise ValueError(f"unknown or filtered Performance cell: {cell_id}")
    for row in rows:
        bundle_dir = workspace / "bundles" / row["job_id"]
        build_performance_job_bundle(
            row["job_id"],
            output_dir=bundle_dir,
            matrix_path=protocol_path,
            execution_profile=execution_profile,
        )
        result = compile_performance_cell(
            bundle_dir,
            cell_id=row["job_id"],
            runtime_registry=runtime_registry,
            matrix_path=protocol_path,
        )
        compiled.append(
            {
                "job_id": row["job_id"],
                "dataset": row["dataset"],
                "model_family": row["model_family"],
                "baseline_id": row["baseline_id"],
                "component_status": result["component_status"]["status"],
                "compile_status": result["flow"]["status"],
                "runtime_cell_root": result["flow"]["runtime_cell_root"],
            }
        )
    return compiled


def run_performance_cell(
    *,
    cell_id: str,
    workspace: Path,
    runtime_registry: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    execution_profile: str = "formal",
    device: str = "cuda",
    freeze: bool = True,
) -> dict[str, Any]:
    load_runtime_registry(runtime_registry)
    return run_cell(
        PERFORMANCE_OPERATIONS,
        cell_id=cell_id,
        workspace=workspace,
        matrix_path=protocol_path,
        runtime_registry=runtime_registry,
        execution_profile=execution_profile,
        bind_runtime=prepare_site_runtime(device=device),
        freeze=freeze,
    )


def run_performance_matrix(
    *,
    workspace: Path,
    runtime_registry: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
    execution_profile: str = "formal",
    device: str = "cuda",
    filters: Mapping[str, str] | None = None,
    max_jobs: int | None = None,
    freeze: bool = True,
) -> list[dict[str, Any]]:
    load_runtime_registry(runtime_registry)
    return run_matrix(
        PERFORMANCE_OPERATIONS,
        workspace=workspace,
        matrix_path=protocol_path,
        runtime_registry=runtime_registry,
        execution_profile=execution_profile,
        filters=filters,
        max_jobs=max_jobs,
        bind_runtime=prepare_site_runtime(device=device),
        freeze=freeze,
    )


def resume_performance_cell(
    *,
    cell_root: Path,
    runtime_registry: Path,
    device: str = "cuda",
    freeze: bool = True,
    execution_lane: str = "all",
) -> dict[str, Any]:
    """Resume one compiled Performance cell through the shared executor."""

    load_runtime_registry(runtime_registry)
    root = cell_root.expanduser().resolve()
    bind_runtime = prepare_site_runtime(device=device)
    bind_runtime({"flow": {"runtime_cell_root": str(root)}})
    return execute_flow(root, freeze=freeze, execution_lane=execution_lane)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the formal non-Tulu Performance matrix")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path,
                        help="Required for execution; omit with --compile-only for portable plans")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cell-id")
    parser.add_argument("--resume-cell-root", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--model-family")
    parser.add_argument("--baseline-id")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", choices=("formal", "formal_scaled_gpu"), default="formal")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--execution-lane", choices=("all", "gpu", "cpu"), default="all")
    args = parser.parse_args()
    if args.execution_lane != "all" and not args.resume_cell_root:
        parser.error("execution lanes require an already compiled --resume-cell-root")
    if not args.compile_only and args.runtime_registry is None:
        parser.error("execution requires --runtime-registry; portable compilation does not")
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if args.resume_cell_root:
        if args.compile_only or args.cell_id:
            parser.error("--resume-cell-root cannot be combined with --compile-only or --cell-id")
        result: Any = resume_performance_cell(
            cell_root=args.resume_cell_root,
            runtime_registry=args.runtime_registry,
            device=args.device,
            freeze=not args.no_freeze,
            execution_lane=args.execution_lane,
        )
    elif args.compile_only:
        filters = {
            key: value for key, value in {
                "dataset": args.dataset,
                "model_family": args.model_family,
                "baseline_id": args.baseline_id,
            }.items() if value is not None
        }
        result = compile_formal_matrix(
            workspace=workspace,
            protocol_path=args.protocol,
            execution_profile=args.profile,
            runtime_registry=args.runtime_registry,
            device=args.device,
            filters=filters,
            max_jobs=args.max_jobs,
            cell_id=args.cell_id,
        )
    elif args.cell_id:
        result = run_performance_cell(
            cell_id=args.cell_id,
            workspace=workspace,
            runtime_registry=args.runtime_registry,
            protocol_path=args.protocol,
            execution_profile=args.profile,
            device=args.device,
            freeze=not args.no_freeze,
        )
    else:
        filters = {
            key: value for key, value in {
                "dataset": args.dataset,
                "model_family": args.model_family,
                "baseline_id": args.baseline_id,
            }.items() if value is not None
        }
        result = run_performance_matrix(
            workspace=workspace,
            runtime_registry=args.runtime_registry,
            protocol_path=args.protocol,
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
