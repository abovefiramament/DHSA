"""Compose formal Performance cells into one validation-selected timing group.

No dataset, training, generation, scoring or position implementation lives here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.performance.run_performance import compile_formal_matrix
from experiments.site.run_site_reproduction import prepare_site_runtime
from experiments.shared.flow_executor import execute_flow
from experiments.shared.runtime_registry import load_runtime_registry
from experiments.shared.external_evidence import onboard_declared_evidence, stage_conditions, load_reuse_manifest
from experiments.shared.candidate_curve_selection import select_candidate_curves
from baseline.implementations.cast_training_evidence import validate_cast_training_evidence


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"existing registration differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def register_prefix(source, cell, through, workspace):
    source = Path(source)
    run, flow = read(source / "run_manifest.json"), read(source / "config/flow_manifest.json")
    order = ["position_search", "position_resolve", "position_freeze"]
    if through in {"bank", "validation"}:
        order.append("bank_train")
        training = read(source / "config/resolved_config.json")["baseline"]["parameters"]["training"]
        execution = read(source / "bank/candidate/bank_execution_manifest.json")
        for item in execution["artifact_closure"]:
            if item.get("field") == "training_manifest":
                validate_cast_training_evidence(read(source / item["relative_path"]), training)
    if through == "validation":
        order.extend(("timing_validation_generate", "timing_validation_evaluate"))
    stages = {}
    for name in order:
        state = run["stages"][name]
        if state["status"] != "completed":
            raise ValueError(f"reuse source stage incomplete: {source}/{name}")
        stage = next(item for item in flow["stages"] if item["stage_id"] == name)
        stages[name] = dict(conditions=stage_conditions(stage),
            outputs={key: value["relative_path"] for key, value in state["outputs"].items()},
            condition_evidence=["config/flow_manifest.json", "config/resolved_config.json"])
    evidence_id = cell + "__" + through
    package = workspace / "evidence" / evidence_id
    declaration = dict(evidence_id=evidence_id, eligibility="exact_stage", target_cell_id=cell,
        source_cell_id=run["identity"]["cell_id"], source_identity=run["identity"],
        stage_order=order, stages=stages,
        source=dict(repository_url="PROJECT_SOURCE",
                    revision=run["identity"]["code_commit"], repository_path="experiments/performance",
                    contributor="RPEC collaborators (original attribution retained in source evidence)",
                    publication_status="run_local_evidence_not_uploaded"),
        review=dict(reviewer="registered_timing_group", decision="exact_real_value_match", reviewed_stages=order))
    declaration_path = package.parent / (evidence_id + ".declaration.json")
    save(declaration_path, declaration)
    manifest = package / "evidence_import_manifest.json"
    if not manifest.exists():
        onboard_declared_evidence(declaration_path=declaration_path, source_root=source, output_dir=package)
    load_reuse_manifest(manifest)
    return evidence_id, manifest


def run_group(config_path, compile_only=False):
    job = read(config_path)
    workspace = Path(job["workspace"])
    protocol = read(job["protocol"])
    settings = protocol["cast_timing_selection"]
    order = settings["baseline_order"]
    runtime = read(job["runtime_registry"])
    output = Path(runtime["entries"]["outputs/experiments"]["path"])
    if compile_only:
        workspace = workspace / "compile_preview"
        output = output / "compile_preview"
        runtime["entries"]["outputs/experiments"]["path"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / "timing_selection.json"
    runtime["entries"]["outputs/timing_selection"] = dict(kind="file", must_exist=False, path=str(decision_path))
    registered_roots = {}
    training_roots = {}
    records = []
    for index, baseline in enumerate(order):
        cell = f"performance__{job['dataset']}__{job['model_family']}__{baseline}"
        candidate_workspace = workspace / baseline
        candidate_protocol = copy.deepcopy(protocol)
        candidate_protocol["evidence_reuse"] = {}
        candidate_runtime = copy.deepcopy(runtime)
        training = "prompt_last" if "prompt_last_to_" in baseline else "all"
        source = training_roots.get(training)
        through = "bank"
        reused_validation = job.get("existing_validation_cells", {}).get(baseline)
        if reused_validation:
            source, through = reused_validation, "validation"
        elif source is None:
            source = job.get("existing_training_cells", {}).get(training)
        if source is None and index == 0:
            source = job.get("existing_all_cell")
            if source is None:
                source, through = job.get("existing_position_cell"), "position"
        elif source is None:
            source = registered_roots[order[0]]
            through = "position"
        # Compilation preview never claims that future source artifacts exist.
        if source and not compile_only:
            evidence_id, manifest = register_prefix(source, cell, through, workspace)
            candidate_runtime["entries"]["evidence/" + evidence_id] = dict(kind="file", must_exist=True, path=str(manifest))
            candidate_protocol.setdefault("evidence_reuse", {})[cell] = dict(
                evidence_id=evidence_id, manifest="registry://evidence/" + evidence_id, reuse_through=through)
        rp = candidate_workspace / "runtime.local.json"
        pp = Path(job["protocol"]).parent / "runtime_local" / (
            f"{settings['status']}__{job['dataset']}__{job['model_family']}__{'preview' if compile_only else 'formal'}__{baseline}.local.json")
        save(rp, candidate_runtime)
        save(pp, candidate_protocol)
        root = output / "performance/formal" / cell
        if not (root / "config/runtime_flow.local.json").exists():
            compiled = compile_formal_matrix(workspace=candidate_workspace, protocol_path=pp,
                execution_profile="formal", runtime_registry=rp, device="cuda:0", cell_id=cell)[0]
            root = Path(compiled["runtime_cell_root"])
        registered_roots[baseline] = root
        records.append(dict(candidate=baseline, root=str(root), runtime_registry=str(rp)))
        if not compile_only:
            load_runtime_registry(rp)
            bind = prepare_site_runtime(device="cuda:0")
            bind({"flow": {"runtime_cell_root": str(root)}})
            execute_flow(root, stop_after="timing_validation_evaluate", freeze=False)
            training_roots.setdefault(training, root)
    if compile_only:
        return records
    if not decision_path.exists():
        curves = {name: read(root / "timing_validation/evaluation/summary_metrics.json")["rows"]
                  for name, root in registered_roots.items()}
        decision = select_candidate_curves(curves, candidate_order=order,
            alpha_grid=settings["alpha_grid"], metric=settings["metrics"][job["dataset"]])
        decision.update(dataset=job["dataset"], model_family=job["model_family"],
                        data_role="validation", candidates=records,
                        final_alpha_policy="independent_alpha_dev" if job["dataset"] == "confiqa" else "full_test_curve")
        save(decision_path, decision)
    winner = read(decision_path)["selected_candidate"]
    record = next(record for record in records if record["candidate"] == winner)
    load_runtime_registry(Path(record["runtime_registry"]))
    prepare_site_runtime(device="cuda:0")({"flow": {"runtime_cell_root": record["root"]}})
    execute_flow(Path(record["root"]), freeze=True)
    return read(decision_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_group(args.config, args.compile_only), indent=2))


if __name__ == "__main__":
    main()
