from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.site.protocol import SiteProtocol, require_files
from screscomp.site.runtime import configure_gpu, resolve_pinned_model
from screscomp.site.sentiment import FrozenSentimentScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked held-out Site generation and evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["dev", "test"], required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def _alpha_name(alpha: float) -> str:
    return "alpha_" + (f"{alpha:.1f}".replace(".", "p"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_contract(protocol, job, input_rows: Path, *, kind: str, stage: str) -> dict:
    return {
        "kind": kind,
        "stage": stage,
        "dataset": job.dataset,
        "subset": job.subset,
        "model_key": job.model_key,
        "model": job.model,
        "model_revision": job.model_revision,
        "use_chat_template": job.use_chat_template,
        "input_sha256": _sha256(input_rows),
        "task_score": job.task["score"],
        "downstream": job.task["downstream"],
    }


def _artifact_hashes(source_dir: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    missing = [str(source_dir / rel) for rel in relative_paths if not (source_dir / rel).is_file()]
    if missing:
        raise RuntimeError(f"registered alpha-zero/native base is incomplete: {missing}")
    return {rel: _sha256(source_dir / rel) for rel in relative_paths}


def _confiqa_base_generation_contract(task: dict, stage: str) -> dict:
    downstream = task["downstream"]
    eval_spec = downstream["alpha_dev"] if stage == "dev" else downstream["heldout_test"]
    return {
        "score": task["score"],
        "val_mod": downstream["val_mod"],
        "generation": downstream["generation"],
        "eval_cap": eval_spec["eval_cap"],
        "generation_prompt_key": "official_rag",
        "generation_apply_mode": "all",
    }


def _historical_base_candidate(
    protocol,
    job,
    paths,
    input_rows: Path,
    *,
    kind: str,
    stage: str,
    relative_paths: tuple[str, ...],
) -> tuple[Path, dict] | None:
    policy = protocol.site["execution"]["historical_alpha_zero_reuse"]
    if not policy["allowed"]:
        return None
    repo_root = protocol.config_path.parent.parent
    old_config = repo_root / policy["source_config"]
    if not old_config.is_file() or _sha256(old_config) != policy["source_config_sha256"]:
        raise RuntimeError("registered historical Site config is missing or has drifted")
    old = json.loads(old_config.read_text(encoding="utf-8"))
    if old.get("protocol_id") != policy["source_protocol_id"]:
        raise RuntimeError("historical Site protocol id drift")
    old_task = old["site"]["tasks"][job.dataset]
    if kind == "confiqa":
        if _confiqa_base_generation_contract(
            old_task, stage
        ) != _confiqa_base_generation_contract(job.task, stage):
            raise RuntimeError("historical ConFiQA alpha-zero generation contract is not identical")
    else:
        for key in ("states", "score", "downstream"):
            if old_task[key] != job.task[key]:
                raise RuntimeError(f"historical alpha-zero {key} contract is not identical")
    matches = [
        row for row in old["site"]["matrix"]
        if row["dataset"] == job.dataset and row["model_key"] == job.model_key
    ]
    if len(matches) != 1:
        raise RuntimeError("historical alpha-zero model row is missing or duplicated")
    old_model = matches[0]
    for key, expected in {
        "model": job.model,
        "revision": job.model_revision,
        "use_chat_template": job.use_chat_template,
    }.items():
        if old_model[key] != expected:
            raise RuntimeError(f"historical alpha-zero model contract drift: {key}")
    old_root = Path(policy["source_run_root"])
    if kind == "confiqa":
        old_input = old_root / "confiqa" / job.subset / input_rows.name
        source_dir = old_root / "shared_base" / "confiqa" / job.subset / job.model_key / stage
    else:
        old_input = old_root / "imdb" / job.model_key / input_rows.name
        source_dir = old_root / "shared_base" / "imdb" / job.model_key
    if not old_input.is_file() or _sha256(old_input) != _sha256(input_rows):
        return None
    try:
        hashes = _artifact_hashes(source_dir, relative_paths)
    except RuntimeError:
        return None
    return source_dir, {
        "source_kind": "historical_exact_contract_reuse",
        "source_protocol_id": policy["source_protocol_id"],
        "source_config": str(old_config),
        "source_config_sha256": policy["source_config_sha256"],
        "artifact_sha256": hashes,
    }


def _resolve_base(
    protocol,
    job,
    paths,
    input_rows: Path,
    shared: Path,
    *,
    kind: str,
    stage: str,
    relative_paths: tuple[str, ...],
    generate,
) -> tuple[Path, dict, Path]:
    contract = _base_contract(protocol, job, input_rows, kind=kind, stage=stage)
    reference = shared / "base_reference_manifest.json"
    if reference.is_file():
        payload = json.loads(reference.read_text(encoding="utf-8"))
        if payload.get("base_contract") != contract:
            raise RuntimeError(f"alpha-zero/native base contract drift in {reference}")
        source_dir = Path(payload["source_dir"])
        hashes = _artifact_hashes(source_dir, relative_paths)
        if payload.get("artifact_sha256") != hashes:
            raise RuntimeError(f"alpha-zero/native base artifact hash drift in {reference}")
        return source_dir, payload, reference
    partial = [str(shared / rel) for rel in relative_paths if (shared / rel).is_file()]
    if partial:
        raise RuntimeError(f"unregistered partial alpha-zero/native base: {partial}")

    historical = _historical_base_candidate(
        protocol,
        job,
        paths,
        input_rows,
        kind=kind,
        stage=stage,
        relative_paths=relative_paths,
    )
    if historical is not None:
        source_dir, provenance = historical
    else:
        shared.mkdir(parents=True, exist_ok=True)
        generate(shared)
        source_dir = shared
        provenance = {
            "source_kind": "generated_once_under_current_protocol",
            "source_protocol_id": protocol.data["protocol_id"],
            "source_config": str(protocol.config_path),
            "source_config_sha256": protocol.config_sha256,
            "artifact_sha256": _artifact_hashes(source_dir, relative_paths),
        }
    payload = {
        "config_sha256": protocol.config_sha256,
        "base_contract": contract,
        "source_dir": str(source_dir),
        **provenance,
        "status": "complete",
    }
    shared.mkdir(parents=True, exist_ok=True)
    dump_json(reference, payload)
    return source_dir, payload, reference


def _reuse_record(source_dir: Path, payload: dict, reference: Path) -> dict:
    return {
        "alpha_zero_reuse_source": str(source_dir),
        "alpha_zero_reuse_manifest": str(reference),
        "alpha_zero_source_kind": payload["source_kind"],
        "alpha_zero_source_config_sha256": payload["source_config_sha256"],
        "alpha_zero_artifact_sha256": payload["artifact_sha256"],
    }


def _run_confiqa_generation(job, downstream, pinned_model: str, input_rows: Path, out_dir: Path, controls: str, actuator: Path | None, max_rows: int) -> None:
    command = [
        sys.executable,
        "-m", "screscomp.cli.cecm_run_joint_actuator_generation",
        "--model", pinned_model,
        "--eval-open-rows", str(input_rows),
        "--controls", controls,
        "--generation-prompt-key", "official_rag",
        "--prior-source", "dataset_orig",
        "--scoring-kind", "source",
        "--split", "",
        "--start", "0",
        "--max-rows", str(max_rows),
        "--val-mod", str(downstream["val_mod"]),
        "--generation-apply-mode", "all",
        "--max-new-tokens", str(downstream["generation"]["max_new_tokens"]),
        "--stop-strings", ",".join(downstream["generation"]["stop_strings"]),
        "--empty-cache-every", "25",
        "--device", "cuda:0",
        "--torch-dtype", "auto",
        "--out-dir", str(out_dir),
    ]
    if actuator is not None:
        command.extend(["--head-actuators", f"site={actuator}"])
    if downstream["generation"]["do_sample"]:
        command.append("--do-sample")
    if job.use_chat_template:
        command.append("--use-chat-template")
    subprocess.run(command, check=True)


def _confiqa_summary(path: Path, alpha_by_control: dict[str, float]) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        control = row["control_name"]
        alpha = alpha_by_control[control]
        logic = (
            float(row["context_only_rate"])
            + float(row["short_exact_context_rate"])
            - float(row["prior_hit_rate"])
            - float(row["neither_rate"])
            - float(row["mean_chars"]) / 400.0
        )
        out.append({**row, "alpha": alpha, "logic_score": logic})
    return out


def _evaluate_confiqa(protocol, job, paths, pinned_model: str, stage: str) -> dict[str, object]:
    downstream = job.task["downstream"]
    shared = paths.root / "shared_base" / "confiqa" / job.subset / job.model_key / stage
    input_rows = paths.alpha_dev_rows if stage == "dev" else paths.heldout_test_rows
    require_files([input_rows, paths.actuator_dir / "head_actuator.pt"])
    max_rows = int(
        downstream["alpha_dev"]["eval_cap"] if stage == "dev"
        else downstream["heldout_test"]["eval_cap"]
    )

    def generate_base(out_dir: Path) -> None:
        _run_confiqa_generation(
            job, downstream, pinned_model, input_rows, out_dir, "base=", None, max_rows
        )

    base_dir, base_reference, reference_path = _resolve_base(
        protocol,
        job,
        paths,
        input_rows,
        shared,
        kind="confiqa",
        stage=stage,
        relative_paths=("generation_summary.csv",),
        generate=generate_base,
    )
    base_rows = _confiqa_summary(base_dir / "generation_summary.csv", {"base": 0.0})
    out_dir = paths.evaluation_dir / stage
    actuator = paths.actuator_dir / "head_actuator.pt"
    if stage == "dev":
        alphas = [float(value) for value in downstream["alpha_nonzero_grid"]]
    else:
        selection = paths.evaluation_dir / "alpha_selection.json"
        require_files([selection])
        alphas = [float(json.loads(selection.read_text(encoding="utf-8"))["selected_alpha"])]
    controls = ";".join(
        f"{_alpha_name(alpha)}=head_act:site:{alpha}:all" for alpha in alphas
    )
    _run_confiqa_generation(job, downstream, pinned_model, input_rows, out_dir, controls, actuator, max_rows)
    alpha_map = {_alpha_name(alpha): alpha for alpha in alphas}
    site_rows = _confiqa_summary(out_dir / "generation_summary.csv", alpha_map)
    dump_csv(out_dir / "site_summary.csv", [*base_rows, *site_rows])
    if stage == "dev":
        selected = sorted(site_rows, key=lambda row: (-float(row["logic_score"]), float(row["alpha"])))[0]
        dump_json(
            paths.evaluation_dir / "alpha_selection.json",
            {
                "selection_metric": downstream["alpha_dev"]["selection_metric"],
                "tie_break": downstream["alpha_dev"]["tie_break"],
                "selected_alpha": float(selected["alpha"]),
                "selected_logic_score": float(selected["logic_score"]),
                "actual_n": int(selected["n"]),
            },
        )
    return _reuse_record(base_dir, base_reference, reference_path)


def _run_imdb_generation(job, downstream, pinned_model: str, prompts: Path, out: Path, alphas: list[float], actuator: Path | None) -> None:
    command = [
        sys.executable,
        "-m", "screscomp.cli.run_imdb_sentiment_actuator_generation",
        "--model", pinned_model,
        "--tokenizer", pinned_model,
        "--prompts-jsonl", str(prompts),
        "--out-jsonl", str(out),
        "--control-name", "site" if actuator is not None else "base",
        "--alpha-sweep", ",".join(str(value) for value in alphas),
        "--generation-apply-mode", "all",
        "--head-apply-mode", "all",
        "--split", "eval",
        "--start", "0",
        "--max-rows", str(downstream["test_prompts"]),
        "--samples-per-prompt", str(downstream["samples_per_prompt"]),
        "--generation-batch-size", str(downstream["generation_batch_size"]),
        "--max-new-tokens", str(downstream["max_new_tokens"]),
        "--temperature", str(downstream["temperature"]),
        "--top-p", str(downstream["top_p"]),
        "--top-k", str(downstream["generation_top_k"]),
        "--seed", str(downstream["generation_seed"]),
        "--same-seed-across-alpha",
        "--device", "cuda:0",
        "--torch-dtype", "auto",
    ]
    if actuator is not None:
        command.extend(["--head-actuator", str(actuator)])
    if not downstream["do_sample"]:
        command.append("--no-do-sample")
    if job.use_chat_template:
        command.append("--use-chat-template")
    subprocess.run(command, check=True)


def _score_imdb(job, input_path: Path, output_dir: Path) -> None:
    rows = load_jsonl(input_path)
    scorer = FrozenSentimentScorer(job.task["score"], device="cuda:0")
    scores = scorer.score([str(row["completion"]) for row in rows])
    scored = [{**row, **score} for row, score in zip(rows, scores, strict=True)]
    dump_jsonl(output_dir / "sentiment_scored.jsonl", scored)
    grouped = defaultdict(list)
    for row in scored:
        grouped[float(row.get("alpha", 0.0))].append(row)
    summary = []
    for alpha, alpha_rows in sorted(grouped.items()):
        completions = [str(row["completion"]) for row in alpha_rows]
        positive = [float(row["positive_sentiment_score"]) for row in alpha_rows]
        summary.append(
            {
                "alpha": alpha,
                "n": len(alpha_rows),
                "mean_positive_sentiment_score": mean(positive),
                "positive_rate": mean(float(value >= 0.5) for value in positive),
                "mean_completion_chars": mean(len(text) for text in completions),
                "empty_rate": mean(float(not text.strip()) for text in completions),
                "actual_generated_tokens": sum(int(row.get("generated_tokens", 0)) for row in alpha_rows),
            }
        )
    dump_csv(output_dir / "sentiment_summary.csv", summary)
    del scorer
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _evaluate_imdb(protocol, job, paths, pinned_model: str) -> dict[str, object]:
    downstream = job.task["downstream"]
    require_files([paths.imdb_eval_prompts, paths.actuator_dir / "head_actuator.pt"])
    shared = paths.root / "shared_base" / "imdb" / job.model_key

    def generate_base(out_dir: Path) -> None:
        base_generation = out_dir / "generation_rows.jsonl"
        _run_imdb_generation(
            job,
            downstream,
            pinned_model,
            paths.imdb_eval_prompts,
            base_generation,
            [0.0],
            None,
        )
        _score_imdb(job, base_generation, out_dir)

    base_dir, base_reference, reference_path = _resolve_base(
        protocol,
        job,
        paths,
        paths.imdb_eval_prompts,
        shared,
        kind="imdb",
        stage="test",
        relative_paths=(
            "generation_rows.jsonl",
            "sentiment_scored.jsonl",
            "sentiment_summary.csv",
        ),
        generate=generate_base,
    )
    out_dir = paths.evaluation_dir / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    generation = out_dir / "generation_rows.jsonl"
    _run_imdb_generation(
        job,
        downstream,
        pinned_model,
        paths.imdb_eval_prompts,
        generation,
        [float(value) for value in downstream["alpha_nonzero_grid"]],
        paths.actuator_dir / "head_actuator.pt",
    )
    _score_imdb(job, generation, out_dir)
    subprocess.run(
        [
            sys.executable,
            "-m", "screscomp.cli.compute_imdb_generation_kl",
            "--input-jsonl", str(generation),
            "--out-jsonl", str(out_dir / "kl_rows.jsonl"),
            "--summary-csv", str(out_dir / "kl_summary.csv"),
            "--model", pinned_model,
            "--device", "cuda:0",
            "--torch-dtype", "auto",
            "--batch-size", str(downstream["kl_batch_size"]),
            "--overwrite",
        ],
        check=True,
    )
    return _reuse_record(base_dir, base_reference, reference_path)


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    job = protocol.job(args.job_id)
    paths = protocol.paths(args.artifact_root, args.job_id)
    gpu = configure_gpu(args.device, protocol.site["execution"])
    pinned_model = resolve_pinned_model(job.model, job.model_revision)
    paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    if job.dataset == "confiqa":
        base_reuse = _evaluate_confiqa(protocol, job, paths, pinned_model, args.stage)
    elif args.stage == "dev":
        raise ValueError("IMDb reports the preregistered full test alpha curve and has no dev alpha stage")
    else:
        base_reuse = _evaluate_imdb(protocol, job, paths, pinned_model)
    dump_json(
        paths.evaluation_dir / f"{args.stage}_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "stage": args.stage,
            "gpu": gpu,
            "model_path": pinned_model,
            **base_reuse,
        },
    )
    print(f"[site-eval] complete job={job.job_id} stage={args.stage}", flush=True)


if __name__ == "__main__":
    main()
