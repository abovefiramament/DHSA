from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from screscomp.data import dump_json, dump_jsonl
from screscomp.site.protocol import SiteProtocol
from screscomp.site.runtime import resolve_pinned_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shared locked Site source and prompt artifacts.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _run(module: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def _validate_confiqa_prompt_contract(path: Path, contract: dict[str, object]) -> dict[str, object]:
    if contract.get("id") != "context_dpo_official_v1":
        raise RuntimeError(f"Unsupported ConFiQA prompt contract: {contract.get('id')}")
    if contract.get("prepared_prompt_key") != "official_rag":
        raise RuntimeError("ConFiQA prepared prompt key drift")
    if contract.get("byte_exact_assertion") is not True:
        raise RuntimeError("ConFiQA byte-exact prompt assertion must be enabled")
    template = str(contract["template"])
    digest = hashlib.sha256()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = template.format(context=row["context"], question=row["question"])
            actual = row["prompts"]["official_rag"]
            if actual != expected:
                raise RuntimeError(
                    f"ConFiQA official prompt mismatch in {path} line {line_number}: "
                    f"expected_tail={expected[-80:]!r} actual_tail={actual[-80:]!r}"
                )
            if "Q:Q:" in actual or "\nA: \nA:" in actual:
                raise RuntimeError(f"ConFiQA prompt is double wrapped in {path} line {line_number}")
            digest.update(actual.encode("utf-8"))
            digest.update(b"\0")
            count += 1
    if count == 0:
        raise RuntimeError(f"ConFiQA prompt audit found no rows in {path}")
    return {
        "contract_id": contract["id"],
        "prompt_key": contract["prepared_prompt_key"],
        "template": template,
        "row_count": count,
        "prompt_bytes_sha256": digest.hexdigest(),
        "byte_exact_assertion": True,
    }


def _audit_confiqa_pair_prompt_chain(
    open_rows_path: Path,
    pairs_csv_path: Path,
    contract: dict[str, object],
) -> dict[str, object]:
    prompt_key = str(contract["prepared_prompt_key"])
    source_prompts: dict[str, str] = {}
    with open_rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_prompts[str(row["sample_id"])] = str(row["prompts"][prompt_key])
    source_digest = hashlib.sha256()
    pair_digest = hashlib.sha256()
    pair_count = 0
    with pairs_csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row["sample_id"])
            expected = source_prompts.get(sample_id)
            if expected is None:
                raise RuntimeError(f"ConFiQA pair has no prepared source row: {sample_id}")
            actual = str(row["prompt"])
            if actual != expected:
                raise RuntimeError(
                    f"ConFiQA pair prompt serialization drift for {sample_id}: "
                    f"expected_tail={expected[-20:]!r} actual_tail={actual[-20:]!r}"
                )
            source_digest.update(expected.encode("utf-8"))
            source_digest.update(b"\0")
            pair_digest.update(actual.encode("utf-8"))
            pair_digest.update(b"\0")
            pair_count += 1
    if pair_count != len(source_prompts):
        raise RuntimeError(
            f"ConFiQA prompt-chain count drift: source={len(source_prompts)} pair={pair_count}"
        )
    return {
        "pair_prompt_normalization": contract["pair_prompt_normalization"],
        "prompt_key": prompt_key,
        "row_count": pair_count,
        "source_prompt_bytes_sha256": source_digest.hexdigest(),
        "pair_prompt_bytes_sha256": pair_digest.hexdigest(),
        "byte_exact_equal": source_digest.digest() == pair_digest.digest(),
    }


def _prepare_confiqa(protocol: SiteProtocol, job, paths) -> dict[str, object]:
    prep = job.task["selector_preprocessing"]["rcm"]
    prompt_contract = job.task["prompt_contract"]
    prompt_contract_args = ["--official_prompt_contract", prompt_contract["id"]]
    target_groups = int(prep["target_admitted_groups"])
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    _run(
        "screscomp.cli.cecm_fetch_confiqa",
        ["--out-dir", str(paths.source_dir), "--tasks", job.subset],
    )
    filename = f"ConFiQA-{job.subset.upper()}.json"
    source = paths.source_dir / filename
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = job.task["source_file_sha256"][job.subset]
    if digest != expected:
        raise RuntimeError(f"ConFiQA {job.subset} hash drift: expected {expected}, got {digest}")
    _run(
        "screscomp.cli.prepare_ckplug_open",
        [
            "--dataset", "confiqa",
            "--data_json", str(source),
            "--out_jsonl", str(paths.open_rows),
            "--start", str(prep["source_start"]),
            "--max_rows", str(int(prep["source_stop"]) - int(prep["source_start"])),
            "--target_rows", str(target_groups),
            "--audit_dir", str(paths.selector_source_audit_dir),
            "--schema", prep["schema"],
            "--alias_policy", prep["alias_policy"],
            *prompt_contract_args,
        ],
    )
    selector_prompt_audit = _validate_confiqa_prompt_contract(paths.open_rows, prompt_contract)
    selector_audit = json.loads(
        paths.selector_source_audit_manifest.read_text(encoding="utf-8")
    )
    if int(selector_audit["admitted_rows"]) != target_groups:
        raise RuntimeError(
            f"ConFiQA selector admission drift: expected {target_groups}, "
            f"got {selector_audit['admitted_rows']}"
        )
    downstream = job.task["downstream"]
    dev = downstream["alpha_dev"]
    if dev["source_start_policy"] != "max_registered_start_and_selector_scan_stop_plus_one":
        raise RuntimeError("ConFiQA alpha-dev start policy drift")
    selector_scan_stop = int(selector_audit["scan_stop_exclusive"])
    dev_start = max(int(dev["source_start"]), selector_scan_stop + 1)
    dev_window_rows = int(dev["source_stop"]) - int(dev["source_start"])
    _run(
        "screscomp.cli.prepare_ckplug_open",
        [
            "--dataset", "confiqa",
            "--data_json", str(source),
            "--out_jsonl", str(paths.alpha_dev_rows),
            "--start", str(dev_start),
            "--max_rows", str(dev_window_rows),
            "--schema", prep["schema"],
            "--alias_policy", prep["alias_policy"],
            *prompt_contract_args,
        ],
    )
    dev_prompt_audit = _validate_confiqa_prompt_contract(paths.alpha_dev_rows, prompt_contract)
    test = downstream["heldout_test"]
    _run(
        "screscomp.cli.prepare_ckplug_open",
        [
            "--dataset", "confiqa",
            "--data_json", str(source),
            "--out_jsonl", str(paths.heldout_test_rows),
            "--start", str(test["source_start"]),
            "--max_rows", str(int(test["source_stop"]) - int(test["source_start"])),
            "--schema", prep["schema"],
            "--alias_policy", prep["alias_policy"],
            *prompt_contract_args,
        ],
    )
    test_prompt_audit = _validate_confiqa_prompt_contract(paths.heldout_test_rows, prompt_contract)
    _run(
        "screscomp.cli.cecm_build_preference_pairs",
        [
            "--input-jsonl", str(paths.open_rows),
            "--out-dir", str(paths.pair_dir),
            "--event", job.task["downstream"]["event"],
            "--prompt-key", "official_rag",
            "--prompt-normalization",
            str(prompt_contract["pair_prompt_normalization"]),
            "--prior-source", "dataset_orig",
            "--val-mod", str(prep["val_mod"]),
            "--max-rows", str(target_groups),
        ],
    )
    pair_manifest = json.loads(
        (paths.pair_dir / "pair_build_manifest.json").read_text(encoding="utf-8")
    )
    if int(pair_manifest["admitted_pairs"]) != target_groups:
        raise RuntimeError(
            f"ConFiQA pair admission drift: expected {target_groups}, "
            f"got {pair_manifest['admitted_pairs']}"
        )
    if pair_manifest.get("prompt_normalization") != prompt_contract["pair_prompt_normalization"]:
        raise RuntimeError("ConFiQA pair prompt-normalization config drift")
    pair_prompt_chain = _audit_confiqa_pair_prompt_chain(
        paths.open_rows,
        paths.pairs_csv,
        prompt_contract,
    )
    return {
        "selector_source_audit_manifest": str(paths.selector_source_audit_manifest),
        "selector_scan_stop_exclusive": selector_scan_stop,
        "selector_admitted_groups": target_groups,
        "selector_rejected_source_rows": int(selector_audit["rejected_rows"]),
        "selector_replacement_rows": int(selector_audit["replacement_rows"]),
        "alpha_dev_actual_source_start": dev_start,
        "alpha_dev_actual_source_stop": dev_start + dev_window_rows,
        "alpha_dev_raw_window_rows": dev_window_rows,
        "prompt_contract": prompt_contract,
        "canonical_prompt_assertion": {
            "selector": selector_prompt_audit,
            "alpha_dev": dev_prompt_audit,
            "heldout_test": test_prompt_audit,
        },
        "pair_prompt_chain": pair_prompt_chain,
    }


def _download_imdb_snapshot(job, paths, split: str) -> Path:
    from datasets import load_dataset

    snapshot = paths.source_dir / f"imdb_{split}_source.jsonl"
    if snapshot.is_file():
        return snapshot
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        job.task["dataset"],
        split=split,
        revision=job.task["dataset_revision"],
    )
    dump_jsonl(snapshot, (dict(dataset[index]) for index in range(len(dataset))))
    return snapshot

def _prepare_imdb_environment(protocol: SiteProtocol, job, paths, source: Path, test_source: Path, pinned_model: str) -> None:
    selector = protocol.selector_state_generation(job)
    if selector is None:
        raise RuntimeError("IMDb selector state-generation contract is missing")
    pair = job.task["pair_preparation"]
    selector_dir = paths.shared_dataset_dir / "selector_environment"
    pair_dir = paths.shared_dataset_dir / "pair_environment"
    common_selector = [
        "--input", str(source),
        "--source-dataset", f"{job.task['dataset']}@{job.task['dataset_revision']}",
        "--start", str(selector["source_start"]),
        "--max-source-rows", str(selector["max_source_rows_before_shuffle"]),
        "--seed", str(selector["shuffle_seed"]),
        "--prefix-token-min", str(selector["prefix_token_min"]),
        "--prefix-token-max", str(selector["prefix_token_max"]),
        "--prefix-mode", selector["prefix_mode"],
        "--tokenizer", pinned_model,
        "--prompt-template", "{prefix}",
        "--scorer-model", job.task["score"]["scorer_model"],
        "--scorer-positive-label", job.task["score"]["positive_label"],
        "--reference-model", pinned_model,
        "--completions-per-prefix", str(selector["samples_per_prompt_per_condition"]),
    ]
    if selector["shuffle"]:
        common_selector.append("--shuffle")
    if selector["normalize_imdb_breaks"]:
        common_selector.append("--normalize-imdb-breaks")
    _run(
        "screscomp.cli.prepare_imdb_sentiment_env",
        [
            *common_selector,
            "--out-dir", str(selector_dir),
            "--train-rows", str(selector["prompt_rows"]),
            "--val-rows", "0",
            "--eval-rows", "0",
        ],
    )
    shutil.copyfile(selector_dir / "prompts.jsonl", paths.selector_prompts)

    pair_args = [
        "--input", str(source),
        "--source-dataset", f"{job.task['dataset']}@{job.task['dataset_revision']}",
        "--out-dir", str(pair_dir),
        "--start", "0",
        "--max-source-rows", str(selector["max_source_rows_before_shuffle"]),
        "--train-rows", str(pair["train_source_prompts"]),
        "--val-rows", str(pair["val_source_prompts"]),
        "--eval-rows", "0",
        "--seed", str(pair["seed"]),
        "--prefix-token-min", str(pair["prefix_token_min"]),
        "--prefix-token-max", str(pair["prefix_token_max"]),
        "--prefix-mode", pair["prefix_mode"],
        "--tokenizer", pinned_model,
        "--prompt-template", pair["prompt_template"],
        "--scorer-model", job.task["score"]["scorer_model"],
        "--scorer-positive-label", job.task["score"]["positive_label"],
        "--reference-model", pinned_model,
        "--completions-per-prefix", str(pair["completions_per_prompt"]),
    ]
    if pair["shuffle"]:
        pair_args.append("--shuffle")
    _run("screscomp.cli.prepare_imdb_sentiment_env", pair_args)

    downstream = job.task["downstream"]
    eval_dir = paths.shared_dataset_dir / "eval_environment"
    eval_args = [
        "--input", str(test_source),
        "--source-dataset", f"{job.task['dataset']}@{job.task['dataset_revision']}",
        "--out-dir", str(eval_dir),
        "--start", str(downstream["test_source_start"]),
        "--max-source-rows", str(downstream["test_max_source_rows_before_shuffle"]),
        "--train-rows", "0",
        "--val-rows", "0",
        "--eval-rows", str(downstream["test_prompts"]),
        "--seed", str(downstream["test_shuffle_seed"]),
        "--prefix-token-min", str(pair["prefix_token_min"]),
        "--prefix-token-max", str(pair["prefix_token_max"]),
        "--prefix-mode", pair["prefix_mode"],
        "--tokenizer", pinned_model,
        "--prompt-template", pair["prompt_template"],
        "--scorer-model", job.task["score"]["scorer_model"],
        "--scorer-positive-label", job.task["score"]["positive_label"],
        "--reference-model", pinned_model,
        "--completions-per-prefix", str(downstream["samples_per_prompt"]),
    ]
    if downstream["test_shuffle"]:
        eval_args.append("--shuffle")
    _run("screscomp.cli.prepare_imdb_sentiment_env", eval_args)
    shutil.copyfile(eval_dir / "prompts.jsonl", paths.imdb_eval_prompts)


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    job = protocol.job(args.job_id)
    paths = protocol.paths(args.artifact_root, args.job_id)
    paths.shared_dataset_dir.mkdir(parents=True, exist_ok=True)
    sample_preparation: dict[str, object] = {}
    if job.dataset == "confiqa":
        sample_preparation = _prepare_confiqa(protocol, job, paths)
    else:
        selector_state = protocol.selector_state_generation(job)
        if selector_state is None:
            raise RuntimeError("IMDb selector state-generation contract is missing")
        source = _download_imdb_snapshot(job, paths, selector_state["source_split"])
        test_source = _download_imdb_snapshot(job, paths, job.task["downstream"]["test_source_split"])
        pinned_model = resolve_pinned_model(job.model, job.model_revision)
        _prepare_imdb_environment(protocol, job, paths, source, test_source, pinned_model)
    dump_json(
        paths.shared_dataset_dir / "input_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "shared_dataset_dir": str(paths.shared_dataset_dir),
            "sample_preparation": sample_preparation,
        },
    )
    print(f"[site-prepare] complete dataset={job.dataset} path={paths.shared_dataset_dir}", flush=True)


if __name__ == "__main__":
    main()
