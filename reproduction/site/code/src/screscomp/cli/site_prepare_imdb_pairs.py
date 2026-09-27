from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from screscomp.data import dump_json, dump_jsonl, load_jsonl
from screscomp.site.protocol import SiteProtocol, require_files
from screscomp.site.runtime import configure_gpu, resolve_pinned_model
from screscomp.site.sentiment import FrozenSentimentScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one shared model-native IMDb pair artifact.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    job = protocol.job(args.job_id)
    if job.dataset != "imdb":
        raise ValueError("site_prepare_imdb_pairs only accepts IMDb jobs")
    paths = protocol.paths(args.artifact_root, args.job_id)
    pair = job.task["pair_preparation"]
    prompt_file = paths.shared_dataset_dir / "pair_environment" / "prompts.jsonl"
    require_files([prompt_file])
    gpu = configure_gpu(args.device, protocol.site["execution"])
    pinned_model = resolve_pinned_model(job.model, job.model_revision)
    generation_file = paths.pair_dir / "model_native_generations.jsonl"
    scored_file = paths.pair_dir / "model_native_scored_generations.jsonl"
    paths.pair_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m", "screscomp.cli.generate_imdb_sentiment_completions",
        "--model", pinned_model,
        "--prompts-jsonl", str(prompt_file),
        "--out-jsonl", str(generation_file),
        "--split", "all",
        "--completions-per-prefix", str(pair["completions_per_prompt"]),
        "--generation-batch-size", "1",
        "--max-new-tokens", str(pair["generation_max_new_tokens"]),
        "--temperature", str(pair["temperature"]),
        "--top-p", str(pair["top_p"]),
        "--top-k", str(pair["top_k"]),
        "--seed", str(pair["generation_seed"]),
        "--device", "cuda:0",
        "--torch-dtype", "auto",
    ]
    if not pair["do_sample"]:
        command.append("--no-do-sample")
    if job.use_chat_template:
        command.append("--use-chat-template")
    subprocess.run(command, check=True)

    generated = load_jsonl(generation_file)
    scorer = FrozenSentimentScorer(job.task["score"], device="cuda:0")
    scores = scorer.score([str(row["completion"]) for row in generated])
    scored = [
        {
            **row,
            **score,
            "scorer_model": job.task["score"]["scorer_model"],
            "scorer_revision": job.task["score"]["scorer_revision"],
        }
        for row, score in zip(generated, scores, strict=True)
    ]
    dump_jsonl(scored_file, scored)
    subprocess.run(
        [
            sys.executable,
            "-m", "screscomp.cli.prepare_imdb_sentiment_pairs",
            "--input", str(scored_file),
            "--out-dir", str(paths.pair_dir),
            "--event", job.task["downstream"]["event"],
            "--score-field", "positive_sentiment_score",
            "--min-score-margin", str(pair["min_score_margin"]),
            "--pair-selection-mode", pair["pair_selection_mode"],
            "--target-train-pairs", str(pair["target_train_pairs"]),
            "--target-val-pairs", str(pair["target_val_pairs"]),
        ],
        check=True,
    )
    with paths.pairs_csv.open(encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    split_counts = {
        split: sum(1 for row in pair_rows if row.get("split") == split)
        for split in ("train", "val")
    }
    if split_counts["train"] != int(pair["target_train_pairs"]) or split_counts["val"] != int(pair["target_val_pairs"]):
        dump_json(paths.pair_dir / "incomplete.json", {"reason": "fixed_window_pair_shortfall", "actual": split_counts})
        raise RuntimeError(f"IMDb fixed pair windows did not admit locked targets: {split_counts}")
    dump_json(
        paths.pair_dir / "site_pair_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "gpu": gpu,
            "model_path": pinned_model,
            "prompt_artifact": str(prompt_file),
            "generation_artifact": str(generation_file),
            "scored_artifact": str(scored_file),
            "pairs_csv": str(paths.pairs_csv),
            "pair_generation_batch_size": 1,
            "pair_generation_batch_size_source": "existing generate_imdb_sentiment_completions default used by the locked IMDb framework",
            "actual_generation_rows": len(generated),
            "actual_pairs": split_counts,
        },
    )
    print(f"[site-imdb-pairs] complete model={job.model_key} pairs={split_counts}", flush=True)


if __name__ == "__main__":
    main()
