from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.cecm_run_joint_actuator_generation import (
    HeadScaleAction,
    JointControl,
    JointGenerationRunner,
)
from screscomp.cli.cecm_scan_component_contributions import (
    ComponentSummaryAccumulator,
    CsvRowStream,
    _batches,
    _completion_format_audit,
    _parse_stop_strings,
    _prompt_samples,
    _score_generation_records,
    _select_prompt_rows,
    _set_seed,
)
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero individual attention heads during open-generation rollout on a fixed prompt sample."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--attn-layers", required=True)
    parser.add_argument("--event", default="imdb_positive_sentiment")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--stop-strings", default="")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-mode", default="all")
    parser.add_argument("--scorer-model", default="siebert/sentiment-roberta-large-english")
    parser.add_argument("--target-label", default="POSITIVE")
    parser.add_argument("--source-label", default="NEGATIVE")
    parser.add_argument("--score-text", choices=["completion", "full_text", "prompt_plus_completion"], default="completion")
    parser.add_argument("--scorer-batch-size", type=int, default=16)
    parser.add_argument("--scorer-max-length", type=int, default=512)
    parser.add_argument("--scorer-device", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reuse-baseline-csv", type=Path, default=None,
                        help="Load full baseline from component discovery delta_samples.csv instead of regenerating")
    return parser.parse_args(argv)


def _parse_layers(raw: str) -> list[int]:
    layers = list(dict.fromkeys(int(item.strip()) for item in raw.split(",") if item.strip()))
    if not layers:
        raise SystemExit("--attn-layers is empty")
    return layers


def _head_control(layer_idx: int, head_idx: int, apply_mode: str) -> JointControl:
    return JointControl(
        name=f"zero_L{layer_idx}.attn.h{head_idx}",
        component_additions=tuple(),
        head_scales=(HeadScaleAction(layer_idx=layer_idx, head_idx=head_idx, factor=0.0, apply_mode=apply_mode),),
        head_vectors=tuple(),
        conditional_components=tuple(),
        conditional_heads=tuple(),
        conditional_group_specs={},
        parts=tuple(),
    )


def _records(samples: list[dict[str, Any]], completions: list[str], *, control_name: str, args: argparse.Namespace, batch_seed: int) -> list[dict[str, Any]]:
    return [
        {
            **sample,
            "control_name": control_name,
            "completion": completion,
            "full_text": f"{sample['prompt']}{completion}",
            "model": args.model,
            "generation_batch_size": args.generation_batch_size,
            "batch_seed": batch_seed,
        }
        for sample, completion in zip(samples, completions)
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_rows != 300 or args.samples_per_prompt != 1:
        print(f"[imdb-rollout-head-scan] WARNING: max_rows={args.max_rows} samples_per_prompt={args.samples_per_prompt} (recommended: 300, 1)", flush=True)
    if args.generation_batch_size != 16 or args.scorer_batch_size != 16:
        print(f"[imdb-rollout-head-scan] WARNING: batch sizes gen={args.generation_batch_size} scorer={args.scorer_batch_size} (recommended: 16)", flush=True)
    if args.apply_mode != "all":
        print(f"[imdb-rollout-head-scan] WARNING: apply_mode={args.apply_mode} (recommended: all)", flush=True)

    layers = _parse_layers(args.attn_layers)
    prompt_rows = _select_prompt_rows(load_jsonl(args.prompts_jsonl), args=args)
    samples = _prompt_samples(prompt_rows, args=args)
    if len(prompt_rows) != 300 or len(samples) != 300:
        print(f"[imdb-rollout-head-scan] WARNING: Expected 300 rollout prompts/samples, got prompts={len(prompt_rows)} samples={len(samples)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop_strings = _parse_stop_strings(args.stop_strings)
    dump_json(
        args.out_dir / "scan_config.json",
        {
            "scan_mode": "rollout_head_zero",
            "model": args.model,
            "prompts_jsonl": str(args.prompts_jsonl),
            "event": args.event,
            "split": args.split,
            "prompt_rows": len(prompt_rows),
            "prompt_samples": len(samples),
            "attn_layers": layers,
            "apply_mode": args.apply_mode,
            "generation_batch_size": args.generation_batch_size,
            "scorer_batch_size": args.scorer_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "scorer_model": args.scorer_model,
            "target_label": args.target_label,
            "source_label": args.source_label,
            "score_text": args.score_text,
            "ablation": "scale one attention head slice to zero before the layer output projection during rollout",
            "selection_input": "the union of reused positive and negative attention-layer top4 pools",
        },
    )

    backend = TransformersABBackend(model_name_or_path=args.model, device=args.device, torch_dtype=args.torch_dtype)
    runner = JointGenerationRunner(backend)
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Rollout scoring requires transformers pipeline support.") from exc
    scorer = pipeline("sentiment-analysis", model=args.scorer_model, device=args.scorer_device, top_k=None)
    generation_batches = _batches(samples, args.generation_batch_size)

    full_by_key: dict[tuple[str, int], dict[str, object]] = {}
    if args.reuse_baseline_csv is not None and args.reuse_baseline_csv.exists():
        print(f"[imdb-rollout-head-scan] Reusing baseline from {args.reuse_baseline_csv}", flush=True)
        import csv as _csv
        # Load full baseline from component discovery delta_samples.csv
        # Take rows from one component (e.g. first layer) to get unique baselines
        seen_keys = set()
        with args.reuse_baseline_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in _csv.DictReader(f):
                key = (str(row.get("sample_id", "")), int(row.get("completion_id", 0)))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                full_by_key[key] = {
                    "sample_id": row["sample_id"],
                    "prompt_id": row.get("prompt_id", ""),
                    "completion_id": row["completion_id"],
                    "split": row.get("split", ""),
                    "prompt": row.get("prompt", ""),
                    "completion": row.get("full_completion", ""),
                    "full_text": f"{row.get('prompt', '')}{row.get('full_completion', '')}",
                    "target_score": row.get("full_target_score", 0),
                    "source_score": row.get("full_source_score", 0),
                    "control_name": "full",
                    "generation_batch_size": row.get("generation_batch_size", args.generation_batch_size),
                    "batch_seed": row.get("batch_seed", 0),
                }
        if len(full_by_key) != 300:
            print(f"[imdb-rollout-head-scan] WARNING: Loaded {len(full_by_key)} baseline records (expected 300)", flush=True)
        dump_jsonl(args.out_dir / "full_baseline.jsonl", full_by_key.values())
    else:
        if args.reuse_baseline_csv is not None:
            print(f"[imdb-rollout-head-scan] WARNING: reuse_baseline_csv not found, regenerating baseline", flush=True)
        for batch_index, batch in enumerate(tqdm(generation_batches, desc="head rollout full")):
            batch_seed = args.seed + batch_index * 1009
            _set_seed(backend, batch_seed)
            completions = backend.generate_many(
                [str(sample["prompt"]) for sample in batch],
                max_new_tokens=args.max_new_tokens,
                stop_strings=stop_strings,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            for row in _score_generation_records(
                _records(batch, completions, control_name="full", args=args, batch_seed=batch_seed),
                args=args,
                scorer=scorer,
                desc="score head full",
                show_progress=False,
            ):
                full_by_key[(str(row["sample_id"]), int(row["completion_id"]))] = row
        if len(full_by_key) != 300:
            raise RuntimeError(f"Full baseline key mismatch: scored={len(full_by_key)} expected=300")
        dump_jsonl(args.out_dir / "full_baseline.jsonl", full_by_key.values())

    total_heads = sum(runner._head_geometry(layer_idx)[1] for layer_idx in layers)
    total_rows = total_heads * len(samples)
    step = 0
    accumulator = ComponentSummaryAccumulator()
    with CsvRowStream(args.out_dir / "head_delta_samples.csv") as writer:
        for layer_idx in layers:
            num_heads = runner._head_geometry(layer_idx)[1]
            for head_idx in range(num_heads):
                head_id = f"L{layer_idx}.attn.h{head_idx}"
                control = _head_control(layer_idx, head_idx, args.apply_mode)
                for batch_index, batch in enumerate(tqdm(generation_batches, desc=f"rollout zero {head_id}")):
                    batch_seed = args.seed + batch_index * 1009
                    _set_seed(backend, batch_seed)
                    completions = runner.generate_many(
                        [str(sample["prompt"]) for sample in batch],
                        control=control,
                        apply_mode=args.apply_mode,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=bool(args.do_sample),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    scored = _score_generation_records(
                        _records(batch, completions, control_name=f"zero_{head_id}", args=args, batch_seed=batch_seed),
                        args=args,
                        scorer=scorer,
                        desc=f"score zero {head_id}",
                        show_progress=False,
                    )
                    for ablated in scored:
                        key = (str(ablated["sample_id"]), int(ablated["completion_id"]))
                        full = full_by_key[key]
                        full_margin = float(full["target_score"]) - float(full["source_score"])
                        ablated_margin = float(ablated["target_score"]) - float(ablated["source_score"])
                        delta = full_margin - ablated_margin
                        full_audit = _completion_format_audit(str(full.get("completion", "")))
                        ablated_audit = _completion_format_audit(str(ablated.get("completion", "")))
                        row = {
                            "sample_id": ablated["sample_id"],
                            "prompt_id": ablated.get("prompt_id", ""),
                            "completion_id": ablated["completion_id"],
                            "split": ablated.get("split", ""),
                            "event": args.event,
                            "component_id": head_id,
                            "head_id": head_id,
                            "layer_idx": layer_idx,
                            "head_idx": head_idx,
                            "component_type": "head",
                            "prompt": ablated.get("prompt", ""),
                            "full_completion": full.get("completion", ""),
                            "ablated_completion": ablated.get("completion", ""),
                            "full_target_score": float(full["target_score"]),
                            "full_source_score": float(full["source_score"]),
                            "ablated_target_score": float(ablated["target_score"]),
                            "ablated_source_score": float(ablated["source_score"]),
                            "full_margin": full_margin,
                            "ablated_margin": ablated_margin,
                            "delta": delta,
                            "target_drop": float(full["target_score"]) - float(ablated["target_score"]),
                            "source_rise": float(ablated["source_score"]) - float(full["source_score"]),
                            "positive_transition": int(delta > 0),
                            "negative_transition": int(delta < 0),
                            "bidirectional": int(delta > 0),
                            "full_format_ok": full_audit["format_ok"],
                            "ablated_format_ok": ablated_audit["format_ok"],
                            "format_collapse": int(full_audit["format_ok"] == 1 and ablated_audit["format_ok"] == 0),
                            "full_completion_words": full_audit["completion_words"],
                            "ablated_completion_words": ablated_audit["completion_words"],
                            "full_completion_unique_word_ratio": full_audit["completion_unique_word_ratio"],
                            "ablated_completion_unique_word_ratio": ablated_audit["completion_unique_word_ratio"],
                            "apply_mode": args.apply_mode,
                            "scan_mode": "rollout_head_zero",
                            "generation_batch_size": args.generation_batch_size,
                            "batch_seed": batch_seed,
                            "max_new_tokens": args.max_new_tokens,
                            "score_text": args.score_text,
                            "scorer_model": args.scorer_model,
                        }
                        writer.write(row)
                        accumulator.update(row)
                    step += len(batch)
                    if step % 300 == 0 or step == total_rows:
                        print(f"[imdb-head-rollout-scan] step={step}/{total_rows} head={head_id}", flush=True)

    screen_rows = accumulator.summary_rows()
    dump_csv(args.out_dir / "head_screen.csv", screen_rows)
    dump_csv(
        args.out_dir / "scan_summary.csv",
        [
            {"metric": "scan_mode", "value": "rollout_head_zero"},
            {"metric": "prompt_samples", "value": len(samples)},
            {"metric": "generation_batch_size", "value": args.generation_batch_size},
            {"metric": "scorer_batch_size", "value": args.scorer_batch_size},
            {"metric": "attention_layers", "value": ",".join(str(value) for value in layers)},
            {"metric": "heads_scanned", "value": total_heads},
            {"metric": "delta_rows", "value": total_rows},
        ],
    )
    print(f"[imdb-head-rollout-scan] complete heads={total_heads} rows={total_rows} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
