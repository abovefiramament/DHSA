from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


PAIR_SPECS = {
    "e1_content": {
        "high_key": "c_p",
        "low_key": "c_e",
        "label_mode": "normal",
        "description": "same task/source/options, only supported answer entity changes",
    },
    "e1_content_swapped": {
        "high_key": "swap_c_p",
        "low_key": "swap_c_e",
        "label_mode": "swapped",
        "description": "option-swapped E1 content pair",
    },
    "e2_task": {
        "high_key": "e2_c1_doc_actual",
        "low_key": "e2_c1_doc_according",
        "label_mode": "normal",
        "description": "same source/claim/options, only task objective changes",
    },
    "e2_source": {
        "high_key": "e2_c2_user_actual",
        "low_key": "e2_c2_doc_actual",
        "label_mode": "normal",
        "description": "same task/claim/options, only source label changes",
    },
    "format_label_control": {
        "high_key": "format_only",
        "low_key": "c_e",
        "label_mode": "normal",
        "description": "pure label cue control against counter-evidence prompt",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pairwise layer-level residual mediation checks.")
    p.add_argument("--rendered_jsonl", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_patches_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--layers", type=str, default="12,16,20,24,28")
    p.add_argument("--splits", type=str, default="discovery,test")
    p.add_argument("--templates", type=str, default="")
    p.add_argument(
        "--pairs",
        type=str,
        default="e1_content,e1_content_swapped,e2_task,e2_source,format_label_control",
        help=f"Comma-separated pair specs. Available: {','.join(PAIR_SPECS)}",
    )
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_random_control", action="store_true")
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_set(raw: str) -> set[str]:
    return set(_parse_csv(raw))


def _parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw == "all":
        return list(range(num_layers))

    layers: set[int] = set()
    for part in _parse_csv(raw):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(part))

    invalid = sorted(x for x in layers if x < 0 or x >= num_layers)
    if invalid:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {invalid}")
    return sorted(layers)


def _opposite_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    raise ValueError(f"Unsupported label: {label}")


def _prior_label_for_pair(row: dict, spec: dict[str, str]) -> str:
    if spec["label_mode"] == "normal":
        return row["prior_label"]
    if spec["label_mode"] == "swapped":
        return _opposite_label(row["prior_label"])
    raise ValueError(f"Unsupported label_mode: {spec['label_mode']}")


def _semantic_b(logit_a: float, logit_b: float, prior_label: str) -> float:
    if prior_label == "A":
        return logit_a - logit_b
    if prior_label == "B":
        return logit_b - logit_a
    raise ValueError(f"Unsupported prior label: {prior_label}")


def _score_b(backend: TransformersABBackend, prompt: str, prior_label: str) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab(prompt)
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _score_patch_b(
    backend: TransformersABBackend,
    prompt: str,
    prior_label: str,
    layer_idx: int,
    replacement: Any,
) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab_with_layer_last_token_patch(
        prompt=prompt,
        layer_idx=layer_idx,
        replacement_last_token=replacement,
    )
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _select_rows(rows: list[dict], splits: set[str], templates: set[str], max_rows: int | None) -> list[dict]:
    selected = [row for row in rows if row["split"] in splits and (not templates or row["template_family"] in templates)]
    if max_rows is not None:
        selected = selected[:max_rows]
    if not selected:
        raise ValueError("No rendered rows selected.")
    return selected


def _random_source_indices(rows: list[dict], seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    by_split: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_split[row["split"]].append(idx)

    result: dict[int, int] = {}
    for idx, row in enumerate(rows):
        candidates = [j for j in by_split[row["split"]] if j != idx]
        if not candidates:
            candidates = [j for j in range(len(rows)) if j != idx]
        result[idx] = rng.choice(candidates) if candidates else idx
    return result


def _capture_pair_activations(
    backend: TransformersABBackend,
    rows: list[dict],
    pair_names: list[str],
    layers: list[int],
) -> dict[tuple[int, str, str], dict[int, Any]]:
    cache: dict[tuple[int, str, str], dict[int, Any]] = {}
    for row_idx, row in enumerate(tqdm(rows, desc="capture pair activations")):
        prompts = row["prompts"]
        for pair_name in pair_names:
            spec = PAIR_SPECS[pair_name]
            for side in ("high", "low"):
                key = spec[f"{side}_key"]
                cache[(row_idx, pair_name, side)] = backend.capture_layer_last_token(prompts[key], layers)
    return cache


def _run_pairwise_patching(
    backend: TransformersABBackend,
    rows: list[dict],
    pair_names: list[str],
    layers: list[int],
    include_random_control: bool,
    seed: int,
) -> list[dict]:
    acts = _capture_pair_activations(backend=backend, rows=rows, pair_names=pair_names, layers=layers)
    random_sources = _random_source_indices(rows=rows, seed=seed)
    patch_rows: list[dict] = []

    for row_idx, row in enumerate(tqdm(rows, desc="pairwise patches")):
        prompts = row["prompts"]
        for pair_name in pair_names:
            spec = PAIR_SPECS[pair_name]
            prior_label = _prior_label_for_pair(row, spec)
            high_key = spec["high_key"]
            low_key = spec["low_key"]

            b_high, logits_high = _score_b(backend, prompts[high_key], prior_label)
            b_low, logits_low = _score_b(backend, prompts[low_key], prior_label)
            delta_b = b_high - b_low
            denom = delta_b if abs(delta_b) > 1e-9 else 1e-9

            for layer_idx in layers:
                b_low_with_high, logits_low_with_high = _score_patch_b(
                    backend=backend,
                    prompt=prompts[low_key],
                    prior_label=prior_label,
                    layer_idx=layer_idx,
                    replacement=acts[(row_idx, pair_name, "high")][layer_idx],
                )
                b_high_with_low, logits_high_with_low = _score_patch_b(
                    backend=backend,
                    prompt=prompts[high_key],
                    prior_label=prior_label,
                    layer_idx=layer_idx,
                    replacement=acts[(row_idx, pair_name, "low")][layer_idx],
                )
                effect_high_to_low = b_low_with_high - b_low
                effect_low_to_high = b_high - b_high_with_low

                out = {
                    "render_id": row["render_id"],
                    "sample_id": row["sample_id"],
                    "fact_id": row["fact_id"],
                    "subject_id": row["subject_id"],
                    "split": row["split"],
                    "template_family": row["template_family"],
                    "pair_name": pair_name,
                    "pair_description": spec["description"],
                    "high_key": high_key,
                    "low_key": low_key,
                    "prior_label_for_pair": prior_label,
                    "layer_idx": layer_idx,
                    "B_high": b_high,
                    "B_low": b_low,
                    "delta_B_high_minus_low": delta_b,
                    "B_low_with_high_layer": b_low_with_high,
                    "B_high_with_low_layer": b_high_with_low,
                    "effect_high_to_low": effect_high_to_low,
                    "effect_low_to_high": effect_low_to_high,
                    "recovery_high_to_low": effect_high_to_low / denom,
                    "recovery_low_to_high": effect_low_to_high / denom,
                    "delta_positive": delta_b > 0,
                    "bidirectional_positive": effect_high_to_low > 0 and effect_low_to_high > 0,
                    "logits": {
                        "high": logits_high,
                        "low": logits_low,
                        "low_with_high_layer": logits_low_with_high,
                        "high_with_low_layer": logits_high_with_low,
                    },
                }

                if include_random_control:
                    random_idx = random_sources[row_idx]
                    b_low_with_random_high, logits_random = _score_patch_b(
                        backend=backend,
                        prompt=prompts[low_key],
                        prior_label=prior_label,
                        layer_idx=layer_idx,
                        replacement=acts[(random_idx, pair_name, "high")][layer_idx],
                    )
                    random_effect = b_low_with_random_high - b_low
                    out.update(
                        {
                            "random_source_render_id": rows[random_idx]["render_id"],
                            "B_low_with_random_high_layer": b_low_with_random_high,
                            "effect_random_high_to_low": random_effect,
                            "matched_minus_random_high_to_low": effect_high_to_low - random_effect,
                        }
                    )
                    out["logits"]["low_with_random_high_layer"] = logits_random

                patch_rows.append(out)
    return patch_rows


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        value = row.get(key)
        hits += bool(value) if isinstance(value, bool) else float(value) > 0
    return hits / len(rows)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(vals) if vals else 0.0


def _summarize_group(name: str, rows: list[dict]) -> dict[str, object]:
    return {
        "group": name,
        "n": len(rows),
        "mean_delta_B_high_minus_low": _mean(rows, "delta_B_high_minus_low"),
        "frac_delta_positive": _rate(rows, "delta_positive"),
        "mean_effect_high_to_low": _mean(rows, "effect_high_to_low"),
        "mean_effect_low_to_high": _mean(rows, "effect_low_to_high"),
        "mean_recovery_high_to_low": _mean(rows, "recovery_high_to_low"),
        "mean_recovery_low_to_high": _mean(rows, "recovery_low_to_high"),
        "frac_high_to_low_positive": _rate(rows, "effect_high_to_low"),
        "frac_low_to_high_positive": _rate(rows, "effect_low_to_high"),
        "frac_bidirectional_positive": _rate(rows, "bidirectional_positive"),
        "mean_effect_random_high_to_low": _mean(rows, "effect_random_high_to_low"),
        "mean_matched_minus_random_high_to_low": _mean(rows, "matched_minus_random_high_to_low"),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pair = row["pair_name"]
        layer = row["layer_idx"]
        split = row["split"]
        template = row["template_family"]
        groups[f"pair:{pair}:layer:{layer}"].append(row)
        groups[f"split:{split}:pair:{pair}:layer:{layer}"].append(row)
        groups[f"template:{template}:pair:{pair}:layer:{layer}"].append(row)

    return [_summarize_group(name, groups[name]) for name in sorted(groups)]


def main() -> None:
    args = parse_args()
    pair_names = _parse_csv(args.pairs)
    unknown_pairs = sorted(set(pair_names) - set(PAIR_SPECS))
    if unknown_pairs:
        raise ValueError(f"Unknown pair specs: {unknown_pairs}")

    rows = _select_rows(
        rows=load_jsonl(args.rendered_jsonl),
        splits=_parse_csv_set(args.splits),
        templates=_parse_csv_set(args.templates),
        max_rows=args.max_rows,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    layers = _parse_layers(args.layers, backend.num_layers)
    patch_rows = _run_pairwise_patching(
        backend=backend,
        rows=rows,
        pair_names=pair_names,
        layers=layers,
        include_random_control=args.include_random_control,
        seed=args.seed,
    )
    dump_jsonl(args.out_patches_jsonl, patch_rows)
    dump_csv(args.out_summary_csv, _summarize(patch_rows))
    print(
        f"[score-pairwise-patching] rows={len(rows)} pairs={len(pair_names)} "
        f"layers={len(layers)} patch_rows={len(patch_rows)}"
    )


if __name__ == "__main__":
    main()
