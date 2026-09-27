from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run small-scale layer-level last-token residual patching.")
    p.add_argument("--rendered_jsonl", type=Path, required=True, help="Input 09_rendered_samples.jsonl.")
    p.add_argument("--model", type=str, required=True, help="Transformers model id/path.")
    p.add_argument("--out_patches_jsonl", type=Path, required=True, help="Output per-row/per-layer patch scores.")
    p.add_argument("--out_summary_csv", type=Path, required=True, help="Output per-layer summary CSV.")
    p.add_argument("--device", type=str, default="auto", help="cpu|cuda|auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument(
        "--layers",
        type=str,
        default="0,4,8,12,16,20,24,28,31",
        help="Comma-separated layer indices, ranges like 8-12, or `all`.",
    )
    p.add_argument("--splits", type=str, default="discovery,test", help="Comma-separated splits to include.")
    p.add_argument("--templates", type=str, default="", help="Optional comma-separated template families to include.")
    p.add_argument("--max_rows", type=int, default=None, help="Optional row limit after split/template filtering.")
    p.add_argument("--seed", type=int, default=42, help="Seed for random-source control pairing.")
    p.add_argument(
        "--include_random_control",
        action="store_true",
        help="Patch c_e with c_p activation from another rendered row at the same layer.",
    )
    return p.parse_args()


def _parse_csv(raw: str) -> set[str]:
    return {x.strip() for x in raw.split(",") if x.strip()}


def _parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw == "all":
        return list(range(num_layers))

    layers: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    invalid = sorted(x for x in layers if x < 0 or x >= num_layers)
    if invalid:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {invalid}")
    return sorted(layers)


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
    replacement,
) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab_with_layer_last_token_patch(
        prompt=prompt,
        layer_idx=layer_idx,
        replacement_last_token=replacement,
    )
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _choose_random_sources(rows: list[dict], seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    by_split: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_split[row["split"]].append(idx)

    source_for_idx: dict[int, int] = {}
    for idx, row in enumerate(rows):
        candidates = [j for j in by_split[row["split"]] if j != idx]
        if not candidates:
            candidates = [j for j in range(len(rows)) if j != idx]
        if not candidates:
            source_for_idx[idx] = idx
        else:
            source_for_idx[idx] = rng.choice(candidates)
    return source_for_idx


def _capture_activations(
    backend: TransformersABBackend,
    rows: list[dict],
    layers: list[int],
) -> tuple[list[dict[int, object]], list[dict[int, object]]]:
    cp_acts: list[dict[int, object]] = []
    ce_acts: list[dict[int, object]] = []
    for row in tqdm(rows, desc="capture activations"):
        prompts = row["prompts"]
        cp_acts.append(backend.capture_layer_last_token(prompts["c_p"], layers))
        ce_acts.append(backend.capture_layer_last_token(prompts["c_e"], layers))
    return cp_acts, ce_acts


def _run_patching(
    backend: TransformersABBackend,
    rows: list[dict],
    layers: list[int],
    include_random_control: bool,
    seed: int,
) -> list[dict]:
    cp_acts, ce_acts = _capture_activations(backend=backend, rows=rows, layers=layers)
    random_source_for_idx = _choose_random_sources(rows=rows, seed=seed)

    output_rows: list[dict] = []
    for idx, row in enumerate(tqdm(rows, desc="patch layers")):
        prompts = row["prompts"]
        prior_label = row["prior_label"]

        b_cp, logits_cp = _score_b(backend, prompts["c_p"], prior_label)
        b_ce, logits_ce = _score_b(backend, prompts["c_e"], prior_label)
        gap = b_cp - b_ce

        for layer_idx in layers:
            patched_p_to_e, logits_p_to_e = _score_patch_b(
                backend=backend,
                prompt=prompts["c_e"],
                prior_label=prior_label,
                layer_idx=layer_idx,
                replacement=cp_acts[idx][layer_idx],
            )
            patched_e_to_p, logits_e_to_p = _score_patch_b(
                backend=backend,
                prompt=prompts["c_p"],
                prior_label=prior_label,
                layer_idx=layer_idx,
                replacement=ce_acts[idx][layer_idx],
            )

            effect_p_to_e = patched_p_to_e - b_ce
            effect_e_to_p = b_cp - patched_e_to_p
            denom = gap if abs(gap) > 1e-9 else 1e-9

            patch_row = {
                "render_id": row["render_id"],
                "sample_id": row["sample_id"],
                "fact_id": row["fact_id"],
                "subject_id": row["subject_id"],
                "split": row["split"],
                "template_family": row["template_family"],
                "layer_idx": layer_idx,
                "B_c_p": b_cp,
                "B_c_e": b_ce,
                "doc_support_gap": gap,
                "B_c_e_with_c_p_layer": patched_p_to_e,
                "B_c_p_with_c_e_layer": patched_e_to_p,
                "effect_p_to_e": effect_p_to_e,
                "effect_e_to_p": effect_e_to_p,
                "recovery_p_to_e": effect_p_to_e / denom,
                "recovery_e_to_p": effect_e_to_p / denom,
                "bidirectional_positive": effect_p_to_e > 0 and effect_e_to_p > 0,
                "logits": {
                    "c_p": logits_cp,
                    "c_e": logits_ce,
                    "c_e_with_c_p_layer": logits_p_to_e,
                    "c_p_with_c_e_layer": logits_e_to_p,
                },
            }

            if include_random_control:
                random_idx = random_source_for_idx[idx]
                random_p_to_e, logits_random = _score_patch_b(
                    backend=backend,
                    prompt=prompts["c_e"],
                    prior_label=prior_label,
                    layer_idx=layer_idx,
                    replacement=cp_acts[random_idx][layer_idx],
                )
                random_effect = random_p_to_e - b_ce
                patch_row.update(
                    {
                        "random_source_render_id": rows[random_idx]["render_id"],
                        "B_c_e_with_random_c_p_layer": random_p_to_e,
                        "effect_random_p_to_e": random_effect,
                        "matched_minus_random_p_to_e": effect_p_to_e - random_effect,
                    }
                )
                patch_row["logits"]["c_e_with_random_c_p_layer"] = logits_random

            output_rows.append(patch_row)

    return output_rows


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        value = row[key]
        hits += bool(value) if isinstance(value, bool) else float(value) > 0
    return hits / len(rows)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(vals) if vals else 0.0


def _summarize_group(name: str, rows: list[dict]) -> dict[str, object]:
    return {
        "group": name,
        "n": len(rows),
        "mean_doc_support_gap": _mean(rows, "doc_support_gap"),
        "mean_effect_p_to_e": _mean(rows, "effect_p_to_e"),
        "mean_effect_e_to_p": _mean(rows, "effect_e_to_p"),
        "mean_recovery_p_to_e": _mean(rows, "recovery_p_to_e"),
        "mean_recovery_e_to_p": _mean(rows, "recovery_e_to_p"),
        "frac_p_to_e_positive": _rate(rows, "effect_p_to_e"),
        "frac_e_to_p_positive": _rate(rows, "effect_e_to_p"),
        "frac_bidirectional_positive": _rate(rows, "bidirectional_positive"),
        "mean_effect_random_p_to_e": _mean(rows, "effect_random_p_to_e"),
        "mean_matched_minus_random_p_to_e": _mean(rows, "matched_minus_random_p_to_e"),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []

    by_layer: dict[int, list[dict]] = defaultdict(list)
    by_split_layer: dict[tuple[str, int], list[dict]] = defaultdict(list)
    by_template_layer: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        layer_idx = row["layer_idx"]
        by_layer[layer_idx].append(row)
        by_split_layer[(row["split"], layer_idx)].append(row)
        by_template_layer[(row["template_family"], layer_idx)].append(row)

    for layer_idx in sorted(by_layer):
        summary.append(_summarize_group(f"layer:{layer_idx}", by_layer[layer_idx]))
    for split, layer_idx in sorted(by_split_layer):
        summary.append(_summarize_group(f"split:{split}:layer:{layer_idx}", by_split_layer[(split, layer_idx)]))
    for template, layer_idx in sorted(by_template_layer):
        summary.append(_summarize_group(f"template:{template}:layer:{layer_idx}", by_template_layer[(template, layer_idx)]))

    return summary


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.rendered_jsonl)

    splits = _parse_csv(args.splits)
    templates = _parse_csv(args.templates)
    rows = [row for row in rows if row["split"] in splits and (not templates or row["template_family"] in templates)]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise ValueError("No rows selected for residual patching.")

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    layers = _parse_layers(args.layers, backend.num_layers)

    patch_rows = _run_patching(
        backend=backend,
        rows=rows,
        layers=layers,
        include_random_control=args.include_random_control,
        seed=args.seed,
    )
    dump_jsonl(args.out_patches_jsonl, patch_rows)
    dump_csv(args.out_summary_csv, _summarize(patch_rows))
    print(f"[score-residual-patching] rows={len(rows)} layers={len(layers)} patch_rows={len(patch_rows)}")


if __name__ == "__main__":
    main()
