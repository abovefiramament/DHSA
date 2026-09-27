from __future__ import annotations

import argparse
import math
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
    "e1_content": ("c_p", "c_e", "normal"),
    "e1_content_swapped": ("swap_c_p", "swap_c_e", "swapped"),
    "ix_content_according": ("ix_source_prior_according", "ix_source_counter_according", "normal"),
    "ix_content_according_swapped": (
        "swap_ix_source_prior_according",
        "swap_ix_source_counter_according",
        "swapped",
    ),
    "ix_content_actual": ("ix_source_prior_actual", "ix_source_counter_actual", "normal"),
    "ix_content_actual_swapped": ("swap_ix_source_prior_actual", "swap_ix_source_counter_actual", "swapped"),
    "e2_task": ("e2_c1_doc_actual", "e2_c1_doc_according", "normal"),
    "e2_task_swapped": ("swap_e2_c1_doc_actual", "swap_e2_c1_doc_according", "swapped"),
    "e2_source": ("e2_c2_user_actual", "e2_c2_doc_actual", "normal"),
    "e2_source_swapped": ("swap_e2_c2_user_actual", "swap_e2_c2_doc_actual", "swapped"),
    "format_label_control": ("format_only", "c_e", "normal"),
    "format_label_control_swapped": ("swap_format_only", "swap_c_e", "swapped"),
}


DIRECTION_SPECS = {
    "e1_content_balanced": ["e1_content", "e1_content_swapped"],
    "ix_content_according_balanced": ["ix_content_according", "ix_content_according_swapped"],
    "ix_content_actual_balanced": ["ix_content_actual", "ix_content_actual_swapped"],
    "e2_task_balanced": ["e2_task", "e2_task_swapped"],
    "e2_source_balanced": ["e2_source", "e2_source_swapped"],
    "format_label_unbalanced": ["format_label_control"],
    "format_label_balanced": ["format_label_control", "format_label_control_swapped"],
}


SIGNED_DIRECTION_SPECS = {
    "ix_interaction_balanced": [
        ("ix_content_according", 1.0),
        ("ix_content_according_swapped", 1.0),
        ("ix_content_actual", -1.0),
        ("ix_content_actual_swapped", -1.0),
    ],
    "ix_interaction_normal": [
        ("ix_content_according", 1.0),
        ("ix_content_actual", -1.0),
    ],
    "ix_interaction_swapped": [
        ("ix_content_according_swapped", 1.0),
        ("ix_content_actual_swapped", -1.0),
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn residual directions on discovery and intervene on held-out prompts.")
    p.add_argument("--rendered_jsonl", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_rows_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--layers", type=str, default="16,20,24,28")
    p.add_argument("--train_splits", type=str, default="discovery")
    p.add_argument("--eval_splits", type=str, default="test")
    p.add_argument("--train_templates", type=str, default="")
    p.add_argument("--eval_templates", type=str, default="")
    p.add_argument(
        "--directions",
        type=str,
        default="e2_task_balanced,e2_source_balanced,e1_content_balanced,format_label_unbalanced,format_label_balanced",
    )
    p.add_argument(
        "--eval_pairs",
        type=str,
        default=(
            "e2_task,e2_task_swapped,e2_source,e2_source_swapped,"
            "e1_content,e1_content_swapped,format_label_control,format_label_control_swapped"
        ),
    )
    p.add_argument("--alphas", type=str, default="1.0")
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_eval_rows", type=int, default=None)
    p.add_argument("--normalize_diffs", action="store_true")
    p.add_argument(
        "--orthogonalize_to_format_label",
        action="store_true",
        help="Project non-format directions away from the unbalanced A/B format-label direction before intervention.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_set(raw: str) -> set[str]:
    return set(_parse_csv(raw))


def _parse_float_csv(raw: str) -> list[float]:
    return [float(x) for x in _parse_csv(raw)]


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


def _known_directions() -> set[str]:
    return set(DIRECTION_SPECS) | set(SIGNED_DIRECTION_SPECS)


def _direction_terms(direction_name: str) -> list[tuple[str, float]]:
    if direction_name in SIGNED_DIRECTION_SPECS:
        return SIGNED_DIRECTION_SPECS[direction_name]
    return [(pair_name, 1.0) for pair_name in DIRECTION_SPECS[direction_name]]


def _opposite_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    raise ValueError(f"Unsupported label: {label}")


def _prior_label(row: dict, label_mode: str) -> str:
    if label_mode == "normal":
        return row["prior_label"]
    if label_mode == "swapped":
        return _opposite_label(row["prior_label"])
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _semantic_b(logit_a: float, logit_b: float, prior_label: str) -> float:
    if prior_label == "A":
        return logit_a - logit_b
    if prior_label == "B":
        return logit_b - logit_a
    raise ValueError(f"Unsupported prior label: {prior_label}")


def _score_b(backend: TransformersABBackend, prompt: str, prior_label: str) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab(prompt)
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _score_add_b(
    backend: TransformersABBackend,
    prompt: str,
    prior_label: str,
    layer_idx: int,
    direction: Any,
    alpha: float,
) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab_with_layer_last_token_add(
        prompt=prompt,
        layer_idx=layer_idx,
        direction=direction,
        alpha=alpha,
    )
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _select_rows(rows: list[dict], splits: set[str], templates: set[str], max_rows: int | None) -> list[dict]:
    selected = [row for row in rows if row["split"] in splits and (not templates or row["template_family"] in templates)]
    if max_rows is not None:
        selected = selected[:max_rows]
    if not selected:
        raise ValueError("No rows selected.")
    return selected


def _validate_keys(rows: list[dict], pair_names: list[str]) -> None:
    for pair_name in pair_names:
        high_key, low_key, _label_mode = PAIR_SPECS[pair_name]
        for row in rows[:1]:
            prompts = row["prompts"]
            missing = [key for key in (high_key, low_key) if key not in prompts]
            if missing:
                raise ValueError(
                    f"Rendered file is missing {missing}; rerun screscomp-render after the swapped-E2 update."
                )


def _capture_pair_diffs(
    backend: TransformersABBackend,
    rows: list[dict],
    pair_names: list[str],
    layers: list[int],
    normalize_diffs: bool,
) -> dict[str, dict[int, list[Any]]]:
    diffs: dict[str, dict[int, list[Any]]] = {pair_name: {layer: [] for layer in layers} for pair_name in pair_names}
    for row in tqdm(rows, desc="capture direction diffs"):
        prompts = row["prompts"]
        for pair_name in pair_names:
            high_key, low_key, _label_mode = PAIR_SPECS[pair_name]
            high_acts = backend.capture_layer_last_token(prompts[high_key], layers)
            low_acts = backend.capture_layer_last_token(prompts[low_key], layers)
            for layer_idx in layers:
                diff = high_acts[layer_idx] - low_acts[layer_idx]
                if normalize_diffs:
                    norm = diff.norm()
                    if float(norm.item()) > 1e-9:
                        diff = diff / norm
                diffs[pair_name][layer_idx].append(diff)
    return diffs


def _mean_tensor(torch_module, tensors: list[Any]):
    if not tensors:
        raise ValueError("Cannot average an empty tensor list.")
    return torch_module.stack(tensors, dim=0).mean(dim=0)


def _learn_directions(
    backend: TransformersABBackend,
    rows: list[dict],
    direction_names: list[str],
    layers: list[int],
    normalize_diffs: bool,
) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, int]]]:
    pair_names = sorted({pair for name in direction_names for pair, _coeff in _direction_terms(name)})
    _validate_keys(rows, pair_names)
    pair_diffs = _capture_pair_diffs(
        backend=backend,
        rows=rows,
        pair_names=pair_names,
        layers=layers,
        normalize_diffs=normalize_diffs,
    )

    directions: dict[str, dict[int, Any]] = {name: {} for name in direction_names}
    counts: dict[str, dict[int, int]] = {name: {} for name in direction_names}
    for direction_name in direction_names:
        for layer_idx in layers:
            tensors = []
            for pair_name, coeff in _direction_terms(direction_name):
                tensors.extend(coeff * diff for diff in pair_diffs[pair_name][layer_idx])
            directions[direction_name][layer_idx] = _mean_tensor(backend._torch, tensors)
            counts[direction_name][layer_idx] = len(tensors)
    return directions, counts


def _cosine(a, b) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = float((a_flat.norm() * b_flat.norm()).item())
    if denom <= 1e-12:
        return 0.0
    return float((a_flat.dot(b_flat) / denom).item())


def _project_away(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = b_flat.dot(b_flat)
    if float(denom.item()) <= 1e-12:
        return a
    scale = a_flat.dot(b_flat) / denom
    return a - scale.to(dtype=a.dtype, device=a.device) * b


def _orthogonalize_to_format_label(
    directions: dict[str, dict[int, Any]],
    direction_counts: dict[str, dict[int, int]],
) -> tuple[dict[str, dict[int, Any]], dict[str, dict[int, int]]]:
    format_by_layer = directions.get("format_label_unbalanced")
    if format_by_layer is None:
        raise ValueError("--orthogonalize_to_format_label requires format_label_unbalanced in --directions.")

    output_directions: dict[str, dict[int, Any]] = {}
    output_counts: dict[str, dict[int, int]] = {}
    for direction_name, by_layer in directions.items():
        if direction_name.startswith("format_label"):
            output_directions[direction_name] = by_layer
            output_counts[direction_name] = direction_counts[direction_name]
            continue

        orth_name = f"{direction_name}_orth_format"
        output_directions[orth_name] = {}
        output_counts[orth_name] = direction_counts[direction_name]
        for layer_idx, direction in by_layer.items():
            output_directions[orth_name][layer_idx] = _project_away(direction, format_by_layer[layer_idx])
    return output_directions, output_counts


def _make_random_directions(backend: TransformersABBackend, directions: dict[str, dict[int, Any]], seed: int):
    rng = random.Random(seed)
    random_dirs: dict[str, dict[int, Any]] = {}
    for direction_name, by_layer in directions.items():
        random_dirs[direction_name] = {}
        for layer_idx, direction in by_layer.items():
            torch_seed = rng.randrange(2**31)
            generator = backend._torch.Generator(device=direction.device)
            generator.manual_seed(torch_seed)
            rand = backend._torch.randn(direction.shape, generator=generator, device=direction.device, dtype=direction.dtype)
            norm = rand.norm()
            target_norm = direction.norm()
            if float(norm.item()) > 1e-12:
                rand = rand / norm * target_norm
            random_dirs[direction_name][layer_idx] = rand
    return random_dirs


def _run_interventions(
    backend: TransformersABBackend,
    rows: list[dict],
    eval_pairs: list[str],
    directions: dict[str, dict[int, Any]],
    direction_counts: dict[str, dict[int, int]],
    random_dirs: dict[str, dict[int, Any]],
    alphas: list[float],
) -> list[dict]:
    _validate_keys(rows, eval_pairs)
    format_direction = directions.get("format_label_unbalanced")
    output_rows: list[dict] = []
    for row in tqdm(rows, desc="direction interventions"):
        prompts = row["prompts"]
        for eval_pair in eval_pairs:
            high_key, low_key, label_mode = PAIR_SPECS[eval_pair]
            prior_label = _prior_label(row, label_mode)
            b_high, logits_high = _score_b(backend, prompts[high_key], prior_label)
            b_low, logits_low = _score_b(backend, prompts[low_key], prior_label)
            delta_b = b_high - b_low

            for direction_name, by_layer in directions.items():
                for layer_idx, direction in by_layer.items():
                    fmt_dir = format_direction.get(layer_idx) if format_direction else None
                    cosine_with_format = _cosine(direction, fmt_dir) if fmt_dir is not None else 0.0
                    norm = float(direction.norm().item())
                    for alpha in alphas:
                        b_plus, logits_plus = _score_add_b(
                            backend=backend,
                            prompt=prompts[low_key],
                            prior_label=prior_label,
                            layer_idx=layer_idx,
                            direction=direction,
                            alpha=alpha,
                        )
                        b_minus, logits_minus = _score_add_b(
                            backend=backend,
                            prompt=prompts[low_key],
                            prior_label=prior_label,
                            layer_idx=layer_idx,
                            direction=direction,
                            alpha=-alpha,
                        )
                        b_random, logits_random = _score_add_b(
                            backend=backend,
                            prompt=prompts[low_key],
                            prior_label=prior_label,
                            layer_idx=layer_idx,
                            direction=random_dirs[direction_name][layer_idx],
                            alpha=alpha,
                        )
                        if fmt_dir is not None:
                            b_format, logits_format = _score_add_b(
                                backend=backend,
                                prompt=prompts[low_key],
                                prior_label=prior_label,
                                layer_idx=layer_idx,
                                direction=fmt_dir,
                                alpha=alpha,
                            )
                        else:
                            b_format, logits_format = 0.0, {}

                        output_rows.append(
                            {
                                "render_id": row["render_id"],
                                "sample_id": row["sample_id"],
                                "fact_id": row["fact_id"],
                                "subject_id": row["subject_id"],
                                "split": row["split"],
                                "template_family": row["template_family"],
                                "direction_name": direction_name,
                                "eval_pair": eval_pair,
                                "high_key": high_key,
                                "low_key": low_key,
                                "layer_idx": layer_idx,
                                "alpha": alpha,
                                "direction_norm": norm,
                                "direction_train_count": direction_counts[direction_name][layer_idx],
                                "cosine_with_format_label_direction": cosine_with_format,
                                "B_high": b_high,
                                "B_low": b_low,
                                "delta_B_high_minus_low": delta_b,
                                "B_low_plus_direction": b_plus,
                                "B_low_minus_direction": b_minus,
                                "B_low_plus_random_direction": b_random,
                                "B_low_plus_format_direction": b_format,
                                "effect_plus": b_plus - b_low,
                                "effect_minus": b_minus - b_low,
                                "effect_random": b_random - b_low,
                                "effect_format": b_format - b_low,
                                "plus_positive": b_plus > b_low,
                                "minus_negative": b_minus < b_low,
                                "plus_beats_random": (b_plus - b_low) > (b_random - b_low),
                                "plus_beats_format": (b_plus - b_low) > (b_format - b_low),
                                "logits": {
                                    "high": logits_high,
                                    "low": logits_low,
                                    "low_plus_direction": logits_plus,
                                    "low_minus_direction": logits_minus,
                                    "low_plus_random_direction": logits_random,
                                    "low_plus_format_direction": logits_format,
                                },
                            }
                        )
    return output_rows


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
        "mean_direction_norm": _mean(rows, "direction_norm"),
        "mean_cosine_with_format_label_direction": _mean(rows, "cosine_with_format_label_direction"),
        "mean_delta_B_high_minus_low": _mean(rows, "delta_B_high_minus_low"),
        "mean_effect_plus": _mean(rows, "effect_plus"),
        "mean_effect_minus": _mean(rows, "effect_minus"),
        "mean_effect_random": _mean(rows, "effect_random"),
        "mean_effect_format": _mean(rows, "effect_format"),
        "frac_plus_positive": _rate(rows, "plus_positive"),
        "frac_minus_negative": _rate(rows, "minus_negative"),
        "frac_plus_beats_random": _rate(rows, "plus_beats_random"),
        "frac_plus_beats_format": _rate(rows, "plus_beats_format"),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        direction = row["direction_name"]
        pair = row["eval_pair"]
        layer = row["layer_idx"]
        alpha = row["alpha"]
        split = row["split"]
        groups[f"direction:{direction}:pair:{pair}:layer:{layer}:alpha:{alpha}"].append(row)
        groups[f"split:{split}:direction:{direction}:pair:{pair}:layer:{layer}:alpha:{alpha}"].append(row)
    return [_summarize_group(name, groups[name]) for name in sorted(groups)]


def main() -> None:
    args = parse_args()
    direction_names = _parse_csv(args.directions)
    unknown_directions = sorted(set(direction_names) - _known_directions())
    if unknown_directions:
        raise ValueError(f"Unknown direction specs: {unknown_directions}")
    eval_pairs = _parse_csv(args.eval_pairs)
    unknown_pairs = sorted(set(eval_pairs) - set(PAIR_SPECS))
    if unknown_pairs:
        raise ValueError(f"Unknown eval pair specs: {unknown_pairs}")

    all_rows = load_jsonl(args.rendered_jsonl)
    train_rows = _select_rows(
        all_rows,
        splits=_parse_csv_set(args.train_splits),
        templates=_parse_csv_set(args.train_templates),
        max_rows=args.max_train_rows,
    )
    eval_rows = _select_rows(
        all_rows,
        splits=_parse_csv_set(args.eval_splits),
        templates=_parse_csv_set(args.eval_templates),
        max_rows=args.max_eval_rows,
    )

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    layers = _parse_layers(args.layers, backend.num_layers)
    directions, direction_counts = _learn_directions(
        backend=backend,
        rows=train_rows,
        direction_names=direction_names,
        layers=layers,
        normalize_diffs=args.normalize_diffs,
    )
    if args.orthogonalize_to_format_label:
        directions, direction_counts = _orthogonalize_to_format_label(directions, direction_counts)
    random_dirs = _make_random_directions(backend=backend, directions=directions, seed=args.seed)
    rows = _run_interventions(
        backend=backend,
        rows=eval_rows,
        eval_pairs=eval_pairs,
        directions=directions,
        direction_counts=direction_counts,
        random_dirs=random_dirs,
        alphas=_parse_float_csv(args.alphas),
    )
    dump_jsonl(args.out_rows_jsonl, rows)
    dump_csv(args.out_summary_csv, _summarize(rows))
    print(
        f"[score-direction-intervention] train_rows={len(train_rows)} eval_rows={len(eval_rows)} "
        f"directions={len(direction_names)} layers={len(layers)} rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
