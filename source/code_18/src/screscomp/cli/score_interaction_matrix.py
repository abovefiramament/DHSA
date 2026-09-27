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

from screscomp.cli.score_direction_intervention import (
    _cosine,
    _known_directions,
    _learn_directions,
    _make_random_directions,
    _opposite_label,
    _orthogonalize_to_format_label,
    _parse_csv,
    _parse_csv_set,
    _parse_float_csv,
    _parse_layers,
    _score_add_b,
    _score_b,
    _select_rows,
)
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


MATRIX_SPECS = {
    "normal": {
        "source_prior_according": "ix_source_prior_according",
        "source_counter_according": "ix_source_counter_according",
        "source_prior_actual": "ix_source_prior_actual",
        "source_counter_actual": "ix_source_counter_actual",
        "label_mode": "normal",
    },
    "swapped": {
        "source_prior_according": "swap_ix_source_prior_according",
        "source_counter_according": "swap_ix_source_counter_according",
        "source_prior_actual": "swap_ix_source_prior_actual",
        "source_counter_actual": "swap_ix_source_counter_actual",
        "label_mode": "swapped",
    },
}


DID_CELL_SIGNS = {
    "source_prior_according": 1.0,
    "source_counter_according": -1.0,
    "source_prior_actual": -1.0,
    "source_counter_actual": 1.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score four-cell source/objective DID under residual direction interventions.")
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
    p.add_argument("--layers", type=str, default="16")
    p.add_argument("--train_splits", type=str, default="discovery")
    p.add_argument("--eval_splits", type=str, default="test")
    p.add_argument("--train_templates", type=str, default="")
    p.add_argument("--eval_templates", type=str, default="")
    p.add_argument(
        "--directions",
        type=str,
        default="ix_interaction_balanced,ix_content_according_balanced,ix_content_actual_balanced,format_label_unbalanced",
    )
    p.add_argument("--matrices", type=str, default="normal,swapped")
    p.add_argument("--alphas", type=str, default="0.5,1.0,2.0")
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_eval_rows", type=int, default=None)
    p.add_argument("--normalize_diffs", action="store_true")
    p.add_argument("--orthogonalize_to_format_label", action="store_true")
    p.add_argument(
        "--signed_cell_intervention",
        action="store_true",
        help="Apply +direction to positive DID cells and -direction to negative DID cells.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _prior_label(row: dict, label_mode: str) -> str:
    if label_mode == "normal":
        return row["prior_label"]
    if label_mode == "swapped":
        return _opposite_label(row["prior_label"])
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _validate_matrix_keys(rows: list[dict], matrix_names: list[str]) -> None:
    for matrix_name in matrix_names:
        spec = MATRIX_SPECS[matrix_name]
        for row in rows[:1]:
            prompts = row["prompts"]
            missing = [key for cell, key in spec.items() if cell != "label_mode" and key not in prompts]
            if missing:
                raise ValueError(f"Rendered file is missing four-cell interaction keys: {missing}")


def _did(scores: dict[str, float]) -> float:
    content_according = scores["source_prior_according"] - scores["source_counter_according"]
    content_actual = scores["source_prior_actual"] - scores["source_counter_actual"]
    return content_according - content_actual


def _matrix_scores(backend: TransformersABBackend, row: dict, matrix_name: str) -> tuple[dict[str, float], dict[str, dict]]:
    spec = MATRIX_SPECS[matrix_name]
    prior_label = _prior_label(row, spec["label_mode"])
    scores: dict[str, float] = {}
    logits: dict[str, dict] = {}
    for cell, prompt_key in spec.items():
        if cell == "label_mode":
            continue
        score, cell_logits = _score_b(backend, row["prompts"][prompt_key], prior_label)
        scores[cell] = score
        logits[cell] = cell_logits
    return scores, logits


def _matrix_scores_add(
    backend: TransformersABBackend,
    row: dict,
    matrix_name: str,
    layer_idx: int,
    direction: Any,
    alpha: float,
    signed_cell_intervention: bool,
) -> tuple[dict[str, float], dict[str, dict]]:
    spec = MATRIX_SPECS[matrix_name]
    prior_label = _prior_label(row, spec["label_mode"])
    scores: dict[str, float] = {}
    logits: dict[str, dict] = {}
    for cell, prompt_key in spec.items():
        if cell == "label_mode":
            continue
        cell_alpha = alpha * DID_CELL_SIGNS[cell] if signed_cell_intervention else alpha
        score, cell_logits = _score_add_b(
            backend=backend,
            prompt=row["prompts"][prompt_key],
            prior_label=prior_label,
            layer_idx=layer_idx,
            direction=direction,
            alpha=cell_alpha,
        )
        scores[cell] = score
        logits[cell] = cell_logits
    return scores, logits


def _run_matrix_interventions(
    backend: TransformersABBackend,
    rows: list[dict],
    matrix_names: list[str],
    directions: dict[str, dict[int, Any]],
    direction_counts: dict[str, dict[int, int]],
    random_dirs: dict[str, dict[int, Any]],
    alphas: list[float],
    signed_cell_intervention: bool,
) -> list[dict]:
    _validate_matrix_keys(rows, matrix_names)
    format_direction = directions.get("format_label_unbalanced")
    output_rows: list[dict] = []
    for row in tqdm(rows, desc="matrix DID interventions"):
        for matrix_name in matrix_names:
            base_scores, base_logits = _matrix_scores(backend=backend, row=row, matrix_name=matrix_name)
            base_did = _did(base_scores)

            for direction_name, by_layer in directions.items():
                for layer_idx, direction in by_layer.items():
                    fmt_dir = format_direction.get(layer_idx) if format_direction else None
                    cosine_with_format = _cosine(direction, fmt_dir) if fmt_dir is not None else 0.0
                    norm = float(direction.norm().item())
                    for alpha in alphas:
                        plus_scores, plus_logits = _matrix_scores_add(
                            backend=backend,
                            row=row,
                            matrix_name=matrix_name,
                            layer_idx=layer_idx,
                            direction=direction,
                            alpha=alpha,
                            signed_cell_intervention=signed_cell_intervention,
                        )
                        minus_scores, minus_logits = _matrix_scores_add(
                            backend=backend,
                            row=row,
                            matrix_name=matrix_name,
                            layer_idx=layer_idx,
                            direction=direction,
                            alpha=-alpha,
                            signed_cell_intervention=signed_cell_intervention,
                        )
                        random_scores, random_logits = _matrix_scores_add(
                            backend=backend,
                            row=row,
                            matrix_name=matrix_name,
                            layer_idx=layer_idx,
                            direction=random_dirs[direction_name][layer_idx],
                            alpha=alpha,
                            signed_cell_intervention=signed_cell_intervention,
                        )
                        if fmt_dir is not None:
                            format_scores, format_logits = _matrix_scores_add(
                                backend=backend,
                                row=row,
                                matrix_name=matrix_name,
                                layer_idx=layer_idx,
                                direction=fmt_dir,
                                alpha=alpha,
                                signed_cell_intervention=signed_cell_intervention,
                            )
                        else:
                            format_scores, format_logits = {}, {}

                        plus_did = _did(plus_scores)
                        minus_did = _did(minus_scores)
                        random_did = _did(random_scores)
                        format_did = _did(format_scores) if format_scores else 0.0
                        output_rows.append(
                            {
                                "render_id": row["render_id"],
                                "sample_id": row["sample_id"],
                                "fact_id": row["fact_id"],
                                "subject_id": row["subject_id"],
                                "split": row["split"],
                                "template_family": row["template_family"],
                                "matrix_name": matrix_name,
                                "direction_name": direction_name,
                                "layer_idx": layer_idx,
                                "alpha": alpha,
                                "intervention_mode": "signed_cell" if signed_cell_intervention else "uniform_cell",
                                "direction_norm": norm,
                                "direction_train_count": direction_counts[direction_name][layer_idx],
                                "cosine_with_format_label_direction": cosine_with_format,
                                "base_did": base_did,
                                "plus_did": plus_did,
                                "minus_did": minus_did,
                                "random_did": random_did,
                                "format_did": format_did,
                                "effect_plus_did": plus_did - base_did,
                                "effect_minus_did": minus_did - base_did,
                                "effect_random_did": random_did - base_did,
                                "effect_format_did": format_did - base_did,
                                "plus_increases_did": plus_did > base_did,
                                "minus_decreases_did": minus_did < base_did,
                                "plus_beats_random_did": (plus_did - base_did) > (random_did - base_did),
                                "plus_beats_format_did": (plus_did - base_did) > (format_did - base_did),
                                "scores": {
                                    "base": base_scores,
                                    "plus": plus_scores,
                                    "minus": minus_scores,
                                    "random": random_scores,
                                    "format": format_scores,
                                },
                                "logits": {
                                    "base": base_logits,
                                    "plus": plus_logits,
                                    "minus": minus_logits,
                                    "random": random_logits,
                                    "format": format_logits,
                                },
                            }
                        )
    return output_rows


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(vals) if vals else 0.0


def _summarize_group(name: str, rows: list[dict]) -> dict[str, object]:
    return {
        "group": name,
        "n": len(rows),
        "mean_direction_norm": _mean(rows, "direction_norm"),
        "mean_cosine_with_format_label_direction": _mean(rows, "cosine_with_format_label_direction"),
        "mean_base_did": _mean(rows, "base_did"),
        "mean_plus_did": _mean(rows, "plus_did"),
        "mean_minus_did": _mean(rows, "minus_did"),
        "mean_random_did": _mean(rows, "random_did"),
        "mean_format_did": _mean(rows, "format_did"),
        "mean_effect_plus_did": _mean(rows, "effect_plus_did"),
        "mean_effect_minus_did": _mean(rows, "effect_minus_did"),
        "mean_effect_random_did": _mean(rows, "effect_random_did"),
        "mean_effect_format_did": _mean(rows, "effect_format_did"),
        "frac_plus_increases_did": _rate(rows, "plus_increases_did"),
        "frac_minus_decreases_did": _rate(rows, "minus_decreases_did"),
        "frac_plus_beats_random_did": _rate(rows, "plus_beats_random_did"),
        "frac_plus_beats_format_did": _rate(rows, "plus_beats_format_did"),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        matrix = row["matrix_name"]
        direction = row["direction_name"]
        layer = row["layer_idx"]
        alpha = row["alpha"]
        mode = row["intervention_mode"]
        split = row["split"]
        groups[f"mode:{mode}:matrix:{matrix}:direction:{direction}:layer:{layer}:alpha:{alpha}"].append(row)
        groups[f"split:{split}:mode:{mode}:matrix:{matrix}:direction:{direction}:layer:{layer}:alpha:{alpha}"].append(row)
    return [_summarize_group(name, groups[name]) for name in sorted(groups)]


def main() -> None:
    args = parse_args()
    direction_names = _parse_csv(args.directions)
    unknown_directions = sorted(set(direction_names) - _known_directions())
    if unknown_directions:
        raise ValueError(f"Unknown direction specs: {unknown_directions}")
    matrix_names = _parse_csv(args.matrices)
    unknown_matrices = sorted(set(matrix_names) - set(MATRIX_SPECS))
    if unknown_matrices:
        raise ValueError(f"Unknown matrix specs: {unknown_matrices}")

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
    rows = _run_matrix_interventions(
        backend=backend,
        rows=eval_rows,
        matrix_names=matrix_names,
        directions=directions,
        direction_counts=direction_counts,
        random_dirs=random_dirs,
        alphas=_parse_float_csv(args.alphas),
        signed_cell_intervention=args.signed_cell_intervention,
    )
    dump_jsonl(args.out_rows_jsonl, rows)
    dump_csv(args.out_summary_csv, _summarize(rows))
    print(
        f"[score-interaction-matrix] train_rows={len(train_rows)} eval_rows={len(eval_rows)} "
        f"directions={len(direction_names)} matrices={len(matrix_names)} layers={len(layers)} rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
