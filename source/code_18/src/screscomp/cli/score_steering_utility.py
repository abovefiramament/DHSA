from __future__ import annotations

import argparse
import csv
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

from screscomp.cli.score_component_competition import (
    PAIR_SPECS,
    _component_id,
    _opposite_label,
    _parse_csv,
    _parse_csv_set,
    _score_b,
    _select_rows,
)
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


TASK_SPECS = {
    "doc_following": ("c_e", "normal", "decrease", -1.0),
    "doc_following_swapped": ("swap_c_e", "swapped", "decrease", -1.0),
    "false_user_actual": ("e2_c2_user_actual", "normal", "increase", 1.0),
    "false_user_actual_swapped": ("swap_e2_c2_user_actual", "swapped", "increase", 1.0),
    "false_doc_actual": ("e2_c1_doc_actual", "normal", "increase", 1.0),
    "false_doc_actual_swapped": ("swap_e2_c1_doc_actual", "swapped", "increase", 1.0),
    "regular_qa": ("prior_only", "normal", "preserve", 1.0),
    "regular_qa_swapped": ("swap_prior_only", "swapped", "preserve", 1.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run held-out steering utility with discovery-selected components.")
    p.add_argument("--rendered_jsonl", type=Path, required=True)
    p.add_argument("--component_summary_csv", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_rows_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--out_manifest_csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--train_splits", type=str, default="discovery")
    p.add_argument("--validation_splits", type=str, default="validation")
    p.add_argument("--eval_splits", type=str, default="test")
    p.add_argument("--train_templates", type=str, default="")
    p.add_argument("--validation_templates", type=str, default="")
    p.add_argument("--eval_templates", type=str, default="")
    p.add_argument("--selection_pairs", type=str, default="e1_content,e1_content_swapped")
    p.add_argument("--tasks", type=str, default=",".join(TASK_SPECS))
    p.add_argument("--alphas", type=str, default="0.25,0.5,1.0,2.0")
    p.add_argument(
        "--top_k_components",
        type=str,
        default="4",
        help="Comma-separated prefix sizes over the discovery-ranked component list, e.g. 1,2,4,8.",
    )
    p.add_argument(
        "--direction_mode",
        type=str,
        default="label_conditional",
        choices=["shared", "label_conditional"],
        help=(
            "`shared` averages all selected E1 directions into one vector. "
            "`label_conditional` learns separate normal/swap directions from discovery only, "
            "using the same selected components."
        ),
    )
    p.add_argument("--normalize_diffs", action="store_true")
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_validation_rows", type=int, default=None)
    p.add_argument("--max_eval_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_float_csv(raw: str) -> list[float]:
    return [float(x) for x in _parse_csv(raw)]


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(x) for x in _parse_csv(raw)]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if any(v <= 0 for v in values):
        raise ValueError("top_k values must be positive integers.")
    return sorted(dict.fromkeys(values))


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


def _score_many_add_b(
    backend: TransformersABBackend,
    prompt: str,
    prior_label: str,
    additions: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab_with_component_last_token_add_many(prompt=prompt, additions=additions)
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rank_components(summary_rows: list[dict[str, str]], selection_pairs: list[str]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("split") != "discovery":
            continue
        if row.get("pair_name") not in selection_pairs:
            continue
        grouped[(int(row["layer_idx"]), row["component_type"])].append(row)

    candidates: list[dict] = []
    for (layer_idx, component_type), rows in grouped.items():
        mean_min_ci = mean(float(row["mean_min_CI"]) for row in rows)
        mean_c_plus_i = mean(float(row["mean_C_plus_I"]) for row in rows)
        frac_both = mean(float(row["frac_both_positive"]) for row in rows)
        candidates.append(
            {
                "layer_idx": layer_idx,
                "component_type": component_type,
                "component_id": _component_id(layer_idx, component_type),
                "selection_score": mean_min_ci,
                "selection_mean_min_CI": mean_min_ci,
                "selection_mean_C_plus_I": mean_c_plus_i,
                "selection_frac_both_positive": frac_both,
                "selection_rows": len(rows),
            }
        )
    if not candidates:
        raise ValueError("No discovery component summary rows found for selection.")
    candidates.sort(
        key=lambda row: (
            -float(row["selection_mean_min_CI"]),
            -float(row["selection_frac_both_positive"]),
            -float(row["selection_mean_C_plus_I"]),
            int(row["layer_idx"]),
            row["component_type"],
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["selection_rank"] = rank
        row["selection_rule"] = "rank_by_mean_min_CI_tiebreak_frac_both_then_mean_C_plus_I"
    return candidates


def _learn_component_directions(
    backend: TransformersABBackend,
    train_rows: list[dict],
    components: list[dict],
    selection_pairs: list[str],
    normalize_diffs: bool,
) -> dict[tuple[int, str], Any]:
    component_specs = [(int(c["layer_idx"]), str(c["component_type"])) for c in components]
    diffs: dict[tuple[int, str], list[Any]] = {spec: [] for spec in component_specs}
    for row in tqdm(train_rows, desc="learn steering components"):
        prompts = row["prompts"]
        for pair_name in selection_pairs:
            high_key, low_key, _label_mode = PAIR_SPECS[pair_name]
            high_acts = backend.capture_component_last_token(prompts[high_key], component_specs)
            low_acts = backend.capture_component_last_token(prompts[low_key], component_specs)
            for spec in component_specs:
                diff = high_acts[spec] - low_acts[spec]
                if normalize_diffs:
                    norm = diff.norm()
                    if float(norm.item()) > 1e-9:
                        diff = diff / norm
                diffs[spec].append(diff)

    directions: dict[tuple[int, str], Any] = {}
    for spec, tensors in diffs.items():
        directions[spec] = backend._torch.stack(tensors, dim=0).mean(dim=0)
    return directions


def _learn_component_directions_by_label_mode(
    backend: TransformersABBackend,
    train_rows: list[dict],
    components: list[dict],
    selection_pairs: list[str],
    normalize_diffs: bool,
) -> dict[str, dict[tuple[int, str], Any]]:
    grouped_pairs: dict[str, list[str]] = defaultdict(list)
    for pair_name in selection_pairs:
        _high_key, _low_key, label_mode = PAIR_SPECS[pair_name]
        grouped_pairs[label_mode].append(pair_name)

    missing = sorted({"normal", "swapped"} - set(grouped_pairs))
    if missing:
        raise ValueError(
            "`label_conditional` direction mode requires both normal and swapped selection pairs; "
            f"missing: {missing}"
        )

    return {
        label_mode: _learn_component_directions(
            backend=backend,
            train_rows=train_rows,
            components=components,
            selection_pairs=pair_names,
            normalize_diffs=normalize_diffs,
        )
        for label_mode, pair_names in sorted(grouped_pairs.items())
    }


def _make_random_directions(backend: TransformersABBackend, directions: dict[tuple[int, str], Any], seed: int):
    rng = random.Random(seed)
    random_dirs: dict[tuple[int, str], Any] = {}
    for spec, direction in directions.items():
        torch_seed = rng.randrange(2**31)
        generator = backend._torch.Generator(device=direction.device)
        generator.manual_seed(torch_seed)
        rand = backend._torch.randn(direction.shape, generator=generator, device=direction.device, dtype=direction.dtype)
        norm = rand.norm()
        target_norm = direction.norm()
        if float(norm.item()) > 1e-12:
            rand = rand / norm * target_norm
        random_dirs[spec] = rand
    return random_dirs


def _make_random_directions_by_label_mode(
    backend: TransformersABBackend,
    directions_by_label_mode: dict[str, dict[tuple[int, str], Any]],
    seed: int,
) -> dict[str, dict[tuple[int, str], Any]]:
    return {
        label_mode: _make_random_directions(backend=backend, directions=directions, seed=seed + idx)
        for idx, (label_mode, directions) in enumerate(sorted(directions_by_label_mode.items()))
    }


def _additions_for(
    directions: dict[tuple[int, str], Any],
    alpha: float,
    task_sign: float,
) -> list[dict[str, Any]]:
    return [
        {
            "layer_idx": layer_idx,
            "component_type": component_type,
            "direction": direction,
            "alpha": alpha * task_sign,
        }
        for (layer_idx, component_type), direction in directions.items()
    ]


def _target_gain(base_b: float, steered_b: float, target: str) -> float:
    if target == "decrease":
        return base_b - steered_b
    return steered_b - base_b


def _target_correct(score_b: float, target: str) -> bool:
    if target == "decrease":
        return score_b < 0
    return score_b > 0


def _score_split(
    backend: TransformersABBackend,
    rows: list[dict],
    task_names: list[str],
    directions_by_label_mode: dict[str, dict[tuple[int, str], Any]],
    random_dirs_by_label_mode: dict[str, dict[tuple[int, str], Any]],
    alphas: list[float],
    phase: str,
    direction_mode: str,
    top_k: int,
) -> list[dict]:
    output_rows: list[dict] = []
    for row in tqdm(rows, desc=f"steering {phase}"):
        prompts = row["prompts"]
        for task_name in task_names:
            prompt_key, label_mode, target, sign = TASK_SPECS[task_name]
            prior_label = _prior_label(row, label_mode)
            base_b, base_logits = _score_b(backend, prompts[prompt_key], prior_label)
            directions = directions_by_label_mode[label_mode]
            random_dirs = random_dirs_by_label_mode[label_mode]
            for alpha in alphas:
                steer_b, steer_logits = _score_many_add_b(
                    backend=backend,
                    prompt=prompts[prompt_key],
                    prior_label=prior_label,
                    additions=_additions_for(directions, alpha=alpha, task_sign=sign),
                )
                random_b, random_logits = _score_many_add_b(
                    backend=backend,
                    prompt=prompts[prompt_key],
                    prior_label=prior_label,
                    additions=_additions_for(random_dirs, alpha=alpha, task_sign=sign),
                )
                steer_gain = _target_gain(base_b, steer_b, target)
                random_gain = _target_gain(base_b, random_b, target)
                output_rows.append(
                    {
                        "phase": phase,
                        "render_id": row["render_id"],
                        "sample_id": row["sample_id"],
                        "fact_id": row["fact_id"],
                        "subject_id": row["subject_id"],
                        "split": row["split"],
                        "template_family": row["template_family"],
                        "task_name": task_name,
                        "prompt_key": prompt_key,
                        "target": target,
                        "alpha": alpha,
                        "direction_mode": direction_mode,
                        "top_k": top_k,
                        "num_components": len(directions),
                        "B_base": base_b,
                        "B_steered": steer_b,
                        "B_random": random_b,
                        "target_gain": steer_gain,
                        "random_target_gain": random_gain,
                        "gain_minus_random": steer_gain - random_gain,
                        "base_target_correct": _target_correct(base_b, target),
                        "steered_target_correct": _target_correct(steer_b, target),
                        "random_target_correct": _target_correct(random_b, target),
                        "regular_margin_drop": base_b - steer_b if target == "preserve" else 0.0,
                        "logits": {
                            "base": base_logits,
                            "steered": steer_logits,
                            "random": random_logits,
                        },
                    }
                )
    return output_rows


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(vals) if vals else 0.0


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def _summarize_group(group: str, rows: list[dict]) -> dict[str, object]:
    first = rows[0]
    return {
        "group": group,
        "phase": first["phase"],
        "split": first["split"],
        "task_name": first["task_name"],
        "target": first["target"],
        "alpha": first["alpha"],
        "direction_mode": first["direction_mode"],
        "top_k": first["top_k"],
        "n": len(rows),
        "mean_B_base": _mean(rows, "B_base"),
        "mean_B_steered": _mean(rows, "B_steered"),
        "mean_B_random": _mean(rows, "B_random"),
        "mean_target_gain": _mean(rows, "target_gain"),
        "mean_random_target_gain": _mean(rows, "random_target_gain"),
        "mean_gain_minus_random": _mean(rows, "gain_minus_random"),
        "mean_regular_margin_drop": _mean(rows, "regular_margin_drop"),
        "base_target_acc": _rate(rows, "base_target_correct"),
        "steered_target_acc": _rate(rows, "steered_target_correct"),
        "random_target_acc": _rate(rows, "random_target_correct"),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["split"], row["task_name"], row["alpha"], row["direction_mode"], row["top_k"])
        groups[key].append(row)
    return [
        _summarize_group(
            f"phase:{phase}:split:{split}:task:{task_name}:alpha:{alpha}:mode:{direction_mode}:topk:{top_k}",
            groups[(phase, split, task_name, alpha, direction_mode, top_k)],
        )
        for phase, split, task_name, alpha, direction_mode, top_k in sorted(groups)
    ]


def _choose_alpha(validation_rows: list[dict], alphas: list[float]) -> float:
    by_alpha: dict[float, list[dict]] = defaultdict(list)
    for row in validation_rows:
        by_alpha[float(row["alpha"])].append(row)

    scored: list[tuple[float, float]] = []
    for alpha in alphas:
        rows = by_alpha[alpha]
        doc_rows = [r for r in rows if r["task_name"].startswith("doc_following")]
        false_rows = [r for r in rows if r["task_name"].startswith("false_")]
        regular_rows = [r for r in rows if r["task_name"].startswith("regular_qa")]
        regular_drop = mean(max(0.0, float(r["regular_margin_drop"])) for r in regular_rows) if regular_rows else 0.0
        regular_acc_drop = (
            _rate(regular_rows, "base_target_correct") - _rate(regular_rows, "steered_target_correct")
            if regular_rows
            else 0.0
        )
        objective = (
            _mean(doc_rows, "target_gain")
            + _mean(false_rows, "target_gain")
            - regular_drop
            - 5.0 * max(0.0, regular_acc_drop)
        )
        scored.append((objective, alpha))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def main() -> None:
    args = parse_args()
    selection_pairs = _parse_csv(args.selection_pairs)
    unknown_pairs = sorted(set(selection_pairs) - set(PAIR_SPECS))
    if unknown_pairs:
        raise ValueError(f"Unknown selection pairs: {unknown_pairs}")
    task_names = _parse_csv(args.tasks)
    unknown_tasks = sorted(set(task_names) - set(TASK_SPECS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    alphas = _parse_float_csv(args.alphas)
    top_k_values = _parse_int_csv(args.top_k_components)

    summary_rows = _load_summary_rows(args.component_summary_csv)
    ranked_components = _rank_components(
        summary_rows=summary_rows,
        selection_pairs=selection_pairs,
    )
    if top_k_values[-1] > len(ranked_components):
        raise ValueError(
            f"Requested top_k={top_k_values[-1]} but only {len(ranked_components)} ranked components are available."
        )
    selected_components = ranked_components[: top_k_values[-1]]

    all_rows = load_jsonl(args.rendered_jsonl)
    train_rows = _select_rows(
        all_rows,
        splits=_parse_csv_set(args.train_splits),
        templates=_parse_csv_set(args.train_templates),
        max_rows=args.max_train_rows,
    )
    validation_rows = _select_rows(
        all_rows,
        splits=_parse_csv_set(args.validation_splits),
        templates=_parse_csv_set(args.validation_templates),
        max_rows=args.max_validation_rows,
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
    if args.direction_mode == "shared":
        shared_directions = _learn_component_directions(
            backend=backend,
            train_rows=train_rows,
            components=selected_components,
            selection_pairs=selection_pairs,
            normalize_diffs=args.normalize_diffs,
        )
        directions_by_label_mode = {"normal": shared_directions, "swapped": shared_directions}
    else:
        directions_by_label_mode = _learn_component_directions_by_label_mode(
            backend=backend,
            train_rows=train_rows,
            components=selected_components,
            selection_pairs=selection_pairs,
            normalize_diffs=args.normalize_diffs,
        )
    random_dirs_by_label_mode = _make_random_directions_by_label_mode(
        backend=backend,
        directions_by_label_mode=directions_by_label_mode,
        seed=args.seed,
    )

    output_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for top_k in top_k_values:
        selected_specs = {
            (int(component["layer_idx"]), str(component["component_type"]))
            for component in ranked_components[:top_k]
        }
        scoped_directions_by_label_mode = {
            label_mode: {
                spec: direction
                for spec, direction in directions.items()
                if spec in selected_specs
            }
            for label_mode, directions in directions_by_label_mode.items()
        }
        scoped_random_dirs_by_label_mode = {
            label_mode: {
                spec: direction
                for spec, direction in directions.items()
                if spec in selected_specs
            }
            for label_mode, directions in random_dirs_by_label_mode.items()
        }
        validation_scored = _score_split(
            backend=backend,
            rows=validation_rows,
            task_names=task_names,
            directions_by_label_mode=scoped_directions_by_label_mode,
            random_dirs_by_label_mode=scoped_random_dirs_by_label_mode,
            alphas=alphas,
            phase="validation_select",
            direction_mode=args.direction_mode,
            top_k=top_k,
        )
        selected_alpha = _choose_alpha(validation_scored, alphas=alphas)
        test_scored = _score_split(
            backend=backend,
            rows=eval_rows,
            task_names=task_names,
            directions_by_label_mode=scoped_directions_by_label_mode,
            random_dirs_by_label_mode=scoped_random_dirs_by_label_mode,
            alphas=[selected_alpha],
            phase="heldout_eval",
            direction_mode=args.direction_mode,
            top_k=top_k,
        )
        output_rows.extend(validation_scored)
        output_rows.extend(test_scored)
        manifest_rows.extend(
            {
                **component,
                "top_k": top_k,
                "selected_alpha": selected_alpha,
                "train_splits": args.train_splits,
                "validation_splits": args.validation_splits,
                "eval_splits": args.eval_splits,
                "selection_pairs": args.selection_pairs,
                "direction_mode": args.direction_mode,
                "normalize_diffs": args.normalize_diffs,
            }
            for component in ranked_components[:top_k]
        )
    dump_jsonl(args.out_rows_jsonl, output_rows)
    dump_csv(args.out_summary_csv, _summarize(output_rows))
    dump_csv(args.out_manifest_csv, manifest_rows)
    print(
        f"[score-steering-utility] train_rows={len(train_rows)} validation_rows={len(validation_rows)} "
        f"eval_rows={len(eval_rows)} ranked_components={len(ranked_components)} top_k_values={top_k_values}"
    )


if __name__ == "__main__":
    main()
