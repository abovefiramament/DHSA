from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from screscomp.cli.score_component_competition import _parse_csv, _parse_csv_set, _select_rows
from screscomp.cli.score_steering_utility import (
    TASK_SPECS,
    _choose_alpha,
    _learn_component_directions,
    _learn_component_directions_by_label_mode,
    _load_summary_rows,
    _make_random_directions_by_label_mode,
    _parse_float_csv,
    _parse_int_csv,
    _rank_components,
    _score_split,
    _summarize,
)
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Learn frozen steering directions on a controlled screscomp discovery split "
            "and evaluate them on an external rendered A/B dataset."
        )
    )
    p.add_argument("--train_rendered_jsonl", type=Path, required=True)
    p.add_argument("--eval_rendered_jsonl", type=Path, required=True)
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
    p.add_argument("--eval_splits", type=str, default="external_test,test")
    p.add_argument("--train_templates", type=str, default="")
    p.add_argument("--validation_templates", type=str, default="")
    p.add_argument("--eval_templates", type=str, default="")
    p.add_argument("--selection_pairs", type=str, default="e1_content,e1_content_swapped")
    p.add_argument(
        "--tasks",
        type=str,
        default="doc_following,doc_following_swapped,regular_qa,regular_qa_swapped",
    )
    p.add_argument("--alphas", type=str, default="0.25,0.5,1.0,2.0")
    p.add_argument("--top_k_components", type=str, default="1,2,4,8")
    p.add_argument(
        "--direction_mode",
        type=str,
        default="label_conditional",
        choices=["shared", "label_conditional"],
    )
    p.add_argument("--normalize_diffs", action="store_true")
    p.add_argument("--max_train_rows", type=int, default=None)
    p.add_argument("--max_validation_rows", type=int, default=None)
    p.add_argument("--max_eval_rows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _scoped(directions_by_label_mode, selected_specs: set[tuple[int, str]]):
    return {
        label_mode: {spec: direction for spec, direction in directions.items() if spec in selected_specs}
        for label_mode, directions in directions_by_label_mode.items()
    }


def main() -> None:
    args = parse_args()
    selection_pairs = _parse_csv(args.selection_pairs)
    task_names = _parse_csv(args.tasks)
    unknown_tasks = sorted(set(task_names) - set(TASK_SPECS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    alphas = _parse_float_csv(args.alphas)
    top_k_values = _parse_int_csv(args.top_k_components)

    ranked_components = _rank_components(
        summary_rows=_load_summary_rows(args.component_summary_csv),
        selection_pairs=selection_pairs,
    )
    if top_k_values[-1] > len(ranked_components):
        raise ValueError(
            f"Requested top_k={top_k_values[-1]} but only {len(ranked_components)} ranked components are available."
        )
    selected_components = ranked_components[: top_k_values[-1]]

    train_source_rows = load_jsonl(args.train_rendered_jsonl)
    eval_source_rows = load_jsonl(args.eval_rendered_jsonl)
    train_rows = _select_rows(
        train_source_rows,
        splits=_parse_csv_set(args.train_splits),
        templates=_parse_csv_set(args.train_templates),
        max_rows=args.max_train_rows,
    )
    validation_rows = _select_rows(
        train_source_rows,
        splits=_parse_csv_set(args.validation_splits),
        templates=_parse_csv_set(args.validation_templates),
        max_rows=args.max_validation_rows,
    )
    eval_rows = _select_rows(
        eval_source_rows,
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
        scoped_directions = _scoped(directions_by_label_mode, selected_specs)
        scoped_random_dirs = _scoped(random_dirs_by_label_mode, selected_specs)
        validation_scored = _score_split(
            backend=backend,
            rows=validation_rows,
            task_names=task_names,
            directions_by_label_mode=scoped_directions,
            random_dirs_by_label_mode=scoped_random_dirs,
            alphas=alphas,
            phase="validation_select",
            direction_mode=args.direction_mode,
            top_k=top_k,
        )
        selected_alpha = _choose_alpha(validation_scored, alphas=alphas)
        external_scored = _score_split(
            backend=backend,
            rows=eval_rows,
            task_names=task_names,
            directions_by_label_mode=scoped_directions,
            random_dirs_by_label_mode=scoped_random_dirs,
            alphas=[selected_alpha],
            phase="external_eval",
            direction_mode=args.direction_mode,
            top_k=top_k,
        )
        output_rows.extend(validation_scored)
        output_rows.extend(external_scored)
        manifest_rows.extend(
            {
                **component,
                "top_k": top_k,
                "selected_alpha": selected_alpha,
                "train_rendered_jsonl": str(args.train_rendered_jsonl),
                "eval_rendered_jsonl": str(args.eval_rendered_jsonl),
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
        f"[score-external-steering-utility] train_rows={len(train_rows)} "
        f"validation_rows={len(validation_rows)} eval_rows={len(eval_rows)} "
        f"top_k_values={top_k_values}"
    )


if __name__ == "__main__":
    main()
