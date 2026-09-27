from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from screscomp.data import dump_csv


E1_SELECTION_PAIRS = ("e1_content", "e1_content_swapped")
E2_PAIRS = ("e2_task", "e2_task_swapped", "e2_source", "e2_source_swapped")
BEHAVIOR_METRICS = (
    "mean_doc_support_gap",
    "mean_e2_task_delta_actual_minus_according",
    "mean_e2_source_delta_user_minus_doc",
    "mean_interaction_did",
    "frac_interaction_did_positive",
)
STEERING_METRICS = (
    "alpha",
    "mean_target_gain",
    "mean_gain_minus_random",
    "base_target_acc",
    "steered_target_acc",
    "random_target_acc",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two large-v1 model runs.")
    p.add_argument("--run_a_dir", type=Path, required=True)
    p.add_argument("--run_b_dir", type=Path, required=True)
    p.add_argument("--label_a", type=str, required=True)
    p.add_argument("--label_b", type=str, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument(
        "--compare_top_k",
        type=int,
        default=4,
        help="Use the top-k E1-ranked discovery components from each model when summarizing selected-component E1/E2 scores.",
    )
    return p.parse_args()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _value_or_blank(value):
    return "" if value is None else value


def _delta_or_blank(a_value, b_value):
    if a_value is None or b_value is None:
        return ""
    return float(b_value) - float(a_value)


def _rank_e1_components(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary_rows:
        if row.get("split") != "discovery":
            continue
        if row.get("pair_name") not in E1_SELECTION_PAIRS:
            continue
        grouped[(int(row["layer_idx"]), row["component_type"])].append(row)

    ranked: list[dict[str, object]] = []
    for (layer_idx, component_type), rows in grouped.items():
        ranked.append(
            {
                "layer_idx": layer_idx,
                "component_type": component_type,
                "component_id": f"L{layer_idx}.{component_type}",
                "selection_mean_min_CI": mean(_as_float(row, "mean_min_CI") for row in rows),
                "selection_mean_C_plus_I": mean(_as_float(row, "mean_C_plus_I") for row in rows),
                "selection_frac_both_positive": mean(_as_float(row, "frac_both_positive") for row in rows),
            }
        )

    ranked.sort(
        key=lambda row: (
            -float(row["selection_mean_min_CI"]),
            -float(row["selection_frac_both_positive"]),
            -float(row["selection_mean_C_plus_I"]),
            int(row["layer_idx"]),
            str(row["component_type"]),
        )
    )
    return ranked


def _top_component_ids(summary_rows: list[dict[str, str]], top_k: int) -> list[str]:
    ranked = _rank_e1_components(summary_rows)
    return [str(row["component_id"]) for row in ranked[:top_k]]


def _component_metric_average(
    summary_rows: list[dict[str, str]],
    component_ids: list[str],
    split: str,
    pair_names: tuple[str, ...],
    metric: str,
) -> float | None:
    subset = [
        row
        for row in summary_rows
        if row.get("split") == split
        and row.get("pair_name") in pair_names
        and row.get("component_id") in component_ids
    ]
    if not subset:
        return None
    return mean(_as_float(row, metric) for row in subset)


def _find_behavior_row(rows: list[dict[str, str]], group: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("group") == group:
            return row
    return None


def _find_steering_row(rows: list[dict[str, str]], top_k: int, task_name: str) -> dict[str, str] | None:
    target_top_k = str(top_k)
    for row in rows:
        if row.get("phase") != "heldout_eval":
            continue
        if row.get("split") != "test":
            continue
        if row.get("task_name") != task_name:
            continue
        if row.get("top_k") != target_top_k:
            continue
        return row
    return None


def _compare_behavior(
    out_rows: list[dict[str, object]],
    behavior_a: list[dict[str, str]],
    behavior_b: list[dict[str, str]],
    label_a: str,
    label_b: str,
) -> None:
    groups: list[str] = ["all"]
    groups.extend(sorted(row["group"] for row in behavior_a if row.get("group", "").startswith("relation:")))
    for group in groups:
        row_a = _find_behavior_row(behavior_a, group)
        row_b = _find_behavior_row(behavior_b, group)
        if row_a is None or row_b is None:
            continue
        for metric in BEHAVIOR_METRICS:
            a_value = _as_float(row_a, metric)
            b_value = _as_float(row_b, metric)
            out_rows.append(
                {
                    "section": "behavior",
                    "group": group,
                    "metric": metric,
                    "label_a": label_a,
                    "label_b": label_b,
                    "value_a": a_value,
                    "value_b": b_value,
                    "delta_b_minus_a": b_value - a_value,
                    "notes": "",
                }
            )


def _compare_e1_selected(
    out_rows: list[dict[str, object]],
    e1_a: list[dict[str, str]],
    e1_b: list[dict[str, str]],
    label_a: str,
    label_b: str,
    compare_top_k: int,
) -> None:
    component_ids_a = _top_component_ids(e1_a, compare_top_k)
    component_ids_b = _top_component_ids(e1_b, compare_top_k)
    out_rows.append(
        {
            "section": "e1_selected_components",
            "group": f"topk:{compare_top_k}",
            "metric": "component_ids",
            "label_a": label_a,
            "label_b": label_b,
            "value_a": ",".join(component_ids_a),
            "value_b": ",".join(component_ids_b),
            "delta_b_minus_a": "",
            "notes": "Discovery-ranked by mean_min_CI.",
        }
    )
    for split in ("discovery", "test"):
        for metric in ("mean_min_CI", "mean_C_t", "mean_I_t", "frac_both_positive"):
            a_value = _component_metric_average(e1_a, component_ids_a, split, E1_SELECTION_PAIRS, metric)
            b_value = _component_metric_average(e1_b, component_ids_b, split, E1_SELECTION_PAIRS, metric)
            out_rows.append(
                {
                    "section": "e1_selected_components",
                    "group": f"topk:{compare_top_k}:split:{split}",
                    "metric": metric,
                    "label_a": label_a,
                    "label_b": label_b,
                    "value_a": _value_or_blank(a_value),
                    "value_b": _value_or_blank(b_value),
                    "delta_b_minus_a": _delta_or_blank(a_value, b_value),
                    "notes": "",
                }
            )


def _compare_e2_selected(
    out_rows: list[dict[str, object]],
    e1_a: list[dict[str, str]],
    e1_b: list[dict[str, str]],
    e2_a: list[dict[str, str]],
    e2_b: list[dict[str, str]],
    label_a: str,
    label_b: str,
    compare_top_k: int,
) -> None:
    component_ids_a = _top_component_ids(e1_a, compare_top_k)
    component_ids_b = _top_component_ids(e1_b, compare_top_k)
    for split in ("discovery", "validation", "test"):
        for pair_name in E2_PAIRS:
            for metric in ("mean_min_CI", "mean_C_t", "mean_I_t", "frac_both_positive"):
                a_value = _component_metric_average(e2_a, component_ids_a, split, (pair_name,), metric)
                b_value = _component_metric_average(e2_b, component_ids_b, split, (pair_name,), metric)
                out_rows.append(
                    {
                        "section": "e2_selected_components",
                        "group": f"topk:{compare_top_k}:split:{split}:pair:{pair_name}",
                        "metric": metric,
                        "label_a": label_a,
                        "label_b": label_b,
                        "value_a": _value_or_blank(a_value),
                        "value_b": _value_or_blank(b_value),
                        "delta_b_minus_a": _delta_or_blank(a_value, b_value),
                        "notes": "Averaged over each model's own discovery-selected E1 components.",
                    }
                )


def _compare_steering(
    out_rows: list[dict[str, object]],
    steering_a: list[dict[str, str]],
    steering_b: list[dict[str, str]],
    label_a: str,
    label_b: str,
) -> None:
    steering_tasks = sorted({row["task_name"] for row in steering_a if row.get("phase") == "heldout_eval"})
    steering_top_ks = sorted({int(row["top_k"]) for row in steering_a if row.get("phase") == "heldout_eval"})
    for top_k in steering_top_ks:
        for task_name in steering_tasks:
            row_a = _find_steering_row(steering_a, top_k, task_name)
            row_b = _find_steering_row(steering_b, top_k, task_name)
            if row_a is None or row_b is None:
                continue
            for metric in STEERING_METRICS:
                a_value = _as_float(row_a, metric)
                b_value = _as_float(row_b, metric)
                out_rows.append(
                    {
                        "section": "steering",
                        "group": f"topk:{top_k}:task:{task_name}",
                        "metric": metric,
                        "label_a": label_a,
                        "label_b": label_b,
                        "value_a": a_value,
                        "value_b": b_value,
                        "delta_b_minus_a": b_value - a_value,
                        "notes": "Held-out eval / test.",
                    }
                )


def main() -> None:
    args = parse_args()
    run_a = args.run_a_dir
    run_b = args.run_b_dir

    behavior_a = _load_csv(run_a / "behavior_summary.csv")
    behavior_b = _load_csv(run_b / "behavior_summary.csv")
    e1_a = _load_csv(run_a / "component_competition_e1_summary.csv")
    e1_b = _load_csv(run_b / "component_competition_e1_summary.csv")
    e2_a = _load_csv(run_a / "component_competition_e2_summary.csv")
    e2_b = _load_csv(run_b / "component_competition_e2_summary.csv")
    steering_a = _load_csv(run_a / "steering_utility_summary.csv")
    steering_b = _load_csv(run_b / "steering_utility_summary.csv")

    out_rows: list[dict[str, object]] = []
    _compare_behavior(out_rows, behavior_a, behavior_b, args.label_a, args.label_b)
    _compare_e1_selected(out_rows, e1_a, e1_b, args.label_a, args.label_b, args.compare_top_k)
    _compare_e2_selected(out_rows, e1_a, e1_b, e2_a, e2_b, args.label_a, args.label_b, args.compare_top_k)
    _compare_steering(out_rows, steering_a, steering_b, args.label_a, args.label_b)

    dump_csv(args.out_csv, out_rows)
    print(
        f"[compare-model-runs] rows={len(out_rows)} compare_top_k={args.compare_top_k} "
        f"run_a={args.run_a_dir} run_b={args.run_b_dir}"
    )


if __name__ == "__main__":
    main()
