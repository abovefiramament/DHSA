from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_ckplug_factor_component_scan import (
    _addition,
    _generation_prompt,
    _output_metrics,
    _score_row,
)
from screscomp.cli.score_ckplug_generation import _classify_generation, _parse_stop_strings
from screscomp.data import dump_csv, dump_jsonl, load_csv, load_jsonl
from screscomp.modeling import TransformersABBackend
from screscomp.rcm.defaults import objective_preset
from screscomp.rcm.objective import weighted_objective


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_map(raw: str) -> dict[str, float]:
    output: dict[str, float] = {}
    if not raw:
        return output
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected metric=weight item: {item}")
        key, value = item.split("=", 1)
        output[key.strip()] = float(value.strip())
    return output


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _edge_rows(path: Path, max_edges: int | None) -> list[dict[str, Any]]:
    rows = load_csv(path)
    if max_edges is not None:
        rows = rows[:max_edges]
    return rows


def _operation_alpha(sign: str, operation: str, base_alpha: float) -> float | None:
    if operation == "target":
        return base_alpha if sign == "up" else -base_alpha
    if operation == "anti_target":
        return -base_alpha if sign == "up" else base_alpha
    if operation == "amplify":
        return base_alpha
    if operation == "suppress":
        return -base_alpha
    if operation == "amplify_up":
        return base_alpha if sign == "up" else None
    if operation == "suppress_up":
        return -base_alpha if sign == "up" else None
    if operation == "amplify_down":
        return base_alpha if sign == "down" else None
    if operation == "suppress_down":
        return -base_alpha if sign == "down" else None
    raise ValueError(f"Unsupported operation: {operation}")


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(bool(row.get(key))) for row in rows) if rows else 0.0


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0.0) or 0.0) for row in rows) if rows else 0.0


def _summarize(rows: list[dict[str, Any]], objective_terms: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["edge_id"]), str(row["operation"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (edge_id, operation), group in sorted(groups.items()):
        first = group[0]
        summary = {
            "edge_id": edge_id,
            "operation": operation,
            "axis": first["axis"],
            "component_id": first["component_id"],
            "layer_idx": first["layer_idx"],
            "component_type": first["component_type"],
            "sign": first["sign"],
            "timing": first["timing"],
            "alpha": first["alpha"],
            "effective_alpha": first["effective_alpha"],
            "n": len(group),
            "context_only": _mean_float(group, "context_only"),
            "prior_only": _mean_float(group, "prior_only"),
            "both": _mean_float(group, "both"),
            "neither": _mean_float(group, "neither"),
            "cf_hit": _mean_bool(group, "cf_hit"),
            "orig_hit": _mean_bool(group, "orig_hit"),
            "cf_em": _mean_bool(group, "cf_em"),
            "first_answer_cf_em": _mean_bool(group, "first_answer_cf_em"),
            "short_output": _mean_bool(group, "short_output"),
            "output_chars": _mean_float(group, "output_chars"),
            "source_score": _mean_float(group, "source_score"),
            "form_score": _mean_float(group, "form_score"),
            "commitment_score": _mean_float(group, "commitment_score"),
            "prior_suppression_score": _mean_float(group, "prior_suppression_score"),
            "joint_score": _mean_float(group, "joint_score"),
        }
        if objective_terms:
            summary["objective_score"] = weighted_objective(summary, objective_terms)
        output.append(summary)
    return output


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate timing-specific RCM candidate edges on validation generations. "
            "Supports positive and negative components via target/anti-target/amplify/suppress operations."
        )
    )
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--candidate_edges_csv", type=Path, required=True)
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
    p.add_argument("--operations", default="target,anti_target")
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--max_edges", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--stop_strings", type=str, default="Q:")
    p.add_argument(
        "--objective_terms",
        default="",
        help="Optional metric=weight terms used to add objective_score to the summary.",
    )
    p.add_argument(
        "--objective_preset",
        default="clean_composite",
        help="Objective preset used when --objective_terms is empty.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_rows = load_jsonl(args.eval_jsonl)
    if args.max_rows is not None:
        eval_rows = eval_rows[: args.max_rows]
    edges = _edge_rows(args.candidate_edges_csv, args.max_edges)
    operations = _parse_csv(args.operations)
    stop_strings = _parse_stop_strings(args.stop_strings)
    objective_terms = _parse_float_map(args.objective_terms) or objective_preset(args.objective_preset)

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    output_rows: list[dict[str, Any]] = []
    work_items = [(edge, operation) for edge in edges for operation in operations]
    for edge, operation in tqdm(work_items, desc="candidate edge eval"):
        axis = str(edge["axis"])
        sign = str(edge["sign"])
        timing = str(edge["timing"])
        component_spec = (_int(edge, "layer_idx"), str(edge["component_type"]))
        base_alpha = _float(edge, "alpha", 1.0)
        effective_alpha = _operation_alpha(sign, operation, base_alpha)
        if effective_alpha is None:
            continue
        for row in eval_rows:
            addition = _addition(
                backend=backend,
                row=row,
                component_spec=component_spec,
                axis=axis,
                alpha=effective_alpha,
            )
            prediction = backend.generate_with_component_last_token_add_many(
                _generation_prompt(row, axis),
                additions=[{**addition, "apply_mode": timing}],
                max_new_tokens=args.max_new_tokens,
                stop_strings=stop_strings,
                apply_mode=timing,
            )
            classified = _classify_generation(prediction, row["orig_answers"], row["cf_answers"])
            metrics = {**classified, **_output_metrics(prediction, row["cf_answers"])}
            axis_scores = _score_row(metrics)
            output_rows.append(
                {
                    "edge_id": edge["edge_id"],
                    "operation": operation,
                    "sample_id": row["sample_id"],
                    "axis": axis,
                    "component_id": edge["component_id"],
                    "layer_idx": component_spec[0],
                    "component_type": component_spec[1],
                    "sign": sign,
                    "timing": timing,
                    "alpha": base_alpha,
                    "effective_alpha": effective_alpha,
                    "prediction": prediction,
                    "context_only": float(metrics.get("outcome") == "context_only"),
                    "prior_only": float(metrics.get("outcome") == "prior_only"),
                    "both": float(metrics.get("outcome") == "both"),
                    "neither": float(metrics.get("outcome") == "neither"),
                    **metrics,
                    **axis_scores,
                }
            )

    dump_jsonl(args.out_rows_jsonl, output_rows)
    dump_csv(args.out_summary_csv, _summarize(output_rows, objective_terms))
    print(
        f"[rcm-eval-candidate-edges] eval_rows={len(eval_rows)} edges={len(edges)} "
        f"operations={','.join(operations)} rows={len(output_rows)} out={args.out_summary_csv}"
    )


if __name__ == "__main__":
    main()
