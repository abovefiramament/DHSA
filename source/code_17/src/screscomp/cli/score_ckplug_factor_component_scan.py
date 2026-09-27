from __future__ import annotations

import argparse
import re
import string
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_ckplug_generation import (
    _classify_generation,
    _no_prior_memory_prompt,
    _parse_stop_strings,
    _prior_memory_prompt,
)
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Task-defined ConFiQA component scan for RCM factor edges. "
            "Each axis defines a target attractor prompt, a contrast attractor prompt, "
            "and a factor-specific validation metric."
        )
    )
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--axis", choices=["source_identity", "commitment", "form", "prior_suppression"], required=True)
    p.add_argument(
        "--direction",
        choices=["forward", "reverse"],
        default="forward",
        help=(
            "Directional EventSpec to scan. `forward` is the canonical target-vs-contrast "
            "edge; `reverse` swaps target and contrast and ranks components from scratch."
        ),
    )
    p.add_argument(
        "--source_contrast_kind",
        choices=["strong_no_rag", "objective_prior"],
        default="strong_no_rag",
        help="Source-axis contrast attractor used for source_identity scans.",
    )
    p.add_argument(
        "--generation_prompt_key",
        default="strong_rag",
        help="Prompt key where the single-component write is evaluated.",
    )
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
    p.add_argument("--max_rows", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--stop_strings", type=str, default="Q:")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--apply_mode", default="prefill")
    p.add_argument("--include_baseline_rows", action="store_true")
    return p.parse_args()


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_PREFIX_PATTERNS = [
    r"^answer\s*[:\-]\s*",
    r"^the answer is\s+",
    r"^the correct answer is\s+",
    r"^it is\s+",
    r"^it's\s+",
]


def _first_answer_span(prediction: str) -> str:
    text = str(prediction).strip()
    if not text:
        return ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = first_line.strip(" \t\"'`")
    lowered = first_line.lower()
    for pattern in _PREFIX_PATTERNS:
        lowered_new = re.sub(pattern, "", lowered, flags=re.IGNORECASE).strip()
        if lowered_new != lowered:
            first_line = re.sub(pattern, "", first_line, flags=re.IGNORECASE).strip()
            lowered = lowered_new
    first_line = re.split(r"\s+(?:because|which|who|that|and it|and was)\b", first_line, maxsplit=1)[0].strip()
    return first_line.strip(" \t.;,")


def _exact_hit(text: str, answers: list[str]) -> bool:
    norm = _normalize_answer(text)
    return any(norm == _normalize_answer(answer) for answer in answers if _normalize_answer(answer))


def _output_metrics(prediction: str, cf_answers: list[str]) -> dict[str, Any]:
    first_span = _first_answer_span(prediction)
    output_chars = len(prediction)
    first_answer_cf_em = _exact_hit(first_span, cf_answers)
    short_output = output_chars <= 48
    return {
        "first_answer_span": first_span,
        "first_answer_cf_em": first_answer_cf_em,
        "short_output": short_output,
        "output_chars": output_chars,
    }


def _all_component_specs(backend: TransformersABBackend) -> list[tuple[int, str]]:
    num_layers = len(backend._model.model.layers)
    specs: list[tuple[int, str]] = []
    for layer_idx in range(num_layers):
        specs.append((layer_idx, "attn"))
        specs.append((layer_idx, "mlp"))
    return specs


def _commitment_contrast_prompt(row: dict[str, Any]) -> str:
    return (
        "Read the given information and answer the question. If there may be conflicting or alternative answers, "
        "mention the alternatives and briefly explain the conflict.\n\n"
        f"{row['context']}\n\nQ: {row['question']}\nA:"
    )


def _prior_objective_prompt(row: dict[str, Any]) -> str:
    return (
        "Read the given information, but answer from your own prior knowledge rather than relying on "
        "the given information. Return only the short answer.\n\n"
        f"{row['context']}\n\nQ: {row['question']}\nA:"
    )


def _target_and_contrast_prompts(
    row: dict[str, Any],
    axis: str,
    direction: str,
    *,
    source_contrast_kind: str,
) -> tuple[str, str, str, str, str]:
    prompts = row["prompts"]
    if axis == "source_identity":
        contrast = (
            _prior_objective_prompt(row)
            if source_contrast_kind == "objective_prior"
            else prompts["strong_no_rag"]
        )
        forward = (prompts["strong_rag"], contrast, "source_context_vs_prior", "context", "prior")
    elif axis == "form":
        forward = (prompts["strong_rag"], prompts["verbose_rag"], "form_short_vs_verbose", "short", "verbose")
    elif axis == "commitment":
        forward = (
            prompts["strong_rag"],
            _commitment_contrast_prompt(row),
            "commitment_single_vs_mixed",
            "single",
            "mixed",
        )
    elif axis == "prior_suppression":
        forward = (
            _no_prior_memory_prompt(row),
            _prior_memory_prompt(row),
            "prior_suppression_no_prior_vs_memory",
            "no_prior",
            "memory",
        )
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    target, contrast, pair_name, target_name, contrast_name = forward
    if direction == "forward":
        return target, contrast, pair_name, target_name, contrast_name
    if direction == "reverse":
        return (
            contrast,
            target,
            f"{axis}_{contrast_name}_vs_{target_name}",
            contrast_name,
            target_name,
        )
    raise ValueError(f"Unsupported direction: {direction}")


def _generation_prompt(row: dict[str, Any], prompt_key: str) -> str:
    prompts = row["prompts"]
    if prompt_key not in prompts:
        raise ValueError(f"Missing generation prompt key `{prompt_key}`. Available keys: {sorted(prompts)}")
    return prompts[prompt_key]


def _addition(
    backend: TransformersABBackend,
    row: dict[str, Any],
    component_spec: tuple[int, str],
    axis: str,
    direction: str,
    alpha: float,
    source_contrast_kind: str,
) -> dict[str, Any]:
    target_prompt, contrast_prompt, _, _, _ = _target_and_contrast_prompts(
        row,
        axis,
        direction,
        source_contrast_kind=source_contrast_kind,
    )
    target_acts = backend.capture_component_last_token(target_prompt, [component_spec])
    contrast_acts = backend.capture_component_last_token(contrast_prompt, [component_spec])
    layer_idx, component_type = component_spec
    return {
        "layer_idx": layer_idx,
        "component_type": component_type,
        "direction": target_acts[component_spec] - contrast_acts[component_spec],
        "alpha": alpha,
    }


def _event_pair_scores(row: dict[str, Any], axis: str, direction: str) -> tuple[float, float]:
    context_only = float(row.get("outcome") == "context_only")
    prior_only = float(row.get("outcome") == "prior_only")
    both = float(row.get("outcome") == "both")
    neither = float(row.get("outcome") == "neither")
    cf_hit = float(bool(row.get("cf_hit")))
    orig_hit = float(bool(row.get("orig_hit")))
    po = float(bool(row.get("orig_hit")))
    first_cf = float(bool(row.get("first_answer_cf_em")))
    short = float(bool(row.get("short_output")))
    chars = float(row.get("output_chars", 0.0) or 0.0)
    clipped_chars = min(chars, 160.0)
    if axis == "source_identity":
        forward = (context_only, prior_only)
    elif axis == "prior_suppression":
        forward = (context_only + cf_hit, prior_only + orig_hit + 0.25 * neither)
    elif axis == "form":
        short_event = first_cf + short - clipped_chars / 200.0
        verbose_event = cf_hit + (1.0 - short) + clipped_chars / 200.0
        forward = (short_event, verbose_event)
    elif axis == "commitment":
        single_event = context_only + first_cf - clipped_chars / 400.0
        mixed_event = both + 0.5 * neither + 0.25 * (1.0 - short) + clipped_chars / 400.0
        forward = (single_event, mixed_event)
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return forward if direction == "forward" else (forward[1], forward[0])


def _score_row(row: dict[str, Any], *, axis: str, direction: str) -> dict[str, float]:
    context_only = float(row.get("outcome") == "context_only")
    prior_only = float(row.get("outcome") == "prior_only")
    both = float(row.get("outcome") == "both")
    neither = float(row.get("outcome") == "neither")
    po = float(bool(row.get("orig_hit")))
    first_cf = float(bool(row.get("first_answer_cf_em")))
    short = float(bool(row.get("short_output")))
    chars = float(row.get("output_chars", 0.0) or 0.0)
    target_score, contrast_score = _event_pair_scores(row, axis, direction)
    return {
        "event_target_score": target_score,
        "event_contrast_score": contrast_score,
        "event_margin_score": target_score - contrast_score,
        "source_score": context_only - prior_only - po,
        "prior_suppression_score": context_only - 2.0 * prior_only - po - 0.25 * neither,
        "commitment_score": context_only + first_cf - both - neither - chars / 400.0,
        "form_score": first_cf + short - chars / 200.0,
        "joint_score": context_only + first_cf - po - neither - chars / 400.0,
    }


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(bool(row.get(key))) for row in rows) if rows else 0.0


def _mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0.0) or 0.0) for row in rows) if rows else 0.0


def _selection_metric(axis: str) -> str:
    if axis in {"source_identity", "prior_suppression", "commitment", "form"}:
        return "event_margin_score"
    raise ValueError(f"Unsupported axis: {axis}")


def _summary_rows(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    direction: str,
    pair_name: str,
    target_attractor: str,
    contrast_attractor: str,
) -> list[dict[str, Any]]:
    by_component: dict[tuple[int, str], list[dict[str, Any]]] = {}
    baseline_by_component: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        spec = (int(row["layer_idx"]), str(row["component_type"]))
        if row["method"] == "baseline":
            baseline_by_component.setdefault(spec, []).append(row)
        else:
            by_component.setdefault(spec, []).append(row)

    metric = _selection_metric(axis)
    output: list[dict[str, Any]] = []
    for spec, group in sorted(by_component.items()):
        baseline = baseline_by_component.get(spec, [])
        mean_metric = _mean_float(group, metric)
        baseline_metric = _mean_float(baseline, metric)
        mean_context = mean(float(row.get("outcome") == "context_only") for row in group) if group else 0.0
        baseline_context = mean(float(row.get("outcome") == "context_only") for row in baseline) if baseline else 0.0
        layer_idx, component_type = spec
        output.append(
            {
                "group": f"{axis}:component:L{layer_idx}.{component_type}",
                "split": "discovery",
                "pair_name": pair_name,
                "axis": axis,
                "direction": direction,
                "target_attractor": target_attractor,
                "contrast_attractor": contrast_attractor,
                "layer_idx": layer_idx,
                "component_type": component_type,
                "component_id": f"L{layer_idx}.{component_type}",
                "n": len(group),
                "mean_event_target_score": _mean_float(group, "event_target_score"),
                "baseline_event_target_score": _mean_float(baseline, "event_target_score"),
                "mean_event_contrast_score": _mean_float(group, "event_contrast_score"),
                "baseline_event_contrast_score": _mean_float(baseline, "event_contrast_score"),
                "mean_event_margin_score": _mean_float(group, "event_margin_score"),
                "baseline_event_margin_score": _mean_float(baseline, "event_margin_score"),
                "mean_source_score": _mean_float(group, "source_score"),
                "baseline_source_score": _mean_float(baseline, "source_score"),
                "mean_prior_suppression_score": _mean_float(group, "prior_suppression_score"),
                "baseline_prior_suppression_score": _mean_float(baseline, "prior_suppression_score"),
                "mean_commitment_score": _mean_float(group, "commitment_score"),
                "baseline_commitment_score": _mean_float(baseline, "commitment_score"),
                "mean_form_score": _mean_float(group, "form_score"),
                "baseline_form_score": _mean_float(baseline, "form_score"),
                "mean_joint_score": _mean_float(group, "joint_score"),
                "baseline_joint_score": _mean_float(baseline, "joint_score"),
                "mean_context_only": mean_context,
                "baseline_context_only": baseline_context,
                "mean_prior_only": mean(float(row.get("outcome") == "prior_only") for row in group) if group else 0.0,
                "mean_both": mean(float(row.get("outcome") == "both") for row in group) if group else 0.0,
                "mean_neither": mean(float(row.get("outcome") == "neither") for row in group) if group else 0.0,
                "mean_cf_em": _mean_bool(group, "cf_em"),
                "mean_first_answer_cf_em": _mean_bool(group, "first_answer_cf_em"),
                "mean_short_output": _mean_bool(group, "short_output"),
                "mean_output_chars": _mean_float(group, "output_chars"),
                "selection_score": mean_metric - baseline_metric,
                "selection_metric": metric,
                "selection_rule": f"rank_by_{axis}_{direction}_{metric}_gain",
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["mean_context_only"]),
            float(row["mean_neither"]),
            float(row["mean_output_chars"]),
            int(row["layer_idx"]),
            str(row["component_type"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["selection_rank"] = rank
    return output


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.eval_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    stop_strings = _parse_stop_strings(args.stop_strings)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    component_specs = _all_component_specs(backend)

    baseline_by_sample: dict[str, dict[str, Any]] = {}
    for row in tqdm(rows, desc=f"{args.axis} baselines"):
        prediction = backend.generate(
            _generation_prompt(row, args.generation_prompt_key),
            max_new_tokens=args.max_new_tokens,
            stop_strings=stop_strings,
        )
        classified = _classify_generation(prediction, row["orig_answers"], row["cf_answers"])
        metrics = {**classified, **_output_metrics(prediction, row["cf_answers"])}
        baseline_by_sample[row["sample_id"]] = {
            "prediction": prediction,
        **metrics,
        **_score_row(metrics, axis=args.axis, direction=args.direction),
        }

    output_rows: list[dict[str, Any]] = []
    pair_name = ""
    target_attractor = ""
    contrast_attractor = ""
    for component_spec in tqdm(component_specs, desc=f"{args.axis} component scan"):
        layer_idx, component_type = component_spec
        for row in rows:
            addition = _addition(
                backend,
                row,
                component_spec,
                args.axis,
                args.direction,
                args.alpha,
                args.source_contrast_kind,
            )
            _, _, pair_name, target_attractor, contrast_attractor = _target_and_contrast_prompts(
                row,
                args.axis,
                args.direction,
                source_contrast_kind=args.source_contrast_kind,
            )
            intervention_prediction = backend.generate_with_component_last_token_add_many(
                _generation_prompt(row, args.generation_prompt_key),
                additions=[addition],
                max_new_tokens=args.max_new_tokens,
                stop_strings=stop_strings,
                apply_mode=args.apply_mode,
            )
            baseline_metrics = baseline_by_sample[row["sample_id"]]
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_index": row["source_index"],
                    "axis": args.axis,
                    "direction": args.direction,
                    "pair_name": pair_name,
                    "target_attractor": target_attractor,
                    "contrast_attractor": contrast_attractor,
                    "layer_idx": layer_idx,
                    "component_type": component_type,
                    "component_id": f"L{layer_idx}.{component_type}",
                    "method": "baseline",
                    "prediction": baseline_metrics["prediction"] if args.include_baseline_rows else "",
                    **{k: v for k, v in baseline_metrics.items() if k != "prediction"},
                }
            )
            classified = _classify_generation(intervention_prediction, row["orig_answers"], row["cf_answers"])
            row_metrics = {**classified, **_output_metrics(intervention_prediction, row["cf_answers"])}
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_index": row["source_index"],
                    "axis": args.axis,
                    "direction": args.direction,
                    "pair_name": pair_name,
                    "target_attractor": target_attractor,
                    "contrast_attractor": contrast_attractor,
                    "layer_idx": layer_idx,
                    "component_type": component_type,
                    "component_id": f"L{layer_idx}.{component_type}",
                    "method": f"{args.axis}_component",
                    "prediction": intervention_prediction,
                **row_metrics,
                **_score_row(row_metrics, axis=args.axis, direction=args.direction),
                }
            )

    summary = _summary_rows(
        output_rows,
        axis=args.axis,
        direction=args.direction,
        pair_name=pair_name,
        target_attractor=target_attractor,
        contrast_attractor=contrast_attractor,
    )
    if not args.include_baseline_rows:
        output_rows = [row for row in output_rows if row["method"] != "baseline"]
    dump_jsonl(args.out_rows_jsonl, output_rows)
    dump_csv(args.out_summary_csv, summary)
    if summary:
        top = ", ".join(row["component_id"] for row in summary[:4])
        print(
            f"[factor-scan] axis={args.axis} direction={args.direction} rows={len(rows)} "
            f"components={len(component_specs)} top4={top}"
        )


if __name__ == "__main__":
    main()
