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

from screscomp.cli.score_ckplug_generation import _classify_generation, _parse_stop_strings
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank components for the CK open-generation answer-form axis.")
    p.add_argument("--eval_jsonl", type=Path, required=True)
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
    p.add_argument("--apply_mode", choices=["prefill", "all", "decode"], default="prefill")
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


def _format_metrics(prediction: str, cf_answers: list[str]) -> dict[str, Any]:
    first_span = _first_answer_span(prediction)
    output_chars = len(prediction)
    first_answer_cf_em = _exact_hit(first_span, cf_answers)
    short_output = output_chars <= 48
    form_score = float(first_answer_cf_em) + float(short_output) - output_chars / 200.0
    return {
        "first_answer_span": first_span,
        "first_answer_cf_em": first_answer_cf_em,
        "short_output": short_output,
        "output_chars": output_chars,
        "form_score": form_score,
    }


def _all_component_specs(backend: TransformersABBackend) -> list[tuple[int, str]]:
    num_layers = len(backend._model.model.layers)
    specs: list[tuple[int, str]] = []
    for layer_idx in range(num_layers):
        specs.append((layer_idx, "attn"))
        specs.append((layer_idx, "mlp"))
    return specs


def _format_addition(
    backend: TransformersABBackend,
    row: dict[str, Any],
    component_spec: tuple[int, str],
    alpha: float,
) -> dict[str, Any]:
    prompts = row["prompts"]
    if "verbose_rag" not in prompts:
        raise ValueError("Format scan requires `verbose_rag` prompts. Re-run prepare_ckplug_open.")
    short_acts = backend.capture_component_last_token(prompts["strong_rag"], [component_spec])
    verbose_acts = backend.capture_component_last_token(prompts["verbose_rag"], [component_spec])
    layer_idx, component_type = component_spec
    return {
        "layer_idx": layer_idx,
        "component_type": component_type,
        "direction": short_acts[component_spec] - verbose_acts[component_spec],
        "alpha": alpha,
    }


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return mean(float(row[key]) for row in rows) if rows else 0.0


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_component: dict[tuple[int, str], list[dict[str, Any]]] = {}
    baseline_by_component: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        spec = (int(row["layer_idx"]), str(row["component_type"]))
        if row["method"] == "baseline":
            baseline_by_component.setdefault(spec, []).append(row)
        else:
            by_component.setdefault(spec, []).append(row)

    output: list[dict[str, Any]] = []
    for spec, group in sorted(by_component.items()):
        baseline = baseline_by_component.get(spec, [])
        mean_form = mean(float(row["form_score"]) for row in group)
        baseline_form = mean(float(row["form_score"]) for row in baseline) if baseline else 0.0
        mean_cf_em = _mean_bool(group, "cf_em")
        baseline_cf_em = _mean_bool(baseline, "cf_em")
        mean_first_cf_em = _mean_bool(group, "first_answer_cf_em")
        baseline_first_cf_em = _mean_bool(baseline, "first_answer_cf_em")
        mean_short = _mean_bool(group, "short_output")
        baseline_short = _mean_bool(baseline, "short_output")
        mean_chars = mean(float(row["output_chars"]) for row in group) if group else 0.0
        baseline_chars = mean(float(row["output_chars"]) for row in baseline) if baseline else 0.0
        layer_idx, component_type = spec
        output.append(
            {
                "group": f"format_axis:component:L{layer_idx}.{component_type}",
                "split": "format_discovery",
                "pair_name": "format_short_vs_verbose",
                "layer_idx": layer_idx,
                "component_type": component_type,
                "component_id": f"L{layer_idx}.{component_type}",
                "n": len(group),
                "mean_format_score": mean_form,
                "baseline_format_score": baseline_form,
                "mean_format_gain": mean_form - baseline_form,
                "mean_cf_em": mean_cf_em,
                "baseline_cf_em": baseline_cf_em,
                "mean_cf_em_gain": mean_cf_em - baseline_cf_em,
                "mean_first_answer_cf_em": mean_first_cf_em,
                "baseline_first_answer_cf_em": baseline_first_cf_em,
                "mean_first_answer_cf_em_gain": mean_first_cf_em - baseline_first_cf_em,
                "mean_short_output": mean_short,
                "baseline_short_output": baseline_short,
                "mean_short_output_gain": mean_short - baseline_short,
                "mean_output_chars": mean_chars,
                "baseline_output_chars": baseline_chars,
                "mean_length_delta": mean_chars - baseline_chars,
                "selection_score": mean_form - baseline_form,
                "selection_rule": "rank_by_format_score_gain",
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["mean_first_answer_cf_em_gain"]),
            float(row["mean_length_delta"]),
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
    for row in tqdm(rows, desc="format scan baselines"):
        prediction = backend.generate(
            row["prompts"]["strong_rag"],
            max_new_tokens=args.max_new_tokens,
            stop_strings=stop_strings,
        )
        classified = _classify_generation(prediction, row["orig_answers"], row["cf_answers"])
        baseline_by_sample[row["sample_id"]] = {
            "prediction": prediction,
            **classified,
            **_format_metrics(prediction, row["cf_answers"]),
        }

    output_rows: list[dict[str, Any]] = []
    for component_spec in tqdm(component_specs, desc="format component scan"):
        layer_idx, component_type = component_spec
        for row in rows:
            addition = _format_addition(backend, row, component_spec, args.alpha)
            intervention_prediction = backend.generate_with_component_last_token_add_many(
                row["prompts"]["strong_rag"],
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
                    "layer_idx": layer_idx,
                    "component_type": component_type,
                    "component_id": f"L{layer_idx}.{component_type}",
                    "method": "baseline",
                    "prediction": baseline_metrics["prediction"] if args.include_baseline_rows else "",
                    **{k: v for k, v in baseline_metrics.items() if k != "prediction"},
                }
            )
            classified = _classify_generation(intervention_prediction, row["orig_answers"], row["cf_answers"])
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_index": row["source_index"],
                    "layer_idx": layer_idx,
                    "component_type": component_type,
                    "component_id": f"L{layer_idx}.{component_type}",
                    "method": "format_component",
                    "prediction": intervention_prediction,
                    **classified,
                    **_format_metrics(intervention_prediction, row["cf_answers"]),
                }
            )

    summary = _summary_rows(output_rows)
    # Unless explicitly requested, keep the JSONL small by removing baseline duplicate rows after summary.
    if not args.include_baseline_rows:
        output_rows = [row for row in output_rows if row["method"] != "baseline"]
    dump_jsonl(args.out_rows_jsonl, output_rows)
    dump_csv(args.out_summary_csv, summary)
    if summary:
        top = ", ".join(row["component_id"] for row in summary[:4])
        print(f"[format-scan] rows={len(rows)} components={len(component_specs)} top4={top}")


if __name__ == "__main__":
    main()
