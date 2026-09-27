from __future__ import annotations

import argparse
import csv
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_component_competition import _parse_csv, _parse_csv_set, _select_rows
from screscomp.cli.score_steering_utility import (
    _additions_for,
    _learn_component_directions_by_label_mode,
    _load_summary_rows,
    _make_random_directions,
    _rank_components,
)
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.eval.health import add_candidate_tagging_instruction, analyze_candidate_health
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score CK-PLUG-style open generation with strict four-way outcomes.")
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--train_rendered_jsonl", type=Path, required=True)
    p.add_argument("--component_summary_csv", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_generations_jsonl", type=Path, required=True)
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
    p.add_argument("--selection_pairs", type=str, default="e1_content,e1_content_swapped")
    p.add_argument("--methods", type=str, default="base_no_rag,base_rag,strong_rag,random_context,ours_context")
    p.add_argument("--prompt_key", type=str, default="base_rag", choices=["base_rag", "strong_rag", "official_rag"])
    p.add_argument("--top_k_components", type=int, default=4)
    p.add_argument(
        "--component_ids",
        type=str,
        default="",
        help="Optional explicit source-axis component ids such as L17.attn,L20.mlp.",
    )
    p.add_argument(
        "--source_prior_component_summary_csv",
        type=Path,
        default=None,
        help="Optional independently discovered source prior-vs-context component summary.",
    )
    p.add_argument(
        "--format_component_summary_csv",
        type=Path,
        default=None,
        help=(
            "Optional component summary for the answer-form axis. If omitted, format-axis methods reuse "
            "the source-axis components unless --format_component_ids is provided."
        ),
    )
    p.add_argument(
        "--format_selection_pairs",
        type=str,
        default="",
        help="Selection pairs for --format_component_summary_csv. Defaults to --selection_pairs when empty.",
    )
    p.add_argument(
        "--format_top_k_components",
        type=int,
        default=None,
        help="Top-k components for the answer-form axis. Defaults to --top_k_components.",
    )
    p.add_argument(
        "--format_component_ids",
        type=str,
        default="",
        help="Explicit comma-separated answer-form components, e.g. L17.attn,L20.mlp. Overrides summary ranking.",
    )
    p.add_argument(
        "--format_verbose_component_summary_csv",
        type=Path,
        default=None,
        help="Optional independently discovered verbose-vs-short form component summary.",
    )
    p.add_argument(
        "--commitment_component_summary_csv",
        type=Path,
        default=None,
        help="Optional component summary for the single-answer-vs-mixed/hedge commitment axis.",
    )
    p.add_argument(
        "--commitment_top_k_components",
        type=int,
        default=None,
        help="Top-k components for the commitment axis. Defaults to --top_k_components.",
    )
    p.add_argument(
        "--commitment_component_ids",
        type=str,
        default="",
        help="Explicit comma-separated commitment-axis components, e.g. L23.attn,L24.mlp.",
    )
    p.add_argument(
        "--commitment_mixed_component_summary_csv",
        type=Path,
        default=None,
        help="Optional independently discovered mixed-vs-single commitment component summary.",
    )
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument(
        "--source_delta_kind",
        type=str,
        default="no_rag_vs_rag",
        choices=["no_rag_vs_rag", "objective_context_vs_prior"],
        help=(
            "How to construct the source-axis residual direction. no_rag_vs_rag uses the existing "
            "no-context vs context prompt contrast. objective_context_vs_prior keeps the same context "
            "and contrasts a context-reliance objective against a prior-memory objective."
        ),
    )
    p.add_argument(
        "--source_apply_mode",
        type=str,
        default="prefill",
        help=(
            "When to apply source-axis residual additions for *_prefill_* methods. "
            "Supported by the backend: prefill, decode, all, first_decode, first_2_decode, ..."
        ),
    )
    p.add_argument(
        "--format_alpha",
        type=float,
        default=1.0,
        help="Strength for short-answer-vs-verbose format-axis additions in multi-axis methods.",
    )
    p.add_argument(
        "--format_apply_mode",
        type=str,
        default="prefill",
        help=(
            "When to apply format-axis residual additions for format methods. "
            "Supported by the backend: prefill, decode, all, first_decode, first_2_decode, ..."
        ),
    )
    p.add_argument("--commitment_alpha", type=float, default=1.0)
    p.add_argument(
        "--commitment_apply_mode",
        type=str,
        default="first_2_decode",
        help=(
            "When to apply commitment-axis residual additions. "
            "Supported by the backend: prefill, decode, all, first_decode, first_2_decode, ..."
        ),
    )
    p.add_argument(
        "--prior_suppression_alpha",
        type=float,
        default=0.0,
        help="Strength for the no-prior-memory-vs-prior-memory suppression edge.",
    )
    p.add_argument(
        "--prior_suppression_component_summary_csv",
        type=Path,
        default=None,
        help="Optional component summary for the no-prior-memory-vs-prior-memory suppression edge.",
    )
    p.add_argument(
        "--prior_suppression_top_k_components",
        type=int,
        default=None,
        help="Top-k components for prior suppression. Defaults to --top_k_components.",
    )
    p.add_argument(
        "--prior_suppression_component_ids",
        type=str,
        default="",
        help="Explicit comma-separated prior-suppression components, e.g. L23.attn,L24.mlp.",
    )
    p.add_argument(
        "--prior_memory_component_summary_csv",
        type=Path,
        default=None,
        help="Optional independently discovered memory-vs-no-prior component summary.",
    )
    p.add_argument(
        "--prior_suppression_apply_mode",
        type=str,
        default="first_2_decode",
        help=(
            "When to apply prior-suppression residual additions. "
            "Supported by the backend: prefill, decode, all, first_decode, first_2_decode, ..."
        ),
    )
    p.add_argument(
        "--candidate_tagging",
        action="store_true",
        help="Ask every method to wrap answer candidates in <cand> and the final answer in <final>.",
    )
    p.add_argument(
        "--score_tagged_final",
        action="store_true",
        help="If candidate tags are present, score the parsed <final> answer instead of the full raw output.",
    )
    p.add_argument(
        "--logit_alpha",
        type=float,
        default=1.0,
        help="Extra multiplier for RCM logit-actuator methods after projecting component directions to vocab space.",
    )
    p.add_argument(
        "--logit_relative_top",
        type=float,
        default=0.01,
        help="Relative-top candidate mask used by RCM logit-actuator methods.",
    )
    p.add_argument(
        "--logit_min_tokens_to_keep",
        type=int,
        default=10,
        help="Minimum number of candidate tokens kept by the RCM logit-actuator mask.",
    )
    p.add_argument(
        "--logit_apply_mode",
        type=str,
        default="all",
        help="When to apply logit actuator: all, prefill, decode, first_decode, first_2_decode, ...",
    )
    p.add_argument("--max_train_rows", type=int, default=96)
    p.add_argument("--max_eval_rows", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--stop_strings", type=str, default="Q:")
    p.add_argument("--out_comparison_csv", type=Path, default=None)
    p.add_argument("--out_transitions_csv", type=Path, default=None)
    p.add_argument(
        "--compare_pairs",
        type=str,
        default=(
            "strong_rag>ours_delta_context,"
            "strong_rag>ours_delta_prefill_context,"
            "random_delta_context>ours_delta_context,"
            "random_delta_prefill_context>ours_delta_prefill_context"
        ),
    )
    p.add_argument("--bootstrap_samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _recall_hit(prediction: str, answers: list[str]) -> bool:
    pred = _normalize_answer(prediction)
    if not pred:
        return False
    return any(_normalize_answer(answer) in pred for answer in answers if _normalize_answer(answer))


def _exact_hit(prediction: str, answers: list[str]) -> bool:
    pred = _normalize_answer(prediction)
    return any(pred == _normalize_answer(answer) for answer in answers if _normalize_answer(answer))


_VERBOSE_CHAR_THRESHOLD = 48


def _classify_generation(prediction: str, orig_answers: list[str], cf_answers: list[str]) -> dict[str, Any]:
    cf_hit = _recall_hit(prediction, cf_answers)
    orig_hit = _recall_hit(prediction, orig_answers)
    cf_em = _exact_hit(prediction, cf_answers)
    orig_em = _exact_hit(prediction, orig_answers)
    output_chars = len(prediction)
    if cf_hit and orig_hit:
        outcome = "both"
    elif cf_hit:
        outcome = "context_only"
    elif orig_hit:
        outcome = "prior_only"
    else:
        outcome = "neither"
    if outcome == "both":
        source_fine = "mixed"
    elif outcome == "neither":
        source_fine = "neither_verbose" if output_chars > _VERBOSE_CHAR_THRESHOLD else "neither_short"
    else:
        source_fine = outcome
    if cf_em and output_chars <= _VERBOSE_CHAR_THRESHOLD:
        form_fine = "short_exact"
    elif cf_hit and output_chars > _VERBOSE_CHAR_THRESHOLD:
        form_fine = "context_verbose"
    elif cf_hit and not cf_em:
        form_fine = "context_inexact"
    elif orig_hit:
        form_fine = "prior_form"
    else:
        form_fine = "neither_form"
    if outcome == "both":
        fine_outcome = "mixed"
    elif outcome == "neither":
        fine_outcome = "neither_verbose" if output_chars > _VERBOSE_CHAR_THRESHOLD else "neither_short"
    elif outcome == "context_only" and cf_em and output_chars <= _VERBOSE_CHAR_THRESHOLD:
        fine_outcome = "short_exact"
    else:
        fine_outcome = outcome
    return {
        "outcome": outcome,
        "source_fine_outcome": source_fine,
        "form_fine_outcome": form_fine,
        "fine_outcome": fine_outcome,
        "cf_hit": cf_hit,
        "orig_hit": orig_hit,
        "cf_em": cf_em,
        "orig_em": orig_em,
        "output_chars": output_chars,
        "short_output": output_chars <= _VERBOSE_CHAR_THRESHOLD,
    }


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_stop_strings(raw: str) -> list[str]:
    stops: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        stops.append(bytes(item, "utf-8").decode("unicode_escape"))
    return stops


def _parse_compare_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in _parse_csv(raw):
        if ">" in item:
            baseline, target = item.split(">", 1)
        elif ":" in item:
            baseline, target = item.split(":", 1)
        else:
            raise ValueError(f"Compare pair must look like baseline>target: {item}")
        baseline = baseline.strip()
        target = target.strip()
        if not baseline or not target:
            raise ValueError(f"Invalid compare pair: {item}")
        pairs.append((baseline, target))
    return pairs


def _mean_label_mode_directions(directions_by_label_mode: dict[str, dict[tuple[int, str], Any]]):
    normal = directions_by_label_mode["normal"]
    swapped = directions_by_label_mode["swapped"]
    out = {}
    for spec in sorted(set(normal) & set(swapped)):
        out[spec] = (normal[spec] + swapped[spec]) / 2.0
    if not out:
        raise ValueError("No shared component specs between normal and swapped directions.")
    return out


def _component_id_from_spec(spec: tuple[int, str]) -> str:
    return f"L{int(spec[0])}.{spec[1]}"


def _parse_component_spec(raw: str) -> tuple[int, str]:
    text = raw.strip()
    match = re.fullmatch(r"L?(\d+)\.(attn|mlp)", text)
    if not match:
        raise ValueError(f"Component id must look like L17.attn or L20.mlp: {raw}")
    return int(match.group(1)), match.group(2)


def _component_rows_from_ids(raw: str, *, axis: str) -> list[dict[str, Any]]:
    if "<" in raw or ">" in raw:
        raise ValueError(
            f"--{axis}_component_ids received a placeholder value ({raw!r}). "
            "Replace it with real component ids such as L17.attn,L20.mlp, or omit the option to reuse source-axis components."
        )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for rank, item in enumerate(_parse_csv(raw), start=1):
        spec = _parse_component_spec(item)
        if spec in seen:
            continue
        seen.add(spec)
        rows.append(
            {
                "layer_idx": spec[0],
                "component_type": spec[1],
                "component_id": _component_id_from_spec(spec),
                "selection_rank": rank,
                "selection_score": "",
                "selection_mean_min_CI": "",
                "selection_mean_C_plus_I": "",
                "selection_frac_both_positive": "",
                "selection_rows": "",
                "selection_rule": "explicit_component_ids",
                "axis": axis,
            }
        )
    return rows


def _select_axis_components(
    *,
    summary_csv: Path | None,
    selection_pairs: str,
    top_k: int,
    explicit_ids: str,
    fallback_components: list[dict[str, Any]],
    axis: str,
) -> tuple[list[dict[str, Any]], str]:
    if explicit_ids:
        components = _component_rows_from_ids(explicit_ids, axis=axis)
        if not components:
            raise ValueError(f"--{axis}_component_ids did not contain any components.")
        return components, "explicit_component_ids"
    if summary_csv is not None:
        summary_rows = _load_summary_rows(summary_csv)
        if summary_rows and "selection_score" in summary_rows[0] and "mean_min_CI" not in summary_rows[0]:
            ranked = sorted(
                summary_rows,
                key=lambda row: (
                    -float(row.get("selection_score", 0.0) or 0.0),
                    int(row.get("selection_rank", 10**9) or 10**9),
                    int(row["layer_idx"]),
                    row["component_type"],
                ),
            )
        else:
            ranked = _rank_components(
                summary_rows=summary_rows,
                selection_pairs=_parse_csv(selection_pairs),
            )
        if top_k > len(ranked):
            raise ValueError(f"Requested {axis}_top_k={top_k}, but only {len(ranked)} components exist.")
        return [{**row, "axis": axis} for row in ranked[:top_k]], f"ranked_summary:{summary_csv}"
    return [{**row, "axis": axis, "selection_rule": f"{row.get('selection_rule', '')}|reused_from_source_axis"} for row in fallback_components], "reuse_source_components"


def _component_specs(components: list[dict[str, Any]]) -> list[tuple[int, str]]:
    return sorted((int(component["layer_idx"]), str(component["component_type"])) for component in components)


def _select_eval_rows(rows: list[dict], max_rows: int | None, seed: int) -> list[dict]:
    if max_rows is None or max_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_rows))
    return [rows[i] for i in indices]


def _component_delta_additions(
    backend: TransformersABBackend,
    base_prompt: str,
    context_prompt: str,
    component_specs: list[tuple[int, str]],
    alpha: float,
) -> list[dict[str, Any]]:
    base_acts = backend.capture_component_last_token(base_prompt, component_specs)
    context_acts = backend.capture_component_last_token(context_prompt, component_specs)
    return [
        {
            "layer_idx": layer_idx,
            "component_type": component_type,
            "direction": context_acts[(layer_idx, component_type)] - base_acts[(layer_idx, component_type)],
            "alpha": alpha,
        }
        for layer_idx, component_type in component_specs
    ]


def _format_delta_additions(
    backend: TransformersABBackend,
    short_prompt: str,
    verbose_prompt: str,
    component_specs: list[tuple[int, str]],
    alpha: float,
) -> list[dict[str, Any]]:
    short_acts = backend.capture_component_last_token(short_prompt, component_specs)
    verbose_acts = backend.capture_component_last_token(verbose_prompt, component_specs)
    return [
        {
            "layer_idx": layer_idx,
            "component_type": component_type,
            "direction": short_acts[(layer_idx, component_type)] - verbose_acts[(layer_idx, component_type)],
            "alpha": alpha,
        }
        for layer_idx, component_type in component_specs
    ]


def _commitment_contrast_prompt(row: dict[str, Any]) -> str:
    return (
        "Read the given information and answer the question. If there may be conflicting or alternative answers, "
        "mention the alternatives and briefly explain the conflict.\n\n"
        f"{row['context']}\n\nQ: {row['question']}\nA:"
    )


def _commitment_delta_additions(
    backend: TransformersABBackend,
    single_prompt: str,
    mixed_prompt: str,
    component_specs: list[tuple[int, str]],
    alpha: float,
) -> list[dict[str, Any]]:
    single_acts = backend.capture_component_last_token(single_prompt, component_specs)
    mixed_acts = backend.capture_component_last_token(mixed_prompt, component_specs)
    return [
        {
            "layer_idx": layer_idx,
            "component_type": component_type,
            "direction": single_acts[(layer_idx, component_type)] - mixed_acts[(layer_idx, component_type)],
            "alpha": alpha,
        }
        for layer_idx, component_type in component_specs
    ]


def _prior_memory_prompt(row: dict[str, Any]) -> str:
    return (
        "Answer the question from your own prior knowledge. Return only the short answer.\n\n"
        f"Q: {row['question']}\nA:"
    )


def _no_prior_memory_prompt(row: dict[str, Any]) -> str:
    return (
        "Do not answer from your own prior knowledge. If the answer is not provided, do not guess. "
        "Return only a short answer.\n\n"
        f"Q: {row['question']}\nA:"
    )


def _prior_suppression_delta_additions(
    backend: TransformersABBackend,
    no_prior_prompt: str,
    prior_prompt: str,
    component_specs: list[tuple[int, str]],
    alpha: float,
) -> list[dict[str, Any]]:
    no_prior_acts = backend.capture_component_last_token(no_prior_prompt, component_specs)
    prior_acts = backend.capture_component_last_token(prior_prompt, component_specs)
    return [
        {
            "layer_idx": layer_idx,
            "component_type": component_type,
            "direction": no_prior_acts[(layer_idx, component_type)] - prior_acts[(layer_idx, component_type)],
            "alpha": alpha,
        }
        for layer_idx, component_type in component_specs
    ]


def _randomized_delta_additions(
    backend: TransformersABBackend,
    additions: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    random_additions: list[dict[str, Any]] = []
    for idx, addition in enumerate(additions):
        direction = addition["direction"]
        generator = backend._torch.Generator(device=direction.device)
        generator.manual_seed(seed + 104729 * (idx + 1))
        rand = backend._torch.randn(
            direction.shape,
            generator=generator,
            device=direction.device,
            dtype=direction.dtype,
        )
        norm = rand.norm()
        target_norm = direction.norm()
        if float(norm.item()) > 1e-12:
            rand = rand / norm * target_norm
        random_additions.append({**addition, "direction": rand})
    return random_additions


def _reverse_additions(additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**addition, "alpha": -float(addition.get("alpha", 1.0))} for addition in additions]


def _with_apply_mode(additions: list[dict[str, Any]], apply_mode: str) -> list[dict[str, Any]]:
    return [{**addition, "apply_mode": apply_mode} for addition in additions]


def _prior_objective_prompt(row: dict[str, Any]) -> str:
    return (
        "Read the given information, but answer from your own prior knowledge rather than relying on "
        "the given information. Return only the short answer.\n\n"
        f"{row['context']}\n\nQ: {row['question']}\nA:"
    )


def _delta_base_prompt(row: dict[str, Any], prompts: dict[str, str], prompt_key: str, source_delta_kind: str) -> str:
    if source_delta_kind == "objective_context_vs_prior":
        return _prior_objective_prompt(row)
    if prompt_key == "strong_rag" and "strong_no_rag" in prompts:
        return prompts["strong_no_rag"]
    if prompt_key == "official_rag" and "official_no_rag" in prompts:
        return prompts["official_no_rag"]
    return prompts["base_no_rag"]


def _generate_for_method(
    backend: TransformersABBackend,
    row: dict,
    method: str,
    prompt_key: str,
    directions,
    random_directions,
    source_component_specs: list[tuple[int, str]],
    source_prior_component_specs: list[tuple[int, str]],
    format_component_specs: list[tuple[int, str]],
    format_verbose_component_specs: list[tuple[int, str]],
    commitment_component_specs: list[tuple[int, str]],
    commitment_mixed_component_specs: list[tuple[int, str]],
    prior_suppression_component_specs: list[tuple[int, str]],
    prior_memory_component_specs: list[tuple[int, str]],
    alpha: float,
    source_delta_kind: str,
    source_apply_mode: str,
    format_alpha: float,
    format_apply_mode: str,
    commitment_alpha: float,
    commitment_apply_mode: str,
    prior_suppression_alpha: float,
    prior_suppression_apply_mode: str,
    candidate_tagging: bool,
    logit_alpha: float,
    logit_relative_top: float,
    logit_min_tokens_to_keep: int,
    logit_apply_mode: str,
    max_new_tokens: int,
    stop_strings: list[str],
    seed: int,
    delta_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, str]:
    prompts = row["prompts"]
    if delta_cache is None:
        delta_cache = {}

    def maybe_tag(prompt: str) -> str:
        return add_candidate_tagging_instruction(prompt) if candidate_tagging else prompt

    generation_prompt = maybe_tag(prompts[prompt_key])

    def get_delta_additions() -> list[dict[str, Any]]:
        key = f"{prompt_key}:source_delta_kind={source_delta_kind}:alpha={alpha}"
        if key not in delta_cache:
            delta_cache[key] = _component_delta_additions(
                backend=backend,
                base_prompt=maybe_tag(_delta_base_prompt(row, prompts, prompt_key, source_delta_kind)),
                context_prompt=maybe_tag(prompts[prompt_key]),
                component_specs=source_component_specs,
                alpha=alpha,
            )
        return delta_cache[key]

    def get_delta_prior_additions() -> list[dict[str, Any]]:
        key = f"{prompt_key}:source_prior_delta_kind={source_delta_kind}:alpha={alpha}"
        if key not in delta_cache:
            delta_cache[key] = _component_delta_additions(
                backend=backend,
                base_prompt=maybe_tag(prompts[prompt_key]),
                context_prompt=maybe_tag(_delta_base_prompt(row, prompts, prompt_key, source_delta_kind)),
                component_specs=source_prior_component_specs,
                alpha=alpha,
            )
        return delta_cache[key]

    def get_format_additions() -> list[dict[str, Any]]:
        if "verbose_rag" not in prompts:
            raise ValueError("Format-axis methods require `verbose_rag` prompts. Re-run prepare_ckplug_open.")
        key = f"{prompt_key}:format_alpha={format_alpha}"
        if key not in delta_cache:
            delta_cache[key] = _format_delta_additions(
                backend=backend,
                short_prompt=maybe_tag(prompts[prompt_key]),
                verbose_prompt=maybe_tag(prompts["verbose_rag"]),
                component_specs=format_component_specs,
                alpha=format_alpha,
            )
        return delta_cache[key]

    def get_format_verbose_additions() -> list[dict[str, Any]]:
        if "verbose_rag" not in prompts:
            raise ValueError("Format-axis methods require `verbose_rag` prompts. Re-run prepare_ckplug_open.")
        key = f"{prompt_key}:format_verbose_alpha={format_alpha}"
        if key not in delta_cache:
            delta_cache[key] = _format_delta_additions(
                backend=backend,
                short_prompt=maybe_tag(prompts["verbose_rag"]),
                verbose_prompt=maybe_tag(prompts[prompt_key]),
                component_specs=format_verbose_component_specs,
                alpha=format_alpha,
            )
        return delta_cache[key]

    def get_commitment_additions() -> list[dict[str, Any]]:
        key = f"{prompt_key}:commitment_alpha={commitment_alpha}"
        if key not in delta_cache:
            delta_cache[key] = _commitment_delta_additions(
                backend=backend,
                single_prompt=maybe_tag(prompts[prompt_key]),
                mixed_prompt=maybe_tag(_commitment_contrast_prompt(row)),
                component_specs=commitment_component_specs,
                alpha=commitment_alpha,
            )
        return delta_cache[key]

    def get_commitment_mixed_additions() -> list[dict[str, Any]]:
        key = f"{prompt_key}:commitment_mixed_alpha={commitment_alpha}"
        if key not in delta_cache:
            delta_cache[key] = _commitment_delta_additions(
                backend=backend,
                single_prompt=maybe_tag(_commitment_contrast_prompt(row)),
                mixed_prompt=maybe_tag(prompts[prompt_key]),
                component_specs=commitment_mixed_component_specs,
                alpha=commitment_alpha,
            )
        return delta_cache[key]

    def get_prior_suppression_additions() -> list[dict[str, Any]]:
        key = f"prior_suppression:alpha={prior_suppression_alpha}"
        if key not in delta_cache:
            delta_cache[key] = _prior_suppression_delta_additions(
                backend=backend,
                no_prior_prompt=maybe_tag(_no_prior_memory_prompt(row)),
                prior_prompt=maybe_tag(_prior_memory_prompt(row)),
                component_specs=prior_suppression_component_specs,
                alpha=prior_suppression_alpha,
            )
        return delta_cache[key]

    def get_prior_memory_additions() -> list[dict[str, Any]]:
        key = f"prior_memory:alpha={prior_suppression_alpha}"
        if key not in delta_cache:
            delta_cache[key] = _prior_suppression_delta_additions(
                backend=backend,
                no_prior_prompt=maybe_tag(_prior_memory_prompt(row)),
                prior_prompt=maybe_tag(_no_prior_memory_prompt(row)),
                component_specs=prior_memory_component_specs,
                alpha=prior_suppression_alpha,
            )
        return delta_cache[key]

    if method == "base_no_rag":
        prompt = maybe_tag(prompts["base_no_rag"])
        return prompt, backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "base_rag":
        prompt = maybe_tag(prompts["base_rag"])
        return prompt, backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "prompt_rag":
        prompt = maybe_tag(prompts[prompt_key])
        return prompt, backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "strong_rag":
        prompt = maybe_tag(prompts["strong_rag"])
        return prompt, backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "prior_objective_rag":
        prompt = maybe_tag(_prior_objective_prompt(row))
        return prompt, backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "ours_context":
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_additions_for(directions, alpha=alpha, task_sign=-1.0),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "ours_prior":
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_additions_for(directions, alpha=alpha, task_sign=1.0),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "random_context":
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_additions_for(random_directions, alpha=alpha, task_sign=-1.0),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "ours_delta_context":
        additions = get_delta_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "ours_delta_prior":
        additions = get_delta_prior_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "random_delta_context":
        additions = get_delta_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "random_delta_prior":
        additions = get_delta_prior_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
        )
    if method == "ours_delta_prefill_context":
        additions = get_delta_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=source_apply_mode,
        )
    if method == "ours_delta_prefill_prior":
        additions = get_delta_prior_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=source_apply_mode,
        )
    if method == "ours_delta_logit_context":
        additions = get_delta_additions()
        return generation_prompt, backend.generate_with_component_logit_delta(
            generation_prompt,
            additions=additions,
            logit_alpha=logit_alpha,
            relative_top=logit_relative_top,
            min_tokens_to_keep=logit_min_tokens_to_keep,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=logit_apply_mode,
        )
    if method == "random_delta_logit_context":
        additions = get_delta_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_logit_delta(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            logit_alpha=logit_alpha,
            relative_top=logit_relative_top,
            min_tokens_to_keep=logit_min_tokens_to_keep,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=logit_apply_mode,
        )
    if method == "ours_delta_logit_context_plus_prior_suppression":
        additions = [*get_delta_additions(), *get_prior_suppression_additions()]
        return generation_prompt, backend.generate_with_component_logit_delta(
            generation_prompt,
            additions=additions,
            logit_alpha=logit_alpha,
            relative_top=logit_relative_top,
            min_tokens_to_keep=logit_min_tokens_to_keep,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=logit_apply_mode,
        )
    if method == "random_delta_prefill_context":
        additions = get_delta_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=source_apply_mode,
        )
    if method == "random_delta_prefill_prior":
        additions = get_delta_prior_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=source_apply_mode,
        )
    if method in {"ours_format_prefill_context", "ours_format_prefill_short"}:
        additions = get_format_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=format_apply_mode,
        )
    if method in {"random_format_prefill_context", "random_format_prefill_short"}:
        additions = get_format_additions()
        sample_seed = seed + 7919 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=format_apply_mode,
        )
    if method == "ours_format_prefill_verbose":
        additions = get_format_verbose_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=format_apply_mode,
        )
    if method == "random_format_prefill_verbose":
        additions = get_format_verbose_additions()
        sample_seed = seed + 7919 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=format_apply_mode,
        )
    if method in {"ours_commitment_prefill_context", "ours_commitment_prefill_single"}:
        additions = get_commitment_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=commitment_apply_mode,
        )
    if method in {"random_commitment_prefill_context", "random_commitment_prefill_single"}:
        additions = get_commitment_additions()
        sample_seed = seed + 15401 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=commitment_apply_mode,
        )
    if method == "ours_commitment_prefill_mixed":
        additions = get_commitment_mixed_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=commitment_apply_mode,
        )
    if method == "random_commitment_prefill_mixed":
        additions = get_commitment_mixed_additions()
        sample_seed = seed + 15401 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=commitment_apply_mode,
        )
    if method == "ours_prior_suppression_prefill_no_prior":
        additions = get_prior_suppression_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=prior_suppression_apply_mode,
        )
    if method == "random_prior_suppression_prefill_no_prior":
        additions = get_prior_suppression_additions()
        sample_seed = seed + 17749 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=prior_suppression_apply_mode,
        )
    if method == "ours_prior_suppression_prefill_memory":
        additions = get_prior_memory_additions()
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=prior_suppression_apply_mode,
        )
    if method == "random_prior_suppression_prefill_memory":
        additions = get_prior_memory_additions()
        sample_seed = seed + 17749 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=_randomized_delta_additions(backend=backend, additions=additions, seed=sample_seed),
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode=prior_suppression_apply_mode,
        )
    if method == "ours_delta_prefill_context_plus_commitment":
        additions = [
            *_with_apply_mode(get_delta_additions(), source_apply_mode),
            *_with_apply_mode(get_commitment_additions(), commitment_apply_mode),
        ]
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "ours_delta_prefill_context_plus_prior_suppression":
        additions = [
            *_with_apply_mode(get_delta_additions(), source_apply_mode),
            *_with_apply_mode(get_prior_suppression_additions(), prior_suppression_apply_mode),
        ]
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "random_delta_prefill_context_plus_prior_suppression":
        source_additions = get_delta_additions()
        prior_suppression_additions = get_prior_suppression_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=[
                *_with_apply_mode(
                    _randomized_delta_additions(backend=backend, additions=source_additions, seed=sample_seed),
                    source_apply_mode,
                ),
                *_with_apply_mode(
                    _randomized_delta_additions(
                        backend=backend,
                        additions=prior_suppression_additions,
                        seed=sample_seed + 17749,
                    ),
                    prior_suppression_apply_mode,
                ),
            ],
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "random_delta_prefill_context_plus_commitment":
        source_additions = get_delta_additions()
        commitment_additions = get_commitment_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=[
                *_with_apply_mode(
                    _randomized_delta_additions(backend=backend, additions=source_additions, seed=sample_seed),
                    source_apply_mode,
                ),
                *_with_apply_mode(
                    _randomized_delta_additions(backend=backend, additions=commitment_additions, seed=sample_seed + 15401),
                    commitment_apply_mode,
                ),
            ],
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "ours_delta_prefill_context_plus_format":
        additions = [
            *_with_apply_mode(get_delta_additions(), source_apply_mode),
            *_with_apply_mode(get_format_additions(), format_apply_mode),
        ]
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "random_delta_prefill_context_plus_format":
        source_additions = get_delta_additions()
        sample_seed = seed + 1009 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=[
                *_with_apply_mode(
                    _randomized_delta_additions(backend=backend, additions=source_additions, seed=sample_seed),
                    source_apply_mode,
                ),
                *_with_apply_mode(get_format_additions(), format_apply_mode),
            ],
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "ours_delta_prefill_context_plus_random_format":
        format_additions = get_format_additions()
        sample_seed = seed + 7919 * int(row.get("source_index", 0))
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=[
                *_with_apply_mode(get_delta_additions(), source_apply_mode),
                *_with_apply_mode(
                    _randomized_delta_additions(backend=backend, additions=format_additions, seed=sample_seed),
                    format_apply_mode,
                ),
            ],
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    if method == "ours_delta_prefill_context_plus_commitment_plus_format":
        additions = [
            *_with_apply_mode(get_delta_additions(), source_apply_mode),
            *_with_apply_mode(get_commitment_additions(), commitment_apply_mode),
            *_with_apply_mode(get_format_additions(), format_apply_mode),
        ]
        return generation_prompt, backend.generate_with_component_last_token_add_many(
            generation_prompt,
            additions=additions,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            apply_mode="all",
        )
    raise ValueError(f"Unknown method: {method}")


_REPORT_METRICS = [
    "ps",
    "po",
    "mr",
    "context_only_rate",
    "prior_only_rate",
    "both_rate",
    "neither_rate",
    "mixed_rate",
    "neither_verbose_rate",
    "neither_short_rate",
    "short_exact_rate",
    "source_fine_mixed_rate",
    "source_fine_neither_verbose_rate",
    "source_fine_neither_short_rate",
    "form_fine_short_exact_rate",
    "form_fine_context_verbose_rate",
    "form_fine_context_inexact_rate",
    "cf_em_rate",
    "orig_em_rate",
    "mean_output_chars",
    "health_tag_format_valid_rate",
    "health_final_context_rate",
    "health_healthy_context_final_rate",
    "health_prior_then_context_rate",
    "health_context_then_prior_rate",
    "health_mixed_candidate_sources_rate",
    "health_extra_candidate_rate",
    "health_mean_candidate_tags",
    "health_open_answer_flip_rate",
    "health_open_prior_then_context_rate",
    "health_open_context_then_prior_rate",
    "health_open_mixed_candidate_sources_rate",
    "health_open_mean_candidate_count",
]


def _method_metrics(group: list[dict]) -> dict[str, Any]:
    n = len(group)
    counts = Counter(row["outcome"] for row in group)
    fine_counts = Counter(row.get("fine_outcome", row["outcome"]) for row in group)
    source_fine_counts = Counter(row.get("source_fine_outcome", row["outcome"]) for row in group)
    form_fine_counts = Counter(row.get("form_fine_outcome", "") for row in group)
    ps = sum(1 for row in group if row["cf_hit"]) / n if n else 0.0
    po = sum(1 for row in group if row["orig_hit"]) / n if n else 0.0
    denom = ps + po
    return {
        "n": n,
        "ps": ps,
        "po": po,
        "mr": po / denom if denom > 0 else 0.0,
        "context_only_rate": counts["context_only"] / n if n else 0.0,
        "prior_only_rate": counts["prior_only"] / n if n else 0.0,
        "both_rate": counts["both"] / n if n else 0.0,
        "neither_rate": counts["neither"] / n if n else 0.0,
        "mixed_rate": fine_counts["mixed"] / n if n else 0.0,
        "neither_verbose_rate": fine_counts["neither_verbose"] / n if n else 0.0,
        "neither_short_rate": fine_counts["neither_short"] / n if n else 0.0,
        "short_exact_rate": fine_counts["short_exact"] / n if n else 0.0,
        "source_fine_mixed_rate": source_fine_counts["mixed"] / n if n else 0.0,
        "source_fine_neither_verbose_rate": source_fine_counts["neither_verbose"] / n if n else 0.0,
        "source_fine_neither_short_rate": source_fine_counts["neither_short"] / n if n else 0.0,
        "form_fine_short_exact_rate": form_fine_counts["short_exact"] / n if n else 0.0,
        "form_fine_context_verbose_rate": form_fine_counts["context_verbose"] / n if n else 0.0,
        "form_fine_context_inexact_rate": form_fine_counts["context_inexact"] / n if n else 0.0,
        "cf_em_rate": sum(1 for row in group if row["cf_em"]) / n if n else 0.0,
        "orig_em_rate": sum(1 for row in group if row["orig_em"]) / n if n else 0.0,
        "mean_output_chars": mean(len(row.get("prediction_scored", row["prediction"])) for row in group) if group else 0.0,
        "health_tag_format_valid_rate": sum(1 for row in group if row.get("health_tag_format_valid")) / n if n else 0.0,
        "health_final_context_rate": (
            sum(1 for row in group if row.get("health_final_answer_source") == "context") / n if n else 0.0
        ),
        "health_healthy_context_final_rate": (
            sum(1 for row in group if row.get("health_healthy_context_final")) / n if n else 0.0
        ),
        "health_prior_then_context_rate": (
            sum(1 for row in group if row.get("health_prior_then_context")) / n if n else 0.0
        ),
        "health_context_then_prior_rate": (
            sum(1 for row in group if row.get("health_context_then_prior")) / n if n else 0.0
        ),
        "health_mixed_candidate_sources_rate": (
            sum(1 for row in group if row.get("health_mixed_candidate_sources")) / n if n else 0.0
        ),
        "health_extra_candidate_rate": (
            sum(1 for row in group if int(row.get("health_extra_candidate_count", 0) or 0) > 0) / n if n else 0.0
        ),
        "health_mean_candidate_tags": mean(
            int(row.get("health_candidate_tag_count", 0) or 0) for row in group
        )
        if group
        else 0.0,
        "health_open_answer_flip_rate": sum(1 for row in group if row.get("health_open_answer_flip")) / n if n else 0.0,
        "health_open_prior_then_context_rate": (
            sum(1 for row in group if row.get("health_open_prior_then_context")) / n if n else 0.0
        ),
        "health_open_context_then_prior_rate": (
            sum(1 for row in group if row.get("health_open_context_then_prior")) / n if n else 0.0
        ),
        "health_open_mixed_candidate_sources_rate": (
            sum(1 for row in group if row.get("health_open_mixed_candidate_sources")) / n if n else 0.0
        ),
        "health_open_mean_candidate_count": mean(
            int(row.get("health_open_candidate_count", 0) or 0) for row in group
        )
        if group
        else 0.0,
    }


def _summarize(rows: list[dict], metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        summary.append({"method": method, **(metadata or {}), **_method_metrics(group)})
    return summary


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_delta(
    baseline_rows: list[dict],
    target_rows: list[dict],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0 or not baseline_rows or not target_rows:
        return 0.0, 0.0
    if len(baseline_rows) != len(target_rows):
        raise ValueError("Bootstrap expects paired row lists with the same length.")
    rng = random.Random(seed)
    n = len(baseline_rows)
    deltas: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        base_sample = [baseline_rows[i] for i in indices]
        target_sample = [target_rows[i] for i in indices]
        deltas.append(_method_metrics(target_sample)[metric] - _method_metrics(base_sample)[metric])
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def _comparison_rows(
    rows: list[dict],
    *,
    compare_pairs: list[tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    by_method_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_method_sample[row["method"]][row["sample_id"]] = row

    output: list[dict[str, Any]] = []
    for pair_idx, (baseline, target) in enumerate(compare_pairs):
        if baseline not in by_method_sample or target not in by_method_sample:
            continue
        sample_ids = sorted(set(by_method_sample[baseline]) & set(by_method_sample[target]))
        baseline_rows = [by_method_sample[baseline][sample_id] for sample_id in sample_ids]
        target_rows = [by_method_sample[target][sample_id] for sample_id in sample_ids]
        baseline_metrics = _method_metrics(baseline_rows)
        target_metrics = _method_metrics(target_rows)
        for metric in _REPORT_METRICS:
            ci_low, ci_high = _bootstrap_delta(
                baseline_rows=baseline_rows,
                target_rows=target_rows,
                metric=metric,
                samples=bootstrap_samples,
                seed=seed + 10007 * (pair_idx + 1) + 97 * (_REPORT_METRICS.index(metric) + 1),
            )
            output.append(
                {
                    **metadata,
                    "baseline": baseline,
                    "target": target,
                    "metric": metric,
                    "n_paired": len(sample_ids),
                    "baseline_value": baseline_metrics[metric],
                    "target_value": target_metrics[metric],
                    "delta_target_minus_baseline": target_metrics[metric] - baseline_metrics[metric],
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    return output


def _transition_rows(
    rows: list[dict],
    *,
    compare_pairs: list[tuple[str, str]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    by_method_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_method_sample[row["method"]][row["sample_id"]] = row

    output: list[dict[str, Any]] = []
    outcomes = ["context_only", "prior_only", "both", "neither"]
    for baseline, target in compare_pairs:
        if baseline not in by_method_sample or target not in by_method_sample:
            continue
        sample_ids = sorted(set(by_method_sample[baseline]) & set(by_method_sample[target]))
        counts = Counter(
            (
                by_method_sample[baseline][sample_id]["outcome"],
                by_method_sample[target][sample_id]["outcome"],
            )
            for sample_id in sample_ids
        )
        for baseline_outcome in outcomes:
            for target_outcome in outcomes:
                count = counts[(baseline_outcome, target_outcome)]
                output.append(
                    {
                        **metadata,
                        "baseline": baseline,
                        "target": target,
                        "baseline_outcome": baseline_outcome,
                        "target_outcome": target_outcome,
                        "count": count,
                        "rate": count / len(sample_ids) if sample_ids else 0.0,
                        "n_paired": len(sample_ids),
                    }
                )
    return output


def main() -> None:
    args = parse_args()
    methods = _parse_csv(args.methods)
    allowed_methods = {
        "base_no_rag",
        "base_rag",
        "prompt_rag",
        "strong_rag",
        "prior_objective_rag",
        "random_context",
        "ours_context",
        "ours_prior",
        "ours_delta_context",
        "ours_delta_prior",
        "random_delta_context",
        "random_delta_prior",
        "ours_delta_prefill_context",
        "ours_delta_prefill_prior",
        "random_delta_prefill_context",
        "random_delta_prefill_prior",
        "ours_delta_logit_context",
        "random_delta_logit_context",
        "ours_delta_logit_context_plus_prior_suppression",
        "ours_format_prefill_context",
        "ours_format_prefill_short",
        "ours_format_prefill_verbose",
        "random_format_prefill_context",
        "random_format_prefill_short",
        "random_format_prefill_verbose",
        "ours_commitment_prefill_context",
        "ours_commitment_prefill_single",
        "ours_commitment_prefill_mixed",
        "random_commitment_prefill_context",
        "random_commitment_prefill_single",
        "random_commitment_prefill_mixed",
        "ours_prior_suppression_prefill_no_prior",
        "ours_prior_suppression_prefill_memory",
        "random_prior_suppression_prefill_no_prior",
        "random_prior_suppression_prefill_memory",
        "ours_delta_prefill_context_plus_commitment",
        "random_delta_prefill_context_plus_commitment",
        "ours_delta_prefill_context_plus_prior_suppression",
        "random_delta_prefill_context_plus_prior_suppression",
        "ours_delta_prefill_context_plus_format",
        "random_delta_prefill_context_plus_format",
        "ours_delta_prefill_context_plus_random_format",
        "ours_delta_prefill_context_plus_commitment_plus_format",
    }
    unknown = sorted(set(methods) - allowed_methods)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    selected_components, source_component_source = _select_axis_components(
        summary_csv=args.component_summary_csv,
        selection_pairs=args.selection_pairs,
        top_k=args.top_k_components,
        explicit_ids=args.component_ids,
        fallback_components=[],
        axis="source",
    )
    selected_source_prior_components, source_prior_component_source = _select_axis_components(
        summary_csv=args.source_prior_component_summary_csv,
        selection_pairs=args.selection_pairs,
        top_k=args.top_k_components,
        explicit_ids="",
        fallback_components=selected_components,
        axis="source_prior",
    )
    format_top_k = args.format_top_k_components if args.format_top_k_components is not None else args.top_k_components
    format_selection_pairs = args.format_selection_pairs if args.format_selection_pairs else args.selection_pairs
    selected_format_components, format_component_source = _select_axis_components(
        summary_csv=args.format_component_summary_csv,
        selection_pairs=format_selection_pairs,
        top_k=format_top_k,
        explicit_ids=args.format_component_ids,
        fallback_components=selected_components,
        axis="format",
    )
    selected_format_verbose_components, format_verbose_component_source = _select_axis_components(
        summary_csv=args.format_verbose_component_summary_csv,
        selection_pairs=format_selection_pairs,
        top_k=format_top_k,
        explicit_ids="",
        fallback_components=selected_format_components,
        axis="format_verbose",
    )
    commitment_top_k = (
        args.commitment_top_k_components if args.commitment_top_k_components is not None else args.top_k_components
    )
    selected_commitment_components, commitment_component_source = _select_axis_components(
        summary_csv=args.commitment_component_summary_csv,
        selection_pairs=args.selection_pairs,
        top_k=commitment_top_k,
        explicit_ids=args.commitment_component_ids,
        fallback_components=selected_components,
        axis="commitment",
    )
    selected_commitment_mixed_components, commitment_mixed_component_source = _select_axis_components(
        summary_csv=args.commitment_mixed_component_summary_csv,
        selection_pairs=args.selection_pairs,
        top_k=commitment_top_k,
        explicit_ids="",
        fallback_components=selected_commitment_components,
        axis="commitment_mixed",
    )
    source_component_specs = _component_specs(selected_components)
    source_prior_component_specs = _component_specs(selected_source_prior_components)
    format_component_specs = _component_specs(selected_format_components)
    format_verbose_component_specs = _component_specs(selected_format_verbose_components)
    commitment_component_specs = _component_specs(selected_commitment_components)
    commitment_mixed_component_specs = _component_specs(selected_commitment_mixed_components)
    prior_suppression_top_k = (
        args.prior_suppression_top_k_components
        if args.prior_suppression_top_k_components is not None
        else args.top_k_components
    )
    selected_prior_suppression_components, prior_suppression_component_source = _select_axis_components(
        summary_csv=args.prior_suppression_component_summary_csv,
        selection_pairs=args.selection_pairs,
        top_k=prior_suppression_top_k,
        explicit_ids=args.prior_suppression_component_ids,
        fallback_components=selected_components,
        axis="prior_suppression",
    )
    selected_prior_memory_components, prior_memory_component_source = _select_axis_components(
        summary_csv=args.prior_memory_component_summary_csv,
        selection_pairs=args.selection_pairs,
        top_k=prior_suppression_top_k,
        explicit_ids="",
        fallback_components=selected_prior_suppression_components,
        axis="prior_memory",
    )
    prior_suppression_component_specs = _component_specs(selected_prior_suppression_components)
    prior_memory_component_specs = _component_specs(selected_prior_memory_components)

    train_rows = _select_rows(
        load_jsonl(args.train_rendered_jsonl),
        splits=_parse_csv_set(args.train_splits),
        templates=set(),
        max_rows=args.max_train_rows,
    )
    eval_rows = _select_eval_rows(load_jsonl(args.eval_jsonl), max_rows=args.max_eval_rows, seed=args.seed)
    stop_strings = _parse_stop_strings(args.stop_strings)
    compare_pairs = _parse_compare_pairs(args.compare_pairs)

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    directions_by_label_mode = _learn_component_directions_by_label_mode(
        backend=backend,
        train_rows=train_rows,
        components=selected_components,
        selection_pairs=_parse_csv(args.selection_pairs),
        normalize_diffs=False,
    )
    directions = _mean_label_mode_directions(directions_by_label_mode)
    random_directions = _make_random_directions(backend=backend, directions=directions, seed=args.seed)

    output_rows: list[dict] = []
    for row in tqdm(eval_rows, desc="ckplug generation"):
        delta_cache: dict[str, list[dict[str, Any]]] = {}
        for method in methods:
            prompt, prediction = _generate_for_method(
                backend=backend,
                row=row,
                method=method,
                prompt_key=args.prompt_key,
                directions=directions,
                random_directions=random_directions,
                source_component_specs=source_component_specs,
                source_prior_component_specs=source_prior_component_specs,
                format_component_specs=format_component_specs,
                format_verbose_component_specs=format_verbose_component_specs,
                commitment_component_specs=commitment_component_specs,
                commitment_mixed_component_specs=commitment_mixed_component_specs,
                prior_suppression_component_specs=prior_suppression_component_specs,
                prior_memory_component_specs=prior_memory_component_specs,
                alpha=args.alpha,
                source_delta_kind=args.source_delta_kind,
                source_apply_mode=args.source_apply_mode,
                format_alpha=args.format_alpha,
                format_apply_mode=args.format_apply_mode,
                commitment_alpha=args.commitment_alpha,
                commitment_apply_mode=args.commitment_apply_mode,
                prior_suppression_alpha=args.prior_suppression_alpha,
                prior_suppression_apply_mode=args.prior_suppression_apply_mode,
                candidate_tagging=args.candidate_tagging,
                logit_alpha=args.logit_alpha,
                logit_relative_top=args.logit_relative_top,
                logit_min_tokens_to_keep=args.logit_min_tokens_to_keep,
                logit_apply_mode=args.logit_apply_mode,
                max_new_tokens=args.max_new_tokens,
                stop_strings=stop_strings,
                seed=args.seed,
                delta_cache=delta_cache,
            )
            health = analyze_candidate_health(
                prediction=prediction,
                orig_answers=row["orig_answers"],
                cf_answers=row["cf_answers"],
            )
            scored_prediction = (
                health["final_text_for_scoring"]
                if args.candidate_tagging and args.score_tagged_final
                else health["stripped_prediction"]
            )
            classified = _classify_generation(
                prediction=scored_prediction,
                orig_answers=row["orig_answers"],
                cf_answers=row["cf_answers"],
            )
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "method": method,
                    "prompt_key": "base_no_rag" if method == "base_no_rag" else args.prompt_key,
                    "source_index": row["source_index"],
                    "alias_risk": row.get("alias_risk", ""),
                    "orig_answer": row["orig_answer"],
                    "cf_answer": row["cf_answer"],
                    "prediction": prediction,
                    "prediction_scored": scored_prediction,
                    "prediction_norm": _normalize_answer(scored_prediction),
                    "prompt": prompt,
                    **classified,
                    **{f"health_{key}": value for key, value in health.items() if key != "final_text_for_scoring"},
                }
            )

    metadata = {
        "top_k": args.top_k_components,
        "source_component_source": source_component_source,
        "source_prior_component_source": source_prior_component_source,
        "format_top_k": len(selected_format_components),
        "format_component_source": format_component_source,
        "format_verbose_top_k": len(selected_format_verbose_components),
        "format_verbose_component_source": format_verbose_component_source,
        "commitment_top_k": len(selected_commitment_components),
        "commitment_component_source": commitment_component_source,
        "commitment_mixed_top_k": len(selected_commitment_mixed_components),
        "commitment_mixed_component_source": commitment_mixed_component_source,
        "alpha": args.alpha,
        "source_delta_kind": args.source_delta_kind,
        "source_apply_mode": args.source_apply_mode,
        "format_alpha": args.format_alpha,
        "format_apply_mode": args.format_apply_mode,
        "commitment_alpha": args.commitment_alpha,
        "commitment_apply_mode": args.commitment_apply_mode,
        "prior_suppression_alpha": args.prior_suppression_alpha,
        "prior_suppression_apply_mode": args.prior_suppression_apply_mode,
        "prior_suppression_top_k": len(selected_prior_suppression_components),
        "prior_suppression_component_source": prior_suppression_component_source,
        "prior_memory_top_k": len(selected_prior_memory_components),
        "prior_memory_component_source": prior_memory_component_source,
        "prompt_key": args.prompt_key,
        "max_new_tokens": args.max_new_tokens,
        "stop_strings": args.stop_strings,
        "candidate_tagging": args.candidate_tagging,
        "score_tagged_final": args.score_tagged_final,
        "logit_alpha": args.logit_alpha,
        "logit_relative_top": args.logit_relative_top,
        "logit_min_tokens_to_keep": args.logit_min_tokens_to_keep,
        "logit_apply_mode": args.logit_apply_mode,
    }

    dump_jsonl(args.out_generations_jsonl, output_rows)
    dump_csv(args.out_summary_csv, _summarize(output_rows, metadata=metadata))
    if args.out_comparison_csv is not None:
        dump_csv(
            args.out_comparison_csv,
            _comparison_rows(
                output_rows,
                compare_pairs=compare_pairs,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                metadata=metadata,
            ),
        )
    if args.out_transitions_csv is not None:
        dump_csv(
            args.out_transitions_csv,
            _transition_rows(output_rows, compare_pairs=compare_pairs, metadata=metadata),
        )
    dump_csv(
        args.out_manifest_csv,
        [
            {
                **component,
                "top_k": args.top_k_components,
                "format_top_k": len(selected_format_components),
                "alpha": args.alpha,
                "format_alpha": args.format_alpha,
                "train_rendered_jsonl": str(args.train_rendered_jsonl),
                "eval_jsonl": str(args.eval_jsonl),
                "selection_pairs": args.selection_pairs,
                "source_component_source": source_component_source,
                "source_prior_component_source": source_prior_component_source,
                "format_selection_pairs": format_selection_pairs,
                "format_component_source": format_component_source,
                "format_verbose_component_source": format_verbose_component_source,
                "commitment_component_source": commitment_component_source,
                "commitment_mixed_component_source": commitment_mixed_component_source,
                "prior_suppression_component_source": prior_suppression_component_source,
                "prior_memory_component_source": prior_memory_component_source,
                "direction_source": "mean_of_normal_and_swapped_discovery_directions",
                "methods": args.methods,
                "prompt_key": args.prompt_key,
                "max_new_tokens": args.max_new_tokens,
                "stop_strings": args.stop_strings,
                "candidate_tagging": args.candidate_tagging,
                "score_tagged_final": args.score_tagged_final,
                "logit_alpha": args.logit_alpha,
                "logit_relative_top": args.logit_relative_top,
                "logit_min_tokens_to_keep": args.logit_min_tokens_to_keep,
                "logit_apply_mode": args.logit_apply_mode,
                "compare_pairs": args.compare_pairs,
                "bootstrap_samples": args.bootstrap_samples,
            }
            for component in [
                *selected_components,
                *selected_source_prior_components,
                *selected_format_components,
                *selected_format_verbose_components,
                *selected_commitment_components,
                *selected_commitment_mixed_components,
                *selected_prior_suppression_components,
                *selected_prior_memory_components,
            ]
        ],
    )
    print(
        f"[score-ckplug-generation] train_rows={len(train_rows)} eval_rows={len(eval_rows)} "
        f"methods={methods} source_top_k={args.top_k_components} format_top_k={len(selected_format_components)} "
        f"commitment_top_k={len(selected_commitment_components)} source={source_component_source} "
        f"source_prior={source_prior_component_source} format_source={format_component_source} "
        f"format_verbose={format_verbose_component_source} commitment_source={commitment_component_source} "
        f"commitment_mixed={commitment_mixed_component_source} "
        f"prior_suppression_source={prior_suppression_component_source} prior_memory={prior_memory_component_source} "
        f"alpha={args.alpha} source_delta_kind={args.source_delta_kind} "
        f"candidate_tagging={args.candidate_tagging} logit_alpha={args.logit_alpha}"
    )


if __name__ == "__main__":
    main()
