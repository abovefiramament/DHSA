from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.actuator import limit_rows, load_actuator_pairs
from screscomp.cecm.contribution import (
    ContributionScanConfig,
    ContributionScorer,
    all_component_specs,
    select_core_edges,
)
from screscomp.cecm.objective import OPTION_SELECTION_MODES, option_selection_description
from screscomp.data import dump_csv, dump_json, load_jsonl
from screscomp.modeling import TransformersABBackend


MARGIN_APPLY_MODES = {"decision_tokens", "boxed_decision", "prompt_last", "prompt", "all"}
ROLLOUT_APPLY_MODE_HELP = "all, prefill, decode, first_decode, or first_N_decode"


def _valid_rollout_apply_mode(value: str) -> bool:
    if value in {"all", "prefill", "decode"}:
        return True
    match = re.fullmatch(r"first(?:_(\d+))?_decode", str(value or ""))
    return bool(match and int(match.group(1) or "1") > 0)


class CsvRowStream:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = None
        self._writer = None
        self._rows = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8", newline="")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._rows == 0:
            self.path.write_text("", encoding="utf-8")

    def write(self, row: dict[str, object]) -> None:
        if self._stream is None:
            raise RuntimeError("CsvRowStream must be used as a context manager.")
        if self._writer is None:
            self._writer = csv.DictWriter(self._stream, fieldnames=list(row.keys()), extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow(row)
        self._rows += 1
        if self._rows % 100 == 0:
            self._stream.flush()


class OnlineScalarStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def stderr(self) -> float:
        return self.std / math.sqrt(self.n) if self.n else math.nan


class ComponentSummaryAccumulator:
    def __init__(self) -> None:
        self.by_component: dict[str, dict[str, Any]] = {}

    def update(self, row: dict[str, object]) -> None:
        component_id = str(row["component_id"])
        entry = self.by_component.setdefault(
            component_id,
            {
                "component_id": component_id,
                "layer_idx": row["layer_idx"],
                "component_type": row["component_type"],
                "event": row["event"],
                "apply_mode": row["apply_mode"],
                "stats": {},
                "positive_count": 0,
                "negative_count": 0,
            },
        )
        stats: dict[str, OnlineScalarStats] = entry["stats"]
        for key in (
            "delta",
            "target_drop",
            "source_rise",
            "bidirectional",
            "positive_transition",
            "negative_transition",
            "full_target_score",
            "full_source_score",
            "ablated_target_score",
            "ablated_source_score",
            "full_format_ok",
            "ablated_format_ok",
            "format_collapse",
            "full_completion_words",
            "ablated_completion_words",
            "full_completion_unique_word_ratio",
            "ablated_completion_unique_word_ratio",
        ):
            if key not in row:
                continue
            stats.setdefault(key, OnlineScalarStats()).update(float(row[key]))
        delta = float(row["delta"])
        if delta > 0:
            entry["positive_count"] += 1
        elif delta < 0:
            entry["negative_count"] += 1

    def summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for entry in self.by_component.values():
            stats: dict[str, OnlineScalarStats] = entry["stats"]
            delta_stats = stats["delta"]
            n = delta_stats.n
            mean_delta = delta_stats.mean
            stderr = delta_stats.stderr
            positive_rate = entry["positive_count"] / n if n else math.nan
            negative_rate = entry["negative_count"] / n if n else math.nan
            sign_consistency = max(positive_rate, negative_rate)
            ci_low = mean_delta - 1.96 * stderr if n else math.nan
            ci_high = mean_delta + 1.96 * stderr if n else math.nan
            row: dict[str, object] = {
                "component_id": entry["component_id"],
                "layer_idx": entry["layer_idx"],
                "component_type": entry["component_type"],
                "event": entry["event"],
                "apply_mode": entry["apply_mode"],
                "n": n,
                "mean_delta": mean_delta,
                "abs_mean_delta": abs(mean_delta),
                "std_delta": delta_stats.std,
                "stderr_delta": stderr,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_rate": positive_rate,
                "negative_rate": negative_rate,
                "sign_consistency": sign_consistency,
                "edge_direction": "positive" if mean_delta >= 0 else "negative",
                "ci_excludes_zero": int((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)),
            }
            if "target_drop" in stats:
                bidirectional_rate = stats["bidirectional"].mean
                positive_transition_rate = stats.get("positive_transition", stats["bidirectional"]).mean
                negative_transition_rate = stats.get("negative_transition", OnlineScalarStats()).mean
                directional_transition_rate = (
                    positive_transition_rate if mean_delta >= 0 else negative_transition_rate
                )
                row.update(
                    {
                        "mean_target_drop": stats["target_drop"].mean,
                        "mean_source_rise": stats["source_rise"].mean,
                        "bidirectional_rate": bidirectional_rate,
                        "positive_transition_rate": positive_transition_rate,
                        "negative_transition_rate": negative_transition_rate,
                        "directional_transition_rate": directional_transition_rate,
                        "mean_full_target_score": stats["full_target_score"].mean,
                        "mean_full_source_score": stats["full_source_score"].mean,
                        "mean_ablated_target_score": stats["ablated_target_score"].mean,
                        "mean_ablated_source_score": stats["ablated_source_score"].mean,
                        "state_transition_score": mean_delta * directional_transition_rate,
                    }
                )
            if "ablated_format_ok" in stats:
                row.update(
                    {
                        "mean_full_format_ok": stats["full_format_ok"].mean,
                        "mean_ablated_format_ok": stats["ablated_format_ok"].mean,
                        "format_collapse_rate": stats["format_collapse"].mean,
                        "mean_full_completion_words": stats["full_completion_words"].mean,
                        "mean_ablated_completion_words": stats["ablated_completion_words"].mean,
                        "mean_full_unique_word_ratio": stats["full_completion_unique_word_ratio"].mean,
                        "mean_ablated_unique_word_ratio": stats["ablated_completion_unique_word_ratio"].mean,
                    }
                )
            rows.append(row)

        if any("bidirectional_rate" in row for row in rows):
            return sorted(
                rows,
                key=lambda row: (
                    -float(row.get("directional_transition_rate", row.get("bidirectional_rate", 0.0))),
                    -float(row.get("state_transition_score", 0.0)),
                    -float(row.get("abs_mean_delta", 0.0)),
                    str(row.get("component_id", "")),
                ),
            )
        return sorted(
            rows,
            key=lambda row: (
                -int(row["ci_excludes_zero"]),
                -float(row["abs_mean_delta"]),
                -float(row["sign_consistency"]),
                str(row["component_id"]),
            ),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan CECM component contributions. The default margin mode is the ConFiQA-style "
            "fixed-continuation zero-ablation scan. Rollout mode zero-ablates a component during "
            "open generation and scores the resulting state transition with a reward classifier."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, default=None)
    p.add_argument("--prompts-jsonl", type=Path, default=None)
    p.add_argument("--event", type=str, required=True)
    p.add_argument("--scan-mode", choices=["margin", "rollout"], default="margin")
    p.add_argument("--split", type=str, default="train", help="Use all to keep every admitted prompt in rollout mode.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=70)
    p.add_argument("--component-types", type=str, default="attn,mlp")
    p.add_argument(
        "--apply-mode",
        type=str,
        default="decision_tokens",
        help=(
            "Margin mode: decision_tokens, boxed_decision, prompt_last, prompt, all. "
            f"Rollout mode: {ROLLOUT_APPLY_MODE_HELP}."
        ),
    )
    p.add_argument(
        "--max-aliases-per-side",
        type=int,
        default=0,
        help="0 means use all aliases. Positive values cap endpoint aliases for faster exploratory scans.",
    )
    p.add_argument(
        "--score-mode",
        type=str,
        default="avglogp",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
        help="Margin-mode candidate answer score.",
    )
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="model_max",
        choices=OPTION_SELECTION_MODES,
        help="How to choose among multiple continuation options for each endpoint in margin mode.",
    )
    p.add_argument("--min-abs-delta", type=float, default=0.0)
    p.add_argument("--min-sign-consistency", type=float, default=0.55)
    p.add_argument("--allow-ci-cross-zero", action="store_true")
    p.add_argument("--max-components-per-direction", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--out-dir", type=Path, required=True)

    # Rollout-only arguments. They are ignored by the fixed-continuation margin scan.
    p.add_argument("--prompt-field", type=str, default="prompt")
    p.add_argument("--sample-id-field", type=str, default="sample_id")
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help=(
            "Batch prompt generation in rollout mode. 1 preserves the original per-sample seed path; "
            "larger values use paired batch seeds for full and zero-ablation generations."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scorer-model", type=str, default="siebert/sentiment-roberta-large-english")
    p.add_argument("--target-label", type=str, default="POSITIVE")
    p.add_argument("--source-label", type=str, default="NEGATIVE")
    p.add_argument(
        "--score-text",
        choices=["completion", "full_text", "prompt_plus_completion"],
        default="completion",
        help="Reward-classifier text in rollout mode. Preference-generation scans usually score the completion.",
    )
    p.add_argument("--scorer-batch-size", type=int, default=16)
    p.add_argument("--scorer-max-length", type=int, default=512)
    p.add_argument("--scorer-device", type=int, default=-1)
    return p.parse_args()


def _parse_component_types(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(values) - {"attn", "mlp"})
    if unknown:
        raise ValueError(f"Unsupported component types: {unknown}")
    if not values:
        raise ValueError("--component-types is empty")
    return values


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def _set_seed(backend: TransformersABBackend, seed: int) -> None:
    backend._torch.manual_seed(int(seed))
    if backend._torch.cuda.is_available():
        backend._torch.cuda.manual_seed_all(int(seed))


def _batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise SystemExit("--generation-batch-size must be positive")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _select_prompt_rows(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("admitted", "1")) not in {"0", "false", "False"}
        and (args.split == "all" or str(row.get("split", "")) == args.split)
    ]
    selected = selected[args.start :]
    if args.max_rows is not None and args.max_rows >= 0:
        selected = selected[: args.max_rows]
    return selected


def _prompt_samples(prompt_rows: list[dict[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row_index, row in enumerate(prompt_rows):
        prompt = str(row.get(args.prompt_field, ""))
        sample_id = str(row.get(args.sample_id_field) or row.get("prompt_id") or f"row_{row_index:06d}")
        if not prompt or not sample_id:
            continue
        row_seed = _as_int(row.get("source_row_index", row.get("raw_index", row_index)), row_index)
        for completion_id in range(args.samples_per_prompt):
            samples.append(
                {
                    "sample_id": sample_id,
                    "prompt_id": row.get("prompt_id", ""),
                    "split": row.get("split", ""),
                    "event": row.get("event", args.event),
                    "prompt": prompt,
                    "prefix": row.get("prefix", ""),
                    "completion_id": completion_id,
                    "source_row_index": row.get("source_row_index", row.get("raw_index", row_index)),
                    "generation_seed": int(args.seed) + row_seed * 1009 + completion_id,
                }
            )
    return samples


def _score_text(row: dict[str, Any], *, mode: str) -> str:
    if mode == "completion":
        return str(row.get("completion", ""))
    if mode == "full_text":
        return str(row.get("full_text", ""))
    return f"{row.get('prompt', '')}{row.get('completion', '')}"


def _max_ngram_repeat(words: list[str], n: int) -> int:
    if len(words) < n:
        return 0
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(0, len(words) - n + 1):
        gram = tuple(words[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) if counts else 0


def _completion_format_audit(text: str) -> dict[str, object]:
    stripped = text.strip()
    words = re.findall(r"[A-Za-z][A-Za-z']*", stripped.lower())
    word_count = len(words)
    unique_word_ratio = len(set(words)) / word_count if word_count else 0.0
    max_trigram_repeat = _max_ngram_repeat(words, 3)
    repeated_char_run = 1 if re.search(r"(.)\1{9,}", stripped) else 0
    has_alpha = 1 if any(ch.isalpha() for ch in stripped) else 0
    too_short = 1 if len(stripped) < 12 or word_count < 3 else 0
    low_diversity = 1 if word_count >= 20 and unique_word_ratio < 0.35 else 0
    repetitive = 1 if max_trigram_repeat >= 4 or repeated_char_run else 0
    format_ok = int(has_alpha and not too_short and not low_diversity and not repetitive)
    return {
        "format_ok": format_ok,
        "completion_chars": len(stripped),
        "completion_words": word_count,
        "completion_unique_word_ratio": unique_word_ratio,
        "completion_max_trigram_repeat": max_trigram_repeat,
        "completion_repeated_char_run": repeated_char_run,
        "completion_too_short": too_short,
        "completion_low_diversity": low_diversity,
        "completion_repetitive": repetitive,
    }


def _normalized_classifier_scores(result: Any) -> list[dict[str, object]]:
    candidates = result if isinstance(result, list) else [result]
    normalized: list[dict[str, object]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized.append({"label": str(item.get("label", "")), "score": float(item.get("score", 0.0))})
    return normalized


def _label_score(result: Any, label: str) -> float | None:
    label_upper = label.upper()
    for item in _normalized_classifier_scores(result):
        if str(item["label"]).upper() == label_upper:
            return float(item["score"])
    return None


def _top_label_score(result: Any) -> tuple[str, float]:
    candidates = _normalized_classifier_scores(result)
    if not candidates:
        return "", 0.0
    best = max(candidates, key=lambda item: float(item["score"]))
    return str(best["label"]), float(best["score"])


def _score_generation_records(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    scorer: Any,
    desc: str,
    show_progress: bool = True,
) -> list[dict[str, object]]:
    if not records:
        return []
    if args.scorer_batch_size <= 0:
        raise SystemExit("--scorer-batch-size must be positive")
    target_label = str(args.target_label)
    source_label = str(args.source_label)
    texts = [_score_text(row, mode=args.score_text) for row in records]
    scored_rows: list[dict[str, object]] = []
    starts = range(0, len(records), args.scorer_batch_size)
    if show_progress:
        starts = tqdm(starts, desc=desc)
    for start in starts:
        batch_rows = records[start : start + args.scorer_batch_size]
        batch_texts = texts[start : start + args.scorer_batch_size]
        score_kwargs: dict[str, object] = {"truncation": True}
        if args.scorer_max_length > 0:
            score_kwargs["max_length"] = args.scorer_max_length
        results = scorer(batch_texts, **score_kwargs)
        for row, text, result in zip(batch_rows, batch_texts, results):
            target_score = _label_score(result, target_label)
            source_score = _label_score(result, source_label)
            if target_score is None and source_score is not None:
                target_score = 1.0 - source_score
            if source_score is None and target_score is not None:
                source_score = 1.0 - target_score
            if target_score is None or source_score is None:
                labels = ", ".join(str(item["label"]) for item in _normalized_classifier_scores(result))
                raise RuntimeError(
                    f"Scorer did not return target/source labels ({target_label!r}, {source_label!r}); got: {labels}"
                )
            top_label, top_score = _top_label_score(result)
            scored_rows.append(
                {
                    **row,
                    "scored_text": text,
                    "scored_text_mode": args.score_text,
                    "target_label": target_label,
                    "source_label": source_label,
                    "target_score": float(target_score),
                    "source_score": float(source_score),
                    "target_minus_source_score": float(target_score) - float(source_score),
                    "top_reward_label": top_label,
                    "top_reward_label_score": top_score,
                    "scorer_model": args.scorer_model,
                }
            )
    return scored_rows


def _run_margin_scan(args: argparse.Namespace) -> None:
    if args.pairs_csv is None:
        raise SystemExit("--pairs-csv is required when --scan-mode=margin")
    if args.apply_mode not in MARGIN_APPLY_MODES:
        raise SystemExit(f"--apply-mode={args.apply_mode!r} is not valid for margin mode")

    all_pairs = load_actuator_pairs(args.pairs_csv, event=args.event)
    split_pairs = [pair for pair in all_pairs if pair.split == args.split]
    split_pairs = split_pairs[args.start :]
    pairs = limit_rows(split_pairs, args.max_rows)
    if not pairs:
        raise SystemExit(f"No pairs found for event={args.event!r} split={args.split!r}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        args.out_dir / "scan_config.json",
        {
            "scan_mode": "margin",
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "event": args.event,
            "split": args.split,
            "start": args.start,
            "max_rows": args.max_rows,
            "component_types": _parse_component_types(args.component_types),
            "apply_mode": args.apply_mode,
            "max_aliases_per_side": args.max_aliases_per_side,
            "score_mode": args.score_mode,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "min_abs_delta": args.min_abs_delta,
            "min_sign_consistency": args.min_sign_consistency,
            "require_ci_excludes_zero": not args.allow_ci_cross_zero,
            "max_components_per_direction": args.max_components_per_direction,
            "score": (
                {
                    "avglogp": (
                        "S = max_alias avglogP(y_plus_alias|x) "
                        "- max_alias avglogP(y_minus_alias|x)"
                    ),
                    "top_logit_gap": (
                        "S = max_alias mean_t[logit(answer_token_t)-max_vocab_logit_t] for y_plus "
                        "minus the same score for static y_minus"
                    ),
                    "answer_rest_margin": (
                        "S = max_alias mean_t[logit(answer_token_t)-max_{v!=answer_token_t} logit(v)] "
                        "for static endpoints; dynamic y_minus recomputes max non-gold logic from current logits"
                    ),
                }[args.score_mode]
            ),
            "competitive_margin": (
                "C(M;x)=S_M(y_plus|x)-S_M(y_minus|x) when y_minus options are present; "
                "when y_minus_mode=dynamic_max_non_gold_logic, y_minus is the strongest non-gold "
                "answer logic recomputed from the current forward logits."
            ),
            "delta": "Delta(c,v;x) = C(M_full;x) - C(M_zero_c;x)",
            "ablation": "zero the selected component write at the configured behavior-scoring positions",
            "semantics": (
                "Component discovery estimates causal contribution to the competitive margin "
                "by zero-ablation. Actuator training uses the same competitive margin but a "
                "different causal operator: vector injection rather than ablation."
            ),
        },
    )

    print(
        f"[cecm-scan] mode=margin loading model={args.model} event={args.event} pairs={len(pairs)} split={args.split}",
        flush=True,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    components = all_component_specs(backend.num_layers, _parse_component_types(args.component_types))
    scorer = ContributionScorer(
        backend=backend,
        config=ContributionScanConfig(
            event=args.event,
            apply_mode=args.apply_mode,
            max_aliases_per_side=args.max_aliases_per_side,
            score_mode=args.score_mode,
            option_selection_mode=args.option_selection_mode,
        ),
    )

    full_margins: dict[str, float] = {}
    for pair in tqdm(pairs, desc="full margins"):
        with scorer.torch.no_grad():
            full_margins[pair.sample_id] = float(scorer.margin(pair).detach().cpu().item())

    total = len(pairs) * len(components)
    step = 0
    summary_accumulator = ComponentSummaryAccumulator()
    with CsvRowStream(args.out_dir / "component_delta_samples.csv") as sample_writer:
        for component in components:
            for pair in pairs:
                step += 1
                if step == 1 or step % 100 == 0 or step == total:
                    print(f"[cecm-scan] step={step}/{total} component={component.component_id}", flush=True)
                row = scorer.scan_one(pair, component, full_margin=full_margins[pair.sample_id])
                sample_writer.write(row)
                summary_accumulator.update(row)

    summary_rows = summary_accumulator.summary_rows()
    core_rows = select_core_edges(
        summary_rows,
        min_abs_delta=args.min_abs_delta,
        min_sign_consistency=args.min_sign_consistency,
        require_ci_excludes_zero=not args.allow_ci_cross_zero,
        max_components_per_direction=args.max_components_per_direction,
    )
    dump_csv(args.out_dir / "component_screen.csv", summary_rows)
    dump_csv(args.out_dir / "core_manifest.csv", core_rows)
    dump_csv(
        args.out_dir / "scan_summary.csv",
        [
            {"metric": "scan_mode", "value": "margin"},
            {"metric": "event", "value": args.event},
            {"metric": "split", "value": args.split},
            {"metric": "pairs", "value": len(pairs)},
            {"metric": "components_scanned", "value": len(components)},
            {"metric": "delta_rows", "value": total},
            {"metric": "core_edges", "value": len(core_rows)},
        ],
    )
    print(
        f"[cecm-scan] done mode=margin components={len(components)} deltas={total} "
        f"core_edges={len(core_rows)} out={args.out_dir}",
        flush=True,
    )


def _run_rollout_scan(args: argparse.Namespace) -> None:
    if args.prompts_jsonl is None:
        raise SystemExit("--prompts-jsonl is required when --scan-mode=rollout")
    if not _valid_rollout_apply_mode(args.apply_mode):
        raise SystemExit(f"--apply-mode={args.apply_mode!r} is not valid for rollout mode; expected {ROLLOUT_APPLY_MODE_HELP}")
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples-per-prompt must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.target_label.upper() == args.source_label.upper():
        raise SystemExit("--target-label and --source-label must be different")

    prompt_rows = _select_prompt_rows(load_jsonl(args.prompts_jsonl), args=args)
    samples = _prompt_samples(prompt_rows, args=args)
    if not samples:
        raise SystemExit(f"No prompt samples selected from {args.prompts_jsonl}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop_strings = _parse_stop_strings(args.stop_strings)
    component_types = _parse_component_types(args.component_types)
    dump_json(
        args.out_dir / "scan_config.json",
        {
            "scan_mode": "rollout",
            "model": args.model,
            "prompts_jsonl": str(args.prompts_jsonl),
            "event": args.event,
            "split": args.split,
            "start": args.start,
            "max_rows": args.max_rows,
            "prompt_rows": len(prompt_rows),
            "samples_per_prompt": args.samples_per_prompt,
            "generation_batch_size": args.generation_batch_size,
            "prompt_samples": len(samples),
            "component_types": component_types,
            "apply_mode": args.apply_mode,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "stop_strings": stop_strings,
            "scorer_model": args.scorer_model,
            "target_label": args.target_label,
            "source_label": args.source_label,
            "score_text": args.score_text,
            "competitive_margin": "C(M;x)=reward_target(generate_M(x))-reward_source(generate_M(x))",
            "delta": "Delta(c;x) = C(M_full;x) - C(M_zero_c;x)",
            "state_transition": (
                "A positive support component should have target_drop=full_target-ablated_target>0 "
                "and source_rise=ablated_source-full_source>0 under component removal."
            ),
            "ablation": "zero the selected component write during rollout generation",
        },
    )

    print(
        f"[cecm-scan] mode=rollout loading model={args.model} prompts={len(prompt_rows)} "
        f"samples={len(samples)} event={args.event}",
        flush=True,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    components = all_component_specs(backend.num_layers, component_types)

    print(f"[cecm-scan] mode=rollout loading scorer={args.scorer_model}", flush=True)
    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Rollout scoring requires transformers pipeline support.") from exc
    reward_scorer = pipeline("sentiment-analysis", model=args.scorer_model, device=args.scorer_device, top_k=None)

    generation_batches = _batches(samples, int(args.generation_batch_size))

    full_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for batch_index, batch in enumerate(tqdm(generation_batches, desc="rollout full")):
        if int(args.generation_batch_size) == 1:
            sample = batch[0]
            batch_seed = int(sample["generation_seed"])
            _set_seed(backend, batch_seed)
            completions = [
                backend.generate(
                    str(sample["prompt"]),
                    max_new_tokens=args.max_new_tokens,
                    stop_strings=stop_strings,
                    do_sample=bool(args.do_sample),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            ]
        else:
            batch_seed = int(args.seed) + batch_index * 1009
            _set_seed(backend, batch_seed)
            completions = backend.generate_many(
                [str(sample["prompt"]) for sample in batch],
                max_new_tokens=args.max_new_tokens,
                stop_strings=stop_strings,
                do_sample=bool(args.do_sample),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        full_records = [
            {
                **sample,
                "control_name": "full",
                "completion": completion,
                "full_text": f"{sample['prompt']}{completion}",
                "model": args.model,
                "generation_batch_size": args.generation_batch_size,
                "batch_seed": batch_seed,
            }
            for sample, completion in zip(batch, completions)
        ]
        for row in _score_generation_records(
            full_records,
            args=args,
            scorer=reward_scorer,
            desc="score full batch",
            show_progress=False,
        ):
            full_by_key[(str(row["sample_id"]), int(row["completion_id"]))] = row
    if len(full_by_key) != len(samples):
        raise RuntimeError(f"Full baseline key mismatch: scored={len(full_by_key)} expected={len(samples)}")

    total = len(components) * len(samples)
    step = 0
    summary_accumulator = ComponentSummaryAccumulator()
    with CsvRowStream(args.out_dir / "component_delta_samples.csv") as sample_writer:
        for component in components:
            for batch_index, batch in enumerate(tqdm(generation_batches, desc=f"rollout zero {component.component_id}")):
                previous_step = step
                step += len(batch)
                if step == len(batch) or step // 100 != previous_step // 100 or step == total:
                    print(f"[cecm-scan] rollout step={step}/{total} component={component.component_id}", flush=True)
                if int(args.generation_batch_size) == 1:
                    sample = batch[0]
                    batch_seed = int(sample["generation_seed"])
                    _set_seed(backend, batch_seed)
                    completions = [
                        backend.generate_with_component_zero(
                            str(sample["prompt"]),
                            layer_idx=int(component.layer_idx),
                            component_type=str(component.component_type),
                            max_new_tokens=args.max_new_tokens,
                            stop_strings=stop_strings,
                            do_sample=bool(args.do_sample),
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            apply_mode=args.apply_mode,
                        )
                    ]
                else:
                    batch_seed = int(args.seed) + batch_index * 1009
                    _set_seed(backend, batch_seed)
                    completions = backend.generate_many_with_component_zero(
                        [str(sample["prompt"]) for sample in batch],
                        layer_idx=int(component.layer_idx),
                        component_type=str(component.component_type),
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=bool(args.do_sample),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        apply_mode=args.apply_mode,
                    )
                ablated_records = [
                    {
                        **sample,
                        "control_name": f"zero_{component.component_id}",
                        "component_id": component.component_id,
                        "layer_idx": component.layer_idx,
                        "component_type": component.component_type,
                        "completion": completion,
                        "full_text": f"{sample['prompt']}{completion}",
                        "model": args.model,
                        "generation_batch_size": args.generation_batch_size,
                        "batch_seed": batch_seed,
                    }
                    for sample, completion in zip(batch, completions)
                ]
                scored_ablated = _score_generation_records(
                    ablated_records,
                    args=args,
                    scorer=reward_scorer,
                    desc=f"score zero {component.component_id} batch",
                    show_progress=False,
                )
                for ablated in scored_ablated:
                    key = (str(ablated["sample_id"]), int(ablated["completion_id"]))
                    full = full_by_key[key]
                    full_margin = float(full["target_score"]) - float(full["source_score"])
                    ablated_margin = float(ablated["target_score"]) - float(ablated["source_score"])
                    delta = full_margin - ablated_margin
                    target_drop = float(full["target_score"]) - float(ablated["target_score"])
                    source_rise = float(ablated["source_score"]) - float(full["source_score"])
                    full_audit = _completion_format_audit(str(full.get("completion", "")))
                    ablated_audit = _completion_format_audit(str(ablated.get("completion", "")))
                    format_collapse = int(int(full_audit["format_ok"]) == 1 and int(ablated_audit["format_ok"]) == 0)
                    row = {
                        "sample_id": ablated["sample_id"],
                        "prompt_id": ablated.get("prompt_id", ""),
                        "completion_id": ablated["completion_id"],
                        "split": ablated.get("split", ""),
                        "event": args.event,
                        "component_id": component.component_id,
                        "layer_idx": component.layer_idx,
                        "component_type": component.component_type,
                        "prompt": ablated.get("prompt", ""),
                        "full_completion": full.get("completion", ""),
                        "ablated_completion": ablated.get("completion", ""),
                        "target_label": args.target_label,
                        "source_label": args.source_label,
                        "full_target_score": float(full["target_score"]),
                        "full_source_score": float(full["source_score"]),
                        "ablated_target_score": float(ablated["target_score"]),
                        "ablated_source_score": float(ablated["source_score"]),
                        "full_margin": full_margin,
                        "ablated_margin": ablated_margin,
                        "delta": delta,
                        "delta_sign": "positive" if delta > 0 else ("negative" if delta < 0 else "zero"),
                        "target_drop": target_drop,
                        "source_rise": source_rise,
                        "bidirectional": int(target_drop > 0 and source_rise > 0),
                        "positive_transition": int(target_drop > 0 and source_rise > 0),
                        "negative_transition": int(target_drop < 0 and source_rise < 0),
                        "full_format_ok": full_audit["format_ok"],
                        "ablated_format_ok": ablated_audit["format_ok"],
                        "format_collapse": format_collapse,
                        "full_completion_chars": full_audit["completion_chars"],
                        "full_completion_words": full_audit["completion_words"],
                        "full_completion_unique_word_ratio": full_audit["completion_unique_word_ratio"],
                        "full_completion_max_trigram_repeat": full_audit["completion_max_trigram_repeat"],
                        "full_completion_repeated_char_run": full_audit["completion_repeated_char_run"],
                        "full_completion_too_short": full_audit["completion_too_short"],
                        "full_completion_low_diversity": full_audit["completion_low_diversity"],
                        "full_completion_repetitive": full_audit["completion_repetitive"],
                        "ablated_completion_chars": ablated_audit["completion_chars"],
                        "ablated_completion_words": ablated_audit["completion_words"],
                        "ablated_completion_unique_word_ratio": ablated_audit["completion_unique_word_ratio"],
                        "ablated_completion_max_trigram_repeat": ablated_audit["completion_max_trigram_repeat"],
                        "ablated_completion_repeated_char_run": ablated_audit["completion_repeated_char_run"],
                        "ablated_completion_too_short": ablated_audit["completion_too_short"],
                        "ablated_completion_low_diversity": ablated_audit["completion_low_diversity"],
                        "ablated_completion_repetitive": ablated_audit["completion_repetitive"],
                        "apply_mode": args.apply_mode,
                        "scan_mode": "rollout",
                        "generation_seed": ablated.get("generation_seed", ""),
                        "batch_seed": ablated.get("batch_seed", ""),
                        "generation_batch_size": args.generation_batch_size,
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": int(bool(args.do_sample)),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "score_text": args.score_text,
                        "scorer_model": args.scorer_model,
                    }
                    sample_writer.write(row)
                    summary_accumulator.update(row)

    summary_rows = summary_accumulator.summary_rows()
    core_rows = select_core_edges(
        summary_rows,
        min_abs_delta=args.min_abs_delta,
        min_sign_consistency=args.min_sign_consistency,
        require_ci_excludes_zero=not args.allow_ci_cross_zero,
        max_components_per_direction=args.max_components_per_direction,
    )
    dump_csv(args.out_dir / "component_screen.csv", summary_rows)
    dump_csv(args.out_dir / "core_manifest.csv", core_rows)
    dump_csv(
        args.out_dir / "scan_summary.csv",
        [
            {"metric": "scan_mode", "value": "rollout"},
            {"metric": "event", "value": args.event},
            {"metric": "split", "value": args.split},
            {"metric": "prompt_rows", "value": len(prompt_rows)},
            {"metric": "prompt_samples", "value": len(samples)},
            {"metric": "generation_batch_size", "value": args.generation_batch_size},
            {"metric": "components_scanned", "value": len(components)},
            {"metric": "delta_rows", "value": total},
            {"metric": "core_edges", "value": len(core_rows)},
            {"metric": "target_label", "value": args.target_label},
            {"metric": "source_label", "value": args.source_label},
        ],
    )
    print(
        f"[cecm-scan] done mode=rollout components={len(components)} deltas={total} "
        f"core_edges={len(core_rows)} out={args.out_dir}",
        flush=True,
    )
    return


def main() -> None:
    args = parse_args()
    if args.scan_mode == "margin":
        _run_margin_scan(args)
        return
    _run_rollout_scan(args)


if __name__ == "__main__":
    main()
