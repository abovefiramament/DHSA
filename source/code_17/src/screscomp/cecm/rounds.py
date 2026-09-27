from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from screscomp.cecm.pairs import PairBuildConfig, build_preference_pairs, prompt_from_row, sample_id_from_row
from screscomp.cecm.specs import load_objective_spec
from screscomp.cli.prepare_imdb_sentiment_pairs import build_imdb_sentiment_pairs_from_args
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.gsm8k import (
    base_math_prompt,
    cot_math_prompt,
    deepseek_math_cot_prompt,
    deepseek_math_prompt,
    extract_boxed_answer_text,
    extract_boxed_span,
    extract_final_number,
    normalize_numeric_answer,
    strong_cot_prompt,
)


ROUND_CONFIG_DIR = Path("configs/round")
SOURCE_PAIR_ADAPTER = "source_pair"
GSM8K_FULL_TRAJECTORY_ADAPTER = "gsm8k_full_trajectory"
GSM8K_TRAJECTORY_PREFERENCE_ADAPTER = "gsm8k_trajectory_preference"
GSM8K_ANSWER_MARGIN_ALL_ADAPTER = "gsm8k_answer_margin_all"
GSM8K_EXPLICIT_ANSWER_MARGIN_ADAPTER = "gsm8k_explicit_answer_margin"
GSM8K_SAMPLED_ANSWER_ARBITRATION_ADAPTER = "gsm8k_sampled_answer_arbitration"
PREBUILT_PAIRS_ADAPTER = "prebuilt_pairs"
IMDB_SENTIMENT_SCORED_GENERATIONS_ADAPTER = "imdb_sentiment_scored_generations"
TASK_ADAPTERS = (
    SOURCE_PAIR_ADAPTER,
    GSM8K_FULL_TRAJECTORY_ADAPTER,
    GSM8K_TRAJECTORY_PREFERENCE_ADAPTER,
    GSM8K_ANSWER_MARGIN_ALL_ADAPTER,
    GSM8K_EXPLICIT_ANSWER_MARGIN_ADAPTER,
    GSM8K_SAMPLED_ANSWER_ARBITRATION_ADAPTER,
    PREBUILT_PAIRS_ADAPTER,
    IMDB_SENTIMENT_SCORED_GENERATIONS_ADAPTER,
)


def load_round_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    task = config.get("task", {})
    adapter = str(task.get("adapter", ""))
    if adapter not in TASK_ADAPTERS:
        raise ValueError(f"Unsupported round task.adapter={adapter!r}; expected one of {TASK_ADAPTERS}")
    return config


def round_dir(config: dict[str, Any], *, round_index: int | None = None, out_dir: Path | None = None) -> Path:
    if out_dir is not None:
        return out_dir
    root = Path(str(config.get("output", {}).get("root", "data/rounds/default")))
    index = round_index if round_index is not None else int(config.get("round", {}).get("index", 1))
    return root / f"round_{index:02d}"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    dump_csv(path, rows)


def build_source_pair_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    task = config["task"]
    input_jsonl = Path(str(task["input_jsonl"]))
    rows = load_jsonl(input_jsonl)
    max_rows = config.get("pairs", {}).get("max_rows")
    if max_rows not in (None, ""):
        rows = rows[: int(max_rows)]
    objective_id = str(task.get("event", "source_context_over_prior"))
    spec_json = task.get("objective_spec")
    spec = load_objective_spec(objective_id=objective_id, spec_json=spec_json)
    pair_config = PairBuildConfig(
        event=spec.objective_id,
        prompt_key=str(task.get("prompt_key") or spec.prompt_key),
        y_plus_keys=spec.y_plus_keys,
        y_minus_keys=spec.y_minus_keys,
        y_minus_fallback_keys=spec.y_minus_fallback_keys,
        val_mod=int(task.get("val_mod", spec.val_mod)),
        continuation_prefix=str(task.get("continuation_prefix", spec.continuation_prefix)),
    )
    pairs, rejected = build_preference_pairs(rows, source_path=str(input_jsonl), config=pair_config)
    pair_dir = out_dir / "pairs"
    _write_rows(pair_dir / "pairs.csv", [pair.to_row() for pair in pairs])
    _write_rows(pair_dir / "rejected_pairs.csv", [row.to_row() for row in rejected])
    dump_jsonl(pair_dir / "pairs.jsonl", ({"pair": pair.to_row(), "raw_row": rows[pair.row_index]} for pair in pairs))
    dump_json(
        pair_dir / "pair_build_manifest.json",
        {
            "adapter": SOURCE_PAIR_ADAPTER,
            "input_jsonl": str(input_jsonl),
            "objective_spec": spec.to_dict(),
            "admitted_pairs": len(pairs),
            "rejected_pairs": len(rejected),
        },
    )
    return {"pairs_csv": str(pair_dir / "pairs.csv"), "admitted_pairs": len(pairs), "rejected_pairs": len(rejected)}


def build_prebuilt_pairs_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    task = config["task"]
    source_pairs = Path(str(task["pairs_csv"]))
    if not source_pairs.exists():
        raise FileNotFoundError(f"Missing prebuilt pairs CSV: {source_pairs}")
    pair_dir = out_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = pair_dir / "pairs.csv"
    shutil.copyfile(source_pairs, pairs_csv)
    source_jsonl = task.get("pairs_jsonl")
    if source_jsonl and Path(str(source_jsonl)).exists():
        shutil.copyfile(Path(str(source_jsonl)), pair_dir / "pairs.jsonl")
    rejected_pairs = task.get("rejected_pairs_csv")
    rejected_count = 0
    if rejected_pairs and Path(str(rejected_pairs)).exists():
        rejected_rows = _read_csv_dicts(Path(str(rejected_pairs)))
        rejected_count = len(rejected_rows)
        dump_csv(pair_dir / "rejected_pairs.csv", rejected_rows)
    pair_rows = _read_csv_dicts(pairs_csv)
    dump_json(
        pair_dir / "pair_build_manifest.json",
        {
            "adapter": PREBUILT_PAIRS_ADAPTER,
            "source_pairs_csv": str(source_pairs),
            "source_pairs_jsonl": str(source_jsonl or ""),
            "source_rejected_pairs_csv": str(rejected_pairs or ""),
            "event": str(task.get("event", "")),
            "admitted_pairs": len(pair_rows),
            "rejected_pairs": rejected_count,
            "semantics": "Reuse an already materialized generic CECM y_plus/y_minus pairs.csv.",
        },
    )
    return {"pairs_csv": str(pairs_csv), "admitted_pairs": len(pair_rows), "rejected_pairs": rejected_count}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_imdb_sentiment_scored_generations_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    import argparse

    task = config["task"]
    pairs_cfg = config.get("pairs", {})
    input_path = Path(str(task.get("scored_generations") or task.get("input") or task.get("input_jsonl") or ""))
    if not input_path.exists():
        raise FileNotFoundError(f"Missing scored IMDb generations: {input_path}")
    pair_dir = out_dir / "pairs"
    args = argparse.Namespace(
        input=input_path,
        out_dir=pair_dir,
        event=str(task.get("event", "imdb_positive_sentiment")),
        group_field=str(task.get("group_field", "sample_id")),
        prompt_id_field=str(task.get("prompt_id_field", "prompt_id")),
        prompt_field=str(task.get("prompt_field", "prompt")),
        completion_field=str(task.get("completion_field", "completion")),
        completion_id_field=str(task.get("completion_id_field", "completion_id")),
        score_field=str(task.get("score_field", pairs_cfg.get("score_field", "positive_sentiment_score"))),
        split_field=str(task.get("split_field", "split")),
        source_row_index_field=str(task.get("source_row_index_field", "source_row_index")),
        min_score_margin=float(pairs_cfg.get("min_score_margin", task.get("min_score_margin", 0.0))),
        max_pairs_per_prompt=int(pairs_cfg.get("max_pairs_per_prompt", task.get("max_pairs_per_prompt", 0))),
        max_prompts_per_split=int(pairs_cfg.get("max_prompts_per_split", task.get("max_prompts_per_split", 0))),
        keep_rejected_rows=bool(pairs_cfg.get("keep_rejected_rows", task.get("keep_rejected_rows", False))),
    )
    result = build_imdb_sentiment_pairs_from_args(args)
    manifest_path = pair_dir / "pair_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "adapter": IMDB_SENTIMENT_SCORED_GENERATIONS_ADAPTER,
            "round_task": str(task.get("name", "")),
        }
    )
    dump_json(manifest_path, manifest)
    return {
        "pairs_csv": str(pair_dir / "pairs.csv"),
        "admitted_pairs": int(result["admitted_pairs"]),
        "rejected_pairs": int(result["rejected_pairs"]),
    }


def _split_for_index(row_index: int, val_mod: int) -> str:
    if val_mod <= 1:
        return "train"
    return "val" if row_index % val_mod == 0 else "train"


def _gsm8k_gold(row: dict[str, Any]) -> str:
    for key in ("gold_answer", "cf_answer", "context_answer"):
        value = normalize_numeric_answer(row.get(key, ""))
        if value:
            return value
    return extract_final_number(row.get("gold_solution", row.get("answer", "")))


def _gsm8k_question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt"):
        text = " ".join(str(row.get(key, "")).strip().split())
        if text:
            return text
    return ""


def _gsm8k_prompt(row: dict[str, Any], prompt_key: str) -> tuple[str, str]:
    prompt, source = prompt_from_row(row, prompt_key)
    if prompt:
        return prompt, source
    question = _gsm8k_question(row)
    if not question:
        return "", ""
    prompts = {
        "base_math": base_math_prompt(question),
        "cot_math": cot_math_prompt(question),
        "deepseek_math_cot": deepseek_math_cot_prompt(question),
        "deepseek_math": deepseek_math_prompt(question),
        "strong_cot": strong_cot_prompt(question),
    }
    return prompts.get(prompt_key, deepseek_math_cot_prompt(question)), f"virtual.{prompt_key}"


def _boxed_answer_continuation(gold: str, prefix: str = " ") -> str:
    gold = normalize_numeric_answer(gold)
    if not gold:
        return ""
    return f"{prefix}\\boxed{{{gold}}}"


def _neighbor_wrong_answer(gold: str) -> str:
    gold = normalize_numeric_answer(gold)
    if not gold:
        return ""
    try:
        value = Decimal(gold)
    except InvalidOperation:
        return f"{gold}_wrong"
    wrong = value + Decimal(1)
    if wrong == wrong.to_integral_value():
        return str(wrong.quantize(Decimal(1)))
    return format(wrong.normalize(), "f").rstrip("0").rstrip(".")


def build_gsm8k_answer_margin_all_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    task = config["task"]
    source_jsonl = Path(str(task["source_jsonl"]))
    prompt_key = str(task.get("prompt_key", "deepseek_math"))
    event = str(task.get("event", "gsm8k_answer_margin_all"))
    val_mod = int(task.get("val_mod", 5))
    continuation_prefix = str(task.get("continuation_prefix", " "))
    rows = load_jsonl(source_jsonl)
    max_rows = config.get("pairs", {}).get("max_rows")
    if max_rows not in (None, ""):
        rows = rows[: int(max_rows)]

    pair_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    summary_counts: defaultdict[str, int] = defaultdict(int)
    for row_index, row in enumerate(rows):
        sample_id = sample_id_from_row(row, row_index)
        prompt, prompt_source = _gsm8k_prompt(row, prompt_key)
        gold = _gsm8k_gold(row)
        question = _gsm8k_question(row)
        continuation = _boxed_answer_continuation(gold, prefix=continuation_prefix)
        if not prompt or not gold or not continuation:
            reason = "missing_prompt_or_gold"
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "event": event,
                    "row_index": row_index,
                    "admitted": 0,
                    "reject_reason": reason,
                    "prompt_key": prompt_key,
                    "prompt_source": prompt_source,
                    "gold_answer": gold,
                }
            )
            summary_counts[reason] += 1
            continue
        summary_counts["admitted"] += 1
        pair_rows.append(
            {
                "sample_id": f"{event}_{row_index:06d}",
                "source_sample_id": sample_id,
                "event": event,
                "split": _split_for_index(row_index, val_mod),
                "prompt": prompt,
                "prompt_key": prompt_key,
                "prompt_source": prompt_source,
                "question": question,
                "gold_answer": gold,
                "sampled_answer": "",
                "raw_open_generation": "",
                "y_plus": continuation,
                "y_minus": "",
                "y_plus_continuation": continuation,
                "y_minus_continuation": "",
                "y_plus_continuations_json": json.dumps([continuation], ensure_ascii=False),
                "y_minus_continuations_json": json.dumps([], ensure_ascii=False),
                "y_plus_score_text": gold,
                "y_minus_score_text": "",
                "y_minus_mode": "dynamic_max_non_gold_logic",
                "score_text": gold,
                "y_plus_alias_count": 1,
                "y_minus_alias_count": 0,
                "admitted": 1,
                "invariants_pass": 1,
                "reject_reason": "",
                "source_path": str(source_jsonl),
                "row_index": row_index,
                "pair_mode": "gsm8k_answer_margin_all_samples",
                "pair_source": "source_gold",
                "preference_evidence": "gold_answer_vs_dynamic_non_gold_vocab_competitor",
                "continuation_policy": "boxed_gold_answer_margin",
                "answer_slot_policy": "score_gold_answer_span_only",
                "competitive_margin": (
                    "C(M;x)=S_M(gold answer tokens | x, boxed-answer prefix) "
                    "- max_non_gold_vocab_logit at the same score span. "
                    "This admits every source sample with a valid prompt and gold answer."
                ),
            }
        )

    pair_dir = out_dir / "pairs"
    _write_rows(pair_dir / "pairs.csv", pair_rows)
    _write_rows(pair_dir / "rejected_pairs.csv", rejected_rows)
    dump_jsonl(
        pair_dir / "pairs.jsonl",
        (
            {"pair": pair_row, "raw_row": rows[int(pair_row["row_index"])]}
            for pair_row in pair_rows
        ),
    )
    dump_json(
        pair_dir / "pair_build_manifest.json",
        {
            "adapter": GSM8K_ANSWER_MARGIN_ALL_ADAPTER,
            "source_jsonl": str(source_jsonl),
            "event": event,
            "prompt_key": prompt_key,
            "val_mod": val_mod,
            "continuation_policy": "boxed_gold_answer_margin",
            "admitted_pairs": len(pair_rows),
            "rejected_pairs": len(rejected_rows),
            "rejection_counts": dict(sorted(summary_counts.items())),
            "semantics": (
                "GSM8K answer-margin-all materializes one pair per source question with a valid "
                "prompt and gold answer. It does not read sampled rollouts, does not require a "
                "current mixed state, and does not build recovery-memory pairs. The shared core "
                "uses score_mode=answer_rest_margin with y_minus_mode=dynamic_max_non_gold_logic, "
                "so the objective pushes the gold answer span over the strongest non-gold vocab "
                "competitor while training on all admitted samples."
            ),
        },
    )
    return {"pairs_csv": str(pair_dir / "pairs.csv"), "admitted_pairs": len(pair_rows), "rejected_pairs": len(rejected_rows)}


def _rollout_answer_counts_by_index(
    paths: list[Path],
    *,
    control_names: set[str],
    require_parse_certified: bool,
) -> dict[int, dict[str, int]]:
    by_index: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in paths:
        if not path.exists():
            continue
        for rollout in load_jsonl(path):
            if control_names and str(rollout.get("control_name", "")) not in control_names:
                continue
            if require_parse_certified and "parse_certified" in rollout and not _as_bool(rollout.get("parse_certified")):
                continue
            row_index = _rollout_row_index(rollout)
            if row_index is None:
                continue
            parsed = _rollout_parsed(rollout)
            if parsed:
                by_index[row_index][parsed] += 1
    return {idx: dict(counts) for idx, counts in by_index.items()}


def build_gsm8k_explicit_answer_margin_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    task = config["task"]
    source_jsonl = Path(str(task["source_jsonl"]))
    prompt_key = str(task.get("prompt_key", "deepseek_math"))
    event = str(task.get("event", "gsm8k_explicit_answer_margin"))
    val_mod = int(task.get("val_mod", 5))
    continuation_prefix = str(task.get("continuation_prefix", " "))
    wrong_rollout_paths = _path_list(
        task.get("explicit_wrong_rollouts_jsonl")
        or task.get("wrong_rollouts_jsonl")
        or task.get("rollouts_jsonl")
        or task.get("explicit_wrong_rollouts_jsonls")
    )
    wrong_control_names = _string_set(task.get("wrong_rollout_control_names") or task.get("rollout_control_names"))
    require_parse_certified = _as_bool(task.get("require_parse_certified"), default=True)
    require_explicit_wrong = _as_bool(task.get("require_explicit_wrong"), default=False)
    fallback_policy = str(task.get("wrong_fallback_policy", "neighbor"))
    max_wrong_options = int(task.get("max_wrong_options", 3))
    rows = load_jsonl(source_jsonl)
    max_rows = config.get("pairs", {}).get("max_rows")
    if max_rows not in (None, ""):
        rows = rows[: int(max_rows)]
    rollout_answer_counts = _rollout_answer_counts_by_index(
        wrong_rollout_paths,
        control_names=wrong_control_names,
        require_parse_certified=require_parse_certified,
    )

    pair_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    summary_counts: defaultdict[str, int] = defaultdict(int)
    for row_index, row in enumerate(rows):
        sample_id = sample_id_from_row(row, row_index)
        prompt, prompt_source = _gsm8k_prompt(row, prompt_key)
        gold = _gsm8k_gold(row)
        question = _gsm8k_question(row)
        y_plus = _boxed_answer_continuation(gold, prefix=continuation_prefix)
        wrong_counts = {
            answer: count
            for answer, count in rollout_answer_counts.get(row_index, {}).items()
            if answer and answer != gold
        }
        wrong_answers = [
            answer
            for answer, _count in sorted(wrong_counts.items(), key=lambda item: (-item[1], item[0]))
        ][:max_wrong_options]
        pair_source = "rollout_wrong"
        if not wrong_answers and not require_explicit_wrong and fallback_policy == "neighbor":
            fallback = _neighbor_wrong_answer(gold)
            if fallback and fallback != gold:
                wrong_answers = [fallback]
                pair_source = "fallback_neighbor"
        if not prompt or not gold or not y_plus:
            reason = "missing_prompt_or_gold"
        elif not wrong_answers:
            reason = "missing_explicit_wrong_answer"
        else:
            reason = ""
        if reason:
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "event": event,
                    "row_index": row_index,
                    "admitted": 0,
                    "reject_reason": reason,
                    "prompt_key": prompt_key,
                    "prompt_source": prompt_source,
                    "gold_answer": gold,
                    "wrong_answer_counts_json": json.dumps(wrong_counts, ensure_ascii=False),
                }
            )
            summary_counts[reason] += 1
            continue
        y_minus_options = [_boxed_answer_continuation(answer, prefix=continuation_prefix) for answer in wrong_answers]
        y_minus_options = [text for text in y_minus_options if text]
        if not y_minus_options:
            summary_counts["missing_explicit_wrong_answer"] += 1
            continue
        summary_counts["admitted"] += 1
        summary_counts[f"admitted_{pair_source}"] += 1
        pair_rows.append(
            {
                "sample_id": f"{event}_{row_index:06d}",
                "source_sample_id": sample_id,
                "event": event,
                "split": _split_for_index(row_index, val_mod),
                "prompt": prompt,
                "prompt_key": prompt_key,
                "prompt_source": prompt_source,
                "question": question,
                "gold_answer": gold,
                "sampled_answer": wrong_answers[0],
                "raw_open_generation": "",
                "y_plus": y_plus,
                "y_minus": y_minus_options[0],
                "y_plus_continuation": y_plus,
                "y_minus_continuation": y_minus_options[0],
                "y_plus_continuations_json": json.dumps([y_plus], ensure_ascii=False),
                "y_minus_continuations_json": json.dumps(y_minus_options, ensure_ascii=False),
                "y_plus_score_text": gold,
                "y_minus_score_text": wrong_answers[0],
                "y_minus_mode": "",
                "score_text": gold,
                "y_plus_alias_count": 1,
                "y_minus_alias_count": len(y_minus_options),
                "admitted": 1,
                "invariants_pass": 1,
                "reject_reason": "",
                "source_path": str(source_jsonl),
                "row_index": row_index,
                "pair_mode": "gsm8k_explicit_answer_margin",
                "pair_source": pair_source,
                "preference_evidence": "boxed_gold_answer_vs_explicit_wrong_answer",
                "continuation_policy": "boxed_gold_vs_boxed_explicit_wrong",
                "answer_slot_policy": "score_gold_and_wrong_answer_spans",
                "explicit_wrong_answers_json": json.dumps(wrong_answers, ensure_ascii=False),
                "wrong_answer_counts_json": json.dumps(wrong_counts, ensure_ascii=False),
                "competitive_margin": (
                    "C(M;x)=S_M(boxed gold answer span|x)-S_M(boxed explicit wrong answer span|x). "
                    "Wrong answers are sourced from model rollouts when available, with an optional "
                    "deterministic neighbor fallback to keep all valid GSM8K source samples admitted."
                ),
            }
        )

    pair_dir = out_dir / "pairs"
    _write_rows(pair_dir / "pairs.csv", pair_rows)
    _write_rows(pair_dir / "rejected_pairs.csv", rejected_rows)
    dump_jsonl(
        pair_dir / "pairs.jsonl",
        (
            {"pair": pair_row, "raw_row": rows[int(pair_row["row_index"])]}
            for pair_row in pair_rows
        ),
    )
    dump_json(
        pair_dir / "pair_build_manifest.json",
        {
            "adapter": GSM8K_EXPLICIT_ANSWER_MARGIN_ADAPTER,
            "source_jsonl": str(source_jsonl),
            "explicit_wrong_rollouts_jsonl": [str(path) for path in wrong_rollout_paths],
            "event": event,
            "prompt_key": prompt_key,
            "val_mod": val_mod,
            "wrong_rollout_control_names": sorted(wrong_control_names),
            "require_parse_certified": require_parse_certified,
            "require_explicit_wrong": require_explicit_wrong,
            "wrong_fallback_policy": fallback_policy,
            "max_wrong_options": max_wrong_options,
            "continuation_policy": "boxed_gold_vs_boxed_explicit_wrong",
            "admitted_pairs": len(pair_rows),
            "rejected_pairs": len(rejected_rows),
            "rejection_counts": dict(sorted(summary_counts.items())),
            "semantics": (
                "GSM8K explicit-answer-margin materializes ConFiQA-style target-vs-competing "
                "answer pairs. y_plus is the boxed gold answer; y_minus is an explicit boxed "
                "wrong answer, preferably from frozen-model rollouts. If configured, a simple "
                "neighbor wrong-answer fallback preserves all-sample coverage. The shared core "
                "still computes the answer_rest_margin endpoint score and optimizes the same "
                "competitive margin machinery."
            ),
        },
    )
    return {"pairs_csv": str(pair_dir / "pairs.csv"), "admitted_pairs": len(pair_rows), "rejected_pairs": len(rejected_rows)}


def build_gsm8k_sampled_answer_arbitration_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    task = config["task"]
    source_jsonl = Path(str(task["source_jsonl"]))
    prompt_key = str(task.get("prompt_key", "deepseek_math"))
    event = str(task.get("event", "gsm8k_sampled_answer_arbitration"))
    val_mod = int(task.get("val_mod", 5))
    continuation_prefix = str(task.get("continuation_prefix", " "))
    rollout_paths = _path_list(task.get("rollouts_jsonl") or task.get("sampled_rollouts_jsonl"))
    rollout_control_names = _string_set(task.get("rollout_control_names"))
    require_parse_certified = _as_bool(task.get("require_parse_certified"), default=True)
    max_pairs_per_question = int(task.get("max_pairs_per_question", 0) or 0)
    dedupe_answers = _as_bool(task.get("dedupe_answers_per_question"), default=True)
    rows = load_jsonl(source_jsonl)
    max_rows = config.get("pairs", {}).get("max_rows")
    if max_rows not in (None, ""):
        rows = rows[: int(max_rows)]

    pair_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    summary_counts: defaultdict[str, int] = defaultdict(int)
    used_by_question: dict[int, set[str]] = defaultdict(set)
    count_by_question: dict[int, int] = defaultdict(int)
    rollouts_by_question: dict[int, list[dict[str, object]]] = defaultdict(list)

    for path in rollout_paths:
        if not path.exists():
            summary_counts["missing_rollout_path"] += 1
            continue
        for rollout in load_jsonl(path):
            control_name = str(rollout.get("control_name", ""))
            if rollout_control_names and control_name not in rollout_control_names:
                continue
            row_index = _rollout_row_index(rollout)
            if row_index is None or row_index >= len(rows):
                summary_counts["missing_source_row"] += 1
                continue
            if require_parse_certified and "parse_certified" in rollout and not _as_bool(rollout.get("parse_certified")):
                summary_counts["rejected_uncertified_parse"] += 1
                continue

            parsed = _rollout_parsed(rollout)
            if not parsed:
                summary_counts["rejected_missing_sampled_answer"] += 1
                continue
            prediction = _rollout_prediction(rollout)
            rollouts_by_question[row_index].append(
                {
                    "parsed": parsed,
                    "prediction": prediction,
                    "control_name": control_name,
                    "path": str(path),
                    "explicit_boxed_answers": _explicit_boxed_answers(prediction),
                }
            )

    if not rollout_paths:
        summary_counts["missing_rollouts_jsonl"] += 1

    def add_pair(
        *,
        source_row: dict[str, Any],
        row_index: int,
        prompt: str,
        prompt_source: str,
        question: str,
        gold: str,
        rollout: dict[str, object],
        y_plus: str,
        y_minus: str,
        y_plus_options: list[str],
        y_minus_options: list[str],
        y_plus_score_text: str,
        y_minus_score_text: str,
        y_plus_mode: str,
        y_minus_mode: str,
        pair_source: str,
        sampled_correct: bool,
        question_state: str,
        preference_evidence: str,
        answer_slot_policy: str,
    ) -> None:
        if max_pairs_per_question > 0 and count_by_question[row_index] >= max_pairs_per_question:
            summary_counts["rejected_question_pair_cap"] += 1
            return
        if dedupe_answers:
            answer_key = f"{pair_source}:{rollout['parsed']}:{y_minus_score_text}:{y_plus_mode}:{y_minus_mode}"
            if answer_key in used_by_question[row_index]:
                summary_counts["rejected_duplicate_answer_for_question"] += 1
                return
            used_by_question[row_index].add(answer_key)
        summary_counts["admitted"] += 1
        summary_counts[f"admitted_{pair_source}"] += 1
        summary_counts[f"admitted_state_{question_state}"] += 1
        count_by_question[row_index] += 1
        pair_rows.append(
            {
                "sample_id": f"{event}_{len(pair_rows):06d}",
                "source_sample_id": sample_id_from_row(source_row, row_index),
                "event": event,
                "split": _split_for_index(row_index, val_mod),
                "prompt": prompt,
                "prompt_key": prompt_key,
                "prompt_source": prompt_source,
                "question": question,
                "gold_answer": gold,
                "sampled_answer": rollout["parsed"],
                "raw_open_generation": rollout["prediction"],
                "rollout_control_name": rollout["control_name"],
                "y_plus": y_plus,
                "y_minus": y_minus,
                "y_plus_continuation": y_plus,
                "y_minus_continuation": y_minus,
                "y_plus_continuations_json": json.dumps(y_plus_options, ensure_ascii=False),
                "y_minus_continuations_json": json.dumps(y_minus_options, ensure_ascii=False),
                "y_plus_score_text": y_plus_score_text,
                "y_minus_score_text": y_minus_score_text,
                "y_plus_mode": y_plus_mode,
                "y_minus_mode": y_minus_mode,
                "score_text": y_plus_score_text,
                "y_plus_alias_count": len(y_plus_options),
                "y_minus_alias_count": len(y_minus_options),
                "admitted": 1,
                "invariants_pass": 1,
                "reject_reason": "",
                "source_path": str(source_jsonl),
                "rollout_path": rollout["path"],
                "row_index": row_index,
                "pair_mode": "gsm8k_sampled_answer_arbitration",
                "pair_source": pair_source,
                "sampled_answer_correct": int(sampled_correct),
                "question_sample_state": question_state,
                "explicit_boxed_answers_json": json.dumps(rollout["explicit_boxed_answers"], ensure_ascii=False),
                "preference_evidence": preference_evidence,
                "continuation_policy": "sampled_answer_correctness_arbitration",
                "answer_slot_policy": answer_slot_policy,
                "competitive_margin": (
                    "Mixed questions use only relative correct-vs-wrong competition. "
                    "Pure-correct questions use absolute gold logic. Pure-wrong questions use absolute "
                    "0-minus-wrong logic unless the sampled logic explicitly contains the gold answer."
                ),
            }
        )

    for row_index, question_rollouts in sorted(rollouts_by_question.items()):
        source_row = rows[row_index]
        prompt, prompt_source = _gsm8k_prompt(source_row, prompt_key)
        gold = _gsm8k_gold(source_row)
        question = _gsm8k_question(source_row)
        if not prompt or not gold:
            summary_counts["rejected_missing_prompt_or_gold"] += len(question_rollouts)
            continue
        gold_continuation = _boxed_answer_continuation(gold, prefix=continuation_prefix)
        if not gold_continuation:
            summary_counts["rejected_missing_continuation"] += len(question_rollouts)
            continue
        has_correct = any(str(rollout["parsed"]) == gold for rollout in question_rollouts)
        has_wrong = any(str(rollout["parsed"]) != gold for rollout in question_rollouts)
        if has_correct and has_wrong:
            question_state = "mixed"
        elif has_correct:
            question_state = "pure_correct"
        else:
            question_state = "pure_wrong"

        for rollout in question_rollouts:
            parsed = str(rollout["parsed"])
            sampled_continuation = _boxed_answer_continuation(parsed, prefix=continuation_prefix)
            if not sampled_continuation:
                summary_counts["rejected_missing_continuation"] += 1
                continue
            explicit_boxed_answers = [str(answer) for answer in rollout["explicit_boxed_answers"]]
            sampled_correct = parsed == gold
            if sampled_correct:
                if question_state == "pure_correct":
                    add_pair(
                        source_row=source_row,
                        row_index=row_index,
                        prompt=prompt,
                        prompt_source=prompt_source,
                        question=question,
                        gold=gold,
                        rollout=rollout,
                        y_plus=gold_continuation,
                        y_minus="",
                        y_plus_options=[gold_continuation],
                        y_minus_options=[],
                        y_plus_score_text=gold,
                        y_minus_score_text="",
                        y_plus_mode="",
                        y_minus_mode="dynamic_max_non_gold_logic",
                        pair_source="pure_correct_absolute_gold",
                        sampled_correct=True,
                        question_state=question_state,
                        preference_evidence="pure_correct_boost_gold_over_dynamic_non_gold",
                        answer_slot_policy="absolute_correct_logic",
                    )
                else:
                    summary_counts["skipped_mixed_correct_absolute_gold"] += 1
                for explicit_wrong in explicit_boxed_answers:
                    if explicit_wrong == gold:
                        continue
                    explicit_wrong_continuation = _boxed_answer_continuation(explicit_wrong, prefix=continuation_prefix)
                    if not explicit_wrong_continuation:
                        continue
                    add_pair(
                        source_row=source_row,
                        row_index=row_index,
                        prompt=prompt,
                        prompt_source=prompt_source,
                        question=question,
                        gold=gold,
                        rollout=rollout,
                        y_plus=gold_continuation,
                        y_minus=explicit_wrong_continuation,
                        y_plus_options=[gold_continuation],
                        y_minus_options=[explicit_wrong_continuation],
                        y_plus_score_text=gold,
                        y_minus_score_text=explicit_wrong,
                        y_plus_mode="",
                        y_minus_mode="",
                        pair_source="sampled_correct_logic_contains_wrong",
                        sampled_correct=True,
                        question_state=question_state,
                        preference_evidence="explicit_wrong_inside_correct_logic_suppressed_by_gold",
                        answer_slot_policy="explicit_wrong_logic_relative_to_observed_gold",
                    )
                continue

            gold_is_observed = has_correct or gold in explicit_boxed_answers
            if gold_is_observed:
                add_pair(
                    source_row=source_row,
                    row_index=row_index,
                    prompt=prompt,
                    prompt_source=prompt_source,
                    question=question,
                    gold=gold,
                    rollout=rollout,
                    y_plus=gold_continuation,
                    y_minus=sampled_continuation,
                    y_plus_options=[gold_continuation],
                    y_minus_options=[sampled_continuation],
                    y_plus_score_text=gold,
                    y_minus_score_text=parsed,
                    y_plus_mode="",
                    y_minus_mode="",
                    pair_source="sampled_wrong_relative_gold_over_wrong",
                    sampled_correct=False,
                    question_state=question_state,
                    preference_evidence="sampled_wrong_suppressed_against_observed_gold_logic",
                    answer_slot_policy="relative_wrong_logic_with_observed_gold",
                )
            else:
                add_pair(
                    source_row=source_row,
                    row_index=row_index,
                    prompt=prompt,
                    prompt_source=prompt_source,
                    question=question,
                    gold=gold,
                    rollout=rollout,
                    y_plus="",
                    y_minus=sampled_continuation,
                    y_plus_options=[],
                    y_minus_options=[sampled_continuation],
                    y_plus_score_text="",
                    y_minus_score_text=parsed,
                    y_plus_mode="constant_zero_logic",
                    y_minus_mode="",
                    pair_source="sampled_wrong_absolute_zero_over_wrong",
                    sampled_correct=False,
                    question_state=question_state,
                    preference_evidence="pure_wrong_suppresses_observed_wrong_without_inserting_gold",
                    answer_slot_policy="absolute_wrong_logic",
                )

    pair_rows.sort(key=lambda row: (str(row.get("rollout_control_name", "")), int(row["row_index"]), str(row["sample_id"])))

    pair_dir = out_dir / "pairs"
    _write_rows(pair_dir / "pairs.csv", pair_rows)
    _write_rows(pair_dir / "rejected_pairs.csv", rejected_rows)
    dump_jsonl(
        pair_dir / "pairs.jsonl",
        (
            {"pair": pair_row, "raw_row": rows[int(pair_row["row_index"])]}
            for pair_row in pair_rows
        ),
    )
    dump_json(
        pair_dir / "pair_build_manifest.json",
        {
            "adapter": GSM8K_SAMPLED_ANSWER_ARBITRATION_ADAPTER,
            "source_jsonl": str(source_jsonl),
            "rollouts_jsonl": [str(path) for path in rollout_paths],
            "event": event,
            "prompt_key": prompt_key,
            "val_mod": val_mod,
            "rollout_control_names": sorted(rollout_control_names),
            "require_parse_certified": require_parse_certified,
            "dedupe_answers_per_question": dedupe_answers,
            "max_pairs_per_question": max_pairs_per_question,
            "continuation_policy": "sampled_answer_correctness_arbitration",
            "admitted_pairs": len(pair_rows),
            "rejected_pairs": len(rejected_rows),
            "rejection_counts": dict(sorted(summary_counts.items())),
            "semantics": (
                "GSM8K sampled-answer arbitration uses observed no-CoT sampled answers as labeled "
                "candidate states. Mixed questions use only relative correct-vs-observed-wrong "
                "competition. Pure-correct questions use absolute gold-vs-dynamic non-gold logic. "
                "Pure-wrong questions use constant-zero-vs-observed-wrong logic, unless the sampled "
                "logic explicitly contains the gold answer. The adapter never fabricates neighbor "
                "negatives and does not change the shared core actuator objective."
            ),
        },
    )
    return {"pairs_csv": str(pair_dir / "pairs.csv"), "admitted_pairs": len(pair_rows), "rejected_pairs": len(rejected_rows)}


def _rollout_prediction(row: dict[str, Any]) -> str:
    for key in ("prediction", "completion", "output", "text", "response"):
        text = str(row.get(key, "")).strip()
        if text:
            return text
    return ""


def _rollout_parsed(row: dict[str, Any]) -> str:
    for key in ("parsed_answer", "final_answer", "answer"):
        value = normalize_numeric_answer(row.get(key, ""))
        if value:
            return value
    return extract_final_number(_rollout_prediction(row))


def _explicit_boxed_answers(text: Any) -> list[str]:
    answers: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"\\boxed\s*\{([^{}]+)\}", str(text or "")):
        normalized = normalize_numeric_answer(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            answers.append(normalized)
    return answers


def _rollout_row_index(row: dict[str, Any]) -> int | None:
    sample_id = str(row.get("sample_id", "") or row.get("source_sample_id", ""))
    if sample_id.startswith("row_"):
        try:
            return int(sample_id.rsplit("_", 1)[-1])
        except ValueError:
            return None
    if row.get("source_row_index") not in (None, ""):
        try:
            return int(row["source_row_index"])
        except Exception:
            return None
    return None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _string_set(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _path_list(value: Any) -> list[Path]:
    if value in (None, ""):
        return []
    if isinstance(value, (str, Path)):
        return [Path(str(value))]
    return [Path(str(item)) for item in value if str(item).strip()]


def _rollout_sort_key(row: dict[str, Any]) -> tuple[int, str, int, int]:
    rollout_index = row.get("rollout_index")
    try:
        rollout_rank = int(rollout_index)
    except Exception:
        rollout_rank = 1_000_000
    prediction = _rollout_prediction(row)
    token_count = row.get("token_count")
    try:
        length = int(token_count)
    except Exception:
        length = len(prediction)
    return rollout_rank, str(row.get("control_name", "")), length, len(prediction)


def _clean_rollout(
    row: dict[str, Any],
    *,
    require_parse_certified: bool,
) -> tuple[str, str] | None:
    prediction = _rollout_prediction(row)
    parsed = _rollout_parsed(row)
    if not prediction or not parsed:
        return None
    if require_parse_certified and "parse_certified" in row and not _as_bool(row.get("parse_certified")):
        return None
    return prediction, parsed


def _cap_unique_with_origin(items: list[tuple[str, str]], *, limit: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for text, origin in items:
        if text in seen:
            continue
        seen.add(text)
        out.append((text, origin))
        if limit > 0 and len(out) >= limit:
            break
    return out


def _state_counter(summary: dict[str, dict[str, int]], state: str, key: str, amount: int = 1) -> None:
    if state not in summary:
        summary[state] = {"clean": 0, "correct": 0, "wrong": 0, "low_quality_correct": 0}
    summary[state][key] = int(summary[state].get(key, 0)) + amount


def _positive_reasoning_quality(
    prediction: str,
    *,
    require_boxed: bool,
    min_reasoning_chars: int,
    min_reasoning_number_mentions: int,
    reject_phrases: set[str],
) -> bool:
    text = str(prediction or "").strip()
    if not text:
        return False
    lower = text.lower()
    if require_boxed and "\\boxed" not in lower and "boxed{" not in lower:
        return False
    if any(phrase in lower for phrase in reject_phrases):
        return False
    markers = ["\\boxed", "final answer", "answer is", "####"]
    cut = len(text)
    for marker in markers:
        idx = lower.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    reasoning = text[:cut].strip()
    if len(reasoning) < min_reasoning_chars:
        return False
    import re

    number_mentions = len(re.findall(r"[-+]?\d+(?:\.\d+)?", reasoning))
    if number_mentions < min_reasoning_number_mentions:
        return False
    return True


def build_gsm8k_full_trajectory_round(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    task = config["task"]
    source_jsonl = Path(str(task["source_jsonl"]))
    rollouts_jsonl = Path(str(task["rollouts_jsonl"]))
    memory_rollouts_paths = _path_list(task.get("memory_rollouts_jsonl") or task.get("memory_rollouts_jsonls"))
    prompt_key = str(task.get("prompt_key", "deepseek_math_cot"))
    event = str(task.get("event", "gsm8k_full_trajectory_preference"))
    val_mod = int(task.get("val_mod", 5))
    rollout_control_names = _string_set(task.get("rollout_control_names") or task.get("rollout_control_name"))
    memory_control_names = _string_set(task.get("memory_control_names") or task.get("memory_control_name"))
    require_complete_rollout_controls = _as_bool(task.get("require_complete_rollout_controls"), default=False)
    enable_recovery_memory = _as_bool(task.get("enable_recovery_memory"), default=bool(memory_rollouts_paths))
    allow_current_mixed_pairs = _as_bool(task.get("allow_current_mixed_pairs"), default=True)
    allow_recovery_memory_pairs = _as_bool(task.get("allow_recovery_memory_pairs"), default=enable_recovery_memory)
    require_current_negative_evidence = _as_bool(task.get("require_current_negative_evidence"), default=True)
    require_parse_certified = _as_bool(task.get("require_parse_certified"), default=True)
    positive_quality_filter = _as_bool(task.get("positive_quality_filter"), default=False)
    require_positive_boxed = _as_bool(task.get("require_positive_boxed"), default=True)
    min_positive_reasoning_chars = int(task.get("min_positive_reasoning_chars", 40))
    min_positive_reasoning_number_mentions = int(task.get("min_positive_reasoning_number_mentions", 2))
    positive_reject_phrases = _string_set(
        task.get(
            "positive_reject_phrases",
            [
                "i guess",
                "guess",
                "not sure",
                "cannot determine",
                "can't determine",
                "insufficient information",
            ],
        )
    )
    min_correct_options = int(task.get("min_correct_options", 1))
    min_wrong_options = int(task.get("min_wrong_options", 1))
    max_correct_options = int(task.get("max_correct_options", 4))
    max_memory_correct_options = int(task.get("max_memory_correct_options", max_correct_options))
    max_wrong_answer_clusters = int(task.get("max_wrong_answer_clusters", task.get("max_wrong_options", 3)))
    max_wrong_options_per_cluster = int(task.get("max_wrong_options_per_cluster", 1))
    rows = load_jsonl(source_jsonl)
    max_rows = config.get("pairs", {}).get("max_rows")
    if max_rows not in (None, ""):
        rows = rows[: int(max_rows)]
    rollouts = load_jsonl(rollouts_jsonl)
    summary_counts: defaultdict[str, int] = defaultdict(int)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        if rollout_control_names and str(rollout.get("control_name", "")) not in rollout_control_names:
            continue
        sample_id = str(rollout.get("sample_id", "") or rollout.get("source_sample_id", ""))
        if sample_id:
            by_sample[sample_id].append(rollout)
        if rollout.get("source_row_index") not in (None, ""):
            by_index[int(rollout["source_row_index"])].append(rollout)
    memory_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    memory_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    memory_rollout_count = 0
    if enable_recovery_memory:
        for memory_index, memory_path in enumerate(memory_rollouts_paths):
            if not memory_path.exists():
                summary_counts["missing_memory_rollout_files"] += 1
                continue
            memory_state = f"memory_{memory_index:02d}"
            for rollout in load_jsonl(memory_path):
                if memory_control_names and str(rollout.get("control_name", "")) not in memory_control_names:
                    continue
                memory_rollout_count += 1
                rollout = dict(rollout)
                rollout["_cecm_rollout_state"] = memory_state
                rollout["_cecm_rollout_source"] = str(memory_path)
                sample_id = str(rollout.get("sample_id", "") or rollout.get("source_sample_id", ""))
                if sample_id:
                    memory_by_sample[sample_id].append(rollout)
                if rollout.get("source_row_index") not in (None, ""):
                    memory_by_index[int(rollout["source_row_index"])].append(rollout)

    pair_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        sample_id = sample_id_from_row(row, row_index)
        prompt, prompt_source = _gsm8k_prompt(row, prompt_key)
        gold = _gsm8k_gold(row)
        candidates = by_sample.get(sample_id, []) or by_index.get(row_index, [])
        if not prompt or not gold:
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "event": event,
                    "row_index": row_index,
                    "admitted": 0,
                    "reject_reason": "missing_prompt_or_gold",
                    "prompt_key": prompt_key,
                    "rollout_count": len(candidates),
                }
            )
            summary_counts["missing_prompt_or_gold"] += 1
            continue
        if require_complete_rollout_controls and rollout_control_names:
            present_controls = {str(rollout.get("control_name", "")) for rollout in candidates}
            missing_controls = sorted(rollout_control_names - present_controls)
            if missing_controls:
                rejected_rows.append(
                    {
                        "sample_id": sample_id,
                        "event": event,
                        "row_index": row_index,
                        "admitted": 0,
                        "reject_reason": "incomplete_rollout_control_set",
                        "prompt_key": prompt_key,
                        "rollout_count": len(candidates),
                        "missing_rollout_controls": ",".join(missing_controls),
                    }
                )
                summary_counts["incomplete_rollout_control_set"] += 1
                continue
        correct_items: list[tuple[str, str]] = []
        wrong_by_answer: dict[str, list[tuple[str, str]]] = defaultdict(list)
        observed_state_summary: dict[str, dict[str, int]] = {}
        clean_rollout_count = 0
        low_quality_correct_count = 0
        for rollout in sorted(candidates, key=_rollout_sort_key):
            clean = _clean_rollout(rollout, require_parse_certified=require_parse_certified)
            if clean is None:
                continue
            prediction, parsed = clean
            clean_rollout_count += 1
            _state_counter(observed_state_summary, "current", "clean")
            if parsed == gold:
                if positive_quality_filter and not _positive_reasoning_quality(
                    prediction,
                    require_boxed=require_positive_boxed,
                    min_reasoning_chars=min_positive_reasoning_chars,
                    min_reasoning_number_mentions=min_positive_reasoning_number_mentions,
                    reject_phrases=positive_reject_phrases,
                ):
                    low_quality_correct_count += 1
                    _state_counter(observed_state_summary, "current", "low_quality_correct")
                    continue
                _state_counter(observed_state_summary, "current", "correct")
                correct_items.append((prediction, "current"))
            else:
                _state_counter(observed_state_summary, "current", "wrong")
                wrong_by_answer[parsed].append((prediction, "current"))
        correct_items = _cap_unique_with_origin(correct_items, limit=max_correct_options)
        correct_options = [text for text, _origin in correct_items]
        correct_origins = [origin for _text, origin in correct_items]
        if low_quality_correct_count:
            summary_counts["low_quality_current_correct_rollouts"] += low_quality_correct_count
        memory_candidates = memory_by_sample.get(sample_id, []) or memory_by_index.get(row_index, [])
        memory_correct_items: list[tuple[str, str]] = []
        low_quality_memory_correct_count = 0
        for rollout in sorted(memory_candidates, key=_rollout_sort_key):
            clean = _clean_rollout(rollout, require_parse_certified=require_parse_certified)
            if clean is None:
                continue
            prediction, parsed = clean
            state = str(rollout.get("_cecm_rollout_state", "memory"))
            _state_counter(observed_state_summary, state, "clean")
            if parsed == gold:
                if positive_quality_filter and not _positive_reasoning_quality(
                    prediction,
                    require_boxed=require_positive_boxed,
                    min_reasoning_chars=min_positive_reasoning_chars,
                    min_reasoning_number_mentions=min_positive_reasoning_number_mentions,
                    reject_phrases=positive_reject_phrases,
                ):
                    low_quality_memory_correct_count += 1
                    _state_counter(observed_state_summary, state, "low_quality_correct")
                    continue
                _state_counter(observed_state_summary, state, "correct")
                memory_correct_items.append((prediction, state))
            else:
                _state_counter(observed_state_summary, state, "wrong")
        memory_correct_items = _cap_unique_with_origin(memory_correct_items, limit=max_memory_correct_options)
        memory_correct_options = [text for text, _origin in memory_correct_items]
        memory_correct_origins = [origin for _text, origin in memory_correct_items]
        if low_quality_memory_correct_count:
            summary_counts["low_quality_memory_correct_rollouts"] += low_quality_memory_correct_count
        if memory_correct_options:
            summary_counts["memory_correct_available"] += 1
        wrong_answer_counts = {answer: len(options) for answer, options in sorted(wrong_by_answer.items())}
        wrong_groups = sorted(wrong_by_answer.items(), key=lambda item: (-len(item[1]), item[0]))
        wrong_items: list[tuple[str, str]] = []
        selected_wrong_answers: list[str] = []
        for answer, options in wrong_groups:
            if max_wrong_answer_clusters > 0 and len(selected_wrong_answers) >= max_wrong_answer_clusters:
                break
            capped = _cap_unique_with_origin(options, limit=max_wrong_options_per_cluster)
            if not capped:
                continue
            selected_wrong_answers.append(answer)
            wrong_items.extend(capped)
        wrong_options = [text for text, _origin in wrong_items]
        wrong_origins = [origin for _text, origin in wrong_items]
        active_current_wrong_count = len(wrong_options)
        current_has_mixed_evidence = (
            allow_current_mixed_pairs
            and len(correct_options) >= min_correct_options
            and active_current_wrong_count >= min_wrong_options
        )
        recovery_has_cross_state_evidence = (
            enable_recovery_memory
            and allow_recovery_memory_pairs
            and len(memory_correct_options) >= min_correct_options
            and active_current_wrong_count >= min_wrong_options
        )
        pair_source = "current_mixed"
        preference_evidence = "same_state_current_correct_vs_current_wrong"
        y_plus_options = correct_options
        y_plus_origins = correct_origins
        if not current_has_mixed_evidence and recovery_has_cross_state_evidence:
            pair_source = "recovery_memory"
            preference_evidence = "cross_state_memory_correct_vs_current_wrong"
            y_plus_options = memory_correct_options
            y_plus_origins = memory_correct_origins
        if not current_has_mixed_evidence and not recovery_has_cross_state_evidence:
            if not candidates:
                reason = "missing_rollouts"
            elif require_current_negative_evidence and active_current_wrong_count < min_wrong_options:
                reason = "no_active_current_wrong_evidence"
            elif correct_options and wrong_options and not allow_current_mixed_pairs:
                reason = "current_mixed_pairs_disabled"
            elif memory_correct_options and wrong_options and not allow_recovery_memory_pairs:
                reason = "recovery_memory_pairs_disabled"
            elif not correct_options and not wrong_options:
                reason = "no_clean_correct_or_wrong_rollouts"
            elif correct_options and not wrong_options and memory_candidates:
                reason = "current_correct_without_current_wrong"
            elif not correct_options and memory_correct_options and not wrong_options:
                reason = "memory_correct_available_but_no_clean_wrong_rollouts"
            elif not correct_options:
                reason = "no_clean_correct_rollouts"
            else:
                reason = "no_clean_wrong_rollouts"
            rejected_rows.append(
                {
                    "sample_id": sample_id,
                    "event": event,
                    "row_index": row_index,
                    "admitted": 0,
                    "reject_reason": reason,
                    "prompt_key": prompt_key,
                    "rollout_count": len(candidates),
                    "clean_rollout_count": clean_rollout_count,
                    "correct_rollout_count": len(correct_options),
                    "low_quality_correct_rollout_count": low_quality_correct_count,
                    "memory_correct_rollout_count": len(memory_correct_options),
                    "low_quality_memory_correct_rollout_count": low_quality_memory_correct_count,
                    "wrong_answer_cluster_count": len(wrong_by_answer),
                    "wrong_rollout_count": sum(len(options) for options in wrong_by_answer.values()),
                    "active_current_wrong_count": active_current_wrong_count,
                    "active_correction": 0,
                    "wrong_answer_counts_json": json.dumps(wrong_answer_counts, ensure_ascii=False),
                    "observed_state_summary_json": json.dumps(observed_state_summary, ensure_ascii=False),
                }
            )
            summary_counts[reason] += 1
            continue
        if pair_source == "recovery_memory":
            summary_counts["admitted_cross_state_recovery"] += 1
        else:
            summary_counts["admitted_same_state_current_mixed"] += 1
        summary_counts["admitted"] += 1
        summary_counts[f"admitted_{pair_source}"] += 1
        continuation_policy = "sampled_full_trajectory_options_grouped_by_final_answer"
        pair_rows.append(
            {
                "sample_id": f"{event}_{row_index:06d}",
                "source_sample_id": sample_id,
                "event": event,
                "split": _split_for_index(row_index, val_mod),
                "prompt": prompt,
                "prompt_key": prompt_key,
                "prompt_source": prompt_source,
                "question": _gsm8k_question(row),
                "gold_answer": gold,
                "sampled_answer": "",
                "raw_open_generation": y_plus_options[0],
                "y_plus": y_plus_options[0],
                "y_minus": wrong_options[0],
                "y_plus_continuation": y_plus_options[0],
                "y_minus_continuation": wrong_options[0],
                "y_plus_continuations_json": json.dumps(y_plus_options, ensure_ascii=False),
                "y_minus_continuations_json": json.dumps(wrong_options, ensure_ascii=False),
                "y_plus_score_text": extract_boxed_answer_text(y_plus_options[0]),
                "y_minus_score_text": extract_boxed_answer_text(wrong_options[0]),
                "y_minus_mode": "",
                "score_text": "",
                "y_plus_alias_count": len(y_plus_options),
                "y_minus_alias_count": len(wrong_options),
                "admitted": 1,
                "invariants_pass": 1,
                "reject_reason": "",
                "source_path": str(source_jsonl),
                "row_index": row_index,
                "pair_mode": "gsm8k_sampled_trajectory_preference",
                "pair_source": pair_source,
                "preference_evidence": preference_evidence,
                "active_correction": 1,
                "active_current_wrong_count": active_current_wrong_count,
                "positive_trajectory_states_json": json.dumps(y_plus_origins, ensure_ascii=False),
                "negative_trajectory_states_json": json.dumps(wrong_origins, ensure_ascii=False),
                "observed_state_summary_json": json.dumps(observed_state_summary, ensure_ascii=False),
                "continuation_policy": continuation_policy,
                "answer_slot_policy": "no_answer_replacement",
                "rollout_count": len(candidates),
                "clean_rollout_count": clean_rollout_count,
                "correct_rollout_count": len(correct_options),
                "low_quality_correct_rollout_count": low_quality_correct_count,
                "memory_correct_rollout_count": len(memory_correct_options),
                "low_quality_memory_correct_rollout_count": low_quality_memory_correct_count,
                "wrong_rollout_count": len(wrong_options),
                "wrong_answer_cluster_count": len(wrong_by_answer),
                "selected_wrong_answers_json": json.dumps(selected_wrong_answers, ensure_ascii=False),
                "wrong_answer_counts_json": json.dumps(wrong_answer_counts, ensure_ascii=False),
                "competitive_margin": (
                    "C(M;x)=S_M(\\boxed{gold}|x) "
                    "- S_M(\\boxed{hardest_wrong}|x). "
                    "score_text targets the boxed answer span; answer_rest_margin is computed only on answer tokens."
                ),
            }
        )
    pair_dir = out_dir / "pairs"
    _write_rows(pair_dir / "pairs.csv", pair_rows)
    _write_rows(pair_dir / "rejected_pairs.csv", rejected_rows)
    dump_json(
        pair_dir / "pair_build_manifest.json",
        {
            "adapter": str(task.get("adapter", GSM8K_FULL_TRAJECTORY_ADAPTER)),
            "source_jsonl": str(source_jsonl),
            "rollouts_jsonl": str(rollouts_jsonl),
            "memory_rollouts_jsonl": [str(path) for path in memory_rollouts_paths],
            "memory_rollout_count": memory_rollout_count,
            "event": event,
            "prompt_key": prompt_key,
            "rollout_control_names": sorted(rollout_control_names),
            "memory_control_names": sorted(memory_control_names),
            "require_complete_rollout_controls": require_complete_rollout_controls,
            "enable_recovery_memory": enable_recovery_memory,
            "allow_current_mixed_pairs": allow_current_mixed_pairs,
            "allow_recovery_memory_pairs": allow_recovery_memory_pairs,
            "require_current_negative_evidence": require_current_negative_evidence,
            "require_parse_certified": require_parse_certified,
            "positive_quality_filter": positive_quality_filter,
            "require_positive_boxed": require_positive_boxed,
            "min_positive_reasoning_chars": min_positive_reasoning_chars,
            "min_positive_reasoning_number_mentions": min_positive_reasoning_number_mentions,
            "positive_reject_phrases": sorted(positive_reject_phrases),
            "min_correct_options": min_correct_options,
            "min_wrong_options": min_wrong_options,
            "max_correct_options": max_correct_options,
            "max_memory_correct_options": max_memory_correct_options,
            "max_wrong_answer_clusters": max_wrong_answer_clusters,
            "max_wrong_options_per_cluster": max_wrong_options_per_cluster,
            "admitted_pairs": len(pair_rows),
            "rejected_pairs": len(rejected_rows),
            "rejection_counts": dict(sorted(summary_counts.items())),
            "semantics": (
                "GSM8K pairs are built from a redundant sampled trajectory pool. A pair is "
                "admitted only when observed generated trajectories provide explicit positive and "
                "negative evidence for the same question. The primary evidence is a same-state "
                "current K-sample mix: at least one clean correct full trajectory and at least one "
                "clean wrong full trajectory. Wrong trajectories are grouped by parsed final answer "
                "before capping representatives. When recovery memory is enabled, a question with "
                "current clean wrong trajectories but no current clean correct trajectory can use a "
                "previous clean correct full trajectory as y_plus; this cross-state anti-regression "
                "pair is still built only from observed trajectories, not gold-answer insertion. "
                "Historical state transitions are only training-active while the current K-sample "
                "pool still has wrong trajectories; already resolved transitions are rejected as "
                "no_active_current_wrong_evidence and remain diagnostics only. "
                "When require_complete_rollout_controls is enabled, incomplete K-sample pools are "
                "kept in the rollout jsonl but rejected from training pair construction. "
                "The adapter never edits the generated final answer and leaves score_text empty "
                "so the shared core optimizes full-trajectory preference margins."
            ),
        },
    )
    return {"pairs_csv": str(pair_dir / "pairs.csv"), "admitted_pairs": len(pair_rows), "rejected_pairs": len(rejected_rows)}


def materialize_round_pairs(config: dict[str, Any], *, out_dir: Path) -> dict[str, object]:
    adapter = str(config["task"]["adapter"])
    if adapter == SOURCE_PAIR_ADAPTER:
        return build_source_pair_round(config, out_dir=out_dir)
    if adapter == GSM8K_ANSWER_MARGIN_ALL_ADAPTER:
        return build_gsm8k_answer_margin_all_round(config, out_dir=out_dir)
    if adapter == GSM8K_EXPLICIT_ANSWER_MARGIN_ADAPTER:
        return build_gsm8k_explicit_answer_margin_round(config, out_dir=out_dir)
    if adapter == GSM8K_SAMPLED_ANSWER_ARBITRATION_ADAPTER:
        return build_gsm8k_sampled_answer_arbitration_round(config, out_dir=out_dir)
    if adapter in {GSM8K_FULL_TRAJECTORY_ADAPTER, GSM8K_TRAJECTORY_PREFERENCE_ADAPTER}:
        return build_gsm8k_full_trajectory_round(config, out_dir=out_dir)
    if adapter == PREBUILT_PAIRS_ADAPTER:
        return build_prebuilt_pairs_round(config, out_dir=out_dir)
    if adapter == IMDB_SENTIMENT_SCORED_GENERATIONS_ADAPTER:
        return build_imdb_sentiment_scored_generations_round(config, out_dir=out_dir)
    raise ValueError(f"Unsupported adapter: {adapter}")


def _csv_path(path: str) -> str:
    return str(Path(path))


def _maybe_chat_template(command: list[str], model_config: dict[str, Any]) -> list[str]:
    if bool(model_config.get("use_chat_template", False)):
        return [*command, "--use-chat-template"]
    return command


def round_plan(config: dict[str, Any], *, out_dir: Path, pairs_csv: str) -> dict[str, object]:
    model = config.get("model", {})
    training = config.get("training", {})
    alpha = config.get("alpha", {})
    discovery = config.get("discovery", {})
    task = config.get("task", {})
    event = str(task.get("event", "source_context_over_prior"))
    scan_mode = str(discovery.get("scan_mode", "margin"))
    score_mode = str(training.get("score_mode", "avglogp"))
    option_selection_mode = str(training.get("option_selection_mode", "model_max"))
    max_aliases_per_side = int(training.get("max_aliases_per_side", 1))
    rollout_cfg = dict(discovery.get("rollout", {}) or {})
    scorer_cfg = dict(discovery.get("scorer", {}) or {})
    alpha_scan = bool(alpha.get("scan", False))
    train_alpha_sweep = alpha.get("train_sweep") or ([0.0, 1.0] if alpha_scan else [0.0, 1.0])
    mlp_alpha_grid = alpha.get("mlp", [])
    head_alpha_grid = alpha.get("att", alpha.get("head", []))
    full_alpha_grid = alpha.get("full", {})
    mlp_timing_grid = training.get("mlp", {}).get("timing_grid") or [
        {
            "name": str(training.get("mlp", {}).get("generation_apply_mode", "prefill")),
            "train_apply_mode": training.get("mlp", {}).get("train_apply_mode", "prompt_last"),
            "generation_apply_mode": training.get("mlp", {}).get("generation_apply_mode", "prefill"),
        }
    ]
    phases: list[dict[str, object]] = []
    if bool(discovery.get("reselect_components", False)):
        component_types = [str(item) for item in discovery.get("component_types", ["attn", "mlp"])]
        type_apply_modes = {
            str(key): str(value)
            for key, value in dict(discovery.get("component_type_apply_modes", {})).items()
        }
        default_apply_modes = {
            "attn": str(training.get("attention", {}).get("train_apply_mode", "all")),
            "mlp": str(training.get("mlp", {}).get("train_apply_mode", "prompt_last")),
        }
        for component_type in component_types:
            apply_mode = type_apply_modes.get(component_type, default_apply_modes.get(component_type, "decision_tokens"))
            scan_dir = out_dir / "discovery" / f"components_{component_type}"
            command = [
                "python",
                "-m",
                "screscomp.cli.cecm_scan_component_contributions",
                "--model",
                str(model.get("path", "")),
                "--event",
                event,
                "--scan-mode",
                scan_mode,
                "--split",
                str(discovery.get("split", "train")),
                "--start",
                str(discovery.get("start", 0)),
                "--max-rows",
                str(discovery.get("rows", 60)),
                "--component-types",
                component_type,
                "--apply-mode",
                apply_mode,
                "--min-abs-delta",
                str(discovery.get("min_abs_delta", 0.0)),
                "--min-sign-consistency",
                str(discovery.get("scan_min_sign_consistency", 0.50)),
                "--max-components-per-direction",
                str(discovery.get("max_components_per_direction", 16)),
                "--torch-dtype",
                str(model.get("torch_dtype", "auto")),
                "--device",
                str(model.get("device", "auto")),
                "--out-dir",
                str(scan_dir),
            ]
            if scan_mode == "rollout":
                prompts_jsonl = (
                    discovery.get("prompts_jsonl")
                    or rollout_cfg.get("prompts_jsonl")
                    or task.get("prompts_jsonl")
                    or task.get("discovery_prompts_jsonl")
                    or ""
                )
                command.extend(
                    [
                        "--prompts-jsonl",
                        str(prompts_jsonl),
                        "--prompt-field",
                        str(rollout_cfg.get("prompt_field", discovery.get("prompt_field", "prompt"))),
                        "--sample-id-field",
                        str(rollout_cfg.get("sample_id_field", discovery.get("sample_id_field", "sample_id"))),
                        "--samples-per-prompt",
                        str(rollout_cfg.get("samples_per_prompt", discovery.get("samples_per_prompt", 1))),
                        "--generation-batch-size",
                        str(rollout_cfg.get("generation_batch_size", discovery.get("generation_batch_size", 1))),
                        "--max-new-tokens",
                        str(rollout_cfg.get("max_new_tokens", discovery.get("max_new_tokens", 64))),
                        "--stop-strings",
                        str(rollout_cfg.get("stop_strings", discovery.get("stop_strings", ""))),
                        "--temperature",
                        str(rollout_cfg.get("temperature", discovery.get("temperature", 1.0))),
                        "--top-p",
                        str(rollout_cfg.get("top_p", discovery.get("top_p", 1.0))),
                        "--top-k",
                        str(rollout_cfg.get("top_k", discovery.get("top_k", 50))),
                        "--seed",
                        str(rollout_cfg.get("seed", discovery.get("seed", 42))),
                        "--scorer-model",
                        str(scorer_cfg.get("model", discovery.get("scorer_model", "siebert/sentiment-roberta-large-english"))),
                        "--target-label",
                        str(scorer_cfg.get("target_label", discovery.get("target_label", "POSITIVE"))),
                        "--source-label",
                        str(scorer_cfg.get("source_label", discovery.get("source_label", "NEGATIVE"))),
                        "--score-text",
                        str(scorer_cfg.get("score_text", discovery.get("score_text", "completion"))),
                        "--scorer-batch-size",
                        str(scorer_cfg.get("batch_size", discovery.get("scorer_batch_size", 16))),
                        "--scorer-max-length",
                        str(scorer_cfg.get("max_length", discovery.get("scorer_max_length", 512))),
                        "--scorer-device",
                        str(scorer_cfg.get("device", discovery.get("scorer_device", -1))),
                    ]
                )
                command.append("--do-sample" if bool(rollout_cfg.get("do_sample", discovery.get("do_sample", True))) else "--no-do-sample")
            else:
                command.extend(
                    [
                        "--pairs-csv",
                        _csv_path(pairs_csv),
                        "--score-mode",
                        score_mode,
                        "--option-selection-mode",
                        option_selection_mode,
                        "--max-aliases-per-side",
                        str(max_aliases_per_side),
                    ]
                )
            if bool(discovery.get("allow_ci_cross_zero", True)):
                command.append("--allow-ci-cross-zero")
            phases.append(
                {
                    "name": f"component_discovery_{component_type}",
                    "enabled": True,
                    "component_type": component_type,
                    "apply_mode": apply_mode,
                    "command": _maybe_chat_template(command, model),
                }
            )
    else:
        phases.append({"name": "component_discovery", "enabled": False, "reuse": discovery.get("reuse_components", "")})
    phases.append(
        {
            "name": "train_mlp_actuator",
            "enabled": bool(training.get("mlp", {}).get("enabled", True)),
            "apply_mode": training.get("mlp", {}).get("train_apply_mode", "prompt_last"),
            "generation_apply_mode": training.get("mlp", {}).get("generation_apply_mode", "prefill"),
            "timing_grid": mlp_timing_grid,
            "alpha_sweep": train_alpha_sweep,
            "generation_alpha_grid": mlp_alpha_grid,
        }
    )
    phases.append(
        {
            "name": "train_attention_actuator",
            "enabled": bool(training.get("attention", {}).get("enabled", True)),
            "apply_mode": training.get("attention", {}).get("train_apply_mode", "all"),
            "generation_apply_mode": training.get("attention", {}).get("generation_apply_mode", "all"),
            "alpha_sweep": train_alpha_sweep,
            "generation_alpha_grid": head_alpha_grid,
        }
    )
    phases.append(
        {
            "name": "select_full_control",
            "enabled": bool(training.get("full", {}).get("enabled", True)),
            "grid_mode": full_alpha_grid.get("grid_mode", "paired_centerline")
            if isinstance(full_alpha_grid, dict)
            else "paired_centerline",
            "alpha_grid": full_alpha_grid,
            "selection_metric": config.get("selection", {}).get("metric", ""),
            "minimum_material_gain": config.get("selection", {}).get("minimum_material_gain", ""),
        }
    )
    return {
        "round_dir": str(out_dir),
        "task": task,
        "pairs_csv": pairs_csv,
        "scan_mode": scan_mode,
        "score_mode": score_mode,
        "option_selection_mode": option_selection_mode,
        "alpha_scan": alpha_scan,
        "mlp_alpha_grid": mlp_alpha_grid,
        "head_alpha_grid": head_alpha_grid,
        "full_alpha_grid": full_alpha_grid,
        "reselect_components": bool(discovery.get("reselect_components", False)),
        "baselines": config.get("baselines", {}),
        "exclude_attn_layers": str(discovery.get("exclude_attn_layers", "")),
        "exclude_mlp_layers": str(discovery.get("exclude_mlp_layers", "")),
        "require_directional_ci": bool(discovery.get("require_directional_ci", False)),
        "mlp_min_sign_consistency": discovery.get("mlp_min_sign_consistency", discovery.get("min_sign_consistency", "")),
        "attn_min_sign_consistency": discovery.get("attn_min_sign_consistency", discovery.get("min_sign_consistency", "")),
        "mlp_require_directional_ci": bool(discovery.get("mlp_require_directional_ci", False)),
        "attn_require_directional_ci": bool(discovery.get("attn_require_directional_ci", False)),
        "min_directional_transition_rate": discovery.get("min_directional_transition_rate", discovery.get("min_bidirectional_rate", "")),
        "min_ablated_format_ok": discovery.get("min_ablated_format_ok", ""),
        "max_format_collapse_rate": discovery.get("max_format_collapse_rate", ""),
        "head_require_directional_ci": bool(discovery.get("head_require_directional_ci", False)),
        "phases": phases,
    }
