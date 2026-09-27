from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_ckplug_generation import _classify_generation
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.eval.health import analyze_candidate_health
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate and score natural prompt operating points for ConFiQA-style open rows. "
            "This runner does not select components or apply interventions."
        )
    )
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument(
        "--methods",
        type=str,
        default="base_rag,strong_rag,prior_objective_rag",
        help=(
            "Comma-separated prompt variants: base_rag,strong_rag,prior_objective_rag,"
            "base_no_rag,strong_no_rag,official_rag,official_no_rag,verbose_rag."
        ),
    )
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=50)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _split_for_index(index: int, val_mod: int) -> str:
    return "val" if index % val_mod == 0 else "train"


def _select_rows(rows: list[dict[str, Any]], *, split: str, start: int, max_rows: int, val_mod: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_split = _split_for_index(row_index, val_mod)
        if split != "all" and row_split != split:
            continue
        row = dict(row)
        row["_row_index"] = row_index
        row["_split"] = row_split
        selected.append(row)
    if start > 0:
        selected = selected[start:]
    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def _prior_objective_prompt(row: dict[str, Any]) -> str:
    return (
        "Read the given information, but answer from your own prior knowledge rather than relying on "
        "the given information. Return only the short answer.\n\n"
        f"{row['context']}\n\nQ: {row['question']}\nA:"
    )


def _prompt_for_method(row: dict[str, Any], method: str) -> tuple[str, str]:
    prompts = row.get("prompts", {})
    if method == "prior_objective_rag":
        return _prior_objective_prompt(row), "prior_objective_same_context"
    if method in prompts:
        return prompts[method], method
    raise ValueError(f"Unknown prompt method {method!r}; available prompt keys={sorted(prompts)}")


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)
    summary: list[dict[str, Any]] = []
    for method, group in sorted(groups.items()):
        n = len(group)
        counts = Counter(row["outcome"] for row in group)
        ps = sum(1 for row in group if row["cf_hit"]) / n if n else 0.0
        po = sum(1 for row in group if row["orig_hit"]) / n if n else 0.0
        denom = ps + po
        summary.append(
            {
                "method": method,
                "n": n,
                "pc": ps,
                "ps": ps,
                "po": po,
                "mr": po / denom if denom > 0 else 0.0,
                "em": sum(1 for row in group if row["cf_em"]) / n if n else 0.0,
                "context_only_rate": counts["context_only"] / n if n else 0.0,
                "prior_only_rate": counts["prior_only"] / n if n else 0.0,
                "both_rate": counts["both"] / n if n else 0.0,
                "neither_rate": counts["neither"] / n if n else 0.0,
                "cf_em_rate": sum(1 for row in group if row["cf_em"]) / n if n else 0.0,
                "orig_em_rate": sum(1 for row in group if row["orig_em"]) / n if n else 0.0,
                "mean_output_chars": mean(len(row["prediction"]) for row in group) if group else 0.0,
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    methods = _parse_csv(args.methods)
    stop_strings = _parse_stop_strings(args.stop_strings)
    rows = _select_rows(load_jsonl(args.eval_jsonl), split=args.split, start=args.start, max_rows=args.max_rows, val_mod=args.val_mod)
    if not rows:
        raise SystemExit("No rows selected.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        args.out_dir / "run_config.json",
        {
            "eval_jsonl": str(args.eval_jsonl),
            "model": args.model,
            "methods": methods,
            "split": args.split,
            "start": args.start,
            "max_rows": args.max_rows,
            "val_mod": args.val_mod,
            "max_new_tokens": args.max_new_tokens,
            "stop_strings": stop_strings,
            "runner": "cecm_score_prompt_operating_points",
        },
    )

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    output_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="prompt operating points"):
        for method in methods:
            prompt, prompt_source = _prompt_for_method(row, method)
            prediction = backend.generate(prompt, max_new_tokens=args.max_new_tokens, stop_strings=stop_strings)
            health = analyze_candidate_health(
                prediction=prediction,
                orig_answers=row["orig_answers"],
                cf_answers=row["cf_answers"],
            )
            scored_prediction = health["stripped_prediction"]
            classified = _classify_generation(
                prediction=scored_prediction,
                orig_answers=row["orig_answers"],
                cf_answers=row["cf_answers"],
            )
            output_rows.append(
                {
                    "sample_id": row.get("sample_id", f"row_{row['_row_index']}"),
                    "source_index": row.get("source_index", row["_row_index"]),
                    "split": row["_split"],
                    "method": method,
                    "prompt_source": prompt_source,
                    "question": row["question"],
                    "orig_answer": row["orig_answer"],
                    "cf_answer": row["cf_answer"],
                    "prediction": scored_prediction,
                    "raw_prediction": prediction,
                    **classified,
                    **{f"health_{key}": value for key, value in health.items() if key != "stripped_prediction"},
                }
            )

    dump_jsonl(args.out_dir / "generations.jsonl", output_rows)
    dump_csv(args.out_dir / "summary.csv", _summarize(output_rows))
    print(f"[cecm-prompt-operating-points] rows={len(rows)} generations={len(output_rows)} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
