from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.pairs import (
    CONTEXT_ANSWER_KEYS,
    DATASET_PRIOR_ANSWER_KEYS,
    all_texts,
    normalize_answer,
    normalize_text,
    prompt_from_row,
    sample_id_from_row,
)
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Discover each sample's model prior from an ordinary no-evidence prompt. "
            "The output JSONL augments rows with model_prior_answer/model_prior_answers, "
            "so downstream CECM source pairs can use the model's exposed prior instead of "
            "silently treating dataset orig_answer as the prior."
        )
    )
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--out-summary-csv", type=Path, required=True)
    p.add_argument("--out-manifest-json", type=Path, default=None)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompt-key", type=str, default="strong_no_rag")
    p.add_argument("--split", choices=["all", "train", "val"], default="all")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--max-free-prior-chars", type=int, default=64)
    p.add_argument(
        "--reject-free-prior",
        action="store_true",
        help="Only admit discovered priors that match a dataset orig/prior alias.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    return p.parse_args()


def _parse_stop_strings(raw: str) -> list[str]:
    stops: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            stops.append(bytes(item, "utf-8").decode("unicode_escape"))
    return stops


def _split_for_index(row_index: int, val_mod: int) -> str:
    if val_mod <= 1:
        return "train"
    return "val" if row_index % val_mod == 0 else "train"


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    start: int,
    max_rows: int | None,
    val_mod: int,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for row_index, row in enumerate(rows):
        row_split = _split_for_index(row_index, val_mod)
        if split != "all" and row_split != split:
            continue
        selected.append((row_index, row))
    if start > 0:
        selected = selected[start:]
    if max_rows is not None:
        selected = selected[:max_rows]
    return selected


def _contains_alias(prediction: str, aliases: tuple[str, ...]) -> tuple[bool, str]:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False, ""
    for alias in aliases:
        alias_norm = normalize_answer(alias)
        if alias_norm and alias_norm in pred_norm:
            return True, alias
    return False, ""


def _clean_free_answer(prediction: str, *, max_chars: int) -> str:
    text = normalize_text(prediction)
    if not text:
        return ""
    for marker in ("\n", "Q:", "Question:", "Context:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    text = text.strip().strip("\"'`")
    if not text or len(text) > max_chars:
        return ""
    return text


def discover_prior_fields(
    row: dict[str, Any],
    *,
    prediction: str,
    prompt_key: str,
    row_index: int,
    max_free_prior_chars: int = 64,
    admit_free_prior: bool = True,
) -> dict[str, Any]:
    context_answers, context_source = all_texts(row, CONTEXT_ANSWER_KEYS)
    dataset_prior_answers, dataset_prior_source = all_texts(row, DATASET_PRIOR_ANSWER_KEYS)
    context_hit, context_hit_alias = _contains_alias(prediction, context_answers)
    dataset_prior_hit, dataset_prior_hit_alias = _contains_alias(prediction, dataset_prior_answers)

    prior_answer = ""
    source = "empty"
    admissible = False
    reject_reason = ""
    if dataset_prior_hit:
        prior_answer = dataset_prior_hit_alias or (dataset_prior_answers[0] if dataset_prior_answers else "")
        source = "dataset_prior_alias_hit"
        admissible = True
    elif context_hit:
        prior_answer = context_hit_alias or (context_answers[0] if context_answers else "")
        source = "context_alias_hit"
        reject_reason = "model_prior_equals_context_answer"
    else:
        free_answer = _clean_free_answer(prediction, max_chars=max_free_prior_chars)
        if free_answer and admit_free_prior:
            prior_answer = free_answer
            source = "free_generation"
            admissible = True
        elif free_answer:
            prior_answer = free_answer
            source = "free_generation_rejected"
            reject_reason = "free_generation_rejected"
        else:
            reject_reason = "empty_or_too_long_generation"

    prior_norm = normalize_answer(prior_answer)
    context_norms = {normalize_answer(answer) for answer in context_answers}
    if admissible and (not prior_norm or prior_norm in context_norms):
        admissible = False
        reject_reason = "model_prior_overlaps_context_answer"

    return {
        "model_prior_prompt_key": prompt_key,
        "model_prior_prediction": prediction,
        "model_prior_candidate": prior_answer,
        "model_prior_answer": prior_answer if admissible else "",
        "model_prior_answers": [prior_answer] if admissible else [],
        "model_prior_valid": admissible,
        "model_prior_source": source,
        "model_prior_reject_reason": "" if admissible else reject_reason,
        "model_prior_matches_dataset_prior": bool(dataset_prior_hit and admissible),
        "model_prior_matches_context": bool(context_hit),
        "model_prior_context_answer_source": context_source,
        "model_prior_dataset_prior_source": dataset_prior_source,
        "model_prior_source_row_index": row_index,
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    total = len(rows)
    valid = sum(1 for row in rows if bool(row.get("model_prior_valid")))
    dataset_match = sum(1 for row in rows if bool(row.get("model_prior_matches_dataset_prior")))
    context_match = sum(1 for row in rows if bool(row.get("model_prior_matches_context")))
    output: list[dict[str, object]] = [
        {"metric": "total_rows", "value": total},
        {"metric": "valid_model_prior_rows", "value": valid},
        {"metric": "valid_model_prior_rate", "value": valid / total if total else 0.0},
        {"metric": "dataset_prior_match_rows", "value": dataset_match},
        {"metric": "dataset_prior_match_rate", "value": dataset_match / total if total else 0.0},
        {"metric": "context_match_rows", "value": context_match},
        {"metric": "context_match_rate", "value": context_match / total if total else 0.0},
    ]
    for key in ("model_prior_source", "model_prior_reject_reason"):
        counts = Counter(str(row.get(key, "")) for row in rows)
        for value, count in sorted(counts.items()):
            output.append({"metric": f"{key}:{value}", "value": count})
    return output


def main() -> None:
    args = parse_args()
    selected = _select_rows(
        load_jsonl(args.input_jsonl),
        split=args.split,
        start=args.start,
        max_rows=args.max_rows,
        val_mod=args.val_mod,
    )
    stop_strings = _parse_stop_strings(args.stop_strings)
    print(f"[cecm-prior] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    out_rows: list[dict[str, Any]] = []
    for row_index, row in tqdm(selected, desc="discover model prior"):
        prompt, prompt_source = prompt_from_row(row, args.prompt_key)
        out_row = dict(row)
        out_row["model_prior_eval_split"] = _split_for_index(row_index, args.val_mod)
        out_row["model_prior_prompt_source"] = prompt_source
        out_row["model_prior_source_row_index"] = row_index
        if not prompt:
            out_row.update(
                {
                    "model_prior_prompt_key": args.prompt_key,
                    "model_prior_prediction": "",
                    "model_prior_candidate": "",
                    "model_prior_answer": "",
                    "model_prior_answers": [],
                    "model_prior_valid": False,
                    "model_prior_source": "missing_prompt",
                    "model_prior_reject_reason": f"missing_prompt:{args.prompt_key}",
                }
            )
            out_rows.append(out_row)
            continue
        prediction = backend.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            stop_strings=stop_strings,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        out_row.update(
            discover_prior_fields(
                row,
                prediction=prediction,
                prompt_key=args.prompt_key,
                row_index=row_index,
                max_free_prior_chars=args.max_free_prior_chars,
                admit_free_prior=not args.reject_free_prior,
            )
        )
        out_rows.append(out_row)

    dump_jsonl(args.out_jsonl, out_rows)
    dump_csv(args.out_summary_csv, _summary(out_rows))
    manifest_path = args.out_manifest_json or args.out_jsonl.with_suffix(".manifest.json")
    dump_json(
        manifest_path,
        {
            "input_jsonl": str(args.input_jsonl),
            "out_jsonl": str(args.out_jsonl),
            "model": args.model,
            "prompt_key": args.prompt_key,
            "split": args.split,
            "start": args.start,
            "max_rows": args.max_rows if args.max_rows is not None else "",
            "val_mod": args.val_mod,
            "max_new_tokens": args.max_new_tokens,
            "reject_free_prior": bool(args.reject_free_prior),
            "semantics": (
                "model_prior_answer is discovered from the model's ordinary no-evidence generation. "
                "dataset orig_answer is only an alignment label unless the generated answer hits its aliases."
            ),
        },
    )
    print(f"[cecm-prior] rows={len(out_rows)} out={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
