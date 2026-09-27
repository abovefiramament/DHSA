from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.data import dump_json
from screscomp.modeling import TransformersABBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate multiple start-policy completions for IMDb sentiment-control prompts."
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompts-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--split", type=str, default="train", help="Use all to keep every split.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--completions-per-prefix", type=int, default=4)
    p.add_argument("--generation-batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--stop-strings", type=str, default="")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--manifest-json", type=Path, default=None)
    return p.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL rows must be objects: {path}")
                rows.append(value)
    return rows


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _select_prompt_rows(rows: list[dict[str, Any]], *, split: str, start: int, max_rows: int | None) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("admitted", "1")) not in {"0", "false", "False"}
        and (split == "all" or str(row.get("split", "")) == split)
    ]
    selected = selected[start:]
    if max_rows is not None:
        selected = selected[:max_rows]
    return selected


def _completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add((str(row.get("sample_id", "")), int(row.get("completion_id", 0))))
    return keys


def _set_seed(backend: TransformersABBackend, seed: int) -> None:
    backend._torch.manual_seed(seed)
    if backend._torch.cuda.is_available():
        backend._torch.cuda.manual_seed_all(seed)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def _batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.completions_per_prefix <= 0:
        raise SystemExit("--completions-per-prefix must be positive")
    if args.generation_batch_size <= 0:
        raise SystemExit("--generation-batch-size must be positive")
    prompt_rows = _select_prompt_rows(
        _load_jsonl(args.prompts_jsonl),
        split=args.split,
        start=args.start,
        max_rows=args.max_rows,
    )
    if not prompt_rows:
        raise SystemExit(f"No prompt rows selected from {args.prompts_jsonl}")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.out_jsonl.exists():
        args.out_jsonl.unlink()
    completed = _completed_keys(args.out_jsonl)
    stop_strings = _parse_stop_strings(args.stop_strings)

    print(f"[imdb-sentiment-generate] loading model={args.model} prompts={len(prompt_rows)}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    new_rows = 0
    with args.out_jsonl.open("a", encoding="utf-8") as stream:
        if args.generation_batch_size == 1:
            work_batches = [[row] for row in prompt_rows]
        else:
            work_batches = _batches(prompt_rows, args.generation_batch_size)
        for completion_id in range(args.completions_per_prefix):
            for batch_index, batch in enumerate(tqdm(work_batches, desc=f"generate imdb sentiment c={completion_id}")):
                active_rows = []
                for row in batch:
                    sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
                    prompt = str(row.get("prompt", ""))
                    if sample_id and prompt and (sample_id, completion_id) not in completed:
                        active_rows.append(row)
                if not active_rows:
                    continue
                if args.generation_batch_size == 1:
                    row = active_rows[0]
                    row_seed = _as_int(row.get("source_row_index", row.get("raw_index", 0)))
                    batch_seed = args.seed + row_seed * 1009 + completion_id
                    _set_seed(backend, batch_seed)
                    completions = [
                        backend.generate(
                            str(row.get("prompt", "")),
                            max_new_tokens=args.max_new_tokens,
                            stop_strings=stop_strings,
                            do_sample=bool(args.do_sample),
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                    ]
                else:
                    batch_seed = args.seed + completion_id * 9176 + batch_index * 1009
                    _set_seed(backend, batch_seed)
                    completions = backend.generate_many(
                        [str(row.get("prompt", "")) for row in active_rows],
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=bool(args.do_sample),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                for row, completion in zip(active_rows, completions):
                    sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
                    prompt = str(row.get("prompt", ""))
                    key = (sample_id, completion_id)
                    output = {
                        "sample_id": sample_id,
                        "prompt_id": row.get("prompt_id", ""),
                        "split": row.get("split", ""),
                        "event": row.get("event", "imdb_positive_sentiment"),
                        "prompt": prompt,
                        "prefix": row.get("prefix", ""),
                        "completion_id": completion_id,
                        "completion": completion,
                        "full_text": prompt + completion,
                        "model": args.model,
                        "source_dataset": row.get("source_dataset", ""),
                        "source_split": row.get("source_split", ""),
                        "source_row_index": row.get("source_row_index", row.get("raw_index", "")),
                        "generation_do_sample": int(bool(args.do_sample)),
                        "generation_temperature": args.temperature,
                        "generation_top_p": args.top_p,
                        "generation_top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                        "generation_batch_size": args.generation_batch_size,
                        "batch_seed": batch_seed,
                    }
                    stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed.add(key)
                    new_rows += 1

    manifest_path = args.manifest_json or (args.out_jsonl.parent / "generation_manifest.json")
    dump_json(
        manifest_path,
        {
            "model": args.model,
            "prompts_jsonl": str(args.prompts_jsonl),
            "out_jsonl": str(args.out_jsonl),
            "split": args.split,
            "prompt_rows": len(prompt_rows),
            "completions_per_prefix": args.completions_per_prefix,
            "generation_batch_size": args.generation_batch_size,
            "new_rows": new_rows,
            "completed_rows": len(completed),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "baseline_status": "not_run",
            "role": "start_policy_completion_source",
        },
    )
    print(f"[imdb-sentiment-generate] wrote new_rows={new_rows} path={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
