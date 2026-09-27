from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from screscomp.modeling.transformers_backend import _maybe_set_cuda_memory_limit, _resolve_model_name_or_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aligned TL;DR samples from a full causal LM checkpoint or PEFT adapter."
    )
    parser.add_argument("--model", default="CarperAI/openai_summarize_tldr_ppo")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--control-name", default="ppo")
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=320)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _local_adapter_config(model_path: str) -> Path | None:
    path = Path(model_path)
    config = path / "adapter_config.json"
    return config if config.exists() else None


def _adapter_base_model(model_path: str) -> str | None:
    local_config = _local_adapter_config(model_path)
    if local_config is not None:
        raw = json.loads(local_config.read_text(encoding="utf-8"))
        base = str(raw.get("base_model_name_or_path", "")).strip()
        return base or None
    try:
        from peft import PeftConfig
    except Exception:
        return None
    try:
        config = PeftConfig.from_pretrained(model_path)
    except Exception:
        return None
    base = str(getattr(config, "base_model_name_or_path", "") or "").strip()
    return base or None


def _load_model(model_path: str, *, dtype: Any):
    override_base = str(os.environ.get("SCRESCOMP_PEFT_BASE_MODEL", "")).strip()
    adapter_base = override_base or _adapter_base_model(model_path)
    if adapter_base == "$$":
        raise ValueError(
            f"PEFT adapter {model_path} does not declare a usable base model. "
            "Set SCRESCOMP_PEFT_BASE_MODEL or fix adapter_config.json."
        )
    if not adapter_base:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    try:
        from peft import PeftModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Loading a PEFT TL;DR baseline requires `peft`. Install with: pip install peft") from exc

    base_path = _resolve_model_name_or_path(adapter_base)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return PeftModel.from_pretrained(base_model, model_path)


def main() -> None:
    args = parse_args()
    do_sample = args.temperature > 0
    rows = [row for row in _load_jsonl(args.prompts_jsonl) if str(row.get("split", "")) == args.split]
    rows = rows[args.start : args.start + args.max_rows]
    done = set()
    if args.out_jsonl.exists():
        done = {str(row.get("sample_id", "")) for row in _load_jsonl(args.out_jsonl)}

    dtype = getattr(torch, args.torch_dtype)
    model_path = _resolve_model_name_or_path(args.model)
    tokenizer_path = _resolve_model_name_or_path(args.tokenizer)
    _maybe_set_cuda_memory_limit(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(model_path, dtype=dtype).to(args.device)
    model.eval()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("a", encoding="utf-8") as output:
        generated = 0
        for index, row in enumerate(rows, start=1):
            sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
            if not sample_id or sample_id in done:
                continue
            prompt = str(row["prompt"])
            row_seed = args.seed + _as_int(row.get("source_row_index", row.get("raw_index", 0))) * 1009
            torch.manual_seed(row_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(row_seed)
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                generation_args: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    generation_args.update(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                result = model.generate(
                    **inputs,
                    **generation_args,
                )
            completion = tokenizer.decode(result[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            record = {
                "sample_id": sample_id,
                "prompt_id": row.get("prompt_id", ""),
                "split": row.get("split", ""),
                "event": row.get("event", "tldr_summary_preference"),
                "prompt": prompt,
                "control_name": args.control_name,
                "alpha": 0.0,
                "sample_index": 0,
                "completion": completion,
                "full_text": prompt + completion,
                "model": model_path,
                "tokenizer": tokenizer_path,
                "source_dataset": row.get("source_dataset", ""),
                "source_split": row.get("source_split", ""),
                "source_row_index": row.get("source_row_index", row.get("raw_index", "")),
                "batch_seed": row_seed,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            generated += 1
            if index % 10 == 0:
                print(f"[tldr-ppo] completed={index}/{len(rows)} newly_generated={generated}", flush=True)
    print(f"[tldr-ppo] done rows={len(rows)} output={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
