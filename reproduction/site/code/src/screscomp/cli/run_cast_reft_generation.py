
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

from screscomp.cecm.actuator import parse_alpha_list
from screscomp.cecm.cast_reft import CastReftController, parse_generation_apply_mode
from screscomp.data import dump_json
from screscomp.modeling import TransformersABBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate open TLDR samples with a CAST-ReFT actuator and alpha sweep.")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--prompts-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--reft-actuator", type=Path, default=None, help="Path to cast_reft.pt or containing directory. Non-zero alpha requires it.")
    p.add_argument(
        "--additional-reft-actuator",
        action="append",
        type=Path,
        default=[],
        help="Additional disjoint CAST-ReFT payload to compose with the primary payload under the same alpha.",
    )
    p.add_argument("--control-name", default="cast_reft")
    p.add_argument("--alpha-sweep", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--sign", type=float, default=1.0)
    p.add_argument("--generation-apply-mode", default="all")
    p.add_argument("--split", default="test", help="Use all to keep every split.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument("--generation-batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--stop-strings", default="")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--same-seed-across-alpha", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    p.add_argument("--manifest-json", type=Path, default=None)
    return p.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL rows must be objects: {path}")
                rows.append(value)
    return rows


def _select_rows(rows: list[dict[str, Any]], *, split: str, start: int, max_rows: int | None) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("admitted", "1")) not in {"0", "false", "False"} and (split == "all" or str(row.get("split", "")) == split)]
    selected = selected[start:]
    if max_rows is not None:
        selected = selected[:max_rows]
    return selected


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in str(raw or "").split(",") if item.strip()]


def _alpha_key(alpha: float) -> str:
    return f"{float(alpha):.8g}"


def _resolve_reft_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        path = path / "cast_reft.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing CAST-ReFT payload: {path}")
    return path


def _completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add((str(row.get("sample_id", "")), str(row.get("control_name", "")), _alpha_key(float(row.get("alpha", 0.0))), int(row.get("sample_index", 0))))
    return keys


def _batches(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        return int(float(text)) if text else default
    except Exception:
        return default


def _set_seed(backend: TransformersABBackend, seed: int) -> None:
    backend._torch.manual_seed(seed)
    if backend._torch.cuda.is_available():
        backend._torch.cuda.manual_seed_all(seed)


def _generate_with_reft(
    backend: TransformersABBackend,
    controllers: list[CastReftController],
    prompts: list[str],
    *,
    alpha: float,
    apply_mode: str,
    max_new_tokens: int,
    stop_strings: list[str],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    handles = []
    if alpha != 0:
        for controller in controllers:
            handles.extend(controller.register_generation_hooks(alpha=alpha, apply_mode=apply_mode))
    try:
        if len(prompts) == 1:
            return [
                backend.generate(
                    prompts[0],
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            ]
        return backend.generate_many(
            prompts,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    finally:
        for handle in handles:
            handle.remove()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    parse_generation_apply_mode(args.generation_apply_mode)
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples-per-prompt must be positive")
    if args.generation_batch_size <= 0:
        raise SystemExit("--generation-batch-size must be positive")
    alphas = parse_alpha_list(args.alpha_sweep)
    reft_path = _resolve_reft_path(args.reft_actuator)
    additional_reft_paths = [_resolve_reft_path(path) for path in args.additional_reft_actuator]
    reft_paths = [path for path in [reft_path, *additional_reft_paths] if path is not None]
    if not reft_paths and any(alpha != 0 for alpha in alphas):
        raise SystemExit("--reft-actuator is required when --alpha-sweep contains non-zero values")
    if reft_path is None and additional_reft_paths:
        raise SystemExit("--additional-reft-actuator requires a primary --reft-actuator")
    rows = _select_rows(_load_jsonl(args.prompts_jsonl), split=args.split, start=args.start, max_rows=args.max_rows)
    if not rows:
        raise SystemExit(f"No prompt rows selected from {args.prompts_jsonl}")
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.out_jsonl.exists():
        args.out_jsonl.unlink()
    completed = _completed_keys(args.out_jsonl)
    stop_strings = _parse_stop_strings(args.stop_strings)
    print(f"[cast-reft-gen] loading model={args.model} prompts={len(rows)} alphas={','.join(_alpha_key(a) for a in alphas)}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    controllers: list[CastReftController] = []
    payloads: list[dict[str, Any]] = []
    seen_sites: set[str] = set()
    for path in reft_paths:
        controller, payload = CastReftController.load_payload(backend=backend, path=path)
        local_sites = {str(site.site_id) for site in controller.sites}
        overlap = seen_sites & local_sites
        if overlap:
            raise ValueError(f"Composed CAST-ReFT payloads must use disjoint sites; overlap={sorted(overlap)}")
        seen_sites.update(local_sites)
        controllers.append(controller)
        payloads.append(payload)
    new_rows = 0
    with args.out_jsonl.open("a", encoding="utf-8") as stream:
        for alpha in alphas:
            effective_alpha = float(args.sign) * float(alpha)
            alpha_text = _alpha_key(alpha)
            for sample_index in range(args.samples_per_prompt):
                active = []
                for row in rows:
                    sid = str(row.get("sample_id") or row.get("prompt_id") or "")
                    prompt = str(row.get("prompt", ""))
                    if sid and prompt and (sid, args.control_name, alpha_text, sample_index) not in completed:
                        active.append(row)
                if not active:
                    continue
                alpha_seed_offset = 0 if args.same_seed_across_alpha else int(round(float(alpha) * 1000.0))
                for batch_index, batch in enumerate(tqdm(_batches(active, args.generation_batch_size), desc=f"cast-reft a={alpha_text} s={sample_index}")):
                    prompts = [str(row.get("prompt", "")) for row in batch]
                    if args.generation_batch_size > 1:
                        batch_seed = args.seed + sample_index * 9176 + batch_index * 1009 + alpha_seed_offset
                    else:
                        row_seed = _as_int(batch[0].get("source_row_index", batch[0].get("raw_index", 0)))
                        batch_seed = args.seed + row_seed * 1009 + sample_index * 9176 + alpha_seed_offset
                    _set_seed(backend, batch_seed)
                    completions = _generate_with_reft(
                        backend,
                        controllers,
                        prompts,
                        alpha=effective_alpha,
                        apply_mode=args.generation_apply_mode,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=bool(args.do_sample),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    for row, completion in zip(batch, completions, strict=False):
                        sid = str(row.get("sample_id") or row.get("prompt_id") or "")
                        prompt = str(row.get("prompt", ""))
                        output = {
                            "sample_id": sid,
                            "prompt_id": row.get("prompt_id", ""),
                            "split": row.get("split", ""),
                            "event": row.get("event", "tldr_summary_preference"),
                            "prompt": prompt,
                            "prefix": row.get("prefix", ""),
                            "control_name": args.control_name,
                            "alpha": float(alpha),
                            "effective_alpha": effective_alpha,
                            "sample_index": sample_index,
                            "completion": completion,
                            "full_text": prompt + completion,
                            "model": backend.model_id,
                            "tokenizer": backend.resolved_tokenizer_name_or_path,
                            "reft_actuator_path": str(reft_path) if reft_path else "",
                            "reft_actuator_paths": [str(path) for path in reft_paths],
                            "reft_mode": "+".join(str(payload.get("reft_mode", "")) for payload in payloads),
                            "reft_sites": [str(site.site_id) for controller in controllers for site in controller.sites],
                            "reft_bank_count": len(controllers),
                            "generation_apply_mode": args.generation_apply_mode,
                            "source_dataset": row.get("source_dataset", ""),
                            "source_split": row.get("source_split", ""),
                            "source_row_index": row.get("source_row_index", row.get("raw_index", "")),
                            "generation_do_sample": int(bool(args.do_sample)),
                            "generation_temperature": args.temperature,
                            "generation_top_p": args.top_p,
                            "generation_top_k": args.top_k,
                            "generation_stop_strings": stop_strings,
                            "max_new_tokens": args.max_new_tokens,
                            "same_seed_across_alpha": int(bool(args.same_seed_across_alpha)),
                            "generation_batch_size": args.generation_batch_size,
                            "batch_seed": batch_seed,
                        }
                        stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                        stream.flush()
                        completed.add((sid, args.control_name, alpha_text, sample_index))
                        new_rows += 1
    manifest = args.manifest_json or (args.out_jsonl.parent / "cast_reft_generation_manifest.json")
    dump_json(
        manifest,
        {
            "model": backend.model_id,
            "tokenizer": backend.resolved_tokenizer_name_or_path,
            "prompts_jsonl": str(args.prompts_jsonl),
            "out_jsonl": str(args.out_jsonl),
            "reft_actuator": str(reft_path) if reft_path else "",
            "reft_actuators": [str(path) for path in reft_paths],
            "reft_bank_count": len(controllers),
            "reft_sites": [str(site.site_id) for controller in controllers for site in controller.sites],
            "control_name": args.control_name,
            "split": args.split,
            "prompt_rows": len(rows),
            "alpha_sweep": alphas,
            "sign": args.sign,
            "generation_apply_mode": args.generation_apply_mode,
            "samples_per_prompt": args.samples_per_prompt,
            "generation_batch_size": args.generation_batch_size,
            "new_rows": new_rows,
            "completed_rows": len(completed),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "stop_strings": stop_strings,
            "seed": args.seed,
            "same_seed_across_alpha": bool(args.same_seed_across_alpha),
            "role": "cast_reft_open_generation",
        },
    )
    print(f"[cast-reft-gen] wrote new_rows={new_rows} path={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
