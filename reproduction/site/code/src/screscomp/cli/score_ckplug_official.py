from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cli.score_ckplug_generation import (
    _classify_generation,
    _component_specs,
    _normalize_answer,
    _parse_stop_strings,
    _select_axis_components,
    _summarize,
)
from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.eval.health import add_candidate_tagging_instruction, analyze_candidate_health


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official CK-PLUG decoding on prepared CK-style open rows.")
    p.add_argument("--eval_jsonl", type=Path, required=True)
    p.add_argument("--ck_core_dir", type=Path, default=Path("external/CK-PLUG-core"))
    p.add_argument("--ck_transformers_dir", type=Path, default=None)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_generations_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--out_manifest_csv", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--num_gpus", type=str, default="1")
    p.add_argument("--max_gpu_memory", type=int, default=80)
    p.add_argument("--mode", choices=["ck", "base_rag", "base_no_rag"], default="ck")
    p.add_argument("--prompt_family", choices=["shared_strong", "official_base"], default="shared_strong")
    p.add_argument("--schema", choices=["base", "attr", "instr", "opin", "instr+opin"], default="base")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--select_top", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--stop_strings", type=str, default="Q:")
    p.add_argument("--max_eval_rows", type=int, default=None)
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--start", type=int, default=0)
    p.add_argument(
        "--val_mod",
        type=int,
        default=5,
        help="Rows with row_index %% val_mod == 0 are val; the rest are train. Use with --split.",
    )
    p.add_argument("--rcm_format_component_summary_csv", type=Path, default=None)
    p.add_argument("--rcm_format_component_ids", type=str, default="")
    p.add_argument("--rcm_format_top_k_components", type=int, default=4)
    p.add_argument("--rcm_format_selection_pairs", type=str, default="")
    p.add_argument("--rcm_format_alpha", type=float, default=0.0)
    p.add_argument("--rcm_format_apply_mode", type=str, default="prefill")
    p.add_argument("--candidate_tagging", action="store_true")
    p.add_argument("--score_tagged_final", action="store_true")
    return p.parse_args()


def _official_context_prompt(query_prompt: str, context: str, schema: str) -> str:
    if schema == "base":
        return f"{context}\nQ:{query_prompt}\nA:"
    if schema == "opin":
        context = context.replace('"', "")
        return f'Bob said "{context}"\nQ: {query_prompt[:-1]} in Bob\'s opinion?\nA:'
    if schema == "instr+opin":
        context = context.replace('"', "")
        return "Instruction: read the given information and answer the corresponding question.\n\n" + (
            f'Bob said "{context}"\nQ: {query_prompt[:-1]} in Bob\'s opinion?\nA:'
        )
    if schema == "attr":
        return f"{context}\nQ:{query_prompt[:-1]} based on the given tex?\nA:"
    if schema == "instr":
        return (
            "Instruction: read the given information and answer the corresponding question.\n\n"
            f"{context}\nQ:{query_prompt}\nA:"
        )
    raise ValueError(f"Unsupported schema: {schema}")


def _prompts_for_row(row: dict[str, Any], prompt_family: str, schema: str) -> tuple[str, str]:
    if prompt_family == "shared_strong":
        return row["prompts"]["strong_no_rag"], row["prompts"]["strong_rag"]
    if prompt_family == "official_base":
        base_prompt = f"Q: {row['question']}\nA: "
        return base_prompt, _official_context_prompt(base_prompt, row["context"], schema)
    raise ValueError(f"Unsupported prompt_family: {prompt_family}")


def _method_name(args: argparse.Namespace) -> str:
    adaptive = "_adaptive" if args.adaptive else ""
    chat = "_chat" if args.use_chat_template else ""
    rcm = ""
    if args.rcm_format_alpha != 0.0:
        rcm = (
            f"_rcmfmt_{args.rcm_format_apply_mode}"
            f"_k{args.rcm_format_top_k_components}_fa{args.rcm_format_alpha:g}"
        )
    return f"ckplug_official_{args.mode}_{args.prompt_family}{chat}{adaptive}_alpha{args.alpha:g}{rcm}"


def _select_eval_rows(rows: list[dict[str, Any]], *, split: str, start: int, max_rows: int | None, val_mod: int) -> list[dict[str, Any]]:
    if val_mod <= 0:
        raise ValueError("--val_mod must be positive")
    selected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_split = "val" if row_index % val_mod == 0 else "train"
        if split != "all" and row_split != split:
            continue
        selected_row = dict(row)
        selected_row["_cecm_row_index"] = row_index
        selected_row["_cecm_split"] = row_split
        selected.append(selected_row)
    if start > 0:
        selected = selected[start:]
    if max_rows is not None:
        selected = selected[:max_rows]
    return selected


def _prepend_transformers_path(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"Missing CK-PLUG transformers directory: {path}")
    if (path / "src" / "transformers").exists():
        import_path = path / "src"
    elif (path / "transformers").exists():
        import_path = path
    else:
        import_path = path
    sys.path.insert(0, str(import_path.resolve()))
    return str(import_path)


def _resolve_model_name_or_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if not path.exists():
        return raw
    if (path / "config.json").exists():
        return str(path)

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.exists():
        return str(path)

    refs_main = path / "refs" / "main"
    if refs_main.exists():
        snapshot = snapshots_dir / refs_main.read_text(encoding="utf-8").strip()
        if (snapshot / "config.json").exists():
            return str(snapshot)

    candidates = [
        snapshot
        for snapshot in snapshots_dir.iterdir()
        if snapshot.is_dir() and (snapshot / "config.json").exists()
    ]
    if not candidates:
        return str(path)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _maybe_apply_chat_template(model, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    tokenizer = model.tokenizer
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not expose apply_chat_template.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_id(tokenizer, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
        return int(token_id)
    return None


def _configure_generation_tokens(model) -> tuple[int | None, list[int] | None, str]:
    tokenizer = model.tokenizer
    eos_token_id = tokenizer.eos_token_id
    eot_token_id = _token_id(tokenizer, "<|eot_id|>")

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == eos_token_id:
        for pad_token in (
            "<|finetune_right_pad_id|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
        ):
            pad_token_id = _token_id(tokenizer, pad_token)
            if pad_token_id is not None and pad_token_id != eos_token_id and pad_token_id != eot_token_id:
                tokenizer.pad_token = pad_token
                break
        if tokenizer.pad_token_id is None and eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None:
        model.model.generation_config.pad_token_id = tokenizer.pad_token_id

    eos_ids: list[int] = []
    for token_id in (eos_token_id, eot_token_id):
        if token_id is not None and token_id not in eos_ids:
            eos_ids.append(int(token_id))
    return tokenizer.pad_token_id, eos_ids or None, str(tokenizer.pad_token or "")


def _component_module(model, layer_idx: int, component_type: str):
    layers = model.model.model.layers
    layer = layers[layer_idx]
    if component_type == "attn":
        for attr in ("self_attn", "attention", "attn"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
    if component_type == "mlp":
        for attr in ("mlp", "feed_forward", "ffn"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
    raise ValueError(f"Could not locate {component_type!r} component on decoder layer {layer_idx}.")


def _encode_prompt(model, prompt: str) -> dict[str, Any]:
    inputs = model.tokenizer(prompt, return_tensors="pt")
    return {key: value.to(model.device) for key, value in inputs.items()}


def _capture_component_last_token(model, prompt: str, component_specs: list[tuple[int, str]]):
    captured: dict[tuple[int, str], Any] = {}
    handles = []

    def make_hook(spec: tuple[int, str]):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[spec] = hidden[0, -1, :].detach().clone()

        return hook

    for layer_idx, component_type in component_specs:
        handles.append(_component_module(model, layer_idx, component_type).register_forward_hook(make_hook((layer_idx, component_type))))

    try:
        with __import__("torch").no_grad():
            model.model(**_encode_prompt(model, prompt), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def _format_delta_additions(
    model,
    short_prompt: str,
    verbose_prompt: str,
    component_specs: list[tuple[int, str]],
    alpha: float,
) -> list[dict[str, Any]]:
    short_acts = _capture_component_last_token(model, short_prompt, component_specs)
    verbose_acts = _capture_component_last_token(model, verbose_prompt, component_specs)
    return [
        {
            "layer_idx": layer_idx,
            "component_type": component_type,
            "direction": short_acts[(layer_idx, component_type)] - verbose_acts[(layer_idx, component_type)],
            "alpha": alpha,
        }
        for layer_idx, component_type in component_specs
    ]


def _register_component_add_hooks(model, additions: list[dict[str, Any]], apply_mode: str):
    handles = []

    def parse_apply_mode(raw: str) -> tuple[str, int | None]:
        first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
        if first_decode_match:
            first_decode_steps = int(first_decode_match.group(1) or "1")
            if first_decode_steps <= 0:
                raise ValueError(f"Unsupported apply_mode: {raw}")
            return raw, first_decode_steps
        if raw not in {"all", "prefill", "decode"}:
            raise ValueError(f"Unsupported apply_mode: {raw}")
        return raw, None

    parse_apply_mode(apply_mode)

    def make_hook(direction, alpha: float, local_apply_mode: str):
        _, first_decode_steps = parse_apply_mode(local_apply_mode)
        decode_steps_seen = 0

        def add_hook(_module, _inputs, output):
            nonlocal decode_steps_seen
            hidden = output[0] if isinstance(output, tuple) else output
            seq_len = int(hidden.shape[1])
            if local_apply_mode == "prefill" and seq_len <= 1:
                return output
            if local_apply_mode == "decode" and seq_len > 1:
                return output
            if first_decode_steps is not None:
                if seq_len > 1:
                    return output
                decode_steps_seen += 1
                if decode_steps_seen > first_decode_steps:
                    return output

            if isinstance(output, tuple):
                patched = output[0].clone()
                patched[:, -1, :] = patched[:, -1, :] + alpha * direction.to(
                    device=patched.device,
                    dtype=patched.dtype,
                )
                return (patched, *output[1:])

            patched = output.clone()
            patched[:, -1, :] = patched[:, -1, :] + alpha * direction.to(
                device=patched.device,
                dtype=patched.dtype,
            )
            return patched

        return add_hook

    for addition in additions:
        local_apply_mode = str(addition.get("apply_mode", apply_mode))
        handles.append(
            _component_module(
                model,
                int(addition["layer_idx"]),
                str(addition["component_type"]),
            ).register_forward_hook(make_hook(addition["direction"], float(addition["alpha"]), local_apply_mode))
        )

    return handles


def main() -> None:
    args = parse_args()
    if not args.ck_core_dir.exists():
        raise FileNotFoundError(f"Missing CK-PLUG core directory: {args.ck_core_dir}")

    transformers_import_path = _prepend_transformers_path(args.ck_transformers_dir)

    # The official ck.py imports vllm at module import time, but the ConFiQA
    # CK.generate path below uses HuggingFace Transformers only.
    if "vllm" not in sys.modules:
        vllm_stub = types.ModuleType("vllm")
        vllm_stub.LLM = object
        vllm_stub.SamplingParams = object
        sys.modules["vllm"] = vllm_stub

    sys.path.insert(0, str(args.ck_core_dir.resolve()))
    try:
        from ck import CK  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Could not import official CK-PLUG from {args.ck_core_dir}") from exc

    rows = _select_eval_rows(
        load_jsonl(args.eval_jsonl),
        split=args.split,
        start=args.start,
        max_rows=args.max_eval_rows,
        val_mod=args.val_mod,
    )

    resolved_model = _resolve_model_name_or_path(args.model)
    model = CK(
        resolved_model,
        args.device,
        args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
    )
    pad_token_id, eos_token_id, pad_token = _configure_generation_tokens(model)
    stop_strings = _parse_stop_strings(args.stop_strings)
    model.set_stop_words(stop_strings)
    selected_format_components: list[dict[str, Any]] = []
    format_component_source = ""
    format_component_specs: list[tuple[int, str]] = []
    if args.rcm_format_alpha != 0.0:
        selected_format_components, format_component_source = _select_axis_components(
            summary_csv=args.rcm_format_component_summary_csv,
            selection_pairs=args.rcm_format_selection_pairs,
            top_k=args.rcm_format_top_k_components,
            explicit_ids=args.rcm_format_component_ids,
            fallback_components=[],
            axis="format",
        )
        if not selected_format_components:
            raise ValueError(
                "RCM-format is enabled, but no format components were selected. "
                "Pass --rcm_format_component_summary_csv or --rcm_format_component_ids."
            )
        format_component_specs = _component_specs(selected_format_components)

    method = _method_name(args)
    output_rows: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="official CK-PLUG generation"):
        base_prompt, context_prompt = _prompts_for_row(row, args.prompt_family, args.schema)
        if args.candidate_tagging:
            base_prompt = add_candidate_tagging_instruction(base_prompt)
            context_prompt = add_candidate_tagging_instruction(context_prompt)
        base_prompt = _maybe_apply_chat_template(model, base_prompt, args.use_chat_template)
        context_prompt = _maybe_apply_chat_template(model, context_prompt, args.use_chat_template)
        hook_handles = []
        try:
            if format_component_specs:
                if args.prompt_family != "shared_strong":
                    raise ValueError("RCM-format currently requires --prompt_family shared_strong.")
                verbose_prompt = row["prompts"].get("verbose_rag")
                if not verbose_prompt:
                    raise ValueError("RCM-format requires rows prepared with a `verbose_rag` prompt.")
                if args.candidate_tagging:
                    verbose_prompt = add_candidate_tagging_instruction(verbose_prompt)
                verbose_prompt = _maybe_apply_chat_template(model, verbose_prompt, args.use_chat_template)
                additions = _format_delta_additions(
                    model=model,
                    short_prompt=context_prompt,
                    verbose_prompt=verbose_prompt,
                    component_specs=format_component_specs,
                    alpha=args.rcm_format_alpha,
                )
                hook_handles = _register_component_add_hooks(
                    model=model,
                    additions=additions,
                    apply_mode=args.rcm_format_apply_mode,
                )
            prediction = model.generate(
                base_prompt,
                context_prompt,
                mode=args.mode,
                alpha=args.alpha,
                adaptive=args.adaptive,
                select_top=args.select_top,
                max_new_tokens=args.max_new_tokens,
                top_p=1.0,
                top_k=100,
                temperature=1.0,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                remove_stop_words=True,
            ).strip()
        except TypeError as exc:
            if "ck_decoding" in str(exc):
                raise RuntimeError(
                    "Official CK-PLUG mode requires the authors' modified transformers with ck_decoding support. "
                    "Install their transformers fork before running this baseline."
                ) from exc
            raise
        finally:
            for handle in hook_handles:
                handle.remove()

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
                "split": row.get("_cecm_split", ""),
                "row_index": row.get("_cecm_row_index", ""),
                "dataset": row["dataset"],
                "method": method,
                "prompt_key": args.prompt_family,
                "source_index": row["source_index"],
                "alias_risk": row.get("alias_risk", ""),
                "orig_answer": row["orig_answer"],
                "cf_answer": row["cf_answer"],
                "prediction": prediction,
                "prediction_scored": scored_prediction,
                "prediction_norm": _normalize_answer(scored_prediction),
                "output_chars": len(prediction),
                "prompt": context_prompt if args.mode != "base_no_rag" else base_prompt,
                **classified,
                **{f"health_{key}": value for key, value in health.items() if key != "final_text_for_scoring"},
            }
        )

    metadata = {
        "top_k": "",
        "alpha": args.alpha,
        "prompt_key": args.prompt_family,
        "max_new_tokens": args.max_new_tokens,
        "stop_strings": args.stop_strings,
        "rcm_format_alpha": args.rcm_format_alpha,
        "rcm_format_top_k": args.rcm_format_top_k_components if format_component_specs else "",
        "rcm_format_component_source": format_component_source,
        "rcm_format_apply_mode": args.rcm_format_apply_mode if format_component_specs else "",
        "candidate_tagging": args.candidate_tagging,
        "score_tagged_final": args.score_tagged_final,
        "split": args.split,
        "start": args.start,
        "max_eval_rows": args.max_eval_rows if args.max_eval_rows is not None else "",
        "val_mod": args.val_mod,
    }
    dump_jsonl(args.out_generations_jsonl, output_rows)
    dump_csv(args.out_summary_csv, _summarize(output_rows, metadata=metadata))
    dump_csv(
        args.out_manifest_csv,
        [
            {
                "method": method,
                "ck_core_dir": str(args.ck_core_dir),
                "ck_transformers_dir": str(args.ck_transformers_dir) if args.ck_transformers_dir else "",
                "ck_transformers_import_path": transformers_import_path,
                "model": args.model,
                "resolved_model": resolved_model,
                "mode": args.mode,
                "prompt_family": args.prompt_family,
                "use_chat_template": args.use_chat_template,
                "schema": args.schema,
                "alpha": args.alpha,
                "adaptive": args.adaptive,
                "select_top": args.select_top,
                "max_new_tokens": args.max_new_tokens,
                "stop_strings": args.stop_strings,
                "candidate_tagging": args.candidate_tagging,
                "score_tagged_final": args.score_tagged_final,
                "pad_token_id": pad_token_id,
                "pad_token": pad_token,
                "eos_token_id": ",".join(str(x) for x in eos_token_id) if eos_token_id else "",
                "eval_jsonl": str(args.eval_jsonl),
                "split": args.split,
                "start": args.start,
                "max_eval_rows": args.max_eval_rows if args.max_eval_rows is not None else "",
                "val_mod": args.val_mod,
                "rcm_format_alpha": args.rcm_format_alpha,
                "rcm_format_top_k": args.rcm_format_top_k_components if format_component_specs else "",
                "rcm_format_component_ids": ",".join(
                    f"L{int(row['layer_idx'])}.{row['component_type']}"
                    for row in selected_format_components
                ),
                "rcm_format_component_source": format_component_source,
                "rcm_format_apply_mode": args.rcm_format_apply_mode if format_component_specs else "",
            }
        ],
    )
    print(f"[score-ckplug-official] rows={len(output_rows)} method={method}")


if __name__ == "__main__":
    main()
