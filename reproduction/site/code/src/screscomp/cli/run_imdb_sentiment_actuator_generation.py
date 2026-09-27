from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.actuator import load_fixed_actuator_additions, parse_alpha_list
from screscomp.cecm.components import parse_component_id
from screscomp.cli.cecm_run_joint_actuator_generation import (
    HeadScaleAction,
    JointControl,
    JointGenerationRunner,
    _load_head_actuator,
)
from screscomp.data import dump_json
from screscomp.modeling import TransformersABBackend


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate held-out IMDb continuations with a trained fixed actuator and alpha sweep."
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer name/path. Defaults to --model.")
    p.add_argument("--prompts-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument(
        "--actuator",
        type=Path,
        default=None,
        help="Path to fixed_actuator.pt or its containing directory. Omit only for alpha=0 baseline generation.",
    )
    p.add_argument(
        "--component-actuators",
        type=str,
        default="",
        help=(
            "Optional semicolon specs for multiple fixed component actuators. "
            "Format: name=path or name=path:apply_mode. These are added in addition to --actuator."
        ),
    )
    p.add_argument(
        "--head-actuator",
        type=Path,
        default=None,
        help="Path to head_actuator.pt or its containing directory. Can be combined with --actuator.",
    )
    p.add_argument("--control-name", type=str, default="mlp_positive")
    p.add_argument("--alpha-sweep", type=str, default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--sign", type=float, default=1.0)
    p.add_argument("--generation-apply-mode", type=str, default="prefill")
    p.add_argument(
        "--component-apply-mode",
        type=str,
        default=None,
        help="Optional local timing for component actuators; defaults to --generation-apply-mode.",
    )
    p.add_argument(
        "--head-apply-mode",
        type=str,
        default=None,
        help="Optional local timing for head actuators; defaults to --generation-apply-mode.",
    )
    p.add_argument(
        "--include-components",
        type=str,
        default="",
        help="Comma-separated component ids to keep from --actuator. Empty keeps all.",
    )
    p.add_argument(
        "--exclude-components",
        type=str,
        default="",
        help="Comma-separated component ids to remove from --actuator.",
    )
    p.add_argument(
        "--include-heads",
        type=str,
        default="",
        help="Comma-separated head ids like L31.attn.h14 to keep from --head-actuator. Empty keeps all.",
    )
    p.add_argument(
        "--exclude-heads",
        type=str,
        default="",
        help="Comma-separated head ids like L31.attn.h14 to remove from --head-actuator.",
    )
    p.add_argument(
        "--zero-heads",
        type=str,
        default="",
        help=(
            "Comma-separated head ids like L31.attn.h14 whose pre-o_proj slices are scaled to zero "
            "before optional head-vector injection."
        ),
    )
    p.add_argument(
        "--zero-head-apply-mode",
        type=str,
        default=None,
        help="Optional timing for --zero-heads; defaults to --head-apply-mode.",
    )
    p.add_argument(
        "--zero-components",
        type=str,
        default="",
        help="Comma-separated component ids like L19.mlp whose residual outputs are scaled to zero.",
    )
    p.add_argument(
        "--zero-component-groups",
        type=str,
        default="",
        help=(
            "Optional semicolon specs mode=Lx.mlp,Ly.attn for zeroing component groups with separate timing. "
            "These are added in addition to --zero-components."
        ),
    )
    p.add_argument(
        "--zero-component-apply-mode",
        type=str,
        default=None,
        help="Optional timing for --zero-components; defaults to --component-apply-mode.",
    )
    p.add_argument("--split", type=str, default="eval", help="Use all to keep every split.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--samples-per-prompt", type=int, default=1)
    p.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help=(
            "Batch prompt generation when possible. "
            "Component-only and non-stepwise joint controls can generate in prompt batches; "
            "boxed-decision and conditional controls fall back to serial generation."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--stop-strings", type=str, default="")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--same-seed-across-alpha",
        action="store_true",
        help="Use the same sampling seed for the same prompt/sample_index across alpha values.",
    )
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


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in str(raw or "").split(",") if item.strip()}


def _parse_zero_head_scales(raw: str, *, apply_mode: str) -> tuple[HeadScaleAction, ...]:
    actions: list[HeadScaleAction] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"L(\d+)\.attn\.h(\d+)", item)
        if not match:
            raise ValueError(f"Unsupported zero head id: {item!r}; expected L<layer>.attn.h<head>")
        actions.append(
            HeadScaleAction(
                layer_idx=int(match.group(1)),
                head_idx=int(match.group(2)),
                factor=0.0,
                apply_mode=apply_mode,
            )
        )
    return tuple(actions)


def _parse_zero_component_scales(raw: str, *, apply_mode: str) -> tuple[dict[str, Any], ...]:
    actions: list[dict[str, Any]] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        component = parse_component_id(item)
        actions.append(
            {
                "component_id": component.component_id,
                "layer_idx": int(component.layer_idx),
                "component_type": component.component_type,
                "scale_factor": 0.0,
                "apply_mode": apply_mode,
            }
        )
    return tuple(actions)


def _parse_zero_component_groups(raw: str) -> tuple[dict[str, Any], ...]:
    actions: list[dict[str, Any]] = []
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected mode=component_ids in --zero-component-groups, got {part!r}")
        mode, ids = part.split("=", 1)
        mode = mode.strip()
        if not mode:
            raise ValueError(f"Empty apply mode in --zero-component-groups: {part!r}")
        actions.extend(_parse_zero_component_scales(ids, apply_mode=mode))
    return tuple(actions)


def _batches(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


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


def _resolve_payload_path(path: Path | None, *, filename: str) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        path = path / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed actuator payload: {path}")
    return path


def _resolve_payload_string(raw: str, *, filename: str) -> Path:
    path = Path(raw)
    if path.is_dir():
        path = path / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed actuator payload: {path}")
    return path


def _parse_component_actuator_specs(raw: str) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected name=path[:apply_mode] in --component-actuators, got {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"Invalid --component-actuators spec: {part!r}")
        apply_mode = ""
        path_text = value
        if ":" in value:
            maybe_path, maybe_mode = value.rsplit(":", 1)
            if maybe_mode in {"all", "prefill", "prompt", "prompt_last", "decode", "decision_tokens", "boxed_decision"} or re.fullmatch(
                r"first(?:_\d+)?_decode", maybe_mode
            ):
                path_text = maybe_path
                apply_mode = maybe_mode
        specs.append(
            {
                "name": name,
                "path": _resolve_payload_string(path_text, filename="fixed_actuator.pt"),
                "apply_mode": apply_mode,
            }
        )
    return tuple(specs)


def _alpha_key(alpha: float) -> str:
    return f"{float(alpha):.8g}"


def _completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str, str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            keys.add(
                (
                    str(row.get("sample_id", "")),
                    str(row.get("control_name", "")),
                    _alpha_key(float(row.get("alpha", 0.0))),
                    int(row.get("sample_index", 0)),
                )
            )
    return keys


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


def _filter_component_additions(
    additions: list[dict[str, Any]],
    *,
    include_components: set[str],
    exclude_components: set[str],
) -> list[dict[str, Any]]:
    filtered = []
    for addition in additions:
        component_id = str(addition.get("component_id", ""))
        if include_components and component_id not in include_components:
            continue
        if exclude_components and component_id in exclude_components:
            continue
        filtered.append(addition)
    return filtered


def _filter_head_actions(
    actions: tuple[Any, ...],
    *,
    include_heads: set[str],
    exclude_heads: set[str],
) -> tuple[Any, ...]:
    filtered = []
    for action in actions:
        head_id = str(getattr(action, "head_id", ""))
        if include_heads and head_id not in include_heads:
            continue
        if exclude_heads and head_id in exclude_heads:
            continue
        filtered.append(action)
    return tuple(filtered)


def _component_ids(additions: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[str]:
    return [str(addition.get("component_id", "")) for addition in additions if str(addition.get("component_id", ""))]


def _component_apply_modes(additions: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> list[str]:
    return sorted({str(addition.get("apply_mode", "")) for addition in additions if str(addition.get("apply_mode", ""))})


def _component_specs_manifest(specs: tuple[dict[str, Any], ...]) -> list[dict[str, str]]:
    return [
        {
            "name": str(spec.get("name", "")),
            "path": str(spec.get("path", "")),
            "apply_mode": str(spec.get("apply_mode", "")),
        }
        for spec in specs
    ]


def _head_ids(actions: tuple[Any, ...]) -> list[str]:
    return [str(getattr(action, "head_id", "")) for action in actions if str(getattr(action, "head_id", ""))]


def _additions_for_alpha(
    path: Path | None,
    *,
    alpha: float,
    sign: float,
    apply_mode: str,
    include_components: set[str],
    exclude_components: set[str],
) -> list[dict[str, Any]]:
    if alpha == 0:
        return []
    if path is None:
        raise ValueError("Non-zero alpha requires --actuator")
    additions = load_fixed_actuator_additions(path, alpha=float(sign) * float(alpha), apply_mode=apply_mode)
    return _filter_component_additions(
        additions,
        include_components=include_components,
        exclude_components=exclude_components,
    )


def _component_spec_additions_for_alpha(
    specs: tuple[dict[str, Any], ...],
    *,
    alpha: float,
    sign: float,
    default_apply_mode: str,
    include_components: set[str],
    exclude_components: set[str],
) -> list[dict[str, Any]]:
    if alpha == 0:
        return []
    additions: list[dict[str, Any]] = []
    for spec in specs:
        path = Path(spec["path"])
        apply_mode = str(spec.get("apply_mode") or default_apply_mode)
        loaded = load_fixed_actuator_additions(path, alpha=float(sign) * float(alpha), apply_mode=apply_mode)
        additions.extend(
            _filter_component_additions(
                loaded,
                include_components=include_components,
                exclude_components=exclude_components,
            )
        )
    return additions


def _joint_control_for_alpha(
    *,
    name: str,
    component_path: Path | None,
    component_specs: tuple[dict[str, Any], ...],
    head_path: Path | None,
    alpha: float,
    sign: float,
    apply_mode: str,
    component_apply_mode: str,
    head_apply_mode: str,
    include_components: set[str],
    exclude_components: set[str],
    include_heads: set[str],
    exclude_heads: set[str],
    zero_component_scales: tuple[dict[str, Any], ...],
    zero_head_scales: tuple[HeadScaleAction, ...],
) -> JointControl | None:
    component_additions: list[dict[str, Any]] = list(zero_component_scales)
    if component_path is not None:
        component_additions.extend(
            _additions_for_alpha(
                component_path,
                alpha=alpha,
                sign=sign,
                apply_mode=component_apply_mode,
                include_components=include_components,
                exclude_components=exclude_components,
            )
        )
    component_additions.extend(
        _component_spec_additions_for_alpha(
            component_specs,
            alpha=alpha,
            sign=sign,
            default_apply_mode=component_apply_mode,
            include_components=include_components,
            exclude_components=exclude_components,
        )
    )
    head_vectors = tuple(
        _filter_head_actions(
            _load_head_actuator(head_path, alpha=float(sign) * float(alpha), apply_mode=head_apply_mode),
            include_heads=include_heads,
            exclude_heads=exclude_heads,
        )
        if alpha != 0 and head_path is not None
        else ()
    )
    if not component_additions and not head_vectors and not zero_head_scales:
        return None
    return JointControl(
        name=name,
        component_additions=tuple(component_additions),
        head_scales=zero_head_scales,
        head_vectors=head_vectors,
        conditional_components=tuple(),
        conditional_heads=tuple(),
        conditional_group_specs={},
        parts=tuple(),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples-per-prompt must be positive")
    if args.generation_batch_size <= 0:
        raise SystemExit("--generation-batch-size must be positive")
    alphas = parse_alpha_list(args.alpha_sweep)
    component_apply_mode = args.component_apply_mode or args.generation_apply_mode
    head_apply_mode = args.head_apply_mode or args.generation_apply_mode
    zero_head_apply_mode = args.zero_head_apply_mode or head_apply_mode
    zero_component_apply_mode = args.zero_component_apply_mode or component_apply_mode
    include_components = _parse_csv_set(args.include_components)
    exclude_components = _parse_csv_set(args.exclude_components)
    include_heads = _parse_csv_set(args.include_heads)
    exclude_heads = _parse_csv_set(args.exclude_heads)
    zero_head_scales = _parse_zero_head_scales(args.zero_heads, apply_mode=zero_head_apply_mode)
    zero_component_scales = tuple(_parse_zero_component_scales(args.zero_components, apply_mode=zero_component_apply_mode)) + tuple(
        _parse_zero_component_groups(args.zero_component_groups)
    )
    actuator_path = _resolve_payload_path(args.actuator, filename="fixed_actuator.pt")
    component_specs = _parse_component_actuator_specs(args.component_actuators)
    head_actuator_path = _resolve_payload_path(args.head_actuator, filename="head_actuator.pt")
    if actuator_path is None and not component_specs and head_actuator_path is None and any(alpha != 0 for alpha in alphas):
        raise SystemExit("--actuator, --component-actuators, or --head-actuator is required when --alpha-sweep contains non-zero values")

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

    print(
        (
            f"[imdb-actuator-gen] loading model={args.model} prompts={len(prompt_rows)} "
            f"alphas={','.join(_alpha_key(alpha) for alpha in alphas)}"
        ),
        flush=True,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    joint_runner = (
        JointGenerationRunner(backend)
        if head_actuator_path is not None or zero_head_scales or zero_component_scales
        else None
    )

    new_rows = 0
    with args.out_jsonl.open("a", encoding="utf-8") as stream:
        for alpha in alphas:
            joint_control = (
                _joint_control_for_alpha(
                    name=args.control_name,
                    component_path=actuator_path,
                    component_specs=component_specs,
                    head_path=head_actuator_path,
                    alpha=alpha,
                    sign=args.sign,
                    apply_mode=args.generation_apply_mode,
                    component_apply_mode=component_apply_mode,
                    head_apply_mode=head_apply_mode,
                    include_components=include_components,
                    exclude_components=exclude_components,
                    include_heads=include_heads,
                    exclude_heads=exclude_heads,
                    zero_component_scales=zero_component_scales,
                    zero_head_scales=zero_head_scales,
                )
                if joint_runner is not None
                else None
            )
            additions: list[dict[str, Any]] = []
            if joint_control is None:
                if actuator_path is not None:
                    additions.extend(
                        _additions_for_alpha(
                            actuator_path,
                            alpha=alpha,
                            sign=args.sign,
                            apply_mode=component_apply_mode,
                            include_components=include_components,
                            exclude_components=exclude_components,
                        )
                    )
                additions.extend(
                    _component_spec_additions_for_alpha(
                        component_specs,
                        alpha=alpha,
                        sign=args.sign,
                        default_apply_mode=component_apply_mode,
                        include_components=include_components,
                        exclude_components=exclude_components,
                    )
                )
            alpha_text = _alpha_key(alpha)
            for sample_index in range(args.samples_per_prompt):
                active_rows: list[dict[str, Any]] = []
                for row in prompt_rows:
                    sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
                    prompt = str(row.get("prompt", ""))
                    if not sample_id or not prompt:
                        continue
                    key = (sample_id, args.control_name, alpha_text, sample_index)
                    if key in completed:
                        continue
                    active_rows.append(row)
                if not active_rows:
                    continue

                use_batched_generation = args.generation_batch_size > 1
                for batch_index, batch in enumerate(
                    tqdm(
                        _batches(active_rows, args.generation_batch_size if use_batched_generation else 1),
                        desc=f"generate imdb actuator a={alpha_text} s={sample_index}",
                    )
                ):
                    alpha_seed_offset = 0 if args.same_seed_across_alpha else int(round(float(alpha) * 1000.0))
                    batch_seed = None
                    if use_batched_generation:
                        batch_seed = args.seed + sample_index * 9176 + batch_index * 1009 + alpha_seed_offset
                        _set_seed(backend, batch_seed)
                        prompts = [str(row.get("prompt", "")) for row in batch]
                        if joint_control is not None and joint_runner is not None:
                            completions = joint_runner.generate_many(
                                prompts,
                                control=joint_control,
                                apply_mode=args.generation_apply_mode,
                                max_new_tokens=args.max_new_tokens,
                                stop_strings=stop_strings,
                                do_sample=bool(args.do_sample),
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                            )
                        elif additions:
                            completions = backend.generate_many_with_component_last_token_add_many(
                                prompts,
                                additions,
                                max_new_tokens=args.max_new_tokens,
                                stop_strings=stop_strings,
                                do_sample=bool(args.do_sample),
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                apply_mode=args.generation_apply_mode,
                            )
                        else:
                            completions = backend.generate_many(
                                prompts,
                                max_new_tokens=args.max_new_tokens,
                                stop_strings=stop_strings,
                                do_sample=bool(args.do_sample),
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                            )
                    else:
                        completions = []
                        for row in batch:
                            sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
                            prompt = str(row.get("prompt", ""))
                            row_seed = _as_int(row.get("source_row_index", row.get("raw_index", 0)))
                            local_seed = args.seed + row_seed * 1009 + sample_index * 9176 + alpha_seed_offset
                            _set_seed(backend, local_seed)
                            batch_seed = local_seed
                            if joint_control is not None and joint_runner is not None:
                                completion = joint_runner.generate(
                                    prompt,
                                    control=joint_control,
                                    apply_mode=args.generation_apply_mode,
                                    max_new_tokens=args.max_new_tokens,
                                    stop_strings=stop_strings,
                                    do_sample=bool(args.do_sample),
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    top_k=args.top_k,
                                )
                            elif additions:
                                completion = backend.generate_with_component_last_token_add_many(
                                    prompt,
                                    additions,
                                    max_new_tokens=args.max_new_tokens,
                                    stop_strings=stop_strings,
                                    do_sample=bool(args.do_sample),
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    top_k=args.top_k,
                                    apply_mode=args.generation_apply_mode,
                                )
                            else:
                                completion = backend.generate(
                                    prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    stop_strings=stop_strings,
                                    do_sample=bool(args.do_sample),
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    top_k=args.top_k,
                                )
                            completions.append(completion)

                    for row, completion in zip(batch, completions):
                        sample_id = str(row.get("sample_id") or row.get("prompt_id") or "")
                        prompt = str(row.get("prompt", ""))
                        key = (sample_id, args.control_name, alpha_text, sample_index)
                        output = {
                            "sample_id": sample_id,
                            "prompt_id": row.get("prompt_id", ""),
                            "split": row.get("split", ""),
                            "event": row.get("event", "imdb_positive_sentiment"),
                            "prompt": prompt,
                            "prefix": row.get("prefix", ""),
                            "control_name": args.control_name,
                            "alpha": float(alpha),
                            "sample_index": sample_index,
                            "completion": completion,
                            "full_text": prompt + completion,
                            "model": backend.model_id,
                            "tokenizer": backend.resolved_tokenizer_name_or_path,
                            "actuator_path": str(actuator_path) if actuator_path else "",
                            "component_actuator_specs": _component_specs_manifest(component_specs),
                            "head_actuator_path": str(head_actuator_path) if head_actuator_path else "",
                            "actuator_sign": args.sign,
                            "generation_apply_mode": args.generation_apply_mode,
                            "component_apply_mode": component_apply_mode,
                            "head_apply_mode": head_apply_mode,
                            "included_components": sorted(include_components),
                            "excluded_components": sorted(exclude_components),
                            "included_heads": sorted(include_heads),
                            "excluded_heads": sorted(exclude_heads),
                            "zeroed_components": _component_ids(zero_component_scales),
                            "zero_component_apply_modes": _component_apply_modes(zero_component_scales),
                            "zero_component_groups": args.zero_component_groups,
                            "zero_component_apply_mode": zero_component_apply_mode,
                            "zeroed_heads": _head_ids(zero_head_scales),
                            "zero_head_apply_mode": zero_head_apply_mode,
                            "effective_components": _component_ids(
                                joint_control.component_additions if joint_control is not None else additions
                            ),
                            "effective_heads": _head_ids(joint_control.head_vectors) if joint_control is not None else [],
                            "source_dataset": row.get("source_dataset", ""),
                            "source_split": row.get("source_split", ""),
                            "source_row_index": row.get("source_row_index", row.get("raw_index", "")),
                            "generation_do_sample": int(bool(args.do_sample)),
                            "generation_temperature": args.temperature,
                            "generation_top_p": args.top_p,
                            "generation_top_k": args.top_k,
                            "max_new_tokens": args.max_new_tokens,
                            "same_seed_across_alpha": int(bool(args.same_seed_across_alpha)),
                            "generation_batch_size": args.generation_batch_size if use_batched_generation else 1,
                            "batch_seed": batch_seed if batch_seed is not None else "",
                        }
                        stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                        stream.flush()
                        completed.add(key)
                        new_rows += 1

    manifest_path = args.manifest_json or (args.out_jsonl.parent / "generation_manifest.json")
    dump_json(
        manifest_path,
        {
            "model": backend.model_id,
            "tokenizer": backend.resolved_tokenizer_name_or_path,
            "prompts_jsonl": str(args.prompts_jsonl),
            "out_jsonl": str(args.out_jsonl),
            "actuator_path": str(actuator_path) if actuator_path else "",
            "component_actuator_specs": _component_specs_manifest(component_specs),
            "head_actuator_path": str(head_actuator_path) if head_actuator_path else "",
            "control_name": args.control_name,
            "split": args.split,
            "prompt_rows": len(prompt_rows),
            "alpha_sweep": alphas,
            "sign": args.sign,
            "samples_per_prompt": args.samples_per_prompt,
            "generation_batch_size": args.generation_batch_size,
            "new_rows": new_rows,
            "completed_rows": len(completed),
            "max_new_tokens": args.max_new_tokens,
            "generation_apply_mode": args.generation_apply_mode,
            "component_apply_mode": component_apply_mode,
            "head_apply_mode": head_apply_mode,
            "zero_component_apply_mode": zero_component_apply_mode,
            "zero_head_apply_mode": zero_head_apply_mode,
            "zero_component_groups": args.zero_component_groups,
            "include_components": sorted(include_components),
            "exclude_components": sorted(exclude_components),
            "include_heads": sorted(include_heads),
            "exclude_heads": sorted(exclude_heads),
            "zero_components": _component_ids(zero_component_scales),
            "zero_component_apply_modes": _component_apply_modes(zero_component_scales),
            "zero_heads": _head_ids(zero_head_scales),
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "same_seed_across_alpha": bool(args.same_seed_across_alpha),
            "score_expectation": "score generated completion text with the sentiment reward model",
            "role": "heldout_actuator_generation",
        },
    )
    print(f"[imdb-actuator-gen] wrote new_rows={new_rows} path={args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
