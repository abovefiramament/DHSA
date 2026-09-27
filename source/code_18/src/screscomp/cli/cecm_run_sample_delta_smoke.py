from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.components import ComponentSpec, parse_component_id
from screscomp.cecm.pairs import (
    CONTEXT_ANSWER_KEYS,
    DATASET_PRIOR_ANSWER_KEYS,
    MODEL_PRIOR_ANSWER_KEYS,
    all_texts,
    normalize_answer,
    prompt_from_row,
    sample_id_from_row,
    split_for_index,
)
from screscomp.cecm.scaling import classify_source_generation, component_universe, summarize_generation_rows
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class SmokeRow:
    sample_id: str
    split: str
    start_prompt: str
    target_prompt: str
    generation_prompt: str
    context_answers: tuple[str, ...]
    prior_answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SmokeControl:
    name: str
    baseline_kind: str
    component_group: str
    components: tuple[ComponentSpec, ...]
    sign: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run old-RCM-style same-sample prompt-delta smoke control.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--eval-open-rows", type=Path, required=True)
    p.add_argument("--start-prompt-key", type=str, default="prior_objective_rag")
    p.add_argument("--target-prompt-key", type=str, default="strong_rag")
    p.add_argument("--generation-prompt-key", type=str, default="base_rag")
    p.add_argument("--prior-source", choices=["auto", "model_prior", "dataset_orig"], default="dataset_orig")
    p.add_argument("--components", type=str, required=True)
    p.add_argument("--controls", type=str, default="force_target,force_start")
    p.add_argument("--random-baselines", type=str, default="")
    p.add_argument("--random-trials", type=int, default=0)
    p.add_argument("--alpha-sweep", type=str, default="0,0.5,1.0,1.5")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=30)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--generation-apply-mode", type=str, default="prefill")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--verbose-char-threshold", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_alphas(raw: str) -> list[float]:
    alphas: list[float] = []
    seen: set[float] = set()
    for item in _parse_csv(raw):
        value = float(item)
        if value not in seen:
            seen.add(value)
            alphas.append(value)
    return alphas


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _parse_components(raw: str) -> tuple[ComponentSpec, ...]:
    components = tuple(parse_component_id(item) for item in _parse_csv(raw))
    if not components:
        raise ValueError("component group is empty")
    return components


def _parse_component_groups(raw: str) -> dict[str, tuple[ComponentSpec, ...]]:
    parts = [part.strip() for part in raw.split(";") if part.strip()]
    if len(parts) == 1 and "=" not in parts[0]:
        return {"cecm": _parse_components(parts[0])}
    groups: dict[str, tuple[ComponentSpec, ...]] = {}
    for idx, part in enumerate(parts, start=1):
        if "=" in part:
            name, values = part.split("=", 1)
            name = name.strip()
            values = values.strip()
        else:
            name = f"group{idx}"
            values = part
        groups[name] = _parse_components(values)
    return groups


def _prior_answers(row: dict[str, Any], prior_source: str) -> tuple[str, ...]:
    if prior_source == "model_prior":
        answers, _source = all_texts(row, MODEL_PRIOR_ANSWER_KEYS)
        return answers
    if prior_source == "dataset_orig":
        answers, _source = all_texts(row, DATASET_PRIOR_ANSWER_KEYS)
        return answers
    answers, _source = all_texts(row, MODEL_PRIOR_ANSWER_KEYS)
    if answers:
        return answers
    answers, _source = all_texts(row, DATASET_PRIOR_ANSWER_KEYS)
    return answers


def _load_rows(
    path: Path,
    *,
    start_prompt_key: str,
    target_prompt_key: str,
    generation_prompt_key: str,
    prior_source: str,
    split: str,
    start: int,
    max_rows: int,
    val_mod: int,
) -> list[SmokeRow]:
    out: list[SmokeRow] = []
    for row_index, row in enumerate(load_jsonl(path)):
        row_split = split_for_index(row_index, val_mod)
        if split != "all" and row_split != split:
            continue
        start_prompt, _ = prompt_from_row(row, start_prompt_key)
        target_prompt, _ = prompt_from_row(row, target_prompt_key)
        generation_prompt, _ = prompt_from_row(row, generation_prompt_key)
        if not start_prompt or not target_prompt or not generation_prompt:
            continue
        context_answers, _ = all_texts(row, CONTEXT_ANSWER_KEYS)
        prior_answers = _prior_answers(row, prior_source)
        if not context_answers or not prior_answers:
            continue
        if {normalize_answer(answer) for answer in context_answers} & {normalize_answer(answer) for answer in prior_answers}:
            continue
        out.append(
            SmokeRow(
                sample_id=sample_id_from_row(row, row_index),
                split=row_split,
                start_prompt=start_prompt,
                target_prompt=target_prompt,
                generation_prompt=generation_prompt,
                context_answers=context_answers,
                prior_answers=prior_answers,
            )
        )
    if start > 0:
        out = out[start:]
    if max_rows > 0:
        out = out[:max_rows]
    return out


def _sample_random_components(
    *,
    universe: tuple[ComponentSpec, ...],
    originals: tuple[ComponentSpec, ...],
    exclude: set[str],
    rng: random.Random,
    random_kind: str,
) -> tuple[ComponentSpec, ...]:
    if random_kind == "random_any":
        candidates = [component for component in universe if component.component_id not in exclude]
        return tuple(rng.sample(candidates, len(originals)))
    if random_kind != "random_type_matched":
        raise ValueError(f"Unsupported random baseline: {random_kind}")
    selected: list[ComponentSpec] = []
    local_exclude = set(exclude)
    for component_type in sorted({component.component_type for component in originals}):
        count = sum(1 for component in originals if component.component_type == component_type)
        candidates = [
            component
            for component in universe
            if component.component_id not in local_exclude and component.component_type == component_type
        ]
        sampled = rng.sample(candidates, count)
        selected.extend(sampled)
        local_exclude.update(component.component_id for component in sampled)
    return tuple(selected)


def _build_controls(
    *,
    component_groups: dict[str, tuple[ComponentSpec, ...]],
    controls: list[str],
    random_baselines: list[str],
    random_trials: int,
    universe: tuple[ComponentSpec, ...],
    seed: int,
) -> list[SmokeControl]:
    out: list[SmokeControl] = []
    for group_name, components in component_groups.items():
        for control in controls:
            if control == "force_target":
                out.append(SmokeControl(f"{group_name}_force_target", "cecm_sample_delta", group_name, components, 1.0))
            elif control == "force_start":
                out.append(SmokeControl(f"{group_name}_force_start", "cecm_sample_delta", group_name, components, -1.0))
            else:
                raise ValueError(f"Unsupported control: {control}")

    originals = {component.component_id for group in component_groups.values() for component in group}
    rng = random.Random(seed)
    for group_name, components in component_groups.items():
        exclude = set(originals)
        for trial_idx in range(random_trials):
            for random_kind in random_baselines:
                sampled = _sample_random_components(
                    universe=universe,
                    originals=components,
                    exclude=exclude,
                    rng=rng,
                    random_kind=random_kind,
                )
                exclude.update(component.component_id for component in sampled)
                for control in controls:
                    sign = 1.0 if control == "force_target" else -1.0
                    out.append(SmokeControl(f"{random_kind}_{group_name}_{control}_s{trial_idx}", random_kind, group_name, sampled, sign))
    return out


def _component_tuples(components: Iterable[ComponentSpec]) -> list[tuple[int, str]]:
    return [(component.layer_idx, component.component_type) for component in components]


def _component_ids(components: Iterable[ComponentSpec]) -> str:
    return ",".join(component.component_id for component in components)


def _additions_for_row(
    *,
    backend: TransformersABBackend,
    row: SmokeRow,
    control: SmokeControl,
    alpha: float,
    apply_mode: str,
) -> list[dict[str, Any]]:
    if alpha == 0:
        return []
    specs = _component_tuples(control.components)
    start_acts = backend.capture_component_last_token(row.start_prompt, specs)
    target_acts = backend.capture_component_last_token(row.target_prompt, specs)
    additions: list[dict[str, Any]] = []
    for component in control.components:
        key = (component.layer_idx, component.component_type)
        additions.append(
            {
                "component_id": component.component_id,
                "layer_idx": component.layer_idx,
                "component_type": component.component_type,
                "direction": control.sign * (target_acts[key] - start_acts[key]),
                "alpha": alpha,
                "apply_mode": apply_mode,
            }
        )
    return additions


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphas = _parse_alphas(args.alpha_sweep)
    stop_strings = _parse_stop_strings(args.stop_strings)
    component_groups = _parse_component_groups(args.components)
    controls = _parse_csv(args.controls)
    random_baselines = _parse_csv(args.random_baselines)

    print(f"[cecm-sample-delta-smoke] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    rows = _load_rows(
        args.eval_open_rows,
        start_prompt_key=args.start_prompt_key,
        target_prompt_key=args.target_prompt_key,
        generation_prompt_key=args.generation_prompt_key,
        prior_source=args.prior_source,
        split=args.split,
        start=args.start,
        max_rows=args.max_rows,
        val_mod=args.val_mod,
    )
    if not rows:
        raise SystemExit("No eval rows selected.")

    control_specs = _build_controls(
        component_groups=component_groups,
        controls=controls,
        random_baselines=random_baselines,
        random_trials=args.random_trials,
        universe=component_universe(backend.num_layers),
        seed=args.seed,
    )
    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "eval_open_rows": str(args.eval_open_rows),
            "mode": "same_sample_delta_smoke",
            "start_prompt_key": args.start_prompt_key,
            "target_prompt_key": args.target_prompt_key,
            "generation_prompt_key": args.generation_prompt_key,
            "prior_source": args.prior_source,
            "eval_rows": len(rows),
            "alpha_sweep": alphas,
            "generation_apply_mode": args.generation_apply_mode,
            "component_groups": {
                name: [component.component_id for component in components]
                for name, components in component_groups.items()
            },
        },
    )
    dump_csv(
        args.out_dir / "control_plan.csv",
        [
            {
                "control_name": control.name,
                "baseline_kind": control.baseline_kind,
                "component_group": control.component_group,
                "sign": control.sign,
                "components": _component_ids(control.components),
            }
            for control in control_specs
        ],
    )

    generation_rows: list[dict[str, object]] = []
    for control in control_specs:
        for alpha in alphas:
            for row in tqdm(rows, desc=f"generate {control.name} a={alpha:g}"):
                additions = _additions_for_row(
                    backend=backend,
                    row=row,
                    control=control,
                    alpha=alpha,
                    apply_mode=args.generation_apply_mode,
                )
                if additions:
                    prediction = backend.generate_with_component_last_token_add_many(
                        row.generation_prompt,
                        additions,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        apply_mode=args.generation_apply_mode,
                    )
                else:
                    prediction = backend.generate(
                        row.generation_prompt,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                metrics = classify_source_generation(
                    prediction,
                    context_answers=row.context_answers,
                    prior_answers=row.prior_answers,
                    verbose_char_threshold=args.verbose_char_threshold,
                )
                generation_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "control_name": control.name,
                        "baseline_kind": control.baseline_kind,
                        "env_kind": "context",
                        "target_direction": "target" if control.sign > 0 else "start",
                        "component_group": control.component_group,
                        "alpha": alpha,
                        "prediction": prediction,
                        "start_prompt_key": args.start_prompt_key,
                        "target_prompt_key": args.target_prompt_key,
                        "generation_prompt_key": args.generation_prompt_key,
                        "prior_source": args.prior_source,
                        "context_answers_json": list(row.context_answers),
                        "prior_answers_json": list(row.prior_answers),
                        "components": _component_ids(control.components),
                        **metrics,
                    }
                )

    dump_jsonl(args.out_dir / "generation_rows.jsonl", generation_rows)
    dump_csv(args.out_dir / "generation_summary.csv", summarize_generation_rows(generation_rows))
    dump_csv(
        args.out_dir / "run_summary.csv",
        [
            {"metric": "eval_rows", "value": len(rows)},
            {"metric": "control_specs", "value": len(control_specs)},
            {"metric": "generation_rows", "value": len(generation_rows)},
        ],
    )
    print(f"[cecm-sample-delta-smoke] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
