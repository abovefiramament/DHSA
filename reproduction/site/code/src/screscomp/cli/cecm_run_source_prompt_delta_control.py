from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.actuator import ActuatorPair, load_actuator_pairs
from screscomp.cecm.components import ComponentSpec, parse_component_id
from screscomp.cecm.pairs import prompt_from_row, sample_id_from_row
from screscomp.cecm.scaling import (
    OpenEvalRow,
    classify_source_generation,
    component_universe,
    load_external_generations,
    load_open_eval_rows,
    summarize_generation_rows,
)
from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class PromptDeltaRow:
    sample_id: str
    split: str
    context_prompt: str
    prior_prompt: str


@dataclass(frozen=True, slots=True)
class PromptDeltaControl:
    name: str
    baseline_kind: str
    target_direction: str
    component_group: str
    components: tuple[ComponentSpec, ...]
    sign: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run no-training CECM source controls with prompt-delta component vectors. "
            "Directions are mean component writes from the configured context prompt minus prior prompt, "
            "then applied as additive vectors at frozen component sites."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--vector-open-rows", type=Path, required=True)
    p.add_argument("--eval-open-rows", type=Path, required=True)
    p.add_argument("--vector-pairs-csv", type=Path, default=None)
    p.add_argument("--ckplug-generations", type=Path, default=None)
    p.add_argument("--ckplug-method-name", type=str, default="ckplug")
    p.add_argument("--context-prompt-key", type=str, default="base_rag")
    p.add_argument("--prior-prompt-key", type=str, default="base_no_rag")
    p.add_argument("--prior-source", choices=["auto", "model_prior", "dataset_orig"], default="model_prior")
    p.add_argument(
        "--components",
        type=str,
        required=True,
        help=(
            "Comma-separated components, or semicolon-separated named groups, e.g. "
            "'l9=L9.attn;l31=L31.attn;l31_l9=L31.attn,L9.attn'."
        ),
    )
    p.add_argument("--controls", type=str, default="force_context,force_prior")
    p.add_argument("--random-baselines", type=str, default="random_type_matched")
    p.add_argument("--random-trials", type=int, default=1)
    p.add_argument("--alpha-sweep", type=str, default="0,0.125,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--vector-split", type=str, default="train")
    p.add_argument("--vector-start", type=int, default=0)
    p.add_argument("--vector-max-rows", type=int, default=140)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=200)
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


def _parse_components(raw: str) -> tuple[ComponentSpec, ...]:
    components = tuple(parse_component_id(item) for item in _parse_csv(raw))
    if not components:
        raise ValueError("--components is empty")
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
        if not name:
            raise ValueError(f"Empty component group name: {part!r}")
        if name in groups:
            raise ValueError(f"Duplicate component group name: {name!r}")
        groups[name] = _parse_components(values)
    if not groups:
        raise ValueError("--components is empty")
    return groups


def _parse_alphas(raw: str) -> list[float]:
    alphas: list[float] = []
    seen: set[float] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        alpha = float(item)
        if alpha not in seen:
            seen.add(alpha)
            alphas.append(alpha)
    if not alphas:
        raise ValueError("--alpha-sweep is empty")
    return alphas


def _parse_stop_strings(raw: str) -> list[str]:
    return [
        bytes(item.strip(), "utf-8").decode("unicode_escape")
        for item in raw.split(",")
        if item.strip()
    ]


def _split_for_index(row_index: int, val_mod: int) -> str:
    if val_mod <= 0:
        raise ValueError("--val-mod must be positive")
    return "val" if row_index % val_mod == 0 else "train"


def _sample_ids_from_pairs(
    path: Path,
    *,
    split: str,
    start: int,
    max_rows: int,
) -> list[str]:
    pairs = [pair for pair in load_actuator_pairs(path) if pair.split == split]
    pairs = pairs[start:]
    if max_rows > 0:
        pairs = pairs[:max_rows]
    return [pair.sample_id for pair in pairs]


def _load_prompt_delta_rows(
    path: Path,
    *,
    context_prompt_key: str,
    prior_prompt_key: str,
    split: str,
    start: int,
    max_rows: int,
    val_mod: int,
    ordered_sample_ids: list[str] | None = None,
) -> list[PromptDeltaRow]:
    raw_rows = load_jsonl(path)
    by_id: dict[str, PromptDeltaRow] = {}
    sequential: list[PromptDeltaRow] = []
    for row_index, row in enumerate(raw_rows):
        row_split = _split_for_index(row_index, val_mod)
        if row_split != split:
            continue
        context_prompt, _context_source = prompt_from_row(row, context_prompt_key)
        prior_prompt, _prior_source = prompt_from_row(row, prior_prompt_key)
        if not context_prompt or not prior_prompt:
            continue
        prompt_row = PromptDeltaRow(
            sample_id=sample_id_from_row(row, row_index),
            split=row_split,
            context_prompt=context_prompt,
            prior_prompt=prior_prompt,
        )
        by_id[prompt_row.sample_id] = prompt_row
        sequential.append(prompt_row)

    if ordered_sample_ids is not None:
        selected = [by_id[sample_id] for sample_id in ordered_sample_ids if sample_id in by_id]
    else:
        selected = sequential[start:]
        if max_rows > 0:
            selected = selected[:max_rows]
    if not selected:
        raise ValueError(f"No prompt-delta rows loaded from {path} split={split!r}")
    return selected


def _component_tuples(components: Iterable[ComponentSpec]) -> list[tuple[int, str]]:
    return [(int(component.layer_idx), str(component.component_type)) for component in components]


def _component_key(component: ComponentSpec) -> tuple[int, str]:
    return int(component.layer_idx), str(component.component_type)


def _sample_random_components(
    *,
    universe: tuple[ComponentSpec, ...],
    originals: tuple[ComponentSpec, ...],
    exclude: set[str],
    rng: random.Random,
    random_kind: str,
) -> tuple[ComponentSpec, ...]:
    selected: list[ComponentSpec] = []
    local_exclude = set(exclude)
    if random_kind == "random_any":
        candidates = [component for component in universe if component.component_id not in local_exclude]
        if len(candidates) < len(originals):
            raise ValueError("Not enough components for random_any baseline.")
        return tuple(rng.sample(candidates, len(originals)))
    if random_kind != "random_type_matched":
        raise ValueError(f"Unsupported random baseline: {random_kind}")

    for component_type in sorted({component.component_type for component in originals}):
        count = sum(1 for component in originals if component.component_type == component_type)
        candidates = [
            component
            for component in universe
            if component.component_id not in local_exclude and component.component_type == component_type
        ]
        if len(candidates) < count:
            raise ValueError(f"Not enough {component_type} components for random_type_matched baseline.")
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
) -> list[PromptDeltaControl]:
    out: list[PromptDeltaControl] = []
    for group_name, components in component_groups.items():
        for control in controls:
            if control == "force_context":
                out.append(
                    PromptDeltaControl(
                        name=f"{group_name}_force_context",
                        baseline_kind="cecm_prompt_delta",
                        target_direction="context",
                        component_group=group_name,
                        components=components,
                        sign=1.0,
                    )
                )
            elif control == "force_prior":
                out.append(
                    PromptDeltaControl(
                        name=f"{group_name}_force_prior",
                        baseline_kind="cecm_prompt_delta",
                        target_direction="prior",
                        component_group=group_name,
                        components=components,
                        sign=-1.0,
                    )
                )
            else:
                raise ValueError(f"Unsupported control: {control}")

    all_original_ids = {
        component.component_id
        for components in component_groups.values()
        for component in components
    }
    rng = random.Random(seed)
    for group_name, components in component_groups.items():
        exclude = set(all_original_ids)
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
                    if control == "force_context":
                        sign = 1.0
                        target = "context"
                    elif control == "force_prior":
                        sign = -1.0
                        target = "prior"
                    else:
                        continue
                    out.append(
                        PromptDeltaControl(
                            name=f"{random_kind}_{group_name}_{control}_s{trial_idx}",
                            baseline_kind=random_kind,
                            target_direction=target,
                            component_group=group_name,
                            components=sampled,
                            sign=sign,
                        )
                    )
    return out


def _all_unique_components(controls: Iterable[PromptDeltaControl]) -> tuple[ComponentSpec, ...]:
    out: list[ComponentSpec] = []
    seen: set[str] = set()
    for control in controls:
        for component in control.components:
            if component.component_id not in seen:
                seen.add(component.component_id)
                out.append(component)
    return tuple(out)


def _cosine(torch_module: Any, a: Any, b: Any) -> float:
    denom = float(a.float().norm().item() * b.float().norm().item())
    if denom <= 0:
        return 0.0
    return float(torch_module.dot(a.float(), b.float()).item() / denom)


def _learn_prompt_delta_vectors(
    *,
    backend: TransformersABBackend,
    rows: list[PromptDeltaRow],
    components: tuple[ComponentSpec, ...],
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    torch = backend._torch
    component_specs = _component_tuples(components)
    by_component: dict[str, list[Any]] = {component.component_id: [] for component in components}
    key_to_component = {_component_key(component): component for component in components}

    for row in tqdm(rows, desc="learn prompt-delta vectors"):
        context_acts = backend.capture_component_last_token(row.context_prompt, component_specs)
        prior_acts = backend.capture_component_last_token(row.prior_prompt, component_specs)
        for key, component in key_to_component.items():
            delta = (context_acts[key] - prior_acts[key]).detach().float().cpu()
            by_component[component.component_id].append(delta)

    vectors: dict[str, Any] = {}
    summary: list[dict[str, object]] = []
    for component in components:
        deltas = by_component[component.component_id]
        if not deltas:
            raise ValueError(f"No prompt deltas for {component.component_id}")
        stacked = torch.stack(deltas, dim=0)
        vector = stacked.mean(dim=0)
        vectors[component.component_id] = vector
        cosines = [_cosine(torch, delta, vector) for delta in deltas]
        summary.append(
            {
                "component_id": component.component_id,
                "layer_idx": component.layer_idx,
                "component_type": component.component_type,
                "n": len(deltas),
                "mean_delta_l2": float(stacked.norm(dim=1).mean().item()),
                "vector_l2": float(vector.norm().item()),
                "mean_cosine_to_mean": mean(cosines),
                "min_cosine_to_mean": min(cosines),
                "max_cosine_to_mean": max(cosines),
            }
        )
    return vectors, summary


def _additions(
    *,
    control: PromptDeltaControl,
    vectors: dict[str, Any],
    alpha: float,
    apply_mode: str,
) -> list[dict[str, Any]]:
    if alpha == 0:
        return []
    return [
        {
            "component_id": component.component_id,
            "layer_idx": int(component.layer_idx),
            "component_type": component.component_type,
            "direction": float(control.sign) * vectors[component.component_id],
            "alpha": float(alpha),
            "apply_mode": apply_mode,
        }
        for component in control.components
    ]


def _component_ids(components: Iterable[ComponentSpec]) -> str:
    return ",".join(component.component_id for component in components)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    component_groups = _parse_component_groups(args.components)
    controls = _parse_csv(args.controls)
    random_baselines = _parse_csv(args.random_baselines)
    alphas = _parse_alphas(args.alpha_sweep)
    stop_strings = _parse_stop_strings(args.stop_strings)

    print(f"[cecm-prompt-delta] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )

    universe = component_universe(backend.num_layers)
    control_specs = _build_controls(
        component_groups=component_groups,
        controls=controls,
        random_baselines=random_baselines,
        random_trials=args.random_trials,
        universe=universe,
        seed=args.seed,
    )
    vector_components = _all_unique_components(control_specs)

    ordered_sample_ids = None
    if args.vector_pairs_csv is not None:
        ordered_sample_ids = _sample_ids_from_pairs(
            args.vector_pairs_csv,
            split=args.vector_split,
            start=args.vector_start,
            max_rows=args.vector_max_rows,
        )
    vector_rows = _load_prompt_delta_rows(
        args.vector_open_rows,
        context_prompt_key=args.context_prompt_key,
        prior_prompt_key=args.prior_prompt_key,
        split=args.vector_split,
        start=args.vector_start,
        max_rows=args.vector_max_rows,
        val_mod=args.val_mod,
        ordered_sample_ids=ordered_sample_ids,
    )
    eval_rows = load_open_eval_rows(
        args.eval_open_rows,
        prompt_key=args.context_prompt_key,
        prior_source=args.prior_source,
        split=args.split,
        start=args.start,
        max_rows=args.max_rows,
        val_mod=args.val_mod,
    )
    if not eval_rows:
        raise SystemExit(f"No eval rows loaded from {args.eval_open_rows}")

    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "vector_mode": "prompt_delta",
            "vector_definition": (
                f"d_c = mean_x z_c({args.context_prompt_key}, x) - "
                f"z_c({args.prior_prompt_key}, x)"
            ),
            "actuation_rule": "z_c <- z_c + alpha * d_c for force_context; z_c <- z_c - alpha * d_c for force_prior",
            "vector_open_rows": str(args.vector_open_rows),
            "eval_open_rows": str(args.eval_open_rows),
            "vector_pairs_csv": str(args.vector_pairs_csv) if args.vector_pairs_csv else "",
            "context_prompt_key": args.context_prompt_key,
            "prior_prompt_key": args.prior_prompt_key,
            "prior_source": args.prior_source,
            "component_groups": {
                name: [component.component_id for component in components]
                for name, components in component_groups.items()
            },
            "vector_components": [component.component_id for component in vector_components],
            "controls": controls,
            "random_baselines": random_baselines,
            "random_trials": args.random_trials,
            "alpha_sweep": alphas,
            "vector_split": args.vector_split,
            "vector_rows": len(vector_rows),
            "eval_split": args.split,
            "eval_rows": len(eval_rows),
            "generation_apply_mode": args.generation_apply_mode,
            "ckplug_generations": str(args.ckplug_generations) if args.ckplug_generations else "",
        },
    )
    dump_csv(
        args.out_dir / "control_plan.csv",
        [
            {
                "control_name": control.name,
                "baseline_kind": control.baseline_kind,
                "target_direction": control.target_direction,
                "component_group": control.component_group,
                "sign": control.sign,
                "components": _component_ids(control.components),
            }
            for control in control_specs
        ],
    )
    dump_csv(
        args.out_dir / "vector_samples.csv",
        [{"sample_id": row.sample_id, "split": row.split} for row in vector_rows],
    )
    dump_csv(
        args.out_dir / "generation_eval_samples.csv",
        [
            {
                "sample_id": row.sample_id,
                "split": row.split,
                "prompt_key": args.context_prompt_key,
                "prior_source": args.prior_source,
            }
            for row in eval_rows
        ],
    )

    vectors, vector_summary = _learn_prompt_delta_vectors(
        backend=backend,
        rows=vector_rows,
        components=vector_components,
    )
    dump_csv(args.out_dir / "vector_summary.csv", vector_summary)
    backend._torch.save(
        {
            "vector_mode": "prompt_delta",
            "components": [
                {
                    "component_id": component.component_id,
                    "layer_idx": component.layer_idx,
                    "component_type": component.component_type,
                }
                for component in vector_components
            ],
            "vectors": vectors,
        },
        args.out_dir / "prompt_delta_vectors.pt",
    )

    generation_rows: list[dict[str, object]] = []
    for control in control_specs:
        for alpha in alphas:
            additions = _additions(
                control=control,
                vectors=vectors,
                alpha=alpha,
                apply_mode=args.generation_apply_mode,
            )
            for row in tqdm(eval_rows, desc=f"generate {control.name} a={alpha:g}"):
                if additions:
                    prediction = backend.generate_with_component_last_token_add_many(
                        row.prompt,
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
                        row.prompt,
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
                        "target_direction": control.target_direction,
                        "component_group": control.component_group,
                        "alpha": alpha,
                        "prediction": prediction,
                        "prompt_key": args.context_prompt_key,
                        "prior_source": args.prior_source,
                        "context_answers_json": list(row.context_answers),
                        "prior_answers_json": list(row.prior_answers),
                        "components": _component_ids(control.components),
                        **metrics,
                    }
                )

    if args.ckplug_generations is not None:
        external = load_external_generations(
            args.ckplug_generations,
            method_name=args.ckplug_method_name,
            env_kind="context",
        )
        row_map: dict[str, OpenEvalRow] = {row.sample_id: row for row in eval_rows}
        for external_row in external:
            row = row_map.get(str(external_row.get("sample_id", "")))
            if row is None:
                continue
            metrics = classify_source_generation(
                str(external_row.get("prediction", "")),
                context_answers=row.context_answers,
                prior_answers=row.prior_answers,
                verbose_char_threshold=args.verbose_char_threshold,
            )
            generation_rows.append(
                {
                    **external_row,
                    "split": row.split,
                    "target_direction": "external",
                    "context_answers_json": list(row.context_answers),
                    "prior_answers_json": list(row.prior_answers),
                    **metrics,
                }
            )

    dump_jsonl(args.out_dir / "generation_rows.jsonl", generation_rows)
    dump_csv(args.out_dir / "generation_summary.csv", summarize_generation_rows(generation_rows))
    dump_csv(
        args.out_dir / "run_summary.csv",
        [
            {"metric": "vector_rows", "value": len(vector_rows)},
            {"metric": "eval_rows", "value": len(eval_rows)},
            {"metric": "control_specs", "value": len(control_specs)},
            {"metric": "generation_rows", "value": len(generation_rows)},
        ],
    )
    print(f"[cecm-prompt-delta] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
