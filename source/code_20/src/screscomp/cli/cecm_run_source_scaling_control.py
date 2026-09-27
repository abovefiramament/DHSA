from __future__ import annotations

import argparse
import random
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.actuator import ActuatorPair, limit_rows, load_actuator_pairs, parse_alpha_list
from screscomp.cecm.scaling import (
    ComponentScalingRunner,
    OpenEvalRow,
    ScalingConfig,
    SourceControlSpec,
    build_source_control_specs,
    classify_source_generation,
    component_universe,
    load_open_eval_rows,
    load_external_generations,
    make_randomized_spec,
    make_wrong_side_spec,
    parse_component_list,
    summarize_endpoint_rows,
    summarize_generation_rows,
)
from screscomp.data import dump_csv, dump_json, dump_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run CECM source component-scaling controls. This is a no-training intervention runner: "
            "force_context, force_prior, and conditional_source all use admitted component groups."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--context-pairs-csv", type=Path, required=True)
    p.add_argument("--prior-pairs-csv", type=Path, required=True)
    p.add_argument(
        "--eval-open-rows",
        type=Path,
        default=None,
        help=(
            "Optional CK-style open rows JSONL. If provided, endpoint evaluation still uses pairs CSVs, "
            "but open-generation evaluation uses these ordinary prompts instead of pairs.csv prompts."
        ),
    )
    p.add_argument("--context-prompt-key", type=str, default="strong_rag")
    p.add_argument("--prior-prompt-key", type=str, default="strong_no_rag")
    p.add_argument(
        "--prior-source",
        choices=["auto", "model_prior", "dataset_orig"],
        default="auto",
        help=(
            "Which answer set should count as prior in open-generation metrics. auto prefers discovered "
            "model_prior_answer/model_prior_answers and falls back to dataset orig/prior fields."
        ),
    )
    p.add_argument(
        "--controls",
        type=str,
        default="force_context,force_prior,conditional_source",
        help="Comma-separated controls: force_context, force_prior, conditional_source.",
    )
    p.add_argument(
        "--env-kinds",
        type=str,
        default="context,prior",
        help="Comma-separated environments to run: context, prior.",
    )
    p.add_argument(
        "--shared-policies",
        type=str,
        default="off",
        help="Comma-separated shared arbitration policies: off,on. Use off,on to compare both.",
    )
    p.add_argument("--context-components", type=str, default="L27.attn,L9.attn")
    p.add_argument("--prior-components", type=str, default="L30.mlp,L28.mlp,L21.mlp")
    p.add_argument("--shared-components", type=str, default="L31.attn,L13.attn,L15.attn")
    p.add_argument("--alpha-sweep", type=str, default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument("--up-factor", type=float, default=1.0)
    p.add_argument("--down-floor", type=float, default=0.0)
    p.add_argument("--shared-alpha-scale", type=float, default=0.5)
    p.add_argument(
        "--score-mode",
        type=str,
        default="answer_rest_margin",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
    )
    p.add_argument(
        "--score-apply-mode",
        type=str,
        default="decision_tokens",
        choices=["decision_tokens", "boxed_decision", "prompt_last", "prompt", "all"],
    )
    p.add_argument(
        "--generation-apply-mode",
        type=str,
        default="prefill",
        help="When scaling acts during generation: prefill, decode, all, first_decode, first_2_decode, ...",
    )
    p.add_argument("--max-aliases-per-side", type=int, default=0)
    p.add_argument("--split", type=str, default="val")
    p.add_argument(
        "--allow-discovery-split",
        action="store_true",
        help=(
            "Allow running controls on train/discovery splits. By default this CLI refuses those splits "
            "so scaling smoke/eval does not accidentally reuse component-discovery samples."
        ),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--endpoint-only", action="store_true")
    p.add_argument("--skip-endpoint", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--verbose-char-threshold", type=int, default=48)
    p.add_argument("--include-wrong-side", action="store_true")
    p.add_argument("--random-trials", type=int, default=1)
    p.add_argument(
        "--random-baselines",
        type=str,
        default="random_any,random_type_matched",
        help="Comma-separated random baselines: random_any, random_type_matched. Empty disables random baselines.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ckplug-generations",
        type=Path,
        default=None,
        help=(
            "Optional external CK-PLUG generations CSV/JSONL with sample_id and prediction/output/text. "
            "They are rescored by the same source/form metrics and merged into generation_summary.csv."
        ),
    )
    p.add_argument("--ckplug-method-name", type=str, default="ckplug")
    p.add_argument("--ckplug-env-kind", type=str, default="context", choices=["context", "prior"])
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


def _parse_stop_strings(raw: str) -> list[str]:
    stops: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        stops.append(bytes(item, "utf-8").decode("unicode_escape"))
    return stops


def _load_env_pairs(
    *,
    context_pairs_csv: Path,
    prior_pairs_csv: Path,
    split: str,
    start: int,
    max_rows: int,
) -> dict[str, list[ActuatorPair]]:
    context_pairs = [pair for pair in load_actuator_pairs(context_pairs_csv) if pair.split == split]
    prior_pairs = [pair for pair in load_actuator_pairs(prior_pairs_csv) if pair.split == split]
    context_pairs = limit_rows(context_pairs[start:], max_rows)
    prior_pairs = limit_rows(prior_pairs[start:], max_rows)
    if not context_pairs:
        raise SystemExit(f"No context pairs for split={split!r}")
    if not prior_pairs:
        raise SystemExit(f"No prior pairs for split={split!r}")
    return {"context": context_pairs, "prior": prior_pairs}


def _answers_for_row(pair: ActuatorPair, *, env_kind: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if env_kind == "context":
        return pair.y_plus_options, pair.y_minus_options
    if env_kind == "prior":
        return pair.y_minus_options, pair.y_plus_options
    raise ValueError(f"Unsupported env_kind: {env_kind}")


def _answers_for_generation_row(
    row: ActuatorPair | OpenEvalRow,
    *,
    env_kind: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(row, OpenEvalRow):
        return row.context_answers, row.prior_answers
    return _answers_for_row(row, env_kind=env_kind)


def _component_ids(components) -> str:
    return ",".join(component.component_id for component in components)


def _plan_rows(specs: list[SourceControlSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in specs:
        rows.append(
            {
                "control_name": spec.name,
                "baseline_kind": spec.baseline_kind,
                "env_kind": spec.env_kind,
                "target_direction": spec.target_direction,
                "shared_policy": spec.shared_policy,
                "up_components": _component_ids(spec.up_components),
                "down_components": _component_ids(spec.down_components),
                "shared_components": _component_ids(spec.shared_components),
            }
        )
    return rows


def _spec_key(spec: SourceControlSpec) -> tuple[str, str, str]:
    return (spec.name, spec.env_kind, spec.shared_policy)


def _opposite_specs(specs: list[SourceControlSpec]) -> dict[tuple[str, str], SourceControlSpec]:
    by_env_policy: dict[tuple[str, str, str], SourceControlSpec] = {}
    for spec in specs:
        by_env_policy[(spec.name, spec.env_kind, spec.shared_policy)] = spec
    out: dict[tuple[str, str], SourceControlSpec] = {}
    for spec in specs:
        if spec.name.startswith("force_context"):
            opposite = by_env_policy.get((spec.name.replace("force_context", "force_prior"), spec.env_kind, spec.shared_policy))
        elif spec.name.startswith("force_prior"):
            opposite = by_env_policy.get((spec.name.replace("force_prior", "force_context"), spec.env_kind, spec.shared_policy))
        elif spec.name.startswith("conditional_source"):
            forced = "force_prior" if spec.env_kind == "context" else "force_context"
            suffix = "_shared" if spec.shared_policy == "on" else ""
            opposite = by_env_policy.get((forced + suffix, spec.env_kind, spec.shared_policy))
        else:
            opposite = None
        if opposite is not None:
            out[(spec.name, spec.env_kind)] = opposite
    return out


def main() -> None:
    args = parse_args()
    if args.split.lower() in {"train", "discovery"} and not args.allow_discovery_split:
        raise SystemExit(
            f"Refusing split={args.split!r}: use val/test/heldout for control evaluation, "
            "or pass --allow-discovery-split for an explicit debugging run."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_alpha_list(args.alpha_sweep)
    controls = _parse_csv(args.controls)
    env_kinds = _parse_csv(args.env_kinds)
    shared_policies = _parse_csv(args.shared_policies)
    random_kinds = _parse_csv(args.random_baselines)
    stop_strings = _parse_stop_strings(args.stop_strings)

    context_components = parse_component_list(args.context_components)
    prior_components = parse_component_list(args.prior_components)
    shared_components = parse_component_list(args.shared_components)
    env_pairs = _load_env_pairs(
        context_pairs_csv=args.context_pairs_csv,
        prior_pairs_csv=args.prior_pairs_csv,
        split=args.split,
        start=args.start,
        max_rows=args.max_rows,
    )
    if args.eval_open_rows is not None:
        generation_rows_by_env: dict[str, list[ActuatorPair | OpenEvalRow]] = {}
        if "context" in env_kinds:
            generation_rows_by_env["context"] = load_open_eval_rows(
                args.eval_open_rows,
                prompt_key=args.context_prompt_key,
                prior_source=args.prior_source,
                split=args.split,
                start=args.start,
                max_rows=args.max_rows,
            )
        if "prior" in env_kinds:
            generation_rows_by_env["prior"] = load_open_eval_rows(
                args.eval_open_rows,
                prompt_key=args.prior_prompt_key,
                prior_source=args.prior_source,
                split=args.split,
                start=args.start,
                max_rows=args.max_rows,
            )
        for env_kind in env_kinds:
            if not generation_rows_by_env.get(env_kind):
                raise SystemExit(
                    f"No open eval rows for env_kind={env_kind!r} split={args.split!r} "
                    f"from {args.eval_open_rows}"
                )
    else:
        generation_rows_by_env = {env_kind: list(env_pairs[env_kind]) for env_kind in env_kinds}

    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "context_pairs_csv": str(args.context_pairs_csv),
            "prior_pairs_csv": str(args.prior_pairs_csv),
            "eval_open_rows": str(args.eval_open_rows) if args.eval_open_rows else "",
            "context_prompt_key": args.context_prompt_key,
            "prior_prompt_key": args.prior_prompt_key,
            "prior_source": args.prior_source,
            "controls": controls,
            "env_kinds": env_kinds,
            "shared_policies": shared_policies,
            "context_components": [component.component_id for component in context_components],
            "prior_components": [component.component_id for component in prior_components],
            "shared_components": [component.component_id for component in shared_components],
            "alpha_sweep": alphas,
            "score_mode": args.score_mode,
            "score_apply_mode": args.score_apply_mode,
            "generation_apply_mode": args.generation_apply_mode,
            "max_rows": args.max_rows,
            "split": args.split,
            "sample_isolation": {
                "policy": "control evaluation should use non-discovery splits by default",
                "allow_discovery_split": bool(args.allow_discovery_split),
                "refused_splits_without_override": ["train", "discovery"],
            },
            "generation_prompt_source": (
                "eval_open_rows prompt_key fields" if args.eval_open_rows else "pairs_csv prompt field"
            ),
            "random_baselines": random_kinds,
            "random_trials": args.random_trials,
            "include_wrong_side": args.include_wrong_side,
            "ckplug_generations": str(args.ckplug_generations) if args.ckplug_generations else "",
            "scaling_rule": {
                "up": "z_c <- (1 + up_factor * alpha) z_c",
                "down": "z_c <- max(down_floor, 1 - alpha) z_c",
                "shared": "z_c <- (1 + shared_alpha_scale * alpha) z_c when shared_policy=on",
            },
        },
    )

    print(f"[cecm-scaling] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    config = ScalingConfig(
        score_mode=args.score_mode,
        score_apply_mode=args.score_apply_mode,
        generation_apply_mode=args.generation_apply_mode,
        max_aliases_per_side=args.max_aliases_per_side,
        up_factor=args.up_factor,
        down_floor=args.down_floor,
        shared_alpha_scale=args.shared_alpha_scale,
    )
    runner = ComponentScalingRunner(backend=backend, config=config)
    base_specs = build_source_control_specs(
        controls=controls,
        env_kinds=env_kinds,
        shared_policies=shared_policies,
        context_components=context_components,
        prior_components=prior_components,
        shared_components=shared_components,
    )

    all_specs = list(base_specs)
    if args.include_wrong_side:
        opposites = _opposite_specs(base_specs)
        for spec in base_specs:
            opposite = opposites.get((spec.name, spec.env_kind))
            if opposite is not None:
                all_specs.append(make_wrong_side_spec(spec, opposite=opposite))

    universe = component_universe(backend.num_layers)
    rng = random.Random(args.seed)
    for trial_idx in range(args.random_trials):
        for random_kind in random_kinds:
            for spec in base_specs:
                all_specs.append(
                    make_randomized_spec(
                        spec,
                        universe=universe,
                        rng=rng,
                        random_kind=random_kind,
                        trial_idx=trial_idx,
                    )
                )

    dump_csv(args.out_dir / "control_plan.csv", _plan_rows(all_specs))
    dump_csv(
        args.out_dir / "eval_samples.csv",
        [
            {
                "env_kind": env_kind,
                "sample_id": pair.sample_id,
                "split": pair.split,
                "prompt_source": "pairs_csv",
            }
            for env_kind, pairs in env_pairs.items()
            for pair in pairs
        ],
    )
    dump_csv(
        args.out_dir / "generation_eval_samples.csv",
        [
            {
                "env_kind": env_kind,
                "sample_id": row.sample_id,
                "split": row.split,
                "prompt_source": "open_rows" if isinstance(row, OpenEvalRow) else "pairs_csv",
                "prompt_key": (
                    args.context_prompt_key
                    if env_kind == "context" and isinstance(row, OpenEvalRow)
                    else (
                        args.prior_prompt_key
                        if env_kind == "prior" and isinstance(row, OpenEvalRow)
                        else ""
                    )
                ),
                "prior_source": args.prior_source if isinstance(row, OpenEvalRow) else "",
            }
            for env_kind, rows in generation_rows_by_env.items()
            for row in rows
        ],
    )

    endpoint_rows: list[dict[str, object]] = []
    base_margins: dict[tuple[str, str], float] = {}
    if not args.skip_endpoint:
        for env_kind in env_kinds:
            for pair in tqdm(env_pairs[env_kind], desc=f"base margins {env_kind}"):
                with runner.torch.no_grad():
                    base_margins[(env_kind, pair.sample_id)] = float(
                        runner.margin(pair, actions=[]).detach().cpu().item()
                    )
        for spec in all_specs:
            pairs = env_pairs[spec.env_kind]
            for alpha in alphas:
                actions = spec.actions(alpha=alpha, config=config)
                for pair in tqdm(pairs, desc=f"endpoint {spec.name} {spec.env_kind} a={alpha}"):
                    base = base_margins[(spec.env_kind, pair.sample_id)]
                    with runner.torch.no_grad():
                        margin = base if not actions else float(
                            runner.margin(pair, actions=actions).detach().cpu().item()
                        )
                    endpoint_rows.append(
                        {
                            "sample_id": pair.sample_id,
                            "split": pair.split,
                            "control_name": spec.name,
                            "baseline_kind": spec.baseline_kind,
                            "env_kind": spec.env_kind,
                            "target_direction": spec.target_direction,
                            "shared_policy": spec.shared_policy,
                            "alpha": alpha,
                            "base_margin": base,
                            "margin": margin,
                            "margin_gain": margin - base,
                            "up_components": _component_ids(spec.up_components),
                            "down_components": _component_ids(spec.down_components),
                            "shared_components": _component_ids(spec.shared_components),
                        }
                    )
        dump_csv(args.out_dir / "endpoint_rows.csv", endpoint_rows)
        dump_csv(args.out_dir / "endpoint_summary.csv", summarize_endpoint_rows(endpoint_rows))

    generation_rows: list[dict[str, object]] = []
    if not args.endpoint_only:
        for spec in all_specs:
            pairs = generation_rows_by_env[spec.env_kind]
            for alpha in alphas:
                actions = spec.actions(alpha=alpha, config=config)
                for pair in tqdm(pairs, desc=f"generate {spec.name} {spec.env_kind} a={alpha}"):
                    prediction = runner.generate(
                        pair.prompt,
                        actions=actions,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    context_answers, prior_answers = _answers_for_generation_row(pair, env_kind=spec.env_kind)
                    metrics = classify_source_generation(
                        prediction,
                        context_answers=context_answers,
                        prior_answers=prior_answers,
                        verbose_char_threshold=args.verbose_char_threshold,
                    )
                    generation_rows.append(
                        {
                            "sample_id": pair.sample_id,
                            "split": pair.split,
                            "control_name": spec.name,
                            "baseline_kind": spec.baseline_kind,
                            "env_kind": spec.env_kind,
                            "target_direction": spec.target_direction,
                            "shared_policy": spec.shared_policy,
                            "alpha": alpha,
                            "prediction": prediction,
                            "prompt_source": "open_rows" if isinstance(pair, OpenEvalRow) else "pairs_csv",
                            "prompt_key": (
                                args.context_prompt_key
                                if spec.env_kind == "context" and isinstance(pair, OpenEvalRow)
                                else (
                                    args.prior_prompt_key
                                    if spec.env_kind == "prior" and isinstance(pair, OpenEvalRow)
                                    else ""
                                )
                            ),
                            "prior_source": args.prior_source if isinstance(pair, OpenEvalRow) else "",
                            "context_answers_json": list(context_answers),
                            "prior_answers_json": list(prior_answers),
                            "up_components": _component_ids(spec.up_components),
                            "down_components": _component_ids(spec.down_components),
                            "shared_components": _component_ids(spec.shared_components),
                            **metrics,
                        }
                    )

        if args.ckplug_generations is not None:
            external = load_external_generations(
                args.ckplug_generations,
                method_name=args.ckplug_method_name,
                env_kind=args.ckplug_env_kind,
            )
            pair_map = {
                (env_kind, pair.sample_id): pair
                for env_kind, pairs in generation_rows_by_env.items()
                for pair in pairs
            }
            for row in external:
                env_kind = str(row.get("env_kind") or args.ckplug_env_kind)
                pair = pair_map.get((env_kind, str(row["sample_id"])))
                if pair is None:
                    continue
                context_answers, prior_answers = _answers_for_generation_row(pair, env_kind=env_kind)
                metrics = classify_source_generation(
                    str(row["prediction"]),
                    context_answers=context_answers,
                    prior_answers=prior_answers,
                    verbose_char_threshold=args.verbose_char_threshold,
                )
                generation_rows.append(
                    {
                        **row,
                        "split": pair.split,
                        "target_direction": "external",
                        "shared_policy": "external",
                        "context_answers_json": list(context_answers),
                        "prior_answers_json": list(prior_answers),
                        **metrics,
                    }
                )

        dump_jsonl(args.out_dir / "generation_rows.jsonl", generation_rows)
        dump_csv(args.out_dir / "generation_summary.csv", summarize_generation_rows(generation_rows))

    dump_csv(
        args.out_dir / "run_summary.csv",
        [
            {"metric": "context_pairs", "value": len(env_pairs["context"])},
            {"metric": "prior_pairs", "value": len(env_pairs["prior"])},
            {"metric": "control_specs", "value": len(all_specs)},
            {"metric": "endpoint_rows", "value": len(endpoint_rows)},
            {"metric": "generation_rows", "value": len(generation_rows)},
        ],
    )
    print(f"[cecm-scaling] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
