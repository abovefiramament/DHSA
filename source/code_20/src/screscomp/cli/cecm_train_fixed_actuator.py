from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.cecm.actuator import (
    PREFERENCE_LOSS_MODES,
    ENDPOINT_OBJECTIVES,
    FixedActuatorConfig,
    FixedActuatorTrainer,
    actuator_loss_description,
    actuator_objective_description,
    actuator_score_description,
    competitive_margin_description,
    filter_pairs_by_source_row_index,
    limit_rows,
    load_actuator_pairs,
    load_component_specs,
    parse_alpha_list,
    load_fixed_actuator_additions,
    load_fixed_actuator_vectors,
    load_head_actuator_additions,
    parse_weight_spec,
)
from screscomp.cecm.control import parse_control_parts
from screscomp.cecm.objective import OPTION_SELECTION_MODES, option_selection_description
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a frozen-component CECM fixed actuator from admitted preference pairs."
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--components-csv", type=Path, required=True)
    p.add_argument("--event", type=str, required=True)
    p.add_argument(
        "--endpoint-objective",
        type=str,
        default="pair_margin",
        choices=ENDPOINT_OBJECTIVES,
        help="pair_margin trains on the admitted y_plus/y_minus endpoints.",
    )
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-norm", type=float, default=1e-4)
    p.add_argument("--alpha-train", type=float, default=1.0)
    p.add_argument(
        "--preference-loss-mode",
        type=str,
        default="margin_gain",
        choices=PREFERENCE_LOSS_MODES,
        help="Preference objective for the injected actuator. Use dpo for standard reference-anchored DPO.",
    )
    p.add_argument(
        "--dpo-beta",
        type=float,
        default=1.0,
        help="Inverse-temperature beta used when --preference-loss-mode=dpo.",
    )
    p.add_argument("--state-margin-weight", type=float, default=1.0)
    p.add_argument("--gain-weight", type=float, default=0.0)
    p.add_argument("--target-margin", type=float, default=0.0)
    p.add_argument("--target-gain", type=float, default=0.0)
    p.add_argument(
        "--apply-mode",
        type=str,
        default="decision_tokens",
        help=(
            "Training intervention timing. Supports fixed-continuation names "
            "decision_tokens, boxed_decision, prompt_last, prompt, all, and generation-aligned "
            "names prefill, decode, first_decode, first_N_decode."
        ),
    )
    p.add_argument(
        "--causal-train-mask",
        action="store_true",
        help="Train/evaluate fixed actuator scores causally: each target token only sees the prompt plus previous tokens.",
    )
    p.add_argument(
        "--score-mode",
        type=str,
        default="avglogp",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
        help="Candidate answer score used in S.",
    )
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="model_max",
        choices=OPTION_SELECTION_MODES,
        help="How to choose among multiple continuation options for each endpoint.",
    )
    p.add_argument("--alpha-sweep", type=str, default="0,0.25,0.5,0.75,1.0,1.5,2.0")
    p.add_argument(
        "--background-component-actuators",
        type=str,
        default="",
        help="Semicolon specs name=path for frozen component actuators applied as the round background.",
    )
    p.add_argument(
        "--background-head-actuators",
        type=str,
        default="",
        help="Semicolon specs name=path for frozen head actuators applied as the round background.",
    )
    p.add_argument(
        "--background-control-parts",
        type=str,
        default="",
        help="Control parts applied before the trainable residual, e.g. comp:prev_mlp:0.04:prefill+head_act:prev_head:0.1:all.",
    )
    p.add_argument(
        "--init-fixed-actuator",
        type=Path,
        default=None,
        help="Optional existing fixed_actuator.pt payload used to warm-start the current residual vector.",
    )
    p.add_argument(
        "--max-aliases-per-side",
        type=int,
        default=1,
        help="Endpoint aliases kept per side while training/evaluating margins; 0 keeps all aliases.",
    )
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--max-val-rows", type=int, default=None)
    p.add_argument(
        "--min-source-row-index",
        type=int,
        default=None,
        help="Keep only pairs whose source row_index is >= this value.",
    )
    p.add_argument(
        "--max-source-row-index",
        type=int,
        default=None,
        help="Keep only pairs whose source row_index is <= this value.",
    )
    p.add_argument(
        "--question-state-weights",
        type=str,
        default="",
        help="Semicolon specs question_sample_state=weight, e.g. pure_correct=0.5;mixed=3;pure_wrong=3.",
    )
    p.add_argument(
        "--empty-cache-every",
        type=int,
        default=25,
        help="Run gc.collect() and torch.cuda.empty_cache() every N train steps; 0 disables.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _parse_semicolon_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected name=value spec, got: {part!r}")
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Empty name in spec: {part!r}")
        out[name] = value
    return out


def _resolve_payload_path(path: str, *, filename: str) -> Path:
    value = Path(path)
    if value.is_dir():
        value = value / filename
    if not value.exists():
        raise FileNotFoundError(f"Missing payload: {value}")
    return value


def _background_additions(args: argparse.Namespace) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    component_paths = {
        name: _resolve_payload_path(path, filename="fixed_actuator.pt")
        for name, path in _parse_semicolon_map(args.background_component_actuators).items()
    }
    head_paths = {
        name: _resolve_payload_path(path, filename="head_actuator.pt")
        for name, path in _parse_semicolon_map(args.background_head_actuators).items()
    }
    component_additions: list[dict[str, object]] = []
    head_additions: list[dict[str, object]] = []
    for part in parse_control_parts(args.background_control_parts):
        if part.kind == "comp":
            if part.name not in component_paths:
                raise ValueError(f"Unknown background component actuator: {part.name}")
            component_additions.extend(
                load_fixed_actuator_additions(
                    component_paths[part.name],
                    alpha=float(part.value or 0.0),
                    apply_mode=part.apply_mode or "prefill",
                )
            )
        elif part.kind == "head_act":
            if part.name not in head_paths:
                raise ValueError(f"Unknown background head actuator: {part.name}")
            head_additions.extend(
                load_head_actuator_additions(
                    head_paths[part.name],
                    alpha=float(part.value or 0.0),
                    apply_mode=part.apply_mode or "all",
                )
            )
        else:
            raise ValueError(f"Training background does not support {part.kind!r}; use comp/head_act controls")
    return tuple(component_additions), tuple(head_additions)


def main() -> None:
    args = parse_args()
    all_pairs = load_actuator_pairs(
        args.pairs_csv,
        event=args.event,
        endpoint_objective=args.endpoint_objective,
    )
    all_pairs = filter_pairs_by_source_row_index(
        all_pairs,
        min_row_index=args.min_source_row_index,
        max_row_index=args.max_source_row_index,
    )
    train_pairs = limit_rows([pair for pair in all_pairs if pair.split == args.train_split], args.max_train_rows)
    val_pairs = limit_rows([pair for pair in all_pairs if pair.split == args.val_split], args.max_val_rows)
    if not train_pairs:
        raise SystemExit(f"No train pairs found for split={args.train_split!r} event={args.event!r}")
    if not val_pairs:
        print("[cecm] no validation pairs found; using train pairs for alpha reporting", flush=True)
        val_pairs = train_pairs

    components = load_component_specs(args.components_csv)
    alphas = parse_alpha_list(args.alpha_sweep)
    background_component_additions, background_head_additions = _background_additions(args)
    question_state_weights = parse_weight_spec(args.question_state_weights)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    objective_description = actuator_objective_description(
        state_margin_weight=args.state_margin_weight,
        gain_weight=args.gain_weight,
        preference_loss_mode=args.preference_loss_mode,
    )
    loss_description = actuator_loss_description(
        state_margin_weight=args.state_margin_weight,
        gain_weight=args.gain_weight,
        norm_label="U_c",
        preference_loss_mode=args.preference_loss_mode,
        dpo_beta=args.dpo_beta,
    )
    score_description = actuator_score_description(args.score_mode)
    discovery_alignment = (
        "Component discovery supplies the selected residual-write sites. Training then "
        "optimizes a reference-anchored DPO preference objective over the same fixed "
        "y_plus/y_minus continuations while only updating the injected residual vectors."
        if args.preference_loss_mode == "dpo"
        else (
            "Component discovery supplies the selected residual-write sites. Training then "
            "optimizes the configured endpoint objective with vector injection. With "
            "state_margin_weight=0 and gain_weight>0 this is pure intervention-induced "
            "competitive-margin gain relative to the unsteered model."
        )
    )

    dump_csv(
        args.out_dir / "actuator_plan.csv",
        [
            {
                "event": args.event,
                "component_id": component.component_id,
                "layer_idx": component.layer_idx,
                "component_type": component.component_type,
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
                "endpoint_objective": args.endpoint_objective,
                "competitive_margin": competitive_margin_description(args.endpoint_objective),
                "preference_loss_mode": args.preference_loss_mode,
                "dpo_beta": args.dpo_beta,
                "option_selection_mode": args.option_selection_mode,
                "option_selection": option_selection_description(args.option_selection_mode),
                "objective": objective_description,
                "loss": loss_description,
                "score": score_description,
            }
            for component in components
        ],
    )
    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "components_csv": str(args.components_csv),
            "event": args.event,
            "endpoint_objective": args.endpoint_objective,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "lr": args.lr,
            "lambda_norm": args.lambda_norm,
            "alpha_train": args.alpha_train,
            "preference_loss_mode": args.preference_loss_mode,
            "dpo_beta": args.dpo_beta,
            "state_margin_weight": args.state_margin_weight,
            "gain_weight": args.gain_weight,
            "target_margin": args.target_margin,
            "target_gain": args.target_gain,
            "apply_mode": args.apply_mode,
            "causal_train_mask": args.causal_train_mask,
            "score_mode": args.score_mode,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "alpha_sweep": alphas,
            "background_component_actuators": args.background_component_actuators,
            "background_head_actuators": args.background_head_actuators,
            "background_control_parts": args.background_control_parts,
            "background_component_count": len(background_component_additions),
            "background_head_count": len(background_head_additions),
            "init_fixed_actuator": str(args.init_fixed_actuator) if args.init_fixed_actuator else "",
            "max_aliases_per_side": args.max_aliases_per_side,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "min_source_row_index": args.min_source_row_index,
            "max_source_row_index": args.max_source_row_index,
            "question_state_weights": dict(question_state_weights),
            "seed": args.seed,
            "objective": objective_description,
            "competitive_margin": competitive_margin_description(args.endpoint_objective),
            "causal_operator": "vector injection on selected residual-write components",
            "loss": loss_description,
            "score": score_description,
            "discovery_alignment": discovery_alignment,
            "semantics_source": (
                "The trainer consumes admitted rows and selected components. pair_margin uses the "
                "row y_plus/y_minus endpoints exactly as built. If alias lists are present, each "
                "endpoint score uses the configured option_selection_mode over aliases. When "
                "causal_train_mask=true, token scores are computed stepwise from causal prefixes only."
            ),
        },
    )

    print(
        (
            f"[cecm] loading model={args.model} components={len(components)} "
            f"train={len(train_pairs)} val={len(val_pairs)}"
        ),
        flush=True,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    trainer = FixedActuatorTrainer(
        backend=backend,
        components=components,
        config=FixedActuatorConfig(
            event=args.event,
            endpoint_objective=args.endpoint_objective,
            apply_mode=args.apply_mode,
            causal_train_mask=args.causal_train_mask,
            score_mode=args.score_mode,
            option_selection_mode=args.option_selection_mode,
            alpha_train=args.alpha_train,
            preference_loss_mode=args.preference_loss_mode,
            dpo_beta=args.dpo_beta,
            state_margin_weight=args.state_margin_weight,
            gain_weight=args.gain_weight,
            target_margin=args.target_margin,
            target_gain=args.target_gain,
            lambda_norm=args.lambda_norm,
            lr=args.lr,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            seed=args.seed,
            empty_cache_every=args.empty_cache_every,
            max_aliases_per_side=args.max_aliases_per_side,
            question_state_weights=question_state_weights,
            background_component_additions=background_component_additions,
            background_head_additions=background_head_additions,
        ),
    )
    init_loaded = 0
    if args.init_fixed_actuator is not None:
        init_loaded = trainer.load_initial_vectors(load_fixed_actuator_vectors(args.init_fixed_actuator))
        print(
            f"[cecm] warm-start fixed actuator={args.init_fixed_actuator} matched_components={init_loaded}/{len(components)}",
            flush=True,
        )
    history, baseline_margins = trainer.train(train_pairs, val_pairs=val_pairs, out_dir=args.out_dir)
    dump_json(
        args.out_dir / "warm_start.json",
        {
            "init_fixed_actuator": str(args.init_fixed_actuator) if args.init_fixed_actuator else "",
            "matched_components": init_loaded,
            "total_components": len(components),
        },
    )
    dump_csv(args.out_dir / "train_history.csv", history)
    dump_csv(args.out_dir / "vector_summary.csv", trainer.vector_summary_rows())
    trainer.save_payload(args.out_dir / "fixed_actuator.pt")

    alpha_rows = []
    for split_name, split_pairs in ((args.train_split, train_pairs), (args.val_split, val_pairs)):
        for alpha in alphas:
            alpha_rows.append(
                trainer.evaluate_alpha(
                    split_pairs,
                    baseline_margins=baseline_margins,
                    alpha=alpha,
                    split=split_name,
                )
            )
    dump_csv(args.out_dir / "alpha_summary.csv", alpha_rows)
    print(f"[cecm] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
