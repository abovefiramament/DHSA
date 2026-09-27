from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.cecm.actuator import ActuatorPair, limit_rows, load_actuator_pairs
from screscomp.cli.cecm_train_actuator_gates import (
    ActuatorGateTrainer,
    ComponentAction,
    HeadAction,
    _load_component_actions,
    _load_head_actions,
    _parse_semicolon_map,
)
from screscomp.cecm.objective import OPTION_SELECTION_MODES, option_selection_description
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Jointly unfreeze CAST actuator vectors and bounded group gates. "
            "The base model remains frozen; optimization keeps the same state-margin gain objective."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="source_context_over_prior")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--max-train-rows", type=int, default=240)
    p.add_argument("--max-val-rows", type=int, default=60)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--gate-lr", type=float, default=0.03)
    p.add_argument("--vector-lr", type=float, default=0.005)
    p.add_argument(
        "--state-margin-weight",
        type=float,
        default=1.0,
        help="Weight for completing the target state: softplus(target_margin - C(M_U;x)).",
    )
    p.add_argument(
        "--gain-weight",
        type=float,
        default=0.25,
        help="Auxiliary weight for improving over the frozen model: softplus(target_gain - gain).",
    )
    p.add_argument(
        "--target-margin",
        type=float,
        default=0.0,
        help="Required competitive margin for completed state transition.",
    )
    p.add_argument(
        "--target-gain",
        type=float,
        default=0.0,
        help="Required gain over the frozen model.",
    )
    p.add_argument("--lambda-gate", type=float, default=1e-3)
    p.add_argument("--lambda-vector", type=float, default=1e-4)
    p.add_argument("--lambda-drift", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--score-mode",
        type=str,
        default="answer_rest_margin",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
    )
    p.add_argument("--max-aliases-per-side", type=int, default=1)
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="model_max",
        choices=OPTION_SELECTION_MODES,
        help="How to choose among multiple continuation options for each endpoint.",
    )
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument(
        "--component-actuators",
        type=str,
        default="",
        help="Semicolon specs group=path/to/fixed_actuator.pt, e.g. prior_mlp=.../fixed_actuator.pt",
    )
    p.add_argument(
        "--head-actuators",
        type=str,
        default="",
        help="Semicolon specs group=path/to/head_actuator.pt, e.g. suppress=...;boost=...",
    )
    p.add_argument(
        "--component-apply-modes",
        type=str,
        default="prior_mlp=prompt_last",
        help="Semicolon specs group=mode. Use prompt_last to approximate generation-time prefill.",
    )
    p.add_argument("--head-apply-modes", type=str, default="suppress=all;boost=all")
    p.add_argument("--component-gate-max", type=float, default=0.2)
    p.add_argument("--head-gate-max", type=float, default=1.0)
    p.add_argument(
        "--init-gates",
        type=str,
        default="prior_mlp=0.055475719;suppress=0.48858309;boost=0.2481145",
    )
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


def _param_key(*parts: str) -> str:
    return "__".join(part.replace(".", "_").replace("-", "_").replace("/", "_") for part in parts)


class JointUnfreezeActuatorTrainer(ActuatorGateTrainer):
    def __init__(
        self,
        *,
        backend: TransformersABBackend,
        component_actions: list[ComponentAction],
        head_actions: list[HeadAction],
        init_gates: dict[str, float],
        component_gate_max: float,
        head_gate_max: float,
        score_mode: str,
        max_aliases_per_side: int,
        option_selection_mode: str,
        lambda_gate: float,
        lambda_vector: float,
        lambda_drift: float,
        state_margin_weight: float,
        gain_weight: float,
        target_margin: float,
        target_gain: float,
        gate_lr: float,
        vector_lr: float,
    ) -> None:
        super().__init__(
            backend=backend,
            component_actions=component_actions,
            head_actions=head_actions,
            init_gates=init_gates,
            component_gate_max=component_gate_max,
            head_gate_max=head_gate_max,
            score_mode=score_mode,
            max_aliases_per_side=max_aliases_per_side,
            lambda_gate=lambda_gate,
            lr=gate_lr,
            option_selection_mode=option_selection_mode,
        )
        self.lambda_vector = float(lambda_vector)
        self.lambda_drift = float(lambda_drift)
        self.state_margin_weight = float(state_margin_weight)
        self.gain_weight = float(gain_weight)
        self.target_margin = float(target_margin)
        self.target_gain = float(target_gain)
        self.gate_lr = float(gate_lr)
        self.vector_lr = float(vector_lr)

        self.component_vectors = self.torch.nn.ParameterDict()
        self.head_vectors = self.torch.nn.ParameterDict()
        self.initial_component_vectors: dict[str, Any] = {}
        self.initial_head_vectors: dict[str, Any] = {}

        trainable_component_actions: list[ComponentAction] = []
        for action in component_actions:
            key = _param_key("component", action.group, action.component_id)
            initial = action.vector.detach().to(device=self.device, dtype=self.torch.float32).clone()
            self.component_vectors[key] = self.torch.nn.Parameter(initial.clone())
            self.initial_component_vectors[key] = initial
            trainable_component_actions.append(
                ComponentAction(
                    component_id=action.component_id,
                    layer_idx=action.layer_idx,
                    component_type=action.component_type,
                    vector=self.component_vectors[key],
                    group=action.group,
                    apply_mode=action.apply_mode,
                )
            )
        self.component_actions = trainable_component_actions

        trainable_head_actions: list[HeadAction] = []
        for action in head_actions:
            key = _param_key("head", action.group, action.head_id)
            initial = action.vector.detach().to(device=self.device, dtype=self.torch.float32).clone()
            self.head_vectors[key] = self.torch.nn.Parameter(initial.clone())
            self.initial_head_vectors[key] = initial
            trainable_head_actions.append(
                HeadAction(
                    head_id=action.head_id,
                    layer_idx=action.layer_idx,
                    head_idx=action.head_idx,
                    vector=self.head_vectors[key],
                    group=action.group,
                    apply_mode=action.apply_mode,
                )
            )
        self.head_actions = trainable_head_actions

    def trainable_vector_parameters(self) -> list[Any]:
        return [*self.component_vectors.parameters(), *self.head_vectors.parameters()]

    def vector_norm_penalty(self):
        values = [parameter.pow(2).mean() for parameter in self.trainable_vector_parameters()]
        if not values:
            return self.torch.tensor(0.0, device=self.device)
        return self.torch.stack(values).mean()

    def vector_drift_penalty(self):
        values = []
        for key, parameter in self.component_vectors.items():
            values.append((parameter - self.initial_component_vectors[key]).pow(2).mean())
        for key, parameter in self.head_vectors.items():
            values.append((parameter - self.initial_head_vectors[key]).pow(2).mean())
        if not values:
            return self.torch.tensor(0.0, device=self.device)
        return self.torch.stack(values).mean()

    def _regularization_penalty(self):
        return (
            self.lambda_gate * self.gate_penalty()
            + self.lambda_vector * self.vector_norm_penalty()
            + self.lambda_drift * self.vector_drift_penalty()
        )

    def vector_summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with self.torch.no_grad():
            for action in self.component_actions:
                key = _param_key("component", action.group, action.component_id)
                vector = self.component_vectors[key].float()
                initial = self.initial_component_vectors[key].float()
                rows.append(
                    {
                        "kind": "component",
                        "group": action.group,
                        "id": action.component_id,
                        "layer_idx": action.layer_idx,
                        "component_type": action.component_type,
                        "l2_norm": float(vector.norm().detach().cpu().item()),
                        "mean_abs": float(vector.abs().mean().detach().cpu().item()),
                        "max_abs": float(vector.abs().max().detach().cpu().item()),
                        "drift_l2": float((vector - initial).norm().detach().cpu().item()),
                    }
                )
            for action in self.head_actions:
                key = _param_key("head", action.group, action.head_id)
                vector = self.head_vectors[key].float()
                initial = self.initial_head_vectors[key].float()
                rows.append(
                    {
                        "kind": "head",
                        "group": action.group,
                        "id": action.head_id,
                        "layer_idx": action.layer_idx,
                        "head_idx": action.head_idx,
                        "head_dim": self.head_dims[action.layer_idx],
                        "l2_norm": float(vector.norm().detach().cpu().item()),
                        "mean_abs": float(vector.abs().mean().detach().cpu().item()),
                        "max_abs": float(vector.abs().max().detach().cpu().item()),
                        "drift_l2": float((vector - initial).norm().detach().cpu().item()),
                    }
                )
        return rows

    def save_payloads(self, out_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
        component_paths: dict[str, Path] = {}
        head_paths: dict[str, Path] = {}
        torch = self.torch

        for group in sorted({action.group for action in self.component_actions}):
            actions = [action for action in self.component_actions if action.group == group]
            payload_dir = out_dir / "component_actuators" / group
            payload_dir.mkdir(parents=True, exist_ok=True)
            path = payload_dir / "fixed_actuator.pt"
            payload = {
                "kind": "cast_joint_unfreeze_component_actuator",
                "event": "source_context_over_prior",
                "score_mode": self.score_mode,
                "objective": "jointly unfreezed actuator vectors under state-margin gain",
                "loss": (
                    "state_margin_weight*softplus(target_margin-C(M_U;x)) "
                    "+ gain_weight*softplus(target_gain-(C(M_U;x)-C(M_full;x))) "
                    "+ lambda_gate*gate_penalty "
                    "+ lambda_vector*mean(||U||^2) + lambda_drift*mean(||U-U0||^2)"
                ),
                "components": [
                    {
                        "component_id": action.component_id,
                        "layer_idx": action.layer_idx,
                        "component_type": action.component_type,
                    }
                    for action in actions
                ],
                "vectors": {
                    action.component_id: self.component_vectors[
                        _param_key("component", action.group, action.component_id)
                    ]
                    .detach()
                    .cpu()
                    for action in actions
                },
            }
            torch.save(payload, path)
            component_paths[group] = path

        for group in sorted({action.group for action in self.head_actions}):
            actions = [action for action in self.head_actions if action.group == group]
            payload_dir = out_dir / "head_actuators" / group
            payload_dir.mkdir(parents=True, exist_ok=True)
            path = payload_dir / "head_actuator.pt"
            payload = {
                "kind": "cast_joint_unfreeze_head_actuator",
                "event": "source_context_over_prior",
                "score_mode": self.score_mode,
                "objective": "jointly unfreezed actuator vectors under state-margin gain",
                "loss": (
                    "state_margin_weight*softplus(target_margin-C(M_U;x)) "
                    "+ gain_weight*softplus(target_gain-(C(M_U;x)-C(M_full;x))) "
                    "+ lambda_gate*gate_penalty "
                    "+ lambda_vector*mean(||U||^2) + lambda_drift*mean(||U-U0||^2)"
                ),
                "heads": [
                    {
                        "head_id": action.head_id,
                        "layer_idx": action.layer_idx,
                        "head_idx": action.head_idx,
                        "head_dim": self.head_dims[action.layer_idx],
                        "num_heads": self.num_heads_by_layer[action.layer_idx],
                    }
                    for action in actions
                ],
                "vectors": {
                    action.head_id: self.head_vectors[_param_key("head", action.group, action.head_id)]
                    .detach()
                    .cpu()
                    for action in actions
                },
            }
            torch.save(payload, path)
            head_paths[group] = path

        return component_paths, head_paths

    def train(
        self,
        train_pairs: list[ActuatorPair],
        *,
        val_pairs: list[ActuatorPair],
        epochs: int,
        seed: int,
        empty_cache_every: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        rng = random.Random(seed)
        optimizer = self.torch.optim.AdamW(
            [
                {"params": list(self.gate_logits.parameters()), "lr": self.gate_lr},
                {"params": self.trainable_vector_parameters(), "lr": self.vector_lr},
            ]
        )
        print(f"[cecm-joint-unfreeze] caching train baselines: n={len(train_pairs)}", flush=True)
        baseline_train = self.cache_baseline_margins(train_pairs)
        print(f"[cecm-joint-unfreeze] caching val baselines: n={len(val_pairs)}", flush=True)
        baseline_val = self.cache_baseline_margins(val_pairs)

        history: list[dict[str, object]] = []
        for epoch in range(1, epochs + 1):
            shuffled = list(train_pairs)
            rng.shuffle(shuffled)
            losses: list[float] = []
            completion_losses: list[float] = []
            gain_losses: list[float] = []
            margins: list[float] = []
            gains: list[float] = []
            for step, pair in enumerate(shuffled, start=1):
                optimizer.zero_grad(set_to_none=True)
                steered_margin = self.margin(pair, use_gates=True)
                base_margin = self.torch.tensor(baseline_train[pair.sample_id], device=self.device)
                gain = steered_margin - base_margin
                completion_loss = self.torch.nn.functional.softplus(
                    self.torch.tensor(self.target_margin, device=self.device) - steered_margin
                )
                gain_loss = self.torch.nn.functional.softplus(
                    self.torch.tensor(self.target_gain, device=self.device) - gain
                )
                loss = (
                    self.state_margin_weight * completion_loss
                    + self.gain_weight * gain_loss
                    + self._regularization_penalty()
                )
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                completion_losses.append(float(completion_loss.detach().cpu().item()))
                gain_losses.append(float(gain_loss.detach().cpu().item()))
                margins.append(float(steered_margin.detach().cpu().item()))
                gains.append(float(gain.detach().cpu().item()))
                if step == 1 or step % 25 == 0 or step == len(shuffled):
                    gates = " ".join(f"{key}={value:.4f}" for key, value in sorted(self.gate_values().items()))
                    print(
                        (
                            f"[cecm-joint-unfreeze] epoch={epoch}/{epochs} step={step}/{len(shuffled)} "
                            f"loss={losses[-1]:.6f} margin={margins[-1]:.6f} gain={gains[-1]:.6f} {gates}"
                        ),
                        flush=True,
                    )
                del loss, completion_loss, gain_loss, gain, base_margin, steered_margin
                if empty_cache_every > 0 and step % empty_cache_every == 0:
                    gc.collect()
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
            history.append(
                {
                    "epoch": epoch,
                    "train_pairs": len(shuffled),
                    "mean_loss": mean(losses) if losses else math.nan,
                    "mean_completion_loss": mean(completion_losses) if completion_losses else math.nan,
                    "mean_gain_loss": mean(gain_losses) if gain_losses else math.nan,
                    "mean_margin": mean(margins) if margins else math.nan,
                    "mean_margin_gain": mean(gains) if gains else math.nan,
                    "state_complete_rate": mean(float(value > self.target_margin) for value in margins)
                    if margins
                    else math.nan,
                    "gain_complete_rate": mean(float(value > self.target_gain) for value in gains)
                    if gains
                    else math.nan,
                    "gate_penalty": float(self.gate_penalty().detach().cpu().item()),
                    "vector_norm_penalty": float(self.vector_norm_penalty().detach().cpu().item()),
                    "vector_drift_penalty": float(self.vector_drift_penalty().detach().cpu().item()),
                    **{f"gate_{group}": value for group, value in self.gate_values().items()},
                }
            )
        summaries = [
            self._summarize_split(train_pairs, baselines=baseline_train, split="train"),
            self._summarize_split(val_pairs, baselines=baseline_val, split="val"),
        ]
        return history, summaries


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = load_actuator_pairs(args.pairs_csv, event=args.event)
    train_pairs = limit_rows([pair for pair in all_pairs if pair.split == args.train_split], args.max_train_rows)
    val_pairs = limit_rows([pair for pair in all_pairs if pair.split == args.val_split], args.max_val_rows)
    if not train_pairs:
        raise SystemExit(f"No train pairs found for split={args.train_split!r} event={args.event!r}")
    if not val_pairs:
        print("[cecm-joint-unfreeze] no validation pairs found; using train pairs for reporting", flush=True)
        val_pairs = train_pairs

    component_specs = _parse_semicolon_map(args.component_actuators)
    head_specs = _parse_semicolon_map(args.head_actuators)
    component_apply_modes = _parse_semicolon_map(args.component_apply_modes)
    head_apply_modes = _parse_semicolon_map(args.head_apply_modes)
    init_gates = {key: float(value) for key, value in _parse_semicolon_map(args.init_gates).items()}
    component_actions = _load_component_actions(component_specs, apply_modes=component_apply_modes)
    head_actions = _load_head_actions(head_specs, apply_modes=head_apply_modes)

    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "event": args.event,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "epochs": args.epochs,
            "gate_lr": args.gate_lr,
            "vector_lr": args.vector_lr,
            "state_margin_weight": args.state_margin_weight,
            "gain_weight": args.gain_weight,
            "target_margin": args.target_margin,
            "target_gain": args.target_gain,
            "lambda_gate": args.lambda_gate,
            "lambda_vector": args.lambda_vector,
            "lambda_drift": args.lambda_drift,
            "score_mode": args.score_mode,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "component_actuators": component_specs,
            "head_actuators": head_specs,
            "component_apply_modes": component_apply_modes,
            "head_apply_modes": head_apply_modes,
            "component_gate_max": args.component_gate_max,
            "head_gate_max": args.head_gate_max,
            "init_gates": init_gates,
            "objective": "joint actuator-vector and gate optimization under completed state transition plus gain",
            "base_model": "frozen",
            "loss": (
                "state_margin_weight*softplus(target_margin-C(M_U;x)) "
                "+ gain_weight*softplus(target_gain-(C(M_U;x)-C(M_full;x))) "
                "+ lambda_gate*mean((gate/max)^2) "
                "+ lambda_vector*mean(||U||^2) + lambda_drift*mean(||U-U0||^2)"
            ),
        },
    )
    dump_csv(
        args.out_dir / "joint_unfreeze_plan.csv",
        [
            {
                "kind": "component",
                "group": action.group,
                "id": action.component_id,
                "layer_idx": action.layer_idx,
                "apply_mode": action.apply_mode,
            }
            for action in component_actions
        ]
        + [
            {
                "kind": "head",
                "group": action.group,
                "id": action.head_id,
                "layer_idx": action.layer_idx,
                "head_idx": action.head_idx,
                "apply_mode": action.apply_mode,
            }
            for action in head_actions
        ],
    )

    print(
        (
            f"[cecm-joint-unfreeze] loading model={args.model} components={len(component_actions)} "
            f"heads={len(head_actions)} train={len(train_pairs)} val={len(val_pairs)}"
        ),
        flush=True,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    trainer = JointUnfreezeActuatorTrainer(
        backend=backend,
        component_actions=component_actions,
        head_actions=head_actions,
        init_gates=init_gates,
        component_gate_max=args.component_gate_max,
        head_gate_max=args.head_gate_max,
        score_mode=args.score_mode,
        max_aliases_per_side=args.max_aliases_per_side,
        option_selection_mode=args.option_selection_mode,
        lambda_gate=args.lambda_gate,
        lambda_vector=args.lambda_vector,
        lambda_drift=args.lambda_drift,
        state_margin_weight=args.state_margin_weight,
        gain_weight=args.gain_weight,
        target_margin=args.target_margin,
        target_gain=args.target_gain,
        gate_lr=args.gate_lr,
        vector_lr=args.vector_lr,
    )
    history, summaries = trainer.train(
        train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        seed=args.seed,
        empty_cache_every=args.empty_cache_every,
    )
    gates = trainer.gate_values()
    component_payloads, head_payloads = trainer.save_payloads(args.out_dir)

    dump_csv(args.out_dir / "train_history.csv", history)
    dump_csv(args.out_dir / "gate_summary.csv", summaries)
    dump_csv(args.out_dir / "vector_summary.csv", trainer.vector_summary_rows())
    dump_json(args.out_dir / "gates.json", gates)
    dump_json(
        args.out_dir / "payload_paths.json",
        {
            "component_actuators": {key: str(value) for key, value in component_payloads.items()},
            "head_actuators": {key: str(value) for key, value in head_payloads.items()},
        },
    )

    controls_parts: list[str] = []
    for group in head_specs:
        if group in gates:
            mode = head_apply_modes.get(group, "all")
            controls_parts.append(f"head_act:{group}:{gates[group]:.8g}:{mode}")
    for group in component_specs:
        if group in gates:
            mode = component_apply_modes.get(group, "prompt_last")
            generation_mode = "prefill" if mode in {"prompt_last", "prefill"} else mode
            controls_parts.append(f"comp:{group}:{gates[group]:.8g}:{generation_mode}")
    cast_control = "cast_joint_unfreeze=" + "+".join(controls_parts)
    env_lines = [
        f"CAST_CONTROLS={_sh_quote(cast_control)}",
        f"CAST_COMPONENT_ACTUATORS={_sh_quote(';'.join(f'{k}={v}' for k, v in component_payloads.items()))}",
        f"CAST_HEAD_ACTUATORS={_sh_quote(';'.join(f'{k}={v}' for k, v in head_payloads.items()))}",
        *[f"CAST_GATE_{group.upper()}={value:.8g}" for group, value in sorted(gates.items())],
    ]
    (args.out_dir / "best_config.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f"[cecm-joint-unfreeze] {cast_control}", flush=True)
    print(f"[cecm-joint-unfreeze] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
