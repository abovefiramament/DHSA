from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.cecm.actuator import (
    ActuatorPair,
    is_constant_zero_y_plus,
    is_dynamic_y_minus,
    limit_rows,
    load_actuator_pairs,
    require_dynamic_answer_rest_margin,
)
from screscomp.cecm.objective import (
    MODEL_MAX_OPTION_SELECTION,
    OPTION_SELECTION_MODES,
    limit_options,
    option_selection_description,
    score_text_token_slice,
    select_option_score,
)
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class ComponentAction:
    component_id: str
    layer_idx: int
    component_type: str
    vector: Any
    group: str
    apply_mode: str


@dataclass(frozen=True, slots=True)
class HeadAction:
    head_id: str
    layer_idx: int
    head_idx: int
    vector: Any
    group: str
    apply_mode: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train bounded scalar gates for already-trained CAST actuators. "
            "Base model and actuator vectors stay frozen; only group alpha/gate parameters are updated."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="source_context_over_prior")
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--max-train-rows", type=int, default=240)
    p.add_argument("--max-val-rows", type=int, default=60)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lambda-gate", type=float, default=1e-3)
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
    p.add_argument(
        "--head-apply-modes",
        type=str,
        default="suppress=all;boost=all",
        help="Semicolon specs group=mode.",
    )
    p.add_argument("--component-gate-max", type=float, default=0.2)
    p.add_argument("--head-gate-max", type=float, default=1.0)
    p.add_argument(
        "--init-gates",
        type=str,
        default="prior_mlp=0.075;suppress=0.5;boost=0.5",
        help="Semicolon specs group=initial_alpha.",
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


def _parse_semicolon_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected name=value spec, got {part!r}")
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


def _parse_head_id(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"L(\d+)\.attn\.h(\d+)", raw.strip())
    if not match:
        raise ValueError(f"Unsupported head id: {raw!r}; expected L<layer>.attn.h<head>")
    return int(match.group(1)), int(match.group(2))


def _limit_aliases(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    if limit <= 0:
        return values
    return values[:limit]


def _logit_from_alpha(torch_module: Any, alpha: float, max_alpha: float, *, device: Any):
    eps = 1e-5
    if max_alpha <= 0:
        raise ValueError(f"max alpha must be positive, got {max_alpha}")
    ratio = min(max(alpha / max_alpha, eps), 1.0 - eps)
    return torch_module.nn.Parameter(
        torch_module.tensor(math.log(ratio / (1.0 - ratio)), device=device, dtype=torch_module.float32)
    )


class ActuatorGateTrainer:
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
        lambda_gate: float,
        lr: float,
        option_selection_mode: str = MODEL_MAX_OPTION_SELECTION,
    ) -> None:
        self.backend = backend
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.component_actions = component_actions
        self.head_actions = head_actions
        self.component_gate_max = float(component_gate_max)
        self.head_gate_max = float(head_gate_max)
        self.score_mode = score_mode
        self.max_aliases_per_side = max_aliases_per_side
        self.option_selection_mode = option_selection_mode
        self.lambda_gate = float(lambda_gate)
        self.lr = float(lr)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.group_kinds: dict[str, str] = {}
        for action in component_actions:
            self.group_kinds[action.group] = "component"
        for action in head_actions:
            self.group_kinds[action.group] = "head"
        if not self.group_kinds:
            raise ValueError("No actuator actions supplied.")

        self.gate_logits = self.torch.nn.ParameterDict()
        for group, kind in sorted(self.group_kinds.items()):
            max_alpha = self.component_gate_max if kind == "component" else self.head_gate_max
            init_alpha = float(init_gates.get(group, 0.5 * max_alpha))
            self.gate_logits[group] = _logit_from_alpha(
                self.torch,
                init_alpha,
                max_alpha,
                device=self.device,
            )

        self.head_dims: dict[int, int] = {}
        self.num_heads_by_layer: dict[int, int] = {}
        for action in head_actions:
            _hidden, num_heads, head_dim = self._head_geometry(action.layer_idx)
            if action.head_idx < 0 or action.head_idx >= num_heads:
                raise ValueError(f"Invalid {action.head_id}; L{action.layer_idx}.attn has {num_heads} heads")
            self.head_dims[action.layer_idx] = head_dim
            self.num_heads_by_layer[action.layer_idx] = num_heads

    def _gate_alpha(self, group: str):
        kind = self.group_kinds[group]
        max_alpha = self.component_gate_max if kind == "component" else self.head_gate_max
        return float(max_alpha) * self.torch.sigmoid(self.gate_logits[group])

    def gate_values(self) -> dict[str, float]:
        with self.torch.no_grad():
            return {group: float(self._gate_alpha(group).detach().cpu().item()) for group in self.group_kinds}

    def _component_module(self, layer_idx: int, component_type: str):
        return self.backend._component_module(layer_idx=layer_idx, component_type=component_type)

    def _attention_module(self, layer_idx: int):
        return self.backend._component_module(layer_idx=layer_idx, component_type="attn")

    def _o_proj_module(self, layer_idx: int):
        attn = self._attention_module(layer_idx)
        for attr in ("o_proj", "out_proj", "dense", "c_proj"):
            if hasattr(attn, attr):
                return getattr(attn, attr)
        raise ValueError(f"Could not locate attention output projection for L{layer_idx}.attn")

    def _head_geometry(self, layer_idx: int) -> tuple[int, int, int]:
        attn = self._attention_module(layer_idx)
        o_proj = self._o_proj_module(layer_idx)
        hidden_size = int(
            getattr(o_proj, "in_features", 0)
            or getattr(o_proj, "out_features", 0)
            or getattr(o_proj, "nf", 0)
            or getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
        )
        num_heads = int(
            getattr(attn, "num_heads", 0)
            or getattr(attn, "num_attention_heads", 0)
            or getattr(self.model.config, "num_attention_heads", 0)
        )
        head_dim = int(getattr(attn, "head_dim", 0) or (hidden_size // num_heads if num_heads else 0))
        if hidden_size <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"Could not infer attention head geometry for layer {layer_idx}")
        if num_heads * head_dim != hidden_size:
            num_heads = hidden_size // head_dim
        return hidden_size, num_heads, head_dim

    def _encode_prompt_and_continuation(self, prompt: str, continuation: str) -> tuple[dict[str, Any], int, int]:
        formatted_prompt = self.backend._format_prompt(prompt)
        prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        cont_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if int(prompt_ids.shape[-1]) <= 0:
            raise ValueError("empty prompt after tokenization")
        if int(cont_ids.shape[-1]) <= 0:
            raise ValueError(f"empty continuation after tokenization: {continuation!r}")
        input_ids = self.torch.cat([prompt_ids, cont_ids], dim=-1).to(self.device)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, int(prompt_ids.shape[-1]), int(cont_ids.shape[-1])

    @staticmethod
    def _slice_for_mode(*, mode: str, prompt_len: int, continuation_len: int, seq_len: int) -> slice:
        if mode in {"prompt_last", "prefill"}:
            start = max(min(prompt_len, seq_len) - 1, 0)
            return slice(start, start + 1)
        if mode == "prompt":
            return slice(0, min(prompt_len, seq_len))
        if mode == "decision_tokens":
            start = max(prompt_len - 1, 0)
            stop = min(prompt_len + continuation_len - 1, seq_len)
            return slice(start, max(stop, start + 1))
        if mode == "all":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported gate apply mode: {mode}")

    def _register_component_hooks(self, *, prompt_len: int, continuation_len: int) -> list[Any]:
        handles: list[Any] = []

        def make_hook(action: ComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                pos = self._slice_for_mode(
                    mode=action.apply_mode,
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    seq_len=int(hidden_new.shape[1]),
                )
                alpha = self._gate_alpha(action.group).to(device=hidden_new.device, dtype=hidden_new.dtype)
                delta = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + alpha * delta
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for action in self.component_actions:
            module = self._component_module(action.layer_idx, action.component_type)
            handles.append(module.register_forward_hook(make_hook(action)))
        return handles

    def _register_head_hooks(self, *, prompt_len: int, continuation_len: int) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted({(action.layer_idx, action.apply_mode) for action in self.head_actions})
        for layer_idx, mode in layer_modes:
            local_actions = [
                action for action in self.head_actions if action.layer_idx == layer_idx and action.apply_mode == mode
            ]
            module = self._o_proj_module(layer_idx)
            head_dim = self.head_dims[layer_idx]

            def make_hook(actions: list[HeadAction], local_head_dim: int, local_mode: str):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    pos = self._slice_for_mode(
                        mode=local_mode,
                        prompt_len=prompt_len,
                        continuation_len=continuation_len,
                        seq_len=int(hidden_new.shape[1]),
                    )
                    for action in actions:
                        start = int(action.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        alpha = self._gate_alpha(action.group).to(device=hidden_new.device, dtype=hidden_new.dtype)
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] + alpha * vector
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(local_actions, head_dim, mode)))
        return handles

    def candidate_score(self, prompt: str, continuation: str, *, use_gates: bool, score_text: str = ""):
        inputs, prompt_len, continuation_len = self._encode_prompt_and_continuation(prompt, continuation)
        handles: list[Any] = []
        if use_gates:
            handles.extend(self._register_component_hooks(prompt_len=prompt_len, continuation_len=continuation_len))
            handles.extend(self._register_head_hooks(prompt_len=prompt_len, continuation_len=continuation_len))
        try:
            logits = self.model(**inputs, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        target_ids = inputs["input_ids"][:, prompt_len : prompt_len + continuation_len]
        pred_logits = logits[:, prompt_len - 1 : prompt_len + continuation_len - 1, :].float()
        score_slice = score_text_token_slice(self.tokenizer, continuation, score_text)
        if score_slice is not None:
            target_ids = target_ids[:, score_slice]
            pred_logits = pred_logits[:, score_slice, :]
        token_logits = pred_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        if self.score_mode == "avglogp":
            log_probs = self.torch.nn.functional.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.mean()
        if self.score_mode == "top_logit_gap":
            top_logits = pred_logits.max(dim=-1).values
            return (token_logits - top_logits).mean()
        if self.score_mode == "answer_rest_margin":
            top_values, top_indices = pred_logits.topk(k=2, dim=-1)
            top1_values = top_values[..., 0]
            top2_values = top_values[..., 1]
            top1_indices = top_indices[..., 0]
            rest_max_logits = self.torch.where(top1_indices == target_ids, top2_values, top1_values)
            return (token_logits - rest_max_logits).mean()
        raise ValueError(f"Unsupported score_mode: {self.score_mode}")

    def endpoint_score(
        self,
        prompt: str,
        continuations: tuple[str, ...],
        *,
        use_gates: bool,
        score_text: str = "",
    ):
        options = limit_options(
            continuations,
            max_aliases_per_side=self.max_aliases_per_side,
        )
        if not options:
            raise ValueError("empty endpoint continuation options")
        scores = [
            self.candidate_score(prompt, option, use_gates=use_gates, score_text=score_text)
            for option in options
        ]
        return select_option_score(
            self.torch,
            scores,
            selection_mode=self.option_selection_mode,
        )

    def margin(self, pair: ActuatorPair, *, use_gates: bool):
        require_dynamic_answer_rest_margin(self.score_mode, pair=pair)
        if is_constant_zero_y_plus(pair):
            plus = self.torch.zeros((), device=self.device)
        else:
            plus = self.endpoint_score(
                pair.prompt,
                pair.y_plus_options,
                use_gates=use_gates,
                score_text=pair.y_plus_score_text,
            )
        if is_dynamic_y_minus(pair) or not pair.y_minus_options:
            return plus
        minus = self.endpoint_score(
            pair.prompt,
            pair.y_minus_options,
            use_gates=use_gates,
            score_text=pair.y_minus_score_text,
        )
        return plus - minus

    def cache_baseline_margins(self, pairs: list[ActuatorPair]) -> dict[str, float]:
        baselines: dict[str, float] = {}
        with self.torch.no_grad():
            for pair in pairs:
                baselines[pair.sample_id] = float(self.margin(pair, use_gates=False).detach().cpu().item())
        return baselines

    def gate_penalty(self):
        values = [self._gate_alpha(group) for group in self.group_kinds]
        if not values:
            return self.torch.tensor(0.0, device=self.device)
        scaled = []
        for group, value in zip(self.group_kinds, values):
            kind = self.group_kinds[group]
            max_alpha = self.component_gate_max if kind == "component" else self.head_gate_max
            scaled.append((value / float(max_alpha)).pow(2))
        return self.torch.stack(scaled).mean()

    def _summarize_split(
        self,
        pairs: list[ActuatorPair],
        *,
        baselines: dict[str, float],
        split: str,
    ) -> dict[str, object]:
        margins: list[float] = []
        base_margins: list[float] = []
        gains: list[float] = []
        with self.torch.no_grad():
            for pair in pairs:
                base = baselines[pair.sample_id]
                margin = float(self.margin(pair, use_gates=True).detach().cpu().item())
                base_margins.append(base)
                margins.append(margin)
                gains.append(margin - base)
        n = len(pairs)
        return {
            "split": split,
            "n": n,
            "base_mean_margin": mean(base_margins) if n else math.nan,
            "mean_margin": mean(margins) if n else math.nan,
            "mean_margin_gain": mean(gains) if n else math.nan,
            "base_pref_rate": mean(float(value > 0) for value in base_margins) if n else math.nan,
            "pref_rate": mean(float(value > 0) for value in margins) if n else math.nan,
            "gain_positive_rate": mean(float(value > 0) for value in gains) if n else math.nan,
            **{f"gate_{group}": value for group, value in self.gate_values().items()},
        }

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
        optimizer = self.torch.optim.AdamW(self.gate_logits.parameters(), lr=self.lr)
        print(f"[cecm-gate] caching train baselines: n={len(train_pairs)}", flush=True)
        baseline_train = self.cache_baseline_margins(train_pairs)
        print(f"[cecm-gate] caching val baselines: n={len(val_pairs)}", flush=True)
        baseline_val = self.cache_baseline_margins(val_pairs)

        history: list[dict[str, object]] = []
        for epoch in range(1, epochs + 1):
            shuffled = list(train_pairs)
            rng.shuffle(shuffled)
            losses: list[float] = []
            gains: list[float] = []
            for step, pair in enumerate(shuffled, start=1):
                optimizer.zero_grad(set_to_none=True)
                steered_margin = self.margin(pair, use_gates=True)
                base_margin = self.torch.tensor(baseline_train[pair.sample_id], device=self.device)
                gain = steered_margin - base_margin
                loss = self.torch.nn.functional.softplus(-gain) + self.lambda_gate * self.gate_penalty()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                gains.append(float(gain.detach().cpu().item()))
                if step == 1 or step % 25 == 0 or step == len(shuffled):
                    gates = " ".join(f"{key}={value:.4f}" for key, value in sorted(self.gate_values().items()))
                    print(
                        (
                            f"[cecm-gate] epoch={epoch}/{epochs} step={step}/{len(shuffled)} "
                            f"loss={losses[-1]:.6f} gain={gains[-1]:.6f} {gates}"
                        ),
                        flush=True,
                    )
                if empty_cache_every > 0 and step % empty_cache_every == 0:
                    import gc

                    gc.collect()
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
            history.append(
                {
                    "epoch": epoch,
                    "train_pairs": len(shuffled),
                    "mean_loss": mean(losses) if losses else math.nan,
                    "mean_margin_gain": mean(gains) if gains else math.nan,
                    "gate_penalty": float(self.gate_penalty().detach().cpu().item()),
                    **{f"gate_{group}": value for group, value in self.gate_values().items()},
                }
            )
        summaries = [
            self._summarize_split(train_pairs, baselines=baseline_train, split="train"),
            self._summarize_split(val_pairs, baselines=baseline_val, split="val"),
        ]
        return history, summaries


def _load_component_actions(
    specs: dict[str, str],
    *,
    apply_modes: dict[str, str],
) -> list[ComponentAction]:
    import torch

    actions: list[ComponentAction] = []
    for group, raw_path in specs.items():
        path = _resolve_payload_path(raw_path, filename="fixed_actuator.pt")
        payload = torch.load(path, map_location="cpu")
        mode = apply_modes.get(group, "prompt_last")
        for row in payload["components"]:
            component_id = str(row["component_id"])
            actions.append(
                ComponentAction(
                    component_id=component_id,
                    layer_idx=int(row["layer_idx"]),
                    component_type=str(row["component_type"]),
                    vector=payload["vectors"][component_id],
                    group=group,
                    apply_mode=mode,
                )
            )
    return actions


def _load_head_actions(
    specs: dict[str, str],
    *,
    apply_modes: dict[str, str],
) -> list[HeadAction]:
    import torch

    actions: list[HeadAction] = []
    for group, raw_path in specs.items():
        path = _resolve_payload_path(raw_path, filename="head_actuator.pt")
        payload = torch.load(path, map_location="cpu")
        mode = apply_modes.get(group, "all")
        for row in payload["heads"]:
            head_id = str(row["head_id"])
            layer_idx, head_idx = _parse_head_id(head_id)
            actions.append(
                HeadAction(
                    head_id=head_id,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    vector=payload["vectors"][head_id],
                    group=group,
                    apply_mode=mode,
                )
            )
    return actions


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = load_actuator_pairs(args.pairs_csv, event=args.event)
    train_pairs = limit_rows([pair for pair in all_pairs if pair.split == args.train_split], args.max_train_rows)
    val_pairs = limit_rows([pair for pair in all_pairs if pair.split == args.val_split], args.max_val_rows)
    if not train_pairs:
        raise SystemExit(f"No train pairs found for split={args.train_split!r} event={args.event!r}")
    if not val_pairs:
        print("[cecm-gate] no validation pairs found; using train pairs for reporting", flush=True)
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
            "lr": args.lr,
            "lambda_gate": args.lambda_gate,
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
            "objective": "bounded actuator-gate optimization under the same state-margin gain loss",
            "loss": "softplus(-(C(M_gated;x)-C(M_full;x))) + lambda_gate * mean((gate/max_gate)^2)",
        },
    )
    dump_csv(
        args.out_dir / "gate_plan.csv",
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
                "apply_mode": action.apply_mode,
            }
            for action in head_actions
        ],
    )

    print(
        (
            f"[cecm-gate] loading model={args.model} components={len(component_actions)} "
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
    trainer = ActuatorGateTrainer(
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
        lr=args.lr,
    )
    history, summaries = trainer.train(
        train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        seed=args.seed,
        empty_cache_every=args.empty_cache_every,
    )
    gates = trainer.gate_values()
    dump_csv(args.out_dir / "train_history.csv", history)
    dump_csv(args.out_dir / "gate_summary.csv", summaries)
    dump_json(args.out_dir / "gates.json", gates)

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
    cast_control = "cast_gate=" + "+".join(controls_parts)
    env_lines = [
        f"CAST_CONTROLS={cast_control}",
        *[f"CAST_GATE_{group.upper()}={value:.8g}" for group, value in sorted(gates.items())],
    ]
    (args.out_dir / "best_config.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f"[cecm-gate] {cast_control}", flush=True)
    print(f"[cecm-gate] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
