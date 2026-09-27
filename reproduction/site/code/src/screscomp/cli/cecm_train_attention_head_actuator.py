from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from screscomp.cecm.actuator import (
    ActuatorPair,
    DPO_LOSS,
    ENDPOINT_OBJECTIVES,
    PREFERENCE_LOSS_MODES,
    actuator_loss_description,
    actuator_objective_description,
    actuator_score_description,
    dpo_preference_logit,
    filter_pairs_by_source_row_index,
    is_constant_zero_y_plus,
    limit_rows,
    is_dynamic_y_minus,
    load_actuator_pairs,
    load_fixed_actuator_additions,
    load_head_actuator_additions,
    parse_alpha_list,
    parse_weight_spec,
    competitive_margin_description,
    pair_question_state_weight,
    require_dynamic_answer_rest_margin,
)
from screscomp.cecm.control import parse_control_parts
from screscomp.cecm.objective import (
    MODEL_MAX_OPTION_SELECTION,
    OPTION_SELECTION_MODES,
    limit_options,
    option_selection_description,
    score_text_start_token_index,
    score_text_token_slice,
    select_option_score,
)
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class HeadSpec:
    head_id: str
    layer_idx: int
    head_idx: int


@dataclass(frozen=True, slots=True)
class HeadActuatorConfig:
    event: str
    endpoint_objective: str = "pair_margin"
    apply_mode: str = "decision_tokens"
    causal_train_mask: bool = False
    score_mode: str = "answer_rest_margin"
    option_selection_mode: str = MODEL_MAX_OPTION_SELECTION
    alpha_train: float = 1.0
    preference_loss_mode: str = "margin_gain"
    dpo_beta: float = 1.0
    state_margin_weight: float = 1.0
    gain_weight: float = 0.0
    target_margin: float = 0.0
    target_gain: float = 0.0
    lambda_norm: float = 1e-4
    lr: float = 5e-2
    epochs: int = 2
    train_batch_size: int = 16
    seed: int = 42
    max_aliases_per_side: int = 1
    question_state_weights: tuple[tuple[str, float], ...] = ()
    empty_cache_every: int = 25
    zero_heads: tuple[HeadSpec, ...] = ()
    zero_head_apply_mode: str = "all"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train reusable vectors on selected attention heads.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="source_context_over_prior")
    p.add_argument(
        "--endpoint-objective",
        type=str,
        default="pair_margin",
        choices=ENDPOINT_OBJECTIVES,
        help="pair_margin trains on admitted y_plus/y_minus endpoints.",
    )
    p.add_argument(
        "--heads",
        type=str,
        required=True,
        help="Comma-separated heads, e.g. L9.attn.h17,L31.attn.h11",
    )
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--max-train-rows", type=int, default=80)
    p.add_argument("--max-val-rows", type=int, default=40)
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
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda-norm", type=float, default=1e-4)
    p.add_argument("--alpha-train", type=float, default=1.0)
    p.add_argument(
        "--preference-loss-mode",
        type=str,
        default="margin_gain",
        choices=PREFERENCE_LOSS_MODES,
        help="Preference objective for the head vector. Use dpo for standard reference-anchored DPO.",
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
        help="Train/evaluate head actuator scores causally: each target token only sees the prompt plus previous tokens.",
    )
    p.add_argument(
        "--score-mode",
        type=str,
        default="answer_rest_margin",
        choices=["avglogp", "top_logit_gap", "answer_rest_margin"],
    )
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="model_max",
        choices=OPTION_SELECTION_MODES,
        help="How to choose among multiple continuation options for each endpoint.",
    )
    p.add_argument("--max-aliases-per-side", type=int, default=1)
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
    p.add_argument("--alpha-sweep", type=str, default="0,0.25,0.5,1.0,1.5")
    p.add_argument(
        "--frozen-component-actuator",
        type=Path,
        default=None,
        help="Optional fixed component actuator to apply while training/evaluating head vectors.",
    )
    p.add_argument("--frozen-component-alpha", type=float, default=0.0)
    p.add_argument(
        "--frozen-component-apply-mode",
        type=str,
        default="prefill",
        help="Timing for the optional frozen component actuator; accepts the same names as --apply-mode.",
    )
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
        "--init-head-actuator",
        type=Path,
        default=None,
        help="Optional existing head_actuator.pt payload used to warm-start the current residual vector.",
    )
    p.add_argument(
        "--zero-heads",
        type=str,
        default="",
        help=(
            "Comma-separated heads whose pre-o_proj slices are set to zero while scoring/training, "
            "e.g. L16.attn.h13,L15.attn.h14. Use 'train_heads' to zero the same heads passed to --heads."
        ),
    )
    p.add_argument(
        "--zero-head-apply-mode",
        type=str,
        default="all",
        help="Timing for --zero-heads; accepts the same fixed-continuation names as --apply-mode.",
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


def _parse_heads(raw: str) -> list[HeadSpec]:
    heads: list[HeadSpec] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"L(\d+)\.attn\.h(\d+)", item)
        if not match:
            raise ValueError(f"Unsupported head id: {item!r}; expected L<layer>.attn.h<head>")
        heads.append(HeadSpec(head_id=item, layer_idx=int(match.group(1)), head_idx=int(match.group(2))))
    if not heads:
        raise ValueError("No heads provided")
    return heads


def _parse_zero_heads(raw: str, *, train_heads: list[HeadSpec]) -> tuple[HeadSpec, ...]:
    text = str(raw or "").strip()
    if not text:
        return tuple()
    if text in {"train_heads", "selected_heads", "same"}:
        return tuple(train_heads)
    return tuple(_parse_heads(text))


def _first_decode_steps(mode: str) -> int | None:
    match = re.fullmatch(r"first(?:_(\d+))?_decode", str(mode or ""))
    if not match:
        return None
    steps = int(match.group(1) or "1")
    if steps <= 0:
        raise ValueError(f"Unsupported apply_mode: {mode}")
    return steps


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


def _background_additions(args: argparse.Namespace) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    component_paths = {
        name: _resolve_payload_path(path, filename="fixed_actuator.pt")
        for name, path in _parse_semicolon_map(args.background_component_actuators).items()
    }
    head_paths = {
        name: _resolve_payload_path(path, filename="head_actuator.pt")
        for name, path in _parse_semicolon_map(args.background_head_actuators).items()
    }
    component_additions: list[dict[str, Any]] = []
    head_additions: list[dict[str, Any]] = []
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


def _limit_aliases(values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    if limit <= 0:
        return values
    return values[:limit]


def _require_dpo_conditional_probability(config: HeadActuatorConfig) -> None:
    if config.preference_loss_mode == DPO_LOSS and config.score_mode != "avglogp":
        raise ValueError(
            "preference_loss_mode='dpo' requires score_mode='avglogp' because the "
            "DPO score must be the continuation conditional log-probability; "
            f"got score_mode={config.score_mode!r}"
        )


class AttentionHeadActuatorTrainer:
    def __init__(
        self,
        *,
        backend: TransformersABBackend,
        heads: list[HeadSpec],
        config: HeadActuatorConfig,
        frozen_component_additions: Iterable[dict[str, Any]] = (),
        frozen_head_additions: Iterable[dict[str, Any]] = (),
    ) -> None:
        self.backend = backend
        self.heads = heads
        self.config = config
        _require_dpo_conditional_probability(config)
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.model.eval()
        self.frozen_component_additions = tuple(frozen_component_additions)
        self.frozen_head_additions = tuple(frozen_head_additions)
        self.zero_heads = tuple(config.zero_heads)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.head_dims: dict[int, int] = {}
        self.num_heads_by_layer: dict[int, int] = {}
        self.vectors = self.torch.nn.ParameterDict()
        for head in heads:
            _hidden, num_heads, head_dim = self._head_geometry(head.layer_idx)
            self.num_heads_by_layer[head.layer_idx] = num_heads
            self.head_dims[head.layer_idx] = head_dim
            if head.head_idx < 0 or head.head_idx >= num_heads:
                raise ValueError(f"Invalid {head.head_id}; L{head.layer_idx}.attn has {num_heads} heads")
            self.vectors[self._param_key(head.head_id)] = self.torch.nn.Parameter(
                self.torch.zeros(head_dim, device=self.device, dtype=self.torch.float32)
            )
        for head in self.zero_heads:
            _hidden, num_heads, head_dim = self._head_geometry(head.layer_idx)
            self.num_heads_by_layer[head.layer_idx] = num_heads
            self.head_dims[head.layer_idx] = head_dim
            if head.head_idx < 0 or head.head_idx >= num_heads:
                raise ValueError(f"Invalid zero head {head.head_id}; L{head.layer_idx}.attn has {num_heads} heads")

    def load_initial_vectors(self, vectors: dict[str, Any]) -> int:
        loaded = 0
        with self.torch.no_grad():
            for head in self.heads:
                source = vectors.get(head.head_id)
                if source is None:
                    continue
                target = self._vector_for(head)
                source_tensor = self.torch.as_tensor(source, dtype=target.dtype, device=self.device)
                if tuple(source_tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Warm-start shape mismatch for {head.head_id}: "
                        f"expected {tuple(target.shape)} got {tuple(source_tensor.shape)}"
                    )
                target.copy_(source_tensor)
                loaded += 1
        return loaded

    @staticmethod
    def _param_key(head_id: str) -> str:
        return head_id.replace(".", "__")

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

    def _vector_for(self, head: HeadSpec):
        return self.vectors[self._param_key(head.head_id)]

    def _encode_prompt_and_continuation(self, prompt: str, continuation: str) -> tuple[dict[str, Any], int, int]:
        prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
        input_ids = self.torch.cat([prompt_ids, cont_ids], dim=-1).to(self.device)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, int(prompt_ids.shape[-1]), int(cont_ids.shape[-1])

    def _tokenize_prompt_and_continuation(self, prompt: str, continuation: str) -> tuple[Any, Any]:
        formatted_prompt = self.backend._format_prompt(prompt)
        prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        cont_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if int(prompt_ids.shape[-1]) <= 0:
            raise ValueError("empty prompt after tokenization")
        if int(cont_ids.shape[-1]) <= 0:
            raise ValueError(f"empty continuation after tokenization: {continuation!r}")
        return prompt_ids.to(self.device), cont_ids.to(self.device)

    def _build_prefix_inputs(self, prompt_ids: Any, cont_ids: Any, *, visible_continuation_len: int) -> dict[str, Any]:
        prefix = cont_ids[:, :visible_continuation_len]
        input_ids = self.torch.cat([prompt_ids, prefix], dim=-1)
        return {
            "input_ids": input_ids,
            "attention_mask": self.torch.ones_like(input_ids, device=input_ids.device),
        }

    def _slice_for_apply_mode(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        seq_len: int,
        continuation: str,
        score_text: str,
    ) -> slice:
        mode = self.config.apply_mode
        if mode == "decision_tokens":
            start = max(prompt_len - 1, 0)
            stop = min(prompt_len + continuation_len - 1, seq_len)
            return slice(start, max(stop, start + 1))
        if mode in {"prefill", "prompt"}:
            return slice(0, min(prompt_len, seq_len))
        if mode == "decode":
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + continuation_len, seq_len)
            return slice(start, max(stop, start))
        first_decode_steps = _first_decode_steps(mode)
        if first_decode_steps is not None:
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + first_decode_steps, seq_len)
            return slice(start, max(stop, start))
        if mode == "boxed_decision":
            token_start = score_text_start_token_index(
                self.tokenizer,
                continuation,
                score_text,
                prefer_boxed=True,
            )
            if token_start is None:
                raise ValueError("boxed_decision requires score_text to occur in the continuation")
            start = max(prompt_len + token_start - 1, 0)
            return slice(start, min(start + 1, seq_len))
        if mode == "prompt_last":
            start = max(prompt_len - 1, 0)
            return slice(start, start + 1)
        if mode == "all":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported apply_mode: {mode}")

    def _causal_step_active(self, *, visible_continuation_len: int) -> bool:
        mode = self.config.apply_mode
        if mode in {"prompt_last", "prefill", "prompt"}:
            return True
        if mode == "decode":
            return visible_continuation_len >= 1
        first_decode_steps = _first_decode_steps(mode)
        if first_decode_steps is not None:
            return visible_continuation_len >= 1
        if mode in {"decision_tokens", "boxed_decision", "all"}:
            return True
        raise ValueError(f"Unsupported apply_mode: {mode}")

    def _register_hooks(
        self,
        *,
        alpha: float,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles: list[Any] = []
        heads_by_layer: dict[int, list[HeadSpec]] = {}
        for head in self.heads:
            heads_by_layer.setdefault(head.layer_idx, []).append(head)

        for layer_idx, heads in heads_by_layer.items():
            module = self._o_proj_module(layer_idx)
            head_dim = self.head_dims[layer_idx]

            def make_hook(local_heads: list[HeadSpec], local_head_dim: int):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    pos = self._slice_for_apply_mode(
                        prompt_len=prompt_len,
                        continuation_len=continuation_len,
                        seq_len=int(hidden_new.shape[1]),
                        continuation=continuation,
                        score_text=score_text,
                    )
                    for head in local_heads:
                        start = int(head.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        vector = self._vector_for(head).to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] + float(alpha) * vector
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(heads, head_dim)))
        return handles

    @staticmethod
    def _frozen_slice_for_apply_mode(
        *,
        mode: str,
        prompt_len: int,
        continuation_len: int,
        seq_len: int,
        tokenizer: Any,
        continuation: str,
        score_text: str,
    ) -> slice:
        if mode in {"prefill", "prompt"}:
            return slice(0, min(prompt_len, seq_len))
        if mode == "prompt_last":
            start = max(min(prompt_len, seq_len) - 1, 0)
            return slice(start, start + 1)
        if mode == "decision_tokens":
            start = max(prompt_len - 1, 0)
            stop = min(prompt_len + continuation_len - 1, seq_len)
            return slice(start, max(stop, start + 1))
        if mode == "decode":
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + continuation_len, seq_len)
            return slice(start, max(stop, start))
        first_decode_steps = _first_decode_steps(mode)
        if first_decode_steps is not None:
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + first_decode_steps, seq_len)
            return slice(start, max(stop, start))
        if mode == "boxed_decision":
            token_start = score_text_start_token_index(
                tokenizer,
                continuation,
                score_text,
                prefer_boxed=True,
            )
            if token_start is None:
                raise ValueError("boxed_decision requires score_text to occur in the continuation")
            start = max(prompt_len + token_start - 1, 0)
            return slice(start, min(start + 1, seq_len))
        if mode == "all":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported frozen component apply_mode: {mode}")

    def _register_frozen_component_hooks(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(addition: dict[str, Any]):
            direction = addition["direction"]
            alpha = float(addition["alpha"])
            apply_mode = str(addition.get("apply_mode") or "prefill")

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                pos = self._frozen_slice_for_apply_mode(
                    mode=apply_mode,
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    seq_len=int(hidden_new.shape[1]),
                    tokenizer=self.tokenizer,
                    continuation=continuation,
                    score_text=score_text,
                )
                delta = direction.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + alpha * delta
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for addition in self.frozen_component_additions:
            module = self.backend._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            handles.append(module.register_forward_hook(make_hook(addition)))
        return handles

    def _register_frozen_head_hooks(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles: list[Any] = []
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for addition in self.frozen_head_additions:
            by_layer.setdefault(int(addition["layer_idx"]), []).append(addition)

        for layer_idx, additions in by_layer.items():
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for addition in additions:
                head_idx = int(addition["head_idx"])
                if head_idx < 0 or head_idx >= num_heads:
                    raise ValueError(f"Invalid L{layer_idx}.attn.h{head_idx}; L{layer_idx}.attn has {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(local_additions: list[dict[str, Any]], local_head_dim: int):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    for addition in local_additions:
                        apply_mode = str(addition.get("apply_mode") or "all")
                        pos = self._frozen_slice_for_apply_mode(
                            mode=apply_mode,
                            prompt_len=prompt_len,
                            continuation_len=continuation_len,
                            seq_len=int(hidden_new.shape[1]),
                            tokenizer=self.tokenizer,
                            continuation=continuation,
                            score_text=score_text,
                        )
                        head_idx = int(addition["head_idx"])
                        start = head_idx * local_head_dim
                        stop = start + local_head_dim
                        direction = addition["direction"].to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = (
                            hidden_new[:, pos, start:stop] + float(addition["alpha"]) * direction
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(additions, head_dim)))
        return handles

    def _register_zero_head_hooks(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles: list[Any] = []
        by_layer: dict[int, list[HeadSpec]] = {}
        for head in self.zero_heads:
            by_layer.setdefault(head.layer_idx, []).append(head)

        for layer_idx, heads in by_layer.items():
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for head in heads:
                if head.head_idx < 0 or head.head_idx >= num_heads:
                    raise ValueError(f"Invalid zero head {head.head_id}; L{layer_idx}.attn has {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(local_heads: list[HeadSpec], local_head_dim: int):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    pos = self._frozen_slice_for_apply_mode(
                        mode=self.config.zero_head_apply_mode,
                        prompt_len=prompt_len,
                        continuation_len=continuation_len,
                        seq_len=int(hidden_new.shape[1]),
                        tokenizer=self.tokenizer,
                        continuation=continuation,
                        score_text=score_text,
                    )
                    for head in local_heads:
                        start = int(head.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        hidden_new[:, pos, start:stop] = 0
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(heads, head_dim)))
        return handles

    def _candidate_score_full_sequence(
        self,
        prompt: str,
        continuation: str,
        *,
        alpha: float | None = None,
        score_text: str = "",
    ):
        inputs, prompt_len, continuation_len = self._encode_prompt_and_continuation(prompt, continuation)
        handles: list[Any] = []
        if self.zero_heads:
            handles.extend(
                self._register_zero_head_hooks(
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    continuation=continuation,
                    score_text=score_text,
                )
            )
        if self.frozen_component_additions:
            handles.extend(
                self._register_frozen_component_hooks(
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    continuation=continuation,
                    score_text=score_text,
                )
            )
        if self.frozen_head_additions:
            handles.extend(
                self._register_frozen_head_hooks(
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    continuation=continuation,
                    score_text=score_text,
                )
            )
        if alpha is not None and alpha != 0:
            handles.extend(
                self._register_hooks(
                    alpha=alpha,
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    continuation=continuation,
                    score_text=score_text,
                )
            )
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
        if self.config.score_mode == "avglogp":
            log_probs = self.torch.nn.functional.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            return token_log_probs.mean()
        if self.config.score_mode == "top_logit_gap":
            top_logits = pred_logits.max(dim=-1).values
            return (token_logits - top_logits).mean()
        if self.config.score_mode == "answer_rest_margin":
            top_values, top_indices = pred_logits.topk(k=2, dim=-1)
            top1_values = top_values[..., 0]
            top2_values = top_values[..., 1]
            top1_indices = top_indices[..., 0]
            rest_max_logits = self.torch.where(top1_indices == target_ids, top2_values, top1_values)
            return (token_logits - rest_max_logits).mean()
        raise ValueError(f"Unsupported score_mode: {self.config.score_mode}")

    def candidate_score(self, prompt: str, continuation: str, *, alpha: float | None = None, score_text: str = ""):
        if self.config.causal_train_mask:
            return self._candidate_score_causal(prompt, continuation, alpha=alpha, score_text=score_text)
        return self._candidate_score_full_sequence(
            prompt,
            continuation,
            alpha=alpha,
            score_text=score_text,
        )

    def _candidate_score_causal(
        self,
        prompt: str,
        continuation: str,
        *,
        alpha: float | None = None,
        score_text: str = "",
    ):
        return self._candidate_score_full_sequence(
            prompt,
            continuation,
            alpha=alpha,
            score_text=score_text,
        )

    def endpoint_score(
        self,
        prompt: str,
        continuations: tuple[str, ...],
        *,
        alpha: float | None = None,
        score_text: str = "",
    ):
        options = limit_options(
            continuations,
            max_aliases_per_side=self.config.max_aliases_per_side,
        )
        if not options:
            raise ValueError("empty endpoint continuation options")
        scores = [
            self.candidate_score(prompt, option, alpha=alpha, score_text=score_text)
            for option in options
        ]
        return select_option_score(
            self.torch,
            scores,
            selection_mode=self.config.option_selection_mode,
        )

    def margin(self, pair: ActuatorPair, *, alpha: float | None = None):
        require_dynamic_answer_rest_margin(self.config.score_mode, pair=pair)
        if is_constant_zero_y_plus(pair):
            plus = self.torch.zeros((), device=self.device)
        else:
            plus = self.endpoint_score(
                pair.prompt,
                pair.y_plus_options,
                alpha=alpha,
                score_text=pair.y_plus_score_text,
            )
        if is_dynamic_y_minus(pair) or not pair.y_minus_options:
            return plus
        minus = self.endpoint_score(
            pair.prompt,
            pair.y_minus_options,
            alpha=alpha,
            score_text=pair.y_minus_score_text,
        )
        return plus - minus

    def cache_baseline_margins(self, pairs: list[ActuatorPair]) -> dict[str, float]:
        baselines: dict[str, float] = {}
        if self._fast_batch_supported(pairs):
            batch_size = max(1, int(self.config.train_batch_size))
            with self.torch.no_grad():
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start : start + batch_size]
                    margins = self._margin_batch(batch_pairs, alpha=None)
                    for pair, value in zip(batch_pairs, margins.detach().cpu().tolist(), strict=False):
                        baselines[pair.sample_id] = float(value)
                    self._memory_cleanup()
            return baselines
        with self.torch.no_grad():
            for pair in pairs:
                baselines[pair.sample_id] = float(self.margin(pair, alpha=None).detach().cpu().item())
                if len(baselines) % 25 == 0:
                    import gc

                    gc.collect()
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
        return baselines

    @staticmethod
    def _softplus_float(value: float) -> float:
        if value > 20.0:
            return value
        if value < -20.0:
            return math.exp(value)
        return math.log1p(math.exp(value))

    @staticmethod
    def _sigmoid_negative(value: float) -> float:
        if value >= 0.0:
            exp_neg = math.exp(-value)
            return exp_neg / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return 1.0 / (1.0 + exp_pos)

    def _memory_cleanup(self) -> None:
        import gc

        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _pad_token_id(self) -> int:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad_token_id or eos_token_id for batched training.")
        return int(pad_id)

    def _fast_batch_supported(self, pairs: list[ActuatorPair]) -> bool:
        if self.frozen_component_additions or self.frozen_head_additions:
            return False
        if self.config.max_aliases_per_side != 1:
            return False
        for pair in pairs:
            if is_constant_zero_y_plus(pair) or is_dynamic_y_minus(pair):
                return False
            if len(pair.y_plus_options) != 1 or len(pair.y_minus_options) != 1:
                return False
        return True

    def _encode_example_batch(
        self,
        examples: list[tuple[str, str, str]],
    ) -> tuple[Any, Any, list[int], list[int], list[int], list[slice | None], list[str], list[str]]:
        prompt_lens: list[int] = []
        continuation_lens: list[int] = []
        seq_lens: list[int] = []
        score_slices: list[slice | None] = []
        prompts: list[str] = []
        continuations: list[str] = []
        prompt_id_rows: list[Any] = []
        continuation_id_rows: list[Any] = []
        for prompt, continuation, score_text in examples:
            prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
            prompt_lens.append(int(prompt_ids.shape[-1]))
            continuation_lens.append(int(cont_ids.shape[-1]))
            seq_lens.append(int(prompt_ids.shape[-1] + cont_ids.shape[-1]))
            score_slices.append(score_text_token_slice(self.tokenizer, continuation, score_text))
            prompts.append(prompt)
            continuations.append(continuation)
            prompt_id_rows.append(prompt_ids.squeeze(0))
            continuation_id_rows.append(cont_ids.squeeze(0))
        batch_size = len(examples)
        max_seq_len = max(seq_lens)
        pad_id = self._pad_token_id()
        input_ids = self.torch.full(
            (batch_size, max_seq_len),
            pad_id,
            dtype=prompt_id_rows[0].dtype,
            device=self.device,
        )
        attention_mask = self.torch.zeros(
            (batch_size, max_seq_len),
            dtype=prompt_id_rows[0].dtype,
            device=self.device,
        )
        for row_idx, (prompt_ids_row, cont_ids_row) in enumerate(zip(prompt_id_rows, continuation_id_rows, strict=False)):
            full_ids = self.torch.cat([prompt_ids_row, cont_ids_row], dim=0)
            seq_len = int(full_ids.shape[0])
            input_ids[row_idx, :seq_len] = full_ids
            attention_mask[row_idx, :seq_len] = 1
        return input_ids, attention_mask, prompt_lens, continuation_lens, seq_lens, score_slices, prompts, continuations

    def _batch_position_mask(
        self,
        *,
        prompt_lens: list[int],
        continuation_lens: list[int],
        seq_lens: list[int],
        continuations: list[str],
        score_texts: list[str],
        mode: str | None = None,
    ) -> Any:
        batch_size = len(prompt_lens)
        max_seq_len = max(seq_lens)
        mask = self.torch.zeros((batch_size, max_seq_len), dtype=self.torch.bool, device=self.device)
        local_mode = mode or self.config.apply_mode
        for row_idx in range(batch_size):
            if local_mode == self.config.apply_mode:
                pos = self._slice_for_apply_mode(
                    prompt_len=prompt_lens[row_idx],
                    continuation_len=continuation_lens[row_idx],
                    seq_len=seq_lens[row_idx],
                    continuation=continuations[row_idx],
                    score_text=score_texts[row_idx],
                )
            else:
                pos = self._frozen_slice_for_apply_mode(
                    mode=local_mode,
                    prompt_len=prompt_lens[row_idx],
                    continuation_len=continuation_lens[row_idx],
                    seq_len=seq_lens[row_idx],
                    tokenizer=self.tokenizer,
                    continuation=continuations[row_idx],
                    score_text=score_texts[row_idx],
                )
            mask[row_idx, pos] = True
        return mask

    def _register_zero_head_hooks_batch(self, *, position_mask: Any) -> list[Any]:
        handles: list[Any] = []
        mask = position_mask
        by_layer: dict[int, list[HeadSpec]] = {}
        for head in self.zero_heads:
            by_layer.setdefault(head.layer_idx, []).append(head)

        for layer_idx, heads in by_layer.items():
            module = self._o_proj_module(layer_idx)
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for head in heads:
                if head.head_idx < 0 or head.head_idx >= num_heads:
                    raise ValueError(f"Invalid zero head {head.head_id}; L{layer_idx}.attn has {num_heads} heads")

            def make_hook(local_heads: list[HeadSpec], local_head_dim: int):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    local_mask = mask.to(device=hidden_new.device, dtype=self.torch.bool)
                    for head in local_heads:
                        start = int(head.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        hidden_new[:, :, start:stop] = hidden_new[:, :, start:stop].masked_fill(
                            local_mask.unsqueeze(-1),
                            0,
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(heads, head_dim)))
        return handles

    def _register_hooks_batch(self, *, alpha: float, position_mask: Any) -> list[Any]:
        handles: list[Any] = []
        mask = position_mask
        heads_by_layer: dict[int, list[HeadSpec]] = {}
        for head in self.heads:
            heads_by_layer.setdefault(head.layer_idx, []).append(head)

        for layer_idx, heads in heads_by_layer.items():
            module = self._o_proj_module(layer_idx)
            head_dim = self.head_dims[layer_idx]

            def make_hook(local_heads: list[HeadSpec], local_head_dim: int):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    local_mask = mask.to(device=hidden_new.device, dtype=hidden_new.dtype).unsqueeze(-1)
                    for head in local_heads:
                        start = int(head.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        vector = self._vector_for(head).to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, :, start:stop] = (
                            hidden_new[:, :, start:stop] + float(alpha) * local_mask * vector.view(1, 1, -1)
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(heads, head_dim)))
        return handles

    def _candidate_scores_batch(
        self,
        examples: list[tuple[str, str, str]],
        *,
        alpha: float | None = None,
    ) -> Any:
        if not examples:
            raise ValueError("empty candidate batch")
        (
            input_ids,
            attention_mask,
            prompt_lens,
            continuation_lens,
            seq_lens,
            score_slices,
            _prompts,
            continuations,
        ) = self._encode_example_batch(examples)
        handles: list[Any] = []
        if self.zero_heads:
            zero_position_mask = self._batch_position_mask(
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
                continuations=continuations,
                score_texts=[score_text for _, _, score_text in examples],
                mode=self.config.zero_head_apply_mode,
            )
            handles.extend(self._register_zero_head_hooks_batch(position_mask=zero_position_mask))
        if alpha is not None and alpha != 0:
            position_mask = self._batch_position_mask(
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
                continuations=continuations,
                score_texts=[score_text for _, _, score_text in examples],
            )
            handles.extend(self._register_hooks_batch(alpha=alpha, position_mask=position_mask))
        try:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        scores: list[Any] = []
        for row_idx, score_slice in enumerate(score_slices):
            prompt_len = prompt_lens[row_idx]
            continuation_len = continuation_lens[row_idx]
            target_ids = input_ids[row_idx, prompt_len : prompt_len + continuation_len]
            pred_logits = logits[row_idx, prompt_len - 1 : prompt_len + continuation_len - 1, :].float()
            if score_slice is not None:
                target_ids = target_ids[score_slice]
                pred_logits = pred_logits[score_slice, :]
            token_logits = pred_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            if self.config.score_mode == "avglogp":
                log_probs = self.torch.nn.functional.log_softmax(pred_logits, dim=-1)
                scores.append(log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).mean())
                continue
            if self.config.score_mode == "top_logit_gap":
                top_logits = pred_logits.max(dim=-1).values
                scores.append((token_logits - top_logits).mean())
                continue
            if self.config.score_mode == "answer_rest_margin":
                top_values, top_indices = pred_logits.topk(k=2, dim=-1)
                top1_values = top_values[..., 0]
                top2_values = top_values[..., 1]
                top1_indices = top_indices[..., 0]
                rest_max_logits = self.torch.where(top1_indices == target_ids, top2_values, top1_values)
                scores.append((token_logits - rest_max_logits).mean())
                continue
            raise ValueError(f"Unsupported score_mode: {self.config.score_mode}")
        return self.torch.stack(scores, dim=0)

    def _margin_batch(self, pairs: list[ActuatorPair], *, alpha: float | None = None) -> Any:
        plus_examples = [(pair.prompt, pair.y_plus_options[0], pair.y_plus_score_text) for pair in pairs]
        minus_examples = [(pair.prompt, pair.y_minus_options[0], pair.y_minus_score_text) for pair in pairs]
        plus_scores = self._candidate_scores_batch(plus_examples, alpha=alpha)
        minus_scores = self._candidate_scores_batch(minus_examples, alpha=alpha)
        return plus_scores - minus_scores

    def _streaming_pair_options(self, pair: ActuatorPair) -> tuple[str, str] | None:
        return None

    def _candidate_score_causal_avglogp_streaming_backward(
        self,
        prompt: str,
        continuation: str,
        *,
        alpha: float | None = None,
        score_text: str = "",
        scale: float,
    ) -> float:
        prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
        prompt_len = int(prompt_ids.shape[-1])
        continuation_len = int(cont_ids.shape[-1])
        score_slice = score_text_token_slice(self.tokenizer, continuation, score_text)
        keep_indices = set(range(continuation_len))
        if score_slice is not None:
            keep_indices = set(range(*score_slice.indices(continuation_len)))
        if not keep_indices:
            raise ValueError(f"Empty score slice after tokenization: continuation={continuation!r} score_text={score_text!r}")

        total_score = 0.0
        denom = len(keep_indices)
        scale_per_token = float(scale) / float(denom)
        for target_token_index in range(continuation_len):
            if target_token_index not in keep_indices:
                continue
            visible_continuation_len = target_token_index
            inputs = self._build_prefix_inputs(
                prompt_ids,
                cont_ids,
                visible_continuation_len=visible_continuation_len,
            )
            handles: list[Any] = []
            if self.zero_heads:
                handles.extend(
                    self._register_zero_head_hooks(
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            if self.frozen_component_additions:
                handles.extend(
                    self._register_frozen_component_hooks(
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            if self.frozen_head_additions:
                handles.extend(
                    self._register_frozen_head_hooks(
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            if self.frozen_component_additions:
                handles.extend(
                    self._register_frozen_component_hooks(
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            if self.frozen_head_additions:
                handles.extend(
                    self._register_frozen_head_hooks(
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            if alpha is not None and alpha != 0 and self._causal_step_active(
                visible_continuation_len=visible_continuation_len
            ):
                handles.extend(
                    self._register_hooks(
                        alpha=alpha,
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            try:
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :].float()
                target_ids = cont_ids[:, target_token_index]
                log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)
                step_score = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).mean()
                total_score += float(step_score.detach().cpu().item())
                if scale_per_token != 0.0:
                    (step_score * scale_per_token).backward()
            finally:
                for handle in handles:
                    handle.remove()
            del inputs, logits, target_ids, log_probs, step_score
        return total_score / float(denom)

    def _streaming_dpo_step(
        self,
        pair: ActuatorPair,
        *,
        base_margin: float,
    ) -> dict[str, float] | None:
        options = self._streaming_pair_options(pair)
        if options is None:
            return None
        plus_continuation, minus_continuation = options
        with self.torch.no_grad():
            plus_score = float(
                self.candidate_score(
                    pair.prompt,
                    plus_continuation,
                    alpha=self.config.alpha_train,
                    score_text=pair.y_plus_score_text,
                ).detach().cpu().item()
            )
            minus_score = float(
                self.candidate_score(
                    pair.prompt,
                    minus_continuation,
                    alpha=self.config.alpha_train,
                    score_text=pair.y_minus_score_text,
                ).detach().cpu().item()
            )
        steered_margin = plus_score - minus_score
        gain = steered_margin - float(base_margin)
        preference_logit = float(self.config.dpo_beta) * gain
        preference_loss = self._softplus_float(-preference_logit)
        completion_loss = self._softplus_float(float(self.config.target_margin) - steered_margin)
        gain_loss = self._softplus_float(float(self.config.target_gain) - gain)
        pair_weight = float(pair_question_state_weight(pair, self.config.question_state_weights))
        margin_grad = pair_weight * (-float(self.config.dpo_beta) * self._sigmoid_negative(preference_logit))
        if margin_grad != 0.0:
            self._candidate_score_causal_avglogp_streaming_backward(
                pair.prompt,
                plus_continuation,
                alpha=self.config.alpha_train,
                score_text=pair.y_plus_score_text,
                scale=margin_grad,
            )
            self._candidate_score_causal_avglogp_streaming_backward(
                pair.prompt,
                minus_continuation,
                alpha=self.config.alpha_train,
                score_text=pair.y_minus_score_text,
                scale=-margin_grad,
            )
        norm_penalty_value = float(self.norm_penalty().detach().cpu().item())
        if float(self.config.lambda_norm) != 0.0:
            (float(self.config.lambda_norm) * self.norm_penalty()).backward()
        return {
            "loss": pair_weight * preference_loss + float(self.config.lambda_norm) * norm_penalty_value,
            "completion_loss": completion_loss,
            "gain_loss": gain_loss,
            "preference_loss": preference_loss,
            "preference_logit": preference_logit,
            "margin": steered_margin,
            "gain": gain,
        }

    def norm_penalty(self):
        total = None
        for parameter in self.vectors.values():
            value = parameter.pow(2).mean()
            total = value if total is None else total + value
        if total is None:
            return self.torch.tensor(0.0, device=self.device)
        return total

    def train(
        self,
        train_pairs: list[ActuatorPair],
        *,
        val_pairs: list[ActuatorPair],
        out_dir: Path,
    ) -> tuple[list[dict[str, object]], dict[str, float]]:
        import random

        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(self.config.seed)
        optimizer = self.torch.optim.AdamW(self.vectors.parameters(), lr=self.config.lr)

        print(f"[cecm-head] caching train baselines: n={len(train_pairs)}", flush=True)
        baseline_train = self.cache_baseline_margins(train_pairs)
        self._memory_cleanup()

        history: list[dict[str, object]] = []
        fast_batch = self._fast_batch_supported(train_pairs)
        train_batch_size = max(1, int(self.config.train_batch_size))
        for epoch in range(1, self.config.epochs + 1):
            shuffled = list(train_pairs)
            rng.shuffle(shuffled)
            losses: list[float] = []
            completion_losses: list[float] = []
            gain_losses: list[float] = []
            preference_losses: list[float] = []
            preference_logits: list[float] = []
            margins: list[float] = []
            gains: list[float] = []
            processed_pairs = 0
            for batch_start in range(0, len(shuffled), train_batch_size if fast_batch else 1):
                batch_pairs = shuffled[batch_start : batch_start + (train_batch_size if fast_batch else 1)]
                step = processed_pairs + len(batch_pairs)
                optimizer.zero_grad(set_to_none=True)
                if fast_batch:
                    margin_batch = self._margin_batch(batch_pairs, alpha=self.config.alpha_train)
                    base_margin_batch = self.torch.tensor(
                        [float(baseline_train[pair.sample_id]) for pair in batch_pairs],
                        device=self.device,
                    )
                    gain_batch = margin_batch - base_margin_batch
                    completion_loss_batch = self.torch.nn.functional.softplus(
                        self.torch.tensor(float(self.config.target_margin), device=self.device) - margin_batch
                    )
                    gain_loss_batch = self.torch.nn.functional.softplus(
                        self.torch.tensor(float(self.config.target_gain), device=self.device) - gain_batch
                    )
                    if self.config.preference_loss_mode == DPO_LOSS:
                        preference_logit_batch = float(self.config.dpo_beta) * gain_batch
                        preference_loss_batch = self.torch.nn.functional.softplus(-preference_logit_batch)
                        task_loss_batch = preference_loss_batch
                    else:
                        preference_logit_batch = None
                        preference_loss_batch = None
                        task_loss_batch = (
                            float(self.config.state_margin_weight) * completion_loss_batch
                            + float(self.config.gain_weight) * gain_loss_batch
                        )
                    pair_weight_batch = self.torch.tensor(
                        [pair_question_state_weight(pair, self.config.question_state_weights) for pair in batch_pairs],
                        device=self.device,
                    )
                    loss = (pair_weight_batch * task_loss_batch).mean() + float(self.config.lambda_norm) * self.norm_penalty()
                    loss.backward()
                    optimizer.step()
                    losses.extend(loss.detach().cpu().repeat(len(batch_pairs)).tolist())
                    completion_losses.extend(completion_loss_batch.detach().cpu().tolist())
                    gain_losses.extend(gain_loss_batch.detach().cpu().tolist())
                    margins.extend(margin_batch.detach().cpu().tolist())
                    gains.extend(gain_batch.detach().cpu().tolist())
                    if preference_loss_batch is not None:
                        preference_losses.extend(preference_loss_batch.detach().cpu().tolist())
                    if preference_logit_batch is not None:
                        preference_logits.extend(preference_logit_batch.detach().cpu().tolist())
                else:
                    pair = batch_pairs[0]
                    steered_margin = self.margin(pair, alpha=self.config.alpha_train)
                    base_margin = self.torch.tensor(baseline_train[pair.sample_id], device=self.device)
                    gain = steered_margin - base_margin
                    completion_loss = self.torch.nn.functional.softplus(
                        self.torch.tensor(self.config.target_margin, device=self.device) - steered_margin
                    )
                    gain_loss = self.torch.nn.functional.softplus(
                        self.torch.tensor(self.config.target_gain, device=self.device) - gain
                    )
                    if self.config.preference_loss_mode == DPO_LOSS:
                        preference_logit = dpo_preference_logit(
                            steered_margin,
                            base_margin,
                            beta=self.config.dpo_beta,
                        )
                        preference_loss = self.torch.nn.functional.softplus(-preference_logit)
                        task_loss = preference_loss
                    else:
                        preference_logit = None
                        preference_loss = None
                        task_loss = (
                            float(self.config.state_margin_weight) * completion_loss
                            + float(self.config.gain_weight) * gain_loss
                        )
                    pair_weight = pair_question_state_weight(pair, self.config.question_state_weights)
                    loss = float(pair_weight) * task_loss + float(self.config.lambda_norm) * self.norm_penalty()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))
                    completion_losses.append(float(completion_loss.detach().cpu().item()))
                    gain_losses.append(float(gain_loss.detach().cpu().item()))
                    if preference_loss is not None:
                        preference_losses.append(float(preference_loss.detach().cpu().item()))
                    if preference_logit is not None:
                        preference_logits.append(float(preference_logit.detach().cpu().item()))
                    margins.append(float(steered_margin.detach().cpu().item()))
                    gains.append(float(gain.detach().cpu().item()))
                processed_pairs = step
                if step == len(batch_pairs) or step % 25 < len(batch_pairs) or step == len(shuffled):
                    progress_suffix = (
                        f" dpo_logit={preference_logits[-1]:.6f}"
                        if preference_logits
                        else ""
                    )
                    print(
                        (
                            f"[cecm-head] epoch={epoch}/{self.config.epochs} "
                            f"step={step}/{len(shuffled)} loss={losses[-1]:.6f} "
                            f"margin={margins[-1]:.6f} gain={gains[-1]:.6f}"
                            f"{progress_suffix}"
                        ),
                        flush=True,
                    )
                if self.config.empty_cache_every > 0 and step % self.config.empty_cache_every < len(batch_pairs):
                    self._memory_cleanup()
            history.append(
                {
                    "epoch": epoch,
                    "train_pairs": len(shuffled),
                    "mean_loss": mean(losses) if losses else math.nan,
                    "mean_completion_loss": mean(completion_losses) if completion_losses else math.nan,
                    "mean_gain_loss": mean(gain_losses) if gain_losses else math.nan,
                    "mean_preference_loss": mean(preference_losses) if preference_losses else math.nan,
                    "mean_preference_logit": mean(preference_logits) if preference_logits else math.nan,
                    "mean_margin": mean(margins) if margins else math.nan,
                    "mean_margin_gain": mean(gains) if gains else math.nan,
                    "state_complete_rate": mean(float(value > self.config.target_margin) for value in margins)
                    if margins
                    else math.nan,
                    "gain_complete_rate": mean(float(value > self.config.target_gain) for value in gains)
                    if gains
                    else math.nan,
                    "dpo_win_rate": mean(float(value > 0) for value in preference_logits)
                    if preference_logits
                    else math.nan,
                    "norm_penalty": float(self.norm_penalty().detach().cpu().item()),
                }
            )
        print(f"[cecm-head] caching val baselines: n={len(val_pairs)}", flush=True)
        baseline_val = self.cache_baseline_margins(val_pairs)
        self._memory_cleanup()
        baseline_all = {**baseline_train, **baseline_val}
        return history, baseline_all

    def evaluate_alpha(
        self,
        pairs: list[ActuatorPair],
        *,
        baseline_margins: dict[str, float],
        alpha: float,
        split: str,
    ) -> dict[str, object]:
        margins: list[float] = []
        base_margins: list[float] = []
        gains: list[float] = []
        preference_logits: list[float] = []
        with self.torch.no_grad():
            if self._fast_batch_supported(pairs):
                batch_size = max(1, int(self.config.train_batch_size))
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start : start + batch_size]
                    if alpha != 0:
                        margin_batch = self._margin_batch(batch_pairs, alpha=alpha).detach().cpu().tolist()
                    else:
                        margin_batch = [float(baseline_margins[pair.sample_id]) for pair in batch_pairs]
                    for pair, margin in zip(batch_pairs, margin_batch, strict=False):
                        base = baseline_margins[pair.sample_id]
                        base_margins.append(base)
                        margins.append(float(margin))
                        gains.append(float(margin) - base)
                        if self.config.preference_loss_mode == DPO_LOSS:
                            preference_logits.append(
                                float(dpo_preference_logit(float(margin), base, beta=self.config.dpo_beta))
                            )
                    self._memory_cleanup()
            else:
                for pair in pairs:
                    base = baseline_margins[pair.sample_id]
                    margin = float(self.margin(pair, alpha=alpha).detach().cpu().item()) if alpha != 0 else base
                    base_margins.append(base)
                    margins.append(margin)
                    gains.append(margin - base)
                    if self.config.preference_loss_mode == DPO_LOSS:
                        preference_logits.append(
                            float(dpo_preference_logit(margin, base, beta=self.config.dpo_beta))
                        )
        n = len(pairs)
        return {
            "split": split,
            "alpha": alpha,
            "n": n,
            "base_mean_margin": mean(base_margins) if n else math.nan,
            "mean_margin": mean(margins) if n else math.nan,
            "mean_margin_gain": mean(gains) if n else math.nan,
            "mean_preference_logit": mean(preference_logits) if preference_logits else math.nan,
            "base_pref_rate": mean(float(value > 0) for value in base_margins) if n else math.nan,
            "pref_rate": mean(float(value > 0) for value in margins) if n else math.nan,
            "gain_positive_rate": mean(float(value > 0) for value in gains) if n else math.nan,
            "dpo_win_rate": mean(float(value > 0) for value in preference_logits)
            if preference_logits
            else math.nan,
        }

    def vector_summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with self.torch.no_grad():
            for head in self.heads:
                vector = self._vector_for(head)
                rows.append(
                    {
                        "head_id": head.head_id,
                        "layer_idx": head.layer_idx,
                        "head_idx": head.head_idx,
                        "head_dim": self.head_dims[head.layer_idx],
                        "l2_norm": float(vector.float().norm().detach().cpu().item()),
                        "mean_abs": float(vector.float().abs().mean().detach().cpu().item()),
                        "max_abs": float(vector.float().abs().max().detach().cpu().item()),
                    }
                )
        return rows

    def save_payload(self, path: Path) -> None:
        vectors = {head.head_id: self._vector_for(head).detach().cpu() for head in self.heads}
        payload = {
            "kind": "attention_head_fixed_actuator",
            "event": self.config.event,
            "endpoint_objective": self.config.endpoint_objective,
            "apply_mode": self.config.apply_mode,
            "score_mode": self.config.score_mode,
            "option_selection_mode": self.config.option_selection_mode,
            "option_selection": option_selection_description(self.config.option_selection_mode),
            "alpha_train": self.config.alpha_train,
            "zero_heads": [head.head_id for head in self.zero_heads],
            "zero_head_apply_mode": self.config.zero_head_apply_mode,
            "preference_loss_mode": self.config.preference_loss_mode,
            "dpo_beta": self.config.dpo_beta,
            "objective": actuator_objective_description(
                state_margin_weight=self.config.state_margin_weight,
                gain_weight=self.config.gain_weight,
                preference_loss_mode=self.config.preference_loss_mode,
            ),
            "competitive_margin": competitive_margin_description(self.config.endpoint_objective),
            "causal_operator": (
                "zero selected attention head slices before o_proj, then inject trainable head vectors"
                if self.zero_heads
                else "vector injection on selected attention head slices before o_proj"
            ),
            "loss": actuator_loss_description(
                state_margin_weight=self.config.state_margin_weight,
                gain_weight=self.config.gain_weight,
                norm_label="U_h",
                preference_loss_mode=self.config.preference_loss_mode,
                dpo_beta=self.config.dpo_beta,
            ),
            "state_margin_weight": self.config.state_margin_weight,
            "gain_weight": self.config.gain_weight,
            "target_margin": self.config.target_margin,
            "target_gain": self.config.target_gain,
            "score": actuator_score_description(self.config.score_mode),
            "conditioning": [
                {
                    "component_id": str(addition.get("component_id", "")),
                    "layer_idx": int(addition["layer_idx"]),
                    "component_type": str(addition["component_type"]),
                    "alpha": float(addition["alpha"]),
                    "apply_mode": str(addition.get("apply_mode", "")),
                }
                for addition in self.frozen_component_additions
            ],
            "conditioning_heads": [
                {
                    "head_id": str(addition.get("head_id", "")),
                    "layer_idx": int(addition["layer_idx"]),
                    "head_idx": int(addition["head_idx"]),
                    "alpha": float(addition["alpha"]),
                    "apply_mode": str(addition.get("apply_mode", "")),
                }
                for addition in self.frozen_head_additions
            ],
            "zeroed_heads": [
                {
                    "head_id": head.head_id,
                    "layer_idx": head.layer_idx,
                    "head_idx": head.head_idx,
                    "apply_mode": self.config.zero_head_apply_mode,
                }
                for head in self.zero_heads
            ],
            "heads": [
                {
                    "head_id": head.head_id,
                    "layer_idx": head.layer_idx,
                    "head_idx": head.head_idx,
                    "head_dim": self.head_dims[head.layer_idx],
                    "num_heads": self.num_heads_by_layer[head.layer_idx],
                }
                for head in self.heads
            ],
            "vectors": vectors,
            "semantics": (
                "Reusable vectors are added to selected attention head slices before o_proj. "
                "When zeroed_heads is non-empty, those head slices are first set to zero under "
                "the listed apply mode, so the learned vector is trained as a replacement/causal "
                "residual in the zeroed component context. "
                "Component/head discovery supplies the selected attention sites; training then "
                "optimizes reference-anchored endpoint preference over the same fixed y_plus/y_minus "
                "pairs by changing only the injected head vectors."
                if self.config.preference_loss_mode == DPO_LOSS
                else (
                    "Reusable vectors are added to selected attention head slices before o_proj. "
                    "Component/head discovery supplies the selected attention sites; training then "
                    "optimizes the configured endpoint objective with vector injection rather than "
                    "zero-ablation as the causal operator. If conditioning is non-empty, the head "
                    "vectors were trained on top of the listed frozen component actuator state."
                )
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(payload, path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    heads = _parse_heads(args.heads)
    zero_heads = _parse_zero_heads(args.zero_heads, train_heads=heads)
    alphas = parse_alpha_list(args.alpha_sweep)
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
        print("[cecm-head] no validation pairs found; using train pairs for alpha reporting", flush=True)
        val_pairs = train_pairs
    frozen_component_additions: tuple[dict[str, Any], ...] = tuple()
    if args.frozen_component_actuator is not None:
        if args.frozen_component_alpha == 0:
            raise ValueError("--frozen-component-alpha must be nonzero when --frozen-component-actuator is set")
        frozen_component_additions = tuple(
            load_fixed_actuator_additions(
                args.frozen_component_actuator,
                alpha=args.frozen_component_alpha,
                apply_mode=args.frozen_component_apply_mode,
            )
        )
    background_component_additions, background_head_additions = _background_additions(args)
    frozen_component_additions = tuple([*frozen_component_additions, *background_component_additions])
    question_state_weights = parse_weight_spec(args.question_state_weights)
    objective_description = actuator_objective_description(
        state_margin_weight=args.state_margin_weight,
        gain_weight=args.gain_weight,
        preference_loss_mode=args.preference_loss_mode,
    )
    loss_description = actuator_loss_description(
        state_margin_weight=args.state_margin_weight,
        gain_weight=args.gain_weight,
        norm_label="U_h",
        preference_loss_mode=args.preference_loss_mode,
        dpo_beta=args.dpo_beta,
    )
    score_description = actuator_score_description(args.score_mode)
    discovery_alignment = (
        "Component/head discovery estimates contribution to C by ablation or scaling; "
        "head actuator training then optimizes a reference-anchored DPO preference objective "
        "over the same fixed y_plus/y_minus continuations while only updating the injected head vectors."
        if args.preference_loss_mode == DPO_LOSS
        else (
            "Component/head discovery estimates contribution to C by ablation or scaling; "
            "head actuator training optimizes the configured endpoint objective. With "
            "state_margin_weight=0 and gain_weight>0 this is pure intervention-induced "
            "competitive-margin gain relative to the unsteered model."
        )
    )

    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "event": args.event,
            "endpoint_objective": args.endpoint_objective,
            "heads": [head.head_id for head in heads],
            "train_split": args.train_split,
            "val_split": args.val_split,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "min_source_row_index": args.min_source_row_index,
            "max_source_row_index": args.max_source_row_index,
            "question_state_weights": dict(question_state_weights),
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
            "score": score_description,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "alpha_sweep": alphas,
            "frozen_component_actuator": str(args.frozen_component_actuator) if args.frozen_component_actuator else "",
            "frozen_component_alpha": args.frozen_component_alpha,
            "frozen_component_apply_mode": args.frozen_component_apply_mode,
            "frozen_component_count": len(frozen_component_additions),
            "background_component_actuators": args.background_component_actuators,
            "background_head_actuators": args.background_head_actuators,
            "background_control_parts": args.background_control_parts,
            "background_component_count": len(background_component_additions),
            "background_head_count": len(background_head_additions),
            "init_head_actuator": str(args.init_head_actuator) if args.init_head_actuator else "",
            "zero_heads": [head.head_id for head in zero_heads],
            "zero_head_apply_mode": args.zero_head_apply_mode,
            "objective": objective_description,
            "competitive_margin": competitive_margin_description(args.endpoint_objective),
            "causal_operator": (
                "zero selected attention head slices before o_proj, then inject trainable head vectors"
                if zero_heads
                else "vector injection on selected attention head slices before o_proj"
            ),
            "loss": loss_description,
            "discovery_alignment": discovery_alignment,
            "semantics_source": (
                "The trainer consumes admitted rows and selected heads. pair_margin uses the "
                "row y_plus/y_minus endpoints exactly as built. If causal_train_mask=true, "
                "token scores are computed stepwise from causal prefixes only."
            ),
        },
    )
    dump_csv(
        args.out_dir / "head_plan.csv",
        [
            {
                "head_id": head.head_id,
                "layer_idx": head.layer_idx,
                "head_idx": head.head_idx,
                "train_pairs": len(train_pairs),
                "val_pairs": len(val_pairs),
            }
            for head in heads
        ],
    )

    print(f"[cecm-head] loading model={args.model} heads={len(heads)}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    trainer = AttentionHeadActuatorTrainer(
        backend=backend,
        heads=heads,
        frozen_component_additions=frozen_component_additions,
        frozen_head_additions=background_head_additions,
        config=HeadActuatorConfig(
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
            max_aliases_per_side=args.max_aliases_per_side,
            question_state_weights=question_state_weights,
            empty_cache_every=args.empty_cache_every,
            zero_heads=zero_heads,
            zero_head_apply_mode=args.zero_head_apply_mode,
        ),
    )
    init_loaded = 0
    if args.init_head_actuator is not None:
        payload = trainer.torch.load(args.init_head_actuator, map_location="cpu")
        vectors = {str(head_id): vector for head_id, vector in dict(payload.get("vectors") or {}).items()}
        init_loaded = trainer.load_initial_vectors(vectors)
        print(
            f"[cecm-head] warm-start head actuator={args.init_head_actuator} matched_heads={init_loaded}/{len(heads)}",
            flush=True,
        )
    history, baseline_margins = trainer.train(train_pairs, val_pairs=val_pairs, out_dir=args.out_dir)
    dump_json(
        args.out_dir / "warm_start.json",
        {
            "init_head_actuator": str(args.init_head_actuator) if args.init_head_actuator else "",
            "matched_heads": init_loaded,
            "total_heads": len(heads),
        },
    )
    dump_csv(args.out_dir / "train_history.csv", history)
    dump_csv(args.out_dir / "vector_summary.csv", trainer.vector_summary_rows())
    trainer.save_payload(args.out_dir / "head_actuator.pt")

    alpha_rows: list[dict[str, object]] = []
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
    print(f"[cecm-head] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
