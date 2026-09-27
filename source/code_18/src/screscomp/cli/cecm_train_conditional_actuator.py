from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shlex
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.cecm.actuator import (
    ActuatorPair,
    filter_pairs_by_source_row_index,
    is_constant_zero_y_plus,
    is_dynamic_y_minus,
    limit_rows,
    load_actuator_pairs,
    pair_question_state_weight,
    parse_weight_spec,
    require_dynamic_answer_rest_margin,
)
from screscomp.cecm.conditional import (
    CONDITIONAL_ACTUATOR_KIND,
    ConditionalComponentAction,
    ConditionalGroupSpec,
    ConditionalHeadAction,
    conditional_gate_values,
    default_generation_apply_mode,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a sample-wise conditional actuator on top of frozen CAST vectors. "
            "Each action learns a condition vector and similarity lower/upper thresholds; "
            "group intervention triggers only when every action in the group passes its gate."
        )
    )
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, required=True)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--val-split", type=str, default="val")
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
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--lambda-alpha", type=float, default=1e-3)
    p.add_argument("--state-margin-weight", type=float, default=0.0)
    p.add_argument("--gain-weight", type=float, default=1.0)
    p.add_argument("--target-margin", type=float, default=0.0)
    p.add_argument("--target-gain", type=float, default=0.0)
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
        help="Semicolon specs group=path/to/fixed_actuator.pt",
    )
    p.add_argument(
        "--head-actuators",
        type=str,
        default="",
        help="Semicolon specs group=path/to/head_actuator.pt",
    )
    p.add_argument(
        "--component-train-apply-modes",
        type=str,
        default="",
        help="Semicolon specs group=mode. prompt_last is the usual training-time prefill proxy.",
    )
    p.add_argument(
        "--head-train-apply-modes",
        type=str,
        default="",
        help="Semicolon specs group=mode.",
    )
    p.add_argument("--component-alpha-max", type=float, default=0.1)
    p.add_argument("--head-alpha-max", type=float, default=0.1)
    p.add_argument(
        "--freeze-alpha-max",
        action="store_true",
        help="Keep each group's alpha fixed at init_alphas and only learn when to trigger it.",
    )
    p.add_argument(
        "--init-alphas",
        type=str,
        default="",
        help="Semicolon specs group=initial_alpha. Defaults to 0.5 * kind-specific alpha max.",
    )
    p.add_argument(
        "--init-direction-thresholds",
        type=str,
        default="",
        help="Semicolon specs group=initial_direction_lower_threshold in [-1, 1].",
    )
    p.add_argument(
        "--init-direction-upper-thresholds",
        type=str,
        default="",
        help="Semicolon specs group=initial_direction_upper_threshold in [-1, 1].",
    )
    p.add_argument(
        "--direction-temperature",
        type=float,
        default=8.0,
        help="Fixed sharpness for the direction-similarity threshold sigmoid.",
    )
    p.add_argument(
        "--init-projection-thresholds",
        type=str,
        default="",
        help="Semicolon specs group=initial_projection_lower_threshold on <h, v_act_unit>.",
    )
    p.add_argument(
        "--init-projection-upper-thresholds",
        type=str,
        default="",
        help="Semicolon specs group=initial_projection_upper_threshold on <h, v_act_unit>.",
    )
    p.add_argument(
        "--projection-temperature",
        type=float,
        default=1.0,
        help="Fixed sharpness for the projection threshold sigmoid.",
    )
    p.add_argument(
        "--save-gate-mode",
        type=str,
        default="hard",
        choices=["soft", "hard"],
        help="Gate mode written into the payload and used during open generation.",
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
    for part in str(raw or "").split(";"):
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
    import re

    match = re.fullmatch(r"L(\d+)\.attn\.h(\d+)", raw.strip())
    if not match:
        raise ValueError(f"Unsupported head id: {raw!r}; expected L<layer>.attn.h<head>")
    return int(match.group(1)), int(match.group(2))


def _logit_from_alpha(torch_module: Any, alpha: float, max_alpha: float, *, device: Any):
    eps = 1e-5
    if max_alpha <= 0:
        raise ValueError(f"max alpha must be positive, got {max_alpha}")
    ratio = min(max(float(alpha) / float(max_alpha), eps), 1.0 - eps)
    return torch_module.nn.Parameter(
        torch_module.tensor(math.log(ratio / (1.0 - ratio)), device=device, dtype=torch_module.float32)
    )


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def _artanh(value: float) -> float:
    value = max(min(float(value), 1.0 - 1e-6), -1.0 + 1e-6)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


def _ordered_threshold_pair(lower: float, upper: float) -> tuple[float, float]:
    low = max(min(float(lower), 1.0), -1.0)
    high = max(min(float(upper), 1.0), -1.0)
    return (min(low, high), max(low, high))


def _first_decode_steps(mode: str) -> int | None:
    import re

    match = re.fullmatch(r"first(?:_(\d+))?_decode", str(mode or ""))
    if not match:
        return None
    steps = int(match.group(1) or "1")
    if steps <= 0:
        raise ValueError(f"Unsupported conditional causal train apply mode: {mode}")
    return steps


def _mean_unit_vector(torch_module: Any, vectors: list[Any], *, device: Any):
    normalized = []
    for vector in vectors:
        value = vector.detach().to(device=device, dtype=torch_module.float32)
        norm = value.norm().clamp_min(1e-6)
        normalized.append(value / norm)
    if not normalized:
        raise ValueError("Cannot build an initial conditional direction from an empty vector list.")
    mean_vector = torch_module.stack(normalized, dim=0).mean(dim=0)
    if float(mean_vector.norm().detach().cpu().item()) < 1e-6:
        mean_vector = normalized[0]
    return mean_vector


def _sh_quote(value: str) -> str:
    return shlex.quote(str(value))


class ConditionalActuatorTrainer:
    def __init__(
        self,
        *,
        backend: TransformersABBackend,
        component_actions: list[ConditionalComponentAction],
        head_actions: list[ConditionalHeadAction],
        group_max_alphas: dict[str, float],
        init_alphas: dict[str, float],
        init_direction_thresholds: dict[str, float],
        init_direction_upper_thresholds: dict[str, float],
        init_projection_thresholds: dict[str, float],
        init_projection_upper_thresholds: dict[str, float],
        score_mode: str,
        max_aliases_per_side: int,
        option_selection_mode: str,
        lambda_alpha: float,
        question_state_weights: tuple[tuple[str, float], ...],
        state_margin_weight: float,
        gain_weight: float,
        target_margin: float,
        target_gain: float,
        lr: float,
        direction_temperature: float,
        projection_temperature: float,
        freeze_alpha_max: bool,
        save_gate_mode: str,
    ) -> None:
        self.backend = backend
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.component_actions = component_actions
        self.head_actions = head_actions
        self.group_max_alphas = {str(key): float(value) for key, value in group_max_alphas.items()}
        self.score_mode = score_mode
        self.max_aliases_per_side = max_aliases_per_side
        self.option_selection_mode = option_selection_mode
        self.lambda_alpha = float(lambda_alpha)
        self.question_state_weights = tuple((str(key), float(value)) for key, value in question_state_weights)
        self.state_margin_weight = float(state_margin_weight)
        self.gain_weight = float(gain_weight)
        self.target_margin = float(target_margin)
        self.target_gain = float(target_gain)
        self.lr = float(lr)
        self.direction_temperature = float(direction_temperature)
        self.projection_temperature = float(projection_temperature)
        self.freeze_alpha_max = bool(freeze_alpha_max)
        self.save_gate_mode = str(save_gate_mode)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        group_vectors: dict[str, list[Any]] = {}
        group_dims: dict[str, int] = {}
        for action in self.component_actions:
            dim = int(action.vector.shape[-1])
            group_vectors.setdefault(action.group, []).append(action.vector)
            if action.group in group_dims and group_dims[action.group] != dim:
                raise ValueError(f"Group {action.group!r} mixes incompatible component dims.")
            group_dims[action.group] = dim
        for action in self.head_actions:
            dim = int(action.vector.shape[-1])
            group_vectors.setdefault(action.group, []).append(action.vector)
            if action.group in group_dims and group_dims[action.group] != dim:
                raise ValueError(f"Group {action.group!r} mixes incompatible head/component dims.")
            group_dims[action.group] = dim
        if not group_vectors:
            raise ValueError("No component/head actuator actions supplied.")

        missing_max = sorted(set(group_vectors) - set(self.group_max_alphas))
        if missing_max:
            raise ValueError(f"Missing alpha max for groups: {missing_max}")

        self.alpha_logits = self.torch.nn.ParameterDict()
        self.condition_vectors = self.torch.nn.ParameterDict()
        self.similarity_lower_threshold_raw = self.torch.nn.ParameterDict()
        self.similarity_upper_threshold_raw = self.torch.nn.ParameterDict()
        self.projection_lower_threshold_raw = self.torch.nn.ParameterDict()
        self.projection_upper_threshold_raw = self.torch.nn.ParameterDict()
        self.action_param_keys: dict[str, str] = {}
        self.fixed_alphas: dict[str, float] = {}
        all_actions: list[ConditionalComponentAction | ConditionalHeadAction] = [
            *self.component_actions,
            *self.head_actions,
        ]
        for group in sorted(group_vectors):
            max_alpha = self.group_max_alphas[group]
            init_alpha = float(init_alphas.get(group, 0.5 * max_alpha))
            if self.freeze_alpha_max:
                self.fixed_alphas[group] = init_alpha
            else:
                self.alpha_logits[group] = _logit_from_alpha(self.torch, init_alpha, max_alpha, device=self.device)
        for index, action in enumerate(all_actions):
            action_key = self._action_key(action)
            param_key = f"a{index:03d}"
            self.action_param_keys[action_key] = param_key
            init_similarity_threshold_lower = float(
                init_direction_thresholds.get(action_key, init_direction_thresholds.get(action.group, 0.0))
            )
            init_similarity_threshold_upper = float(
                init_direction_upper_thresholds.get(action_key, init_direction_upper_thresholds.get(action.group, 1.0))
            )
            init_similarity_threshold_lower, init_similarity_threshold_upper = _ordered_threshold_pair(
                init_similarity_threshold_lower,
                init_similarity_threshold_upper,
            )
            init_projection_threshold_lower = float(
                init_projection_thresholds.get(action_key, init_projection_thresholds.get(action.group, -100.0))
            )
            init_projection_threshold_upper = float(
                init_projection_upper_thresholds.get(action_key, init_projection_upper_thresholds.get(action.group, 100.0))
            )
            init_projection_threshold_lower, init_projection_threshold_upper = _ordered_threshold_pair(
                init_projection_threshold_lower,
                init_projection_threshold_upper,
            )
            vector_init = action.vector.detach().to(device=self.device, dtype=self.torch.float32)
            vector_init = vector_init / vector_init.norm().clamp_min(1e-6)
            self.condition_vectors[param_key] = self.torch.nn.Parameter(vector_init.clone())
            self.similarity_lower_threshold_raw[param_key] = self.torch.nn.Parameter(
                self.torch.tensor(
                    _artanh(init_similarity_threshold_lower),
                    device=self.device,
                    dtype=self.torch.float32,
                )
            )
            self.similarity_upper_threshold_raw[param_key] = self.torch.nn.Parameter(
                self.torch.tensor(
                    _artanh(init_similarity_threshold_upper),
                    device=self.device,
                    dtype=self.torch.float32,
                )
            )
            self.projection_lower_threshold_raw[param_key] = self.torch.nn.Parameter(
                self.torch.tensor(
                    init_projection_threshold_lower,
                    device=self.device,
                    dtype=self.torch.float32,
                )
            )
            self.projection_upper_threshold_raw[param_key] = self.torch.nn.Parameter(
                self.torch.tensor(
                    init_projection_threshold_upper,
                    device=self.device,
                    dtype=self.torch.float32,
                )
            )

        self.head_dims: dict[int, int] = {}
        self.num_heads_by_layer: dict[int, int] = {}
        for action in self.head_actions:
            _hidden, num_heads, head_dim = self._head_geometry(action.layer_idx)
            if action.head_idx < 0 or action.head_idx >= num_heads:
                raise ValueError(f"Invalid {action.head_id}; L{action.layer_idx}.attn has {num_heads} heads")
            self.head_dims[action.layer_idx] = head_dim
            self.num_heads_by_layer[action.layer_idx] = num_heads

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
        raise ValueError(f"Unsupported conditional gate apply mode: {mode}")

    def _alpha_max(self, group: str):
        if self.freeze_alpha_max:
            return self.torch.tensor(self.fixed_alphas[group], device=self.device, dtype=self.torch.float32)
        return float(self.group_max_alphas[group]) * self.torch.sigmoid(self.alpha_logits[group])

    @staticmethod
    def _action_key(action: ConditionalComponentAction | ConditionalHeadAction) -> str:
        if action.vector_key:
            return str(action.vector_key)
        if isinstance(action, ConditionalComponentAction):
            return f"{action.group}:{action.component_id}"
        return f"{action.group}:{action.head_id}"

    @staticmethod
    def _summary_action_label(action: ConditionalComponentAction | ConditionalHeadAction) -> str:
        raw = ConditionalActuatorTrainer._action_key(action)
        return "".join(ch if ch.isalnum() else "_" for ch in raw)

    def _action_param_key(self, action: ConditionalComponentAction | ConditionalHeadAction) -> str:
        return self.action_param_keys[self._action_key(action)]

    def _similarity_threshold_bounds(
        self,
        action: ConditionalComponentAction | ConditionalHeadAction,
    ) -> tuple[Any, Any]:
        param_key = self._action_param_key(action)
        lower = self.torch.tanh(self.similarity_lower_threshold_raw[param_key])
        upper = self.torch.tanh(self.similarity_upper_threshold_raw[param_key])
        return (
            self.torch.minimum(lower, upper),
            self.torch.maximum(lower, upper),
        )

    def _projection_threshold_bounds(
        self,
        action: ConditionalComponentAction | ConditionalHeadAction,
    ) -> tuple[Any, Any]:
        param_key = self._action_param_key(action)
        lower = self.projection_lower_threshold_raw[param_key]
        upper = self.projection_upper_threshold_raw[param_key]
        return (
            self.torch.minimum(lower, upper),
            self.torch.maximum(lower, upper),
        )

    def current_group_specs(self) -> dict[str, ConditionalGroupSpec]:
        with self.torch.no_grad():
            return {
                group: ConditionalGroupSpec(
                    group=group,
                    alpha_max=float(self._alpha_max(group).detach().cpu().item()),
                    similarity_temperature=self.direction_temperature,
                    projection_temperature=self.projection_temperature,
                    gate_mode=self.save_gate_mode,
                )
                for group in sorted(self.group_max_alphas)
            }

    def current_action_specs(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for action in [*self.component_actions, *self.head_actions]:
            similarity_threshold_lower, similarity_threshold_upper = self._similarity_threshold_bounds(action)
            projection_threshold_lower, projection_threshold_upper = self._projection_threshold_bounds(action)
            rows.append(
                {
                    "action_key": self._action_key(action),
                    "group": action.group,
                    "kind": "component" if isinstance(action, ConditionalComponentAction) else "head",
                    "id": action.component_id if isinstance(action, ConditionalComponentAction) else action.head_id,
                    "similarity_threshold_lower": float(
                        similarity_threshold_lower.detach().cpu().item()
                    ),
                    "similarity_threshold_upper": float(
                        similarity_threshold_upper.detach().cpu().item()
                    ),
                    "similarity_temperature": self.direction_temperature,
                    "projection_threshold_lower": float(
                        projection_threshold_lower.detach().cpu().item()
                    ),
                    "projection_threshold_upper": float(
                        projection_threshold_upper.detach().cpu().item()
                    ),
                    "projection_temperature": self.projection_temperature,
                }
            )
        return rows

    def _action_gate(
        self,
        *,
        action: ConditionalComponentAction | ConditionalHeadAction,
        hidden_slice: Any,
        gate_mode: str,
    ) -> Any:
        similarity_threshold_lower, similarity_threshold_upper = self._similarity_threshold_bounds(action)
        projection_threshold_lower, projection_threshold_upper = self._projection_threshold_bounds(action)
        values = conditional_gate_values(
            self.torch,
            hidden_slice=hidden_slice,
            condition_vector=self.condition_vectors[self._action_param_key(action)],
            projection_vector=action.vector,
            similarity_threshold_lower=similarity_threshold_lower,
            similarity_threshold_upper=similarity_threshold_upper,
            similarity_temperature=self.direction_temperature,
            projection_threshold_lower=projection_threshold_lower,
            projection_threshold_upper=projection_threshold_upper,
            projection_temperature=self.projection_temperature,
            gate_mode=gate_mode,
        )
        return values["gate"]

    def _probe_group_gates(
        self,
        *,
        inputs: dict[str, Any],
        prompt_len: int,
        continuation_len: int,
        gate_mode: str,
    ) -> dict[str, Any]:
        action_gates: dict[str, Any] = {}
        handles: list[Any] = []

        def make_component_probe(action: ConditionalComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                pos = self._slice_for_mode(
                    mode=action.train_apply_mode,
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    seq_len=int(hidden.shape[1]),
                )
                action_gates[self._action_key(action)] = self._action_gate(
                    action=action,
                    hidden_slice=hidden[:, pos, :],
                    gate_mode=gate_mode,
                )
                return output

            return hook

        def make_head_probe(actions: list[ConditionalHeadAction], local_head_dim: int, local_mode: str):
            def hook(_module, inputs):
                hidden = inputs[0]
                pos = self._slice_for_mode(
                    mode=local_mode,
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    seq_len=int(hidden.shape[1]),
                )
                for action in actions:
                    start = int(action.head_idx) * local_head_dim
                    stop = start + local_head_dim
                    action_gates[self._action_key(action)] = self._action_gate(
                        action=action,
                        hidden_slice=hidden[:, pos, start:stop],
                        gate_mode=gate_mode,
                    )
                return inputs

            return hook

        for action in self.component_actions:
            module = self._component_module(action.layer_idx, action.component_type)
            handles.append(module.register_forward_hook(make_component_probe(action)))
        layer_modes = sorted({(action.layer_idx, action.train_apply_mode) for action in self.head_actions})
        for layer_idx, mode in layer_modes:
            local_actions = [
                action for action in self.head_actions if action.layer_idx == layer_idx and action.train_apply_mode == mode
            ]
            module = self._o_proj_module(layer_idx)
            handles.append(
                module.register_forward_pre_hook(make_head_probe(local_actions, self.head_dims[layer_idx], mode))
            )
        try:
            _ = self.model(**inputs, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()

        group_gates: dict[str, Any] = {}
        for group in sorted(self.group_max_alphas):
            gates = [
                action_gates[self._action_key(action)]
                for action in [*self.component_actions, *self.head_actions]
                if action.group == group
            ]
            if gates:
                group_gates[group] = self.torch.stack(gates, dim=0).prod(dim=0)
        return group_gates

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

    @staticmethod
    def _train_step_slice(*, mode: str, prompt_len: int, seq_len: int) -> slice:
        if mode in {"prompt_last", "prefill"}:
            start = max(min(prompt_len, seq_len) - 1, 0)
            return slice(start, start + 1)
        if mode == "prompt":
            return slice(0, min(prompt_len, seq_len))
        if mode == "decision_tokens":
            return slice(max(seq_len - 1, 0), seq_len)
        first_decode_steps = _first_decode_steps(mode)
        if first_decode_steps is not None:
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + first_decode_steps, seq_len)
            return slice(start, max(stop, start))
        if mode == "all":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported conditional causal train apply mode: {mode}")

    @staticmethod
    def _train_step_active(*, mode: str, target_token_index: int) -> bool:
        if mode in {"prompt_last", "prefill", "prompt"}:
            return True
        first_decode_steps = _first_decode_steps(mode)
        if first_decode_steps is not None:
            return target_token_index >= 1
        if mode in {"decision_tokens", "all"}:
            return True
        raise ValueError(f"Unsupported conditional causal train apply mode: {mode}")

    def _select_active_train_actions(
        self,
        *,
        target_token_index: int,
    ) -> tuple[tuple[ConditionalComponentAction, ...], tuple[ConditionalHeadAction, ...]]:
        group_flags: dict[str, list[bool]] = {}
        for action in [*self.component_actions, *self.head_actions]:
            group_flags.setdefault(action.group, []).append(
                self._train_step_active(mode=action.train_apply_mode, target_token_index=target_token_index)
            )
        active_groups = {
            group
            for group, flags in group_flags.items()
            if flags and all(bool(flag) for flag in flags)
        }
        return (
            tuple(action for action in self.component_actions if action.group in active_groups),
            tuple(action for action in self.head_actions if action.group in active_groups),
        )

    def _probe_group_gates_step(
        self,
        *,
        inputs: dict[str, Any],
        prompt_len: int,
        component_actions: tuple[ConditionalComponentAction, ...],
        head_actions: tuple[ConditionalHeadAction, ...],
        gate_mode: str,
    ) -> dict[str, Any]:
        action_gates: dict[str, Any] = {}
        handles: list[Any] = []

        def make_component_probe(action: ConditionalComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                pos = self._train_step_slice(
                    mode=action.train_apply_mode,
                    prompt_len=prompt_len,
                    seq_len=int(hidden.shape[1]),
                )
                action_gates[self._action_key(action)] = self._action_gate(
                    action=action,
                    hidden_slice=hidden[:, pos, :],
                    gate_mode=gate_mode,
                )
                return output

            return hook

        def make_head_probe(actions: list[ConditionalHeadAction], local_head_dim: int, local_mode: str):
            def hook(_module, inputs):
                hidden = inputs[0]
                pos = self._train_step_slice(
                    mode=local_mode,
                    prompt_len=prompt_len,
                    seq_len=int(hidden.shape[1]),
                )
                for action in actions:
                    start = int(action.head_idx) * local_head_dim
                    stop = start + local_head_dim
                    action_gates[self._action_key(action)] = self._action_gate(
                        action=action,
                        hidden_slice=hidden[:, pos, start:stop],
                        gate_mode=gate_mode,
                    )
                return inputs

            return hook

        for action in component_actions:
            module = self._component_module(action.layer_idx, action.component_type)
            handles.append(module.register_forward_hook(make_component_probe(action)))
        layer_modes = sorted({(action.layer_idx, action.train_apply_mode) for action in head_actions})
        for layer_idx, mode in layer_modes:
            local_actions = [
                action for action in head_actions if action.layer_idx == layer_idx and action.train_apply_mode == mode
            ]
            module = self._o_proj_module(layer_idx)
            handles.append(
                module.register_forward_pre_hook(make_head_probe(local_actions, self.head_dims[layer_idx], mode))
            )
        try:
            _ = self.model(**inputs, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()

        group_gates: dict[str, Any] = {}
        for group in sorted(self.group_max_alphas):
            gates = [
                action_gates[self._action_key(action)]
                for action in [*component_actions, *head_actions]
                if action.group == group
            ]
            if gates:
                group_gates[group] = self.torch.stack(gates, dim=0).prod(dim=0)
        return group_gates

    def _register_step_component_hooks(
        self,
        *,
        prompt_len: int,
        component_actions: tuple[ConditionalComponentAction, ...],
        group_gates: dict[str, Any],
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(action: ConditionalComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                pos = self._train_step_slice(
                    mode=action.train_apply_mode,
                    prompt_len=prompt_len,
                    seq_len=int(hidden_new.shape[1]),
                )
                alpha = self._alpha_max(action.group) * group_gates[action.group]
                delta = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + alpha.to(hidden_new.dtype).view(-1, 1, 1) * delta.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for action in component_actions:
            module = self._component_module(action.layer_idx, action.component_type)
            handles.append(module.register_forward_hook(make_hook(action)))
        return handles

    def _register_step_head_hooks(
        self,
        *,
        prompt_len: int,
        head_actions: tuple[ConditionalHeadAction, ...],
        group_gates: dict[str, Any],
    ) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted({(action.layer_idx, action.train_apply_mode) for action in head_actions})
        for layer_idx, mode in layer_modes:
            local_actions = [
                action for action in head_actions if action.layer_idx == layer_idx and action.train_apply_mode == mode
            ]
            module = self._o_proj_module(layer_idx)
            head_dim = self.head_dims[layer_idx]

            def make_hook(actions: list[ConditionalHeadAction], local_head_dim: int, local_mode: str):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    pos = self._train_step_slice(
                        mode=local_mode,
                        prompt_len=prompt_len,
                        seq_len=int(hidden_new.shape[1]),
                    )
                    for action in actions:
                        start = int(action.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        alpha = self._alpha_max(action.group) * group_gates[action.group]
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = (
                            hidden_new[:, pos, start:stop]
                            + alpha.to(hidden_new.dtype).view(-1, 1, 1) * vector.view(1, 1, -1)
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(local_actions, head_dim, mode)))
        return handles

    def _register_component_hooks(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        group_gates: dict[str, Any],
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(action: ConditionalComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                pos = self._slice_for_mode(
                    mode=action.train_apply_mode,
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    seq_len=int(hidden_new.shape[1]),
                )
                alpha = self._alpha_max(action.group) * group_gates[action.group]
                delta = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + alpha.to(hidden_new.dtype).view(-1, 1, 1) * delta.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for action in self.component_actions:
            module = self._component_module(action.layer_idx, action.component_type)
            handles.append(module.register_forward_hook(make_hook(action)))
        return handles

    def _register_head_hooks(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        group_gates: dict[str, Any],
    ) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted({(action.layer_idx, action.train_apply_mode) for action in self.head_actions})
        for layer_idx, mode in layer_modes:
            local_actions = [
                action for action in self.head_actions if action.layer_idx == layer_idx and action.train_apply_mode == mode
            ]
            module = self._o_proj_module(layer_idx)
            head_dim = self.head_dims[layer_idx]

            def make_hook(actions: list[ConditionalHeadAction], local_head_dim: int, local_mode: str):
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
                        alpha = self._alpha_max(action.group) * group_gates[action.group]
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = (
                            hidden_new[:, pos, start:stop]
                            + alpha.to(hidden_new.dtype).view(-1, 1, 1) * vector.view(1, 1, -1)
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(local_actions, head_dim, mode)))
        return handles

    def candidate_score(self, prompt: str, continuation: str, *, use_conditionals: bool, score_text: str = ""):
        prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
        prompt_len = int(prompt_ids.shape[-1])
        continuation_len = int(cont_ids.shape[-1])
        score_slice = score_text_token_slice(self.tokenizer, continuation, score_text)
        keep_indices = set(range(continuation_len))
        if score_slice is not None:
            keep_indices = set(range(*score_slice.indices(continuation_len)))

        step_scores: list[Any] = []
        for target_token_index in range(continuation_len):
            if target_token_index not in keep_indices:
                continue
            inputs = self._build_prefix_inputs(
                prompt_ids,
                cont_ids,
                visible_continuation_len=target_token_index,
            )
            handles: list[Any] = []
            if use_conditionals:
                active_components, active_heads = self._select_active_train_actions(
                    target_token_index=target_token_index,
                )
                if active_components or active_heads:
                    group_gates = self._probe_group_gates_step(
                        inputs=inputs,
                        prompt_len=prompt_len,
                        component_actions=active_components,
                        head_actions=active_heads,
                        gate_mode="soft",
                    )
                    handles.extend(
                        self._register_step_component_hooks(
                            prompt_len=prompt_len,
                            component_actions=active_components,
                            group_gates=group_gates,
                        )
                    )
                    handles.extend(
                        self._register_step_head_hooks(
                            prompt_len=prompt_len,
                            head_actions=active_heads,
                            group_gates=group_gates,
                        )
                    )
            try:
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :].float()
            finally:
                for handle in handles:
                    handle.remove()

            target_ids = cont_ids[:, target_token_index]
            token_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            if self.score_mode == "avglogp":
                log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)
                step_scores.append(log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1))
                continue
            if self.score_mode == "top_logit_gap":
                top_logits = logits.max(dim=-1).values
                step_scores.append(token_logits - top_logits)
                continue
            if self.score_mode == "answer_rest_margin":
                top_values, top_indices = logits.topk(k=2, dim=-1)
                top1_values = top_values[..., 0]
                top2_values = top_values[..., 1]
                top1_indices = top_indices[..., 0]
                rest_max_logits = self.torch.where(top1_indices == target_ids, top2_values, top1_values)
                step_scores.append(token_logits - rest_max_logits)
                continue
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")
        if not step_scores:
            raise ValueError(f"Empty score slice after tokenization: continuation={continuation!r} score_text={score_text!r}")
        return self.torch.stack(step_scores, dim=0).mean()

    def endpoint_score(
        self,
        prompt: str,
        continuations: tuple[str, ...],
        *,
        use_conditionals: bool,
        score_text: str = "",
    ):
        options = limit_options(
            continuations,
            max_aliases_per_side=self.max_aliases_per_side,
        )
        if not options:
            raise ValueError("empty endpoint continuation options")
        scores = [
            self.candidate_score(prompt, option, use_conditionals=use_conditionals, score_text=score_text)
            for option in options
        ]
        return select_option_score(
            self.torch,
            scores,
            selection_mode=self.option_selection_mode,
        )

    def margin(self, pair: ActuatorPair, *, use_conditionals: bool):
        require_dynamic_answer_rest_margin(self.score_mode, pair=pair)
        if is_constant_zero_y_plus(pair):
            plus = self.torch.zeros((), device=self.device)
        else:
            plus = self.endpoint_score(
                pair.prompt,
                pair.y_plus_options,
                use_conditionals=use_conditionals,
                score_text=pair.y_plus_score_text,
            )
        if is_dynamic_y_minus(pair) or not pair.y_minus_options:
            return plus
        minus = self.endpoint_score(
            pair.prompt,
            pair.y_minus_options,
            use_conditionals=use_conditionals,
            score_text=pair.y_minus_score_text,
        )
        return plus - minus

    def cache_baseline_margins(self, pairs: list[ActuatorPair]) -> dict[str, float]:
        baselines: dict[str, float] = {}
        with self.torch.no_grad():
            for pair in pairs:
                baselines[pair.sample_id] = float(self.margin(pair, use_conditionals=False).detach().cpu().item())
        return baselines

    def alpha_penalty(self):
        if self.freeze_alpha_max:
            return self.torch.tensor(0.0, device=self.device)
        values = [self._alpha_max(group) / float(self.group_max_alphas[group]) for group in self.group_max_alphas]
        if not values:
            return self.torch.tensor(0.0, device=self.device)
        return self.torch.stack([value.pow(2) for value in values]).mean()

    def _summary_row(
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
                margin = float(self.margin(pair, use_conditionals=True).detach().cpu().item())
                base_margins.append(base)
                margins.append(margin)
                gains.append(margin - base)
        row = {
            "split": split,
            "n": len(pairs),
            "base_mean_margin": mean(base_margins) if base_margins else math.nan,
            "mean_margin": mean(margins) if margins else math.nan,
            "mean_margin_gain": mean(gains) if gains else math.nan,
            "base_pref_rate": mean(float(value > 0) for value in base_margins) if base_margins else math.nan,
            "pref_rate": mean(float(value > 0) for value in margins) if margins else math.nan,
            "gain_positive_rate": mean(float(value > 0) for value in gains) if gains else math.nan,
        }
        for group, spec in self.current_group_specs().items():
            row[f"alpha_max_{group}"] = spec.alpha_max
        for action in [*self.component_actions, *self.head_actions]:
            similarity_threshold_lower, similarity_threshold_upper = self._similarity_threshold_bounds(action)
            projection_threshold_lower, projection_threshold_upper = self._projection_threshold_bounds(action)
            row[f"sim_tau_low_{self._summary_action_label(action)}"] = float(
                similarity_threshold_lower.detach().cpu().item()
            )
            row[f"sim_tau_high_{self._summary_action_label(action)}"] = float(
                similarity_threshold_upper.detach().cpu().item()
            )
            row[f"proj_tau_low_{self._summary_action_label(action)}"] = float(
                projection_threshold_lower.detach().cpu().item()
            )
            row[f"proj_tau_high_{self._summary_action_label(action)}"] = float(
                projection_threshold_upper.detach().cpu().item()
            )
        return row

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
        parameters = [
            *self.condition_vectors.parameters(),
            *self.similarity_lower_threshold_raw.parameters(),
            *self.similarity_upper_threshold_raw.parameters(),
            *self.projection_lower_threshold_raw.parameters(),
            *self.projection_upper_threshold_raw.parameters(),
        ]
        if not self.freeze_alpha_max:
            parameters.extend(self.alpha_logits.parameters())
        optimizer = self.torch.optim.AdamW(parameters, lr=self.lr)

        print(f"[cecm-cond] caching train baselines: n={len(train_pairs)}", flush=True)
        baseline_train = self.cache_baseline_margins(train_pairs)
        print(f"[cecm-cond] caching val baselines: n={len(val_pairs)}", flush=True)
        baseline_val = self.cache_baseline_margins(val_pairs)

        history: list[dict[str, object]] = []
        for epoch in range(1, epochs + 1):
            shuffled = list(train_pairs)
            rng.shuffle(shuffled)
            losses: list[float] = []
            completion_losses: list[float] = []
            gain_losses: list[float] = []
            gains: list[float] = []
            margins: list[float] = []
            for step, pair in enumerate(shuffled, start=1):
                optimizer.zero_grad(set_to_none=True)
                steered_margin = self.margin(pair, use_conditionals=True)
                base_margin = self.torch.tensor(baseline_train[pair.sample_id], device=self.device)
                gain = steered_margin - base_margin
                completion_loss = self.torch.nn.functional.softplus(
                    self.torch.tensor(self.target_margin, device=self.device) - steered_margin
                )
                gain_loss = self.torch.nn.functional.softplus(
                    self.torch.tensor(self.target_gain, device=self.device) - gain
                )
                task_loss = (
                    self.state_margin_weight * completion_loss
                    + self.gain_weight * gain_loss
                )
                pair_weight = pair_question_state_weight(pair, self.question_state_weights)
                loss = float(pair_weight) * task_loss + self.lambda_alpha * self.alpha_penalty()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                completion_losses.append(float(completion_loss.detach().cpu().item()))
                gain_losses.append(float(gain_loss.detach().cpu().item()))
                gains.append(float(gain.detach().cpu().item()))
                margins.append(float(steered_margin.detach().cpu().item()))
                if step == 1 or step % 25 == 0 or step == len(shuffled):
                    specs = self.current_group_specs()
                    group_text = " ".join(
                        f"{group}=amax:{spec.alpha_max:.4f}"
                        for group, spec in sorted(specs.items())
                    )
                    action_text = " ".join(
                        (
                            f"{self._summary_action_label(action)}:["
                            f"{float(self._similarity_threshold_bounds(action)[0].detach().cpu().item()):.4f},"
                            f"{float(self._similarity_threshold_bounds(action)[1].detach().cpu().item()):.4f}]"
                            f"/["
                            f"{float(self._projection_threshold_bounds(action)[0].detach().cpu().item()):.4f},"
                            f"{float(self._projection_threshold_bounds(action)[1].detach().cpu().item()):.4f}]"
                        )
                        for action in [*self.component_actions, *self.head_actions]
                    )
                    print(
                        (
                            f"[cecm-cond] epoch={epoch}/{epochs} step={step}/{len(shuffled)} "
                            f"loss={losses[-1]:.6f} margin={margins[-1]:.6f} gain={gains[-1]:.6f} "
                            f"{group_text} {action_text}"
                        ),
                        flush=True,
                    )
                del loss, completion_loss, gain_loss, gain, base_margin, steered_margin
                if empty_cache_every > 0 and step % empty_cache_every == 0:
                    gc.collect()
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
            history_row = {
                "epoch": epoch,
                "train_pairs": len(shuffled),
                "mean_loss": mean(losses) if losses else math.nan,
                "mean_completion_loss": mean(completion_losses) if completion_losses else math.nan,
                "mean_gain_loss": mean(gain_losses) if gain_losses else math.nan,
                "mean_margin": mean(margins) if margins else math.nan,
                "mean_margin_gain": mean(gains) if gains else math.nan,
                "alpha_penalty": float(self.alpha_penalty().detach().cpu().item()),
            }
            for group, spec in self.current_group_specs().items():
                history_row[f"alpha_max_{group}"] = spec.alpha_max
            for action in [*self.component_actions, *self.head_actions]:
                similarity_threshold_lower, similarity_threshold_upper = self._similarity_threshold_bounds(action)
                projection_threshold_lower, projection_threshold_upper = self._projection_threshold_bounds(action)
                history_row[f"sim_tau_low_{self._summary_action_label(action)}"] = float(
                    similarity_threshold_lower.detach().cpu().item()
                )
                history_row[f"sim_tau_high_{self._summary_action_label(action)}"] = float(
                    similarity_threshold_upper.detach().cpu().item()
                )
                history_row[f"proj_tau_low_{self._summary_action_label(action)}"] = float(
                    projection_threshold_lower.detach().cpu().item()
                )
                history_row[f"proj_tau_high_{self._summary_action_label(action)}"] = float(
                    projection_threshold_upper.detach().cpu().item()
                )
            history.append(history_row)

        summaries = [
            self._summary_row(train_pairs, baselines=baseline_train, split="train"),
            self._summary_row(val_pairs, baselines=baseline_val, split="val"),
        ]
        return history, summaries

    def group_summary_rows(self) -> list[dict[str, object]]:
        return [
            {
                "group": group,
                "alpha_max": spec.alpha_max,
                "similarity_temperature": spec.similarity_temperature,
                "projection_temperature": spec.projection_temperature,
                "gate_mode": spec.gate_mode,
                "group_alpha_max_cap": self.group_max_alphas[group],
            }
            for group, spec in sorted(self.current_group_specs().items())
        ]

    def save_payload(self, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        group_specs = self.current_group_specs()
        payload = {
            "payload_format_version": 7,
            "kind": CONDITIONAL_ACTUATOR_KIND,
            "score_mode": self.score_mode,
            "objective": "causal group-and conditional actuator with per-action learned condition vectors plus similarity/projection lower/upper thresholds",
            "loss": (
                "state_margin_weight*softplus(target_margin-C(M_cond;x)) "
                "+ gain_weight*softplus(target_gain-(C(M_cond;x)-C(M_full;x))) "
                + (
                    "+ lambda_alpha*mean((alpha_max/max_alpha)^2)"
                    if not self.freeze_alpha_max
                    else ""
                )
            ),
            "gate_formula": (
                "train: for each causal prefix x_{<=t}, group_gate(x_{<=t})="
                "prod_i sigmoid(temp_dir*(cos(h_i,c_i)-tau_dir_low_i))*sigmoid(temp_dir*(tau_dir_high_i-cos(h_i,c_i)))"
                "*sigmoid(temp_proj*(proj(h_i,v_i)-tau_proj_low_i))*sigmoid(temp_proj*(tau_proj_high_i-proj(h_i,v_i))); "
                "eval: group_gate(x)=prod_i 1[tau_dir_low_i<=cos(h_i,c_i)<=tau_dir_high_i and tau_proj_low_i<=proj(h_i,v_i)<=tau_proj_high_i]"
                if self.save_gate_mode == "hard"
                else (
                    "for each causal prefix x_{<=t}, group_gate(x_{<=t})="
                    "prod_i sigmoid(temp_dir*(cos(h_i,c_i)-tau_dir_low_i))*sigmoid(temp_dir*(tau_dir_high_i-cos(h_i,c_i)))"
                    "*sigmoid(temp_proj*(proj(h_i,v_i)-tau_proj_low_i))*sigmoid(temp_proj*(tau_proj_high_i-proj(h_i,v_i)))"
                )
            ),
            "alpha_mode": "fixed" if self.freeze_alpha_max else "learned",
            "gate_mode": self.save_gate_mode,
            "group_specs": {
                group: {
                    "alpha_max": spec.alpha_max,
                    "similarity_temperature": spec.similarity_temperature,
                    "projection_temperature": spec.projection_temperature,
                    "gate_mode": spec.gate_mode,
                }
                for group, spec in group_specs.items()
            },
            "condition_vectors": {
                self._action_key(action): self.condition_vectors[self._action_param_key(action)].detach().cpu()
                for action in [*self.component_actions, *self.head_actions]
            },
            "components": [
                {
                    "component_id": action.component_id,
                    "vector_key": action.vector_key or f"{action.group}:{action.component_id}",
                    "layer_idx": action.layer_idx,
                    "component_type": action.component_type,
                    "group": action.group,
                    "similarity_threshold_lower": float(
                        self._similarity_threshold_bounds(action)[0].detach().cpu().item()
                    ),
                    "similarity_threshold_upper": float(
                        self._similarity_threshold_bounds(action)[1].detach().cpu().item()
                    ),
                    "similarity_temperature": self.direction_temperature,
                    "projection_threshold_lower": float(
                        self._projection_threshold_bounds(action)[0].detach().cpu().item()
                    ),
                    "projection_threshold_upper": float(
                        self._projection_threshold_bounds(action)[1].detach().cpu().item()
                    ),
                    "projection_temperature": self.projection_temperature,
                    "train_apply_mode": action.train_apply_mode,
                    "generation_apply_mode": action.generation_apply_mode,
                }
                for action in self.component_actions
            ],
            "heads": [
                {
                    "head_id": action.head_id,
                    "vector_key": action.vector_key or f"{action.group}:{action.head_id}",
                    "layer_idx": action.layer_idx,
                    "head_idx": action.head_idx,
                    "group": action.group,
                    "similarity_threshold_lower": float(
                        self._similarity_threshold_bounds(action)[0].detach().cpu().item()
                    ),
                    "similarity_threshold_upper": float(
                        self._similarity_threshold_bounds(action)[1].detach().cpu().item()
                    ),
                    "similarity_temperature": self.direction_temperature,
                    "projection_threshold_lower": float(
                        self._projection_threshold_bounds(action)[0].detach().cpu().item()
                    ),
                    "projection_threshold_upper": float(
                        self._projection_threshold_bounds(action)[1].detach().cpu().item()
                    ),
                    "projection_temperature": self.projection_temperature,
                    "train_apply_mode": action.train_apply_mode,
                    "generation_apply_mode": action.generation_apply_mode,
                }
                for action in self.head_actions
            ],
            "vectors": {
                **{
                    (action.vector_key or f"{action.group}:{action.component_id}"): action.vector.detach().cpu()
                    for action in self.component_actions
                },
                **{
                    (action.vector_key or f"{action.group}:{action.head_id}"): action.vector.detach().cpu()
                    for action in self.head_actions
                },
            },
        }
        torch.save(payload, path)


def _load_component_actions(
    specs: dict[str, str],
    *,
    train_apply_modes: dict[str, str],
) -> tuple[list[ConditionalComponentAction], dict[str, float]]:
    import torch

    actions: list[ConditionalComponentAction] = []
    group_caps: dict[str, float] = {}
    for group, raw_path in specs.items():
        path = _resolve_payload_path(raw_path, filename="fixed_actuator.pt")
        payload = torch.load(path, map_location="cpu")
        mode = str(train_apply_modes.get(group, "prompt_last") or "prompt_last")
        for row in payload["components"]:
            component_id = str(row["component_id"])
            actions.append(
                ConditionalComponentAction(
                    component_id=component_id,
                    layer_idx=int(row["layer_idx"]),
                    component_type=str(row["component_type"]),
                    vector=payload["vectors"][component_id],
                    group=group,
                    vector_key=f"{group}:{component_id}",
                    train_apply_mode=mode,
                    generation_apply_mode=default_generation_apply_mode(mode),
                )
            )
    return actions, group_caps


def _load_head_actions(
    specs: dict[str, str],
    *,
    train_apply_modes: dict[str, str],
) -> tuple[list[ConditionalHeadAction], dict[str, float]]:
    import torch

    actions: list[ConditionalHeadAction] = []
    group_caps: dict[str, float] = {}
    for group, raw_path in specs.items():
        path = _resolve_payload_path(raw_path, filename="head_actuator.pt")
        payload = torch.load(path, map_location="cpu")
        mode = str(train_apply_modes.get(group, "all") or "all")
        for row in payload["heads"]:
            head_id = str(row["head_id"])
            layer_idx, head_idx = _parse_head_id(head_id)
            actions.append(
                ConditionalHeadAction(
                    head_id=head_id,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    vector=payload["vectors"][head_id],
                    group=group,
                    vector_key=f"{group}:{head_id}",
                    train_apply_mode=mode,
                    generation_apply_mode=default_generation_apply_mode(mode),
                )
            )
    return actions, group_caps


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = load_actuator_pairs(args.pairs_csv, event=args.event)
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
        print("[cecm-cond] no validation pairs found; using train pairs for reporting", flush=True)
        val_pairs = train_pairs

    component_specs = _parse_semicolon_map(args.component_actuators)
    head_specs = _parse_semicolon_map(args.head_actuators)
    component_train_modes = _parse_semicolon_map(args.component_train_apply_modes)
    head_train_modes = _parse_semicolon_map(args.head_train_apply_modes)
    init_alphas = {key: float(value) for key, value in _parse_semicolon_map(args.init_alphas).items()}
    init_direction_thresholds = {
        key: float(value) for key, value in _parse_semicolon_map(args.init_direction_thresholds).items()
    }
    init_direction_upper_thresholds = {
        key: float(value) for key, value in _parse_semicolon_map(args.init_direction_upper_thresholds).items()
    }
    init_projection_thresholds = {
        key: float(value) for key, value in _parse_semicolon_map(args.init_projection_thresholds).items()
    }
    init_projection_upper_thresholds = {
        key: float(value) for key, value in _parse_semicolon_map(args.init_projection_upper_thresholds).items()
    }
    question_state_weights = parse_weight_spec(args.question_state_weights)

    component_actions, _component_caps = _load_component_actions(component_specs, train_apply_modes=component_train_modes)
    head_actions, _head_caps = _load_head_actions(head_specs, train_apply_modes=head_train_modes)

    group_max_alphas = {
        **{group: float(args.component_alpha_max) for group in component_specs},
        **{group: float(args.head_alpha_max) for group in head_specs},
    }
    if not component_actions and not head_actions:
        raise SystemExit("No component/head actuator payloads provided.")

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
            "min_source_row_index": args.min_source_row_index,
            "max_source_row_index": args.max_source_row_index,
            "question_state_weights": dict(question_state_weights),
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_alpha": args.lambda_alpha,
            "state_margin_weight": args.state_margin_weight,
            "gain_weight": args.gain_weight,
            "target_margin": args.target_margin,
            "target_gain": args.target_gain,
            "score_mode": args.score_mode,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "component_actuators": component_specs,
            "head_actuators": head_specs,
            "component_train_apply_modes": component_train_modes,
            "head_train_apply_modes": head_train_modes,
            "group_max_alphas": group_max_alphas,
            "init_alphas": init_alphas,
            "init_direction_thresholds": init_direction_thresholds,
            "init_direction_upper_thresholds": init_direction_upper_thresholds,
            "direction_temperature": args.direction_temperature,
            "init_projection_thresholds": init_projection_thresholds,
            "init_projection_upper_thresholds": init_projection_upper_thresholds,
            "projection_temperature": args.projection_temperature,
            "freeze_alpha_max": args.freeze_alpha_max,
            "save_gate_mode": args.save_gate_mode,
            "objective": "causal sample-wise conditional actuator over frozen CAST vectors",
        },
    )
    dump_csv(
        args.out_dir / "conditional_plan.csv",
        [
            {
                "kind": "component",
                "group": action.group,
                "id": action.component_id,
                "layer_idx": action.layer_idx,
                "component_type": action.component_type,
                "train_apply_mode": action.train_apply_mode,
                "generation_apply_mode": action.generation_apply_mode,
                "alpha_cap": group_max_alphas[action.group],
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
                "train_apply_mode": action.train_apply_mode,
                "generation_apply_mode": action.generation_apply_mode,
                "alpha_cap": group_max_alphas[action.group],
            }
            for action in head_actions
        ],
    )

    print(
        (
            f"[cecm-cond] loading model={args.model} components={len(component_actions)} "
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
    trainer = ConditionalActuatorTrainer(
        backend=backend,
        component_actions=component_actions,
        head_actions=head_actions,
        group_max_alphas=group_max_alphas,
        init_alphas=init_alphas,
        init_direction_thresholds=init_direction_thresholds,
        init_direction_upper_thresholds=init_direction_upper_thresholds,
        init_projection_thresholds=init_projection_thresholds,
        init_projection_upper_thresholds=init_projection_upper_thresholds,
        score_mode=args.score_mode,
        max_aliases_per_side=args.max_aliases_per_side,
        option_selection_mode=args.option_selection_mode,
        lambda_alpha=args.lambda_alpha,
        question_state_weights=question_state_weights,
        state_margin_weight=args.state_margin_weight,
        gain_weight=args.gain_weight,
        target_margin=args.target_margin,
        target_gain=args.target_gain,
        lr=args.lr,
        direction_temperature=args.direction_temperature,
        projection_temperature=args.projection_temperature,
        freeze_alpha_max=args.freeze_alpha_max,
        save_gate_mode=args.save_gate_mode,
    )
    history, summaries = trainer.train(
        train_pairs,
        val_pairs=val_pairs,
        epochs=args.epochs,
        seed=args.seed,
        empty_cache_every=args.empty_cache_every,
    )

    payload_path = args.out_dir / "conditional_actuator.pt"
    trainer.save_payload(payload_path)
    dump_csv(args.out_dir / "train_history.csv", history)
    dump_csv(args.out_dir / "conditional_summary.csv", summaries)
    dump_csv(args.out_dir / "group_summary.csv", trainer.group_summary_rows())
    dump_csv(args.out_dir / "action_gate_summary.csv", trainer.current_action_specs())
    dump_json(args.out_dir / "payload_paths.json", {"conditional_actuators": {"conditional": str(payload_path)}})

    cast_control = "cast_conditional=cond:conditional"
    env_lines = [
        f"CAST_CONTROLS={_sh_quote(cast_control)}",
        f"CAST_CONDITIONAL_ACTUATORS={_sh_quote(f'conditional={payload_path}')}",
    ]
    (args.out_dir / "best_config.env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f"[cecm-cond] {cast_control}", flush=True)
    print(f"[cecm-cond] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
