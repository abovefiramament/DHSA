from __future__ import annotations

import csv
import gc
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from screscomp.cecm.components import ComponentSpec, parse_component_id
from screscomp.cecm.objective import (
    MODEL_MAX_OPTION_SELECTION,
    OPTION_SELECTION_MODES,
    limit_options,
    option_selection_description,
    score_text_start_token_index,
    score_text_token_slice,
    select_option_score,
)


PAIR_MARGIN_OBJECTIVE = "pair_margin"
ENDPOINT_OBJECTIVES = (PAIR_MARGIN_OBJECTIVE,)
DYNAMIC_MAX_NON_GOLD_LOGIC = "dynamic_max_non_gold_logic"
DYNAMIC_Y_MINUS_MODES = (DYNAMIC_MAX_NON_GOLD_LOGIC,)
CONSTANT_ZERO_LOGIC = "constant_zero_logic"
CONSTANT_Y_PLUS_MODES = (CONSTANT_ZERO_LOGIC,)
MARGIN_GAIN_LOSS = "margin_gain"
DPO_LOSS = "dpo"
PREFERENCE_LOSS_MODES = (MARGIN_GAIN_LOSS, DPO_LOSS)


@dataclass(frozen=True, slots=True)
class ActuatorPair:
    sample_id: str
    split: str
    prompt: str
    y_plus: str
    y_minus: str
    y_plus_options: tuple[str, ...]
    y_minus_options: tuple[str, ...]
    y_plus_score_text: str = ""
    y_minus_score_text: str = ""
    y_plus_mode: str = ""
    y_minus_mode: str = ""
    pair_mode: str = ""
    question_sample_state: str = ""
    sampled_answer_correct: str = ""
    row_index: int | None = None


@dataclass(frozen=True, slots=True)
class FixedActuatorConfig:
    event: str
    endpoint_objective: str = PAIR_MARGIN_OBJECTIVE
    apply_mode: str = "decision_tokens"
    causal_train_mask: bool = False
    score_mode: str = "avglogp"
    option_selection_mode: str = MODEL_MAX_OPTION_SELECTION
    alpha_train: float = 1.0
    preference_loss_mode: str = MARGIN_GAIN_LOSS
    dpo_beta: float = 1.0
    state_margin_weight: float = 1.0
    gain_weight: float = 0.0
    target_margin: float = 0.0
    target_gain: float = 0.0
    lambda_norm: float = 1e-4
    lr: float = 5e-2
    epochs: int = 3
    train_batch_size: int = 16
    seed: int = 42
    empty_cache_every: int = 25
    max_aliases_per_side: int = 1
    question_state_weights: tuple[tuple[str, float], ...] = ()
    background_component_additions: tuple[dict[str, Any], ...] = ()
    background_head_additions: tuple[dict[str, Any], ...] = ()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_alpha_list(raw: str) -> list[float]:
    alphas = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not alphas:
        raise ValueError("alpha list is empty")
    return alphas


def parse_weight_spec(raw: str) -> tuple[tuple[str, float], ...]:
    weights: list[tuple[str, float]] = []
    for part in str(raw or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected state=weight in question-state weights, got {part!r}")
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty state name in question-state weights: {part!r}")
        weights.append((key, float(value.strip())))
    return tuple(weights)


def pair_question_state_weight(pair: ActuatorPair, weights: tuple[tuple[str, float], ...]) -> float:
    if not weights:
        return 1.0
    table = {str(key): float(value) for key, value in weights}
    return float(table.get(pair.question_sample_state, table.get("*", 1.0)))


def _first_decode_steps(mode: str) -> int | None:
    match = re.fullmatch(r"first(?:_(\d+))?_decode", str(mode or ""))
    if not match:
        return None
    steps = int(match.group(1) or "1")
    if steps <= 0:
        raise ValueError(f"Unsupported apply_mode: {mode}")
    return steps


def _parse_json_string_list(raw: str) -> tuple[str, ...]:
    if not raw:
        return tuple()
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list, got: {raw[:80]}")
    return tuple(str(item) for item in value if str(item).strip())


def _parse_optional_int(raw: str) -> int | None:
    value = str(raw or "").strip()
    if not value:
        return None
    return int(value)


def competitive_margin_description(endpoint_objective: str) -> str:
    return (
        "C(M;x) = S_M(y_plus|x) - S_M(y_minus|x) when y_minus options are present; "
        "when y_plus_mode=constant_zero_logic, S_M(y_plus|x)=0; "
        "when y_minus_mode=dynamic_max_non_gold_logic, y_minus is recomputed from the current "
        "forward pass as the maximum non-gold answer logic at the score span. With "
        "score_mode=answer_rest_margin this is gold token logic minus the current maximum "
        "non-gold vocab competitor."
    )


def actuator_score_description(score_mode: str) -> str:
    if score_mode == "avglogp":
        return (
            "S = max_alias mean_t[log p(y_plus_token_t|x,y_plus_<t>)] minus the same "
            "length-normalized continuation conditional log-probability score for y_minus"
        )
    if score_mode == "top_logit_gap":
        return "S = max_alias mean_t[logit(answer_token_t)-max_vocab_logit_t] for y_plus minus the same score for y_minus"
    if score_mode == "answer_rest_margin":
        return (
            "S = max_alias mean_t[logit(answer_token_t)-max_{v!=answer_token_t} logit(v)] "
            "for static endpoints; dynamic y_minus recomputes max non-gold logic from current logits"
        )
    raise ValueError(f"Unsupported score_mode: {score_mode!r}")


def require_dpo_conditional_probability(config: FixedActuatorConfig) -> None:
    if config.preference_loss_mode not in PREFERENCE_LOSS_MODES:
        raise ValueError(
            f"Unsupported preference_loss_mode={config.preference_loss_mode!r}; "
            f"expected one of {PREFERENCE_LOSS_MODES}"
        )
    if config.preference_loss_mode == DPO_LOSS and config.score_mode != "avglogp":
        raise ValueError(
            "preference_loss_mode='dpo' requires score_mode='avglogp' because the "
            "DPO score must be the continuation conditional log-probability; "
            f"got score_mode={config.score_mode!r}"
        )


def dpo_preference_logit(steered_margin, base_margin, *, beta: float):
    return float(beta) * (steered_margin - base_margin)


def actuator_loss_description(
    *,
    state_margin_weight: float,
    gain_weight: float,
    norm_label: str,
    preference_loss_mode: str = MARGIN_GAIN_LOSS,
    dpo_beta: float = 1.0,
) -> str:
    if preference_loss_mode == DPO_LOSS:
        return (
            f"softplus(-{dpo_beta:g}*(C(M_U;x)-C(M_full;x))) "
            f"+ lambda_norm * mean(||{norm_label}||^2)"
        )
    if preference_loss_mode != MARGIN_GAIN_LOSS:
        raise ValueError(
            f"Unsupported preference_loss_mode={preference_loss_mode!r}; expected one of {PREFERENCE_LOSS_MODES}"
        )
    state_active = abs(float(state_margin_weight)) > 1e-12
    gain_active = abs(float(gain_weight)) > 1e-12
    if state_active and not gain_active:
        return f"softplus(target_margin-C(M_U;x)) + lambda_norm * mean(||{norm_label}||^2)"
    if gain_active and not state_active:
        return (
            f"softplus(target_gain-(C(M_U;x)-C(M_full;x))) "
            f"+ lambda_norm * mean(||{norm_label}||^2)"
        )
    return (
        "state_margin_weight*softplus(target_margin-C(M_U;x)) "
        "+ gain_weight*softplus(target_gain-(C(M_U;x)-C(M_full;x))) "
        f"+ lambda_norm * mean(||{norm_label}||^2)"
    )


def actuator_objective_description(
    *,
    state_margin_weight: float,
    gain_weight: float,
    preference_loss_mode: str = MARGIN_GAIN_LOSS,
) -> str:
    if preference_loss_mode == DPO_LOSS:
        return "reference-anchored DPO preference objective over continuation conditional probability"
    if preference_loss_mode != MARGIN_GAIN_LOSS:
        raise ValueError(
            f"Unsupported preference_loss_mode={preference_loss_mode!r}; expected one of {PREFERENCE_LOSS_MODES}"
        )
    state_active = abs(float(state_margin_weight)) > 1e-12
    gain_active = abs(float(gain_weight)) > 1e-12
    if state_active and not gain_active:
        return "target competitive-margin state objective"
    if gain_active and not state_active:
        return "pure intervention-induced competitive-margin gain objective"
    if state_active and gain_active:
        return "target competitive-margin state objective plus intervention-induced margin gain"
    return "norm-only diagnostic objective"


def is_dynamic_y_minus(pair: ActuatorPair) -> bool:
    return pair.y_minus_mode in DYNAMIC_Y_MINUS_MODES


def is_constant_zero_y_plus(pair: ActuatorPair) -> bool:
    return pair.y_plus_mode in CONSTANT_Y_PLUS_MODES


def require_dynamic_answer_rest_margin(score_mode: str, *, pair: ActuatorPair) -> None:
    if is_dynamic_y_minus(pair) and score_mode != "answer_rest_margin":
        raise ValueError(
            "y_minus_mode=dynamic_max_non_gold_logic requires score_mode='answer_rest_margin' "
            f"for sample_id={pair.sample_id!r}; got {score_mode!r}"
        )


def load_actuator_pairs(
    path: Path,
    *,
    event: str | None = None,
    endpoint_objective: str = PAIR_MARGIN_OBJECTIVE,
) -> list[ActuatorPair]:
    if endpoint_objective not in ENDPOINT_OBJECTIVES:
        raise ValueError(f"Unsupported endpoint_objective={endpoint_objective!r}; expected one of {ENDPOINT_OBJECTIVES}")
    rows = _read_csv(path)
    pairs: list[ActuatorPair] = []
    for row in rows:
        if row.get("admitted", "1") in {"0", "false", "False"}:
            continue
        if event and row.get("event") and row.get("event") != event:
            continue
        prompt = row.get("prompt", "")
        y_plus = row.get("y_plus_continuation") or row.get("y_plus") or ""
        y_minus = row.get("y_minus_continuation") or row.get("y_minus") or ""
        y_plus_options = _parse_json_string_list(row.get("y_plus_continuations_json", ""))
        y_minus_options = _parse_json_string_list(row.get("y_minus_continuations_json", ""))
        y_plus_score_text = row.get("y_plus_score_text") or row.get("score_text") or ""
        y_minus_score_text = row.get("y_minus_score_text") or ""
        y_plus_mode = row.get("y_plus_mode") or row.get("y_plus_endpoint_type") or ""
        y_minus_mode = row.get("y_minus_mode") or row.get("y_minus_endpoint_type") or ""
        if y_plus_mode in CONSTANT_Y_PLUS_MODES:
            y_plus_options = tuple()
        elif not y_plus_options:
            y_plus_options = (y_plus,)
        if y_minus_mode in DYNAMIC_Y_MINUS_MODES:
            y_minus_options = tuple()
        elif not y_minus_options and y_minus:
            y_minus_options = (y_minus,)
        if not prompt or (not y_plus and y_plus_mode not in CONSTANT_Y_PLUS_MODES):
            continue
        pairs.append(
            ActuatorPair(
                sample_id=row.get("sample_id", ""),
                split=row.get("split", "train"),
                prompt=prompt,
                y_plus=y_plus,
                y_minus=y_minus,
                y_plus_options=y_plus_options,
                y_minus_options=y_minus_options,
                y_plus_score_text=y_plus_score_text,
                y_minus_score_text=y_minus_score_text,
                y_plus_mode=y_plus_mode,
                y_minus_mode=y_minus_mode,
                pair_mode=row.get("pair_mode", ""),
                question_sample_state=row.get("question_sample_state", ""),
                sampled_answer_correct=row.get("sampled_answer_correct", ""),
                row_index=_parse_optional_int(row.get("row_index", "")),
            )
        )
    return pairs


def filter_pairs_by_source_row_index(
    rows: list[ActuatorPair],
    *,
    min_row_index: int | None = None,
    max_row_index: int | None = None,
) -> list[ActuatorPair]:
    if min_row_index is None and max_row_index is None:
        return rows
    out: list[ActuatorPair] = []
    for pair in rows:
        if pair.row_index is None:
            continue
        if min_row_index is not None and pair.row_index < min_row_index:
            continue
        if max_row_index is not None and pair.row_index > max_row_index:
            continue
        out.append(pair)
    return out


def load_component_specs(path: Path) -> list[ComponentSpec]:
    rows = _read_csv(path)
    specs: list[ComponentSpec] = []
    for row in rows:
        component_id = row.get("component_id", "")
        if not component_id:
            continue
        parsed = parse_component_id(component_id)
        specs.append(
            ComponentSpec(
                component_id=parsed.component_id,
                layer_idx=int(row.get("layer_idx") or parsed.layer_idx),
                component_type=row.get("component_type") or parsed.component_type,
            )
        )
    if not specs:
        raise ValueError(f"No components found in {path}")
    return specs


def limit_rows(rows: list[ActuatorPair], limit: int | None) -> list[ActuatorPair]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def load_fixed_actuator_additions(path: Path, *, alpha: float = 1.0, apply_mode: str = "decision_tokens"):
    import torch

    payload = torch.load(path, map_location="cpu")
    additions = []
    for row in payload["components"]:
        vector = payload["vectors"][row["component_id"]]
        additions.append(
            {
                "component_id": row["component_id"],
                "layer_idx": int(row["layer_idx"]),
                "component_type": row["component_type"],
                "direction": vector,
                "alpha": alpha,
                "apply_mode": apply_mode,
            }
        )
    return additions


def load_fixed_actuator_vectors(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    vectors = dict(payload.get("vectors") or {})
    return {str(component_id): vector for component_id, vector in vectors.items()}


def load_head_actuator_additions(path: Path, *, alpha: float = 1.0, apply_mode: str = "decision_tokens"):
    import torch

    payload = torch.load(path, map_location="cpu")
    additions = []
    for row in payload["heads"]:
        head_id = str(row["head_id"])
        vector = payload["vectors"][head_id]
        additions.append(
            {
                "head_id": head_id,
                "layer_idx": int(row["layer_idx"]),
                "head_idx": int(row["head_idx"]),
                "direction": vector,
                "alpha": alpha,
                "apply_mode": apply_mode,
            }
        )
    return additions


class FixedActuatorTrainer:
    def __init__(
        self,
        *,
        backend: Any,
        components: list[ComponentSpec],
        config: FixedActuatorConfig,
    ) -> None:
        self.backend = backend
        self.components = components
        self.config = config
        require_dpo_conditional_probability(config)
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.model.eval()
        self.background_component_additions = tuple(config.background_component_additions)
        self.background_head_additions = tuple(config.background_head_additions)
        for param in self.model.parameters():
            param.requires_grad_(False)

        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or getattr(self.model.config, "n_embd", 0))
        if hidden_size <= 0:
            raise ValueError("Could not infer hidden size from model config.")

        self.vectors = self.torch.nn.ParameterDict()
        for component in components:
            key = self._param_key(component.component_id)
            self.vectors[key] = self.torch.nn.Parameter(
                self.torch.zeros(hidden_size, device=self.device, dtype=self.torch.float32)
            )

    def load_initial_vectors(self, vectors: dict[str, Any]) -> int:
        loaded = 0
        with self.torch.no_grad():
            for component in self.components:
                source = vectors.get(component.component_id)
                if source is None:
                    continue
                target = self._vector_for(component)
                source_tensor = self.torch.as_tensor(source, dtype=target.dtype, device=self.device)
                if tuple(source_tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Warm-start shape mismatch for {component.component_id}: "
                        f"expected {tuple(target.shape)} got {tuple(source_tensor.shape)}"
                    )
                target.copy_(source_tensor)
                loaded += 1
        return loaded

    @staticmethod
    def _param_key(component_id: str) -> str:
        return component_id.replace(".", "__")

    def _vector_for(self, component: ComponentSpec):
        return self.vectors[self._param_key(component.component_id)]

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

    def _encode_prompt_and_continuation(self, prompt: str, continuation: str) -> tuple[Any, int, int]:
        prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
        input_ids = self.torch.cat([prompt_ids, cont_ids], dim=-1).to(self.device)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        return inputs, int(prompt_ids.shape[-1]), int(cont_ids.shape[-1])

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

    @staticmethod
    def _background_slice_for_apply_mode(
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
        raise ValueError(f"Unsupported background apply_mode: {mode}")

    def _register_background_component_hooks(
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
                pos = self._background_slice_for_apply_mode(
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

        for addition in self.background_component_additions:
            module = self.backend._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            handles.append(module.register_forward_hook(make_hook(addition)))
        return handles

    def _register_background_head_hooks(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles: list[Any] = []
        by_layer: dict[int, list[dict[str, Any]]] = {}
        for addition in self.background_head_additions:
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
                        pos = self._background_slice_for_apply_mode(
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

    def _register_hooks(
        self,
        *,
        alpha: float,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles = []

        def make_hook(component: ComponentSpec):
            vector = self._vector_for(component)

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                pos = self._slice_for_apply_mode(
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    seq_len=int(hidden_new.shape[1]),
                    continuation=continuation,
                    score_text=score_text,
                )
                delta = vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + float(alpha) * delta
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for component in self.components:
            module = self.backend._component_module(
                layer_idx=int(component.layer_idx),
                component_type=str(component.component_type),
            )
            handles.append(module.register_forward_hook(make_hook(component)))
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
        if self.background_component_additions:
            handles.extend(
                self._register_background_component_hooks(
                    prompt_len=prompt_len,
                    continuation_len=continuation_len,
                    continuation=continuation,
                    score_text=score_text,
                )
            )
        if self.background_head_additions:
            handles.extend(
                self._register_background_head_hooks(
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
        pred_logits = logits[:, prompt_len - 1 : prompt_len + continuation_len - 1, :]
        pred_logits = pred_logits.float()
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

    def avg_logprob(self, prompt: str, continuation: str, *, alpha: float | None = None):
        if self.config.score_mode != "avglogp":
            raise ValueError("avg_logprob is only available when score_mode='avglogp'")
        return self.candidate_score(prompt, continuation, alpha=alpha)

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
            self.candidate_score(prompt, continuation, alpha=alpha, score_text=score_text)
            for continuation in options
        ]
        return select_option_score(
            self.torch,
            scores,
            selection_mode=self.config.option_selection_mode,
        )

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
        if self.background_component_additions or self.background_head_additions:
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
    ) -> Any:
        batch_size = len(prompt_lens)
        max_seq_len = max(seq_lens)
        mask = self.torch.zeros((batch_size, max_seq_len), dtype=self.torch.bool, device=self.device)
        for row_idx in range(batch_size):
            pos = self._slice_for_apply_mode(
                prompt_len=prompt_lens[row_idx],
                continuation_len=continuation_lens[row_idx],
                seq_len=seq_lens[row_idx],
                continuation=continuations[row_idx],
                score_text=score_texts[row_idx],
            )
            mask[row_idx, pos] = True
        return mask

    def _register_hooks_batch(self, *, alpha: float, position_mask: Any) -> list[Any]:
        handles: list[Any] = []
        mask = position_mask

        def make_hook(component: ComponentSpec):
            vector = self._vector_for(component)

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                delta = vector.to(device=hidden_new.device, dtype=hidden_new.dtype).view(1, 1, -1)
                local_mask = mask.to(device=hidden_new.device, dtype=hidden_new.dtype).unsqueeze(-1)
                hidden_new = hidden_new + float(alpha) * local_mask * delta
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for component in self.components:
            module = self.backend._component_module(
                layer_idx=int(component.layer_idx),
                component_type=str(component.component_type),
            )
            handles.append(module.register_forward_hook(make_hook(component)))
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
        plus_examples = [
            (pair.prompt, pair.y_plus_options[0], pair.y_plus_score_text)
            for pair in pairs
        ]
        minus_examples = [
            (pair.prompt, pair.y_minus_options[0], pair.y_minus_score_text)
            for pair in pairs
        ]
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
            if self.background_component_additions:
                handles.extend(
                    self._register_background_component_hooks(
                        prompt_len=prompt_len,
                        continuation_len=visible_continuation_len,
                        continuation=continuation,
                        score_text=score_text,
                    )
                )
            if self.background_head_additions:
                handles.extend(
                    self._register_background_head_hooks(
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
                    self._memory_cleanup()
        return baselines

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
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(self.config.seed)
        optimizer = self.torch.optim.AdamW(self.vectors.parameters(), lr=self.config.lr)

        print(f"[cecm] caching train baselines: n={len(train_pairs)}", flush=True)
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
            improvements: list[float] = []
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
                    improvements.extend(gain_batch.detach().cpu().tolist())
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
                    improvements.append(float(gain.detach().cpu().item()))
                processed_pairs = step
                if step == len(batch_pairs) or step % 25 < len(batch_pairs) or step == len(shuffled):
                    progress_suffix = (
                        f" dpo_logit={preference_logits[-1]:.6f}"
                        if preference_logits
                        else ""
                    )
                    print(
                        (
                            f"[cecm] epoch={epoch}/{self.config.epochs} "
                            f"step={step}/{len(shuffled)} loss={losses[-1]:.6f} "
                            f"margin={margins[-1]:.6f} gain={improvements[-1]:.6f}"
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
                    "mean_loss": sum(losses) / len(losses) if losses else math.nan,
                    "mean_completion_loss": sum(completion_losses) / len(completion_losses)
                    if completion_losses
                    else math.nan,
                    "mean_gain_loss": sum(gain_losses) / len(gain_losses) if gain_losses else math.nan,
                    "mean_preference_loss": sum(preference_losses) / len(preference_losses)
                    if preference_losses
                    else math.nan,
                    "mean_preference_logit": sum(preference_logits) / len(preference_logits)
                    if preference_logits
                    else math.nan,
                    "mean_margin": sum(margins) / len(margins) if margins else math.nan,
                    "mean_margin_gain": sum(improvements) / len(improvements) if improvements else math.nan,
                    "state_complete_rate": sum(
                        1 for value in margins if value > float(self.config.target_margin)
                    )
                    / len(margins)
                    if margins
                    else math.nan,
                    "gain_complete_rate": sum(
                        1 for value in improvements if value > float(self.config.target_gain)
                    )
                    / len(improvements)
                    if improvements
                    else math.nan,
                    "dpo_win_rate": sum(1 for value in preference_logits if value > 0)
                    / len(preference_logits)
                    if preference_logits
                    else math.nan,
                    "norm_penalty": float(self.norm_penalty().detach().cpu().item()),
                }
            )
        print(f"[cecm] caching val baselines: n={len(val_pairs)}", flush=True)
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
            "base_mean_margin": sum(base_margins) / n if n else math.nan,
            "mean_margin": sum(margins) / n if n else math.nan,
            "mean_margin_gain": sum(gains) / n if n else math.nan,
            "mean_preference_logit": sum(preference_logits) / n if preference_logits else math.nan,
            "base_pref_rate": sum(1 for value in base_margins if value > 0) / n if n else math.nan,
            "pref_rate": sum(1 for value in margins if value > 0) / n if n else math.nan,
            "gain_positive_rate": sum(1 for value in gains if value > 0) / n if n else math.nan,
            "dpo_win_rate": sum(1 for value in preference_logits if value > 0) / n
            if preference_logits
            else math.nan,
        }

    def vector_summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with self.torch.no_grad():
            for component in self.components:
                vector = self._vector_for(component)
                rows.append(
                    {
                        "component_id": component.component_id,
                        "layer_idx": component.layer_idx,
                        "component_type": component.component_type,
                        "l2_norm": float(vector.float().norm().detach().cpu().item()),
                        "mean_abs": float(vector.float().abs().mean().detach().cpu().item()),
                        "max_abs": float(vector.float().abs().max().detach().cpu().item()),
                    }
                )
        return rows

    def save_payload(self, path: Path) -> None:
        vectors = {
            component.component_id: self._vector_for(component).detach().cpu()
            for component in self.components
        }
        payload = {
            "event": self.config.event,
            "endpoint_objective": self.config.endpoint_objective,
            "apply_mode": self.config.apply_mode,
            "score_mode": self.config.score_mode,
            "option_selection_mode": self.config.option_selection_mode,
            "alpha_train": self.config.alpha_train,
            "preference_loss_mode": self.config.preference_loss_mode,
            "dpo_beta": self.config.dpo_beta,
            "objective": actuator_objective_description(
                state_margin_weight=self.config.state_margin_weight,
                gain_weight=self.config.gain_weight,
                preference_loss_mode=self.config.preference_loss_mode,
            ),
            "competitive_margin": competitive_margin_description(self.config.endpoint_objective),
            "option_selection": option_selection_description(self.config.option_selection_mode),
            "causal_operator": "vector injection on selected residual-write components",
            "loss": actuator_loss_description(
                state_margin_weight=self.config.state_margin_weight,
                gain_weight=self.config.gain_weight,
                norm_label="U_c",
                preference_loss_mode=self.config.preference_loss_mode,
                dpo_beta=self.config.dpo_beta,
            ),
            "state_margin_weight": self.config.state_margin_weight,
            "gain_weight": self.config.gain_weight,
            "target_margin": self.config.target_margin,
            "target_gain": self.config.target_gain,
            "max_aliases_per_side": self.config.max_aliases_per_side,
            "score": actuator_score_description(self.config.score_mode),
            "discovery_alignment": (
                "Component discovery supplies the selected residual-write sites. Training then "
                "optimizes reference-anchored endpoint preference over the same fixed y_plus/y_minus "
                "pairs by changing only the injected residual vectors."
                if self.config.preference_loss_mode == DPO_LOSS
                else (
                    "Component discovery supplies the selected residual-write sites. Training then "
                    "optimizes the configured endpoint objective with vector injection: "
                    "gain(U,x)=C(M_U;x)-C(M_full;x)."
                )
            ),
            "guard_policy": "health/form/neither are validation reports, not training-loss terms",
            "background_component_count": len(self.background_component_additions),
            "background_head_count": len(self.background_head_additions),
            "background_controls": {
                "components": [
                    {
                        "component_id": str(addition.get("component_id", "")),
                        "layer_idx": int(addition["layer_idx"]),
                        "component_type": str(addition["component_type"]),
                        "alpha": float(addition["alpha"]),
                        "apply_mode": str(addition.get("apply_mode", "")),
                    }
                    for addition in self.background_component_additions
                ],
                "heads": [
                    {
                        "head_id": str(addition.get("head_id", "")),
                        "layer_idx": int(addition["layer_idx"]),
                        "head_idx": int(addition["head_idx"]),
                        "alpha": float(addition["alpha"]),
                        "apply_mode": str(addition.get("apply_mode", "")),
                    }
                    for addition in self.background_head_additions
                ],
            },
            "components": [
                {
                    "component_id": component.component_id,
                    "layer_idx": component.layer_idx,
                    "component_type": component.component_type,
                }
                for component in self.components
            ],
            "vectors": vectors,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(payload, path)
