from __future__ import annotations

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.actuator import load_fixed_actuator_additions
from screscomp.cecm.conditional import (
    ConditionalComponentAction,
    ConditionalGroupSpec,
    ConditionalHeadAction,
    conditional_gate_values,
    load_conditional_actuator_payload,
)
from screscomp.cecm.pairs import prompt_from_row, sample_id_from_row
from screscomp.cecm.scaling import classify_source_generation, load_open_eval_rows, summarize_generation_rows
from screscomp.data import dump_csv, dump_json, load_jsonl
from screscomp.gsm8k import (
    base_math_prompt,
    classify_gsm8k_prediction,
    cot_math_prompt,
    deepseek_math_cot_prompt,
    deepseek_math_prompt,
    extract_final_number,
    normalize_numeric_answer,
    strong_cot_prompt,
    summarize_gsm8k_rows,
    unique_numbers,
)
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class HeadScaleAction:
    layer_idx: int
    head_idx: int
    factor: float
    apply_mode: str = ""

    @property
    def head_id(self) -> str:
        return f"L{self.layer_idx}.attn.h{self.head_idx}"


@dataclass(frozen=True, slots=True)
class HeadVectorAction:
    layer_idx: int
    head_idx: int
    vector: Any
    alpha: float
    apply_mode: str = ""

    @property
    def head_id(self) -> str:
        return f"L{self.layer_idx}.attn.h{self.head_idx}"


@dataclass(frozen=True, slots=True)
class JointControl:
    name: str
    component_additions: tuple[dict[str, Any], ...]
    head_scales: tuple[HeadScaleAction, ...]
    head_vectors: tuple[HeadVectorAction, ...]
    conditional_components: tuple[ConditionalComponentAction, ...]
    conditional_heads: tuple[ConditionalHeadAction, ...]
    conditional_group_specs: dict[str, ConditionalGroupSpec]
    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Gsm8kEvalRow:
    sample_id: str
    split: str
    prompt: str
    gold_answer: str
    wrong_answers: tuple[str, ...]
    source_path: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run joint CECM controls: component vectors, head vectors, and head scaling.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--eval-open-rows", type=Path, required=True)
    p.add_argument(
        "--component-actuators",
        type=str,
        default="",
        help="Semicolon specs name=path/to/fixed_actuator.pt",
    )
    p.add_argument(
        "--component-scalings",
        type=str,
        default="",
        help=(
            "Semicolon specs name=L1.mlp:0,L2.attn:0 for component output scaling. "
            "Use with controls part comp_scale:<name>[:apply_mode]."
        ),
    )
    p.add_argument(
        "--head-actuators",
        type=str,
        default="",
        help="Semicolon specs name=path/to/head_actuator.pt",
    )
    p.add_argument(
        "--head-scalings",
        type=str,
        default="",
        help="Semicolon specs name=L9.attn.h17:0,L31.attn.h11:0",
    )
    p.add_argument(
        "--conditional-actuators",
        type=str,
        default="",
        help="Semicolon specs name=path/to/conditional_actuator.pt",
    )
    p.add_argument(
        "--controls",
        type=str,
        required=True,
        help=(
            "Semicolon controls. Example: base=;mlp=comp:prior_mlp:0.5;"
            "mlp_hs=comp:prior_mlp:0.5+head_scale:suppress;"
            "ha=head_act:suppress:0.5;cond=cond:conditional. Optional per-part timing: "
            "comp:prior_mlp:0.5:prefill+head_act:suppress:0.5:all"
        ),
    )
    p.add_argument("--generation-prompt-key", type=str, default="base_rag")
    p.add_argument("--prior-source", choices=["auto", "model_prior", "dataset_orig"], default="dataset_orig")
    p.add_argument(
        "--scoring-kind",
        choices=["source", "gsm8k"],
        default="source",
        help="source uses context/prior alias metrics; gsm8k uses final numeric answer metrics.",
    )
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=80)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--generation-apply-mode", type=str, default="prefill")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--verbose-char-threshold", type=int, default=48)
    p.add_argument(
        "--empty-cache-every",
        type=int,
        default=25,
        help="Run gc.collect() and torch.cuda.empty_cache() every N generations; 0 disables.",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush streamed generation_rows.jsonl every N new rows; 1 is safest.",
    )
    p.add_argument(
        "--summary-every",
        type=int,
        default=100,
        help="Refresh summary CSVs every N new rows; 0 only refreshes after each control and at the end.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing streamed generation rows instead of resuming them.",
    )
    p.add_argument(
        "--force-stepwise",
        action="store_true",
        help="Use the causal stepwise generation path for every control, including base.",
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


def _parse_head_id(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"L(\d+)\.attn\.h(\d+)", raw.strip())
    if not match:
        raise ValueError(f"Unsupported head id: {raw!r}; expected L<layer>.attn.h<head>")
    return int(match.group(1)), int(match.group(2))


def _parse_head_scalings(raw: str) -> dict[str, tuple[HeadScaleAction, ...]]:
    groups: dict[str, tuple[HeadScaleAction, ...]] = {}
    for name, values in _parse_semicolon_map(raw).items():
        actions: list[HeadScaleAction] = []
        for item in values.split(","):
            item = item.strip()
            if not item:
                continue
            head_id, factor_raw = item.split(":", 1)
            layer_idx, head_idx = _parse_head_id(head_id)
            actions.append(HeadScaleAction(layer_idx=layer_idx, head_idx=head_idx, factor=float(factor_raw)))
        groups[name] = tuple(actions)
    return groups


def _parse_component_scalings(raw: str) -> dict[str, tuple[dict[str, Any], ...]]:
    from screscomp.cecm.components import parse_component_id

    groups: dict[str, tuple[dict[str, Any], ...]] = {}
    for name, values in _parse_semicolon_map(raw).items():
        actions: list[dict[str, Any]] = []
        for item in values.split(","):
            item = item.strip()
            if not item:
                continue
            component_id, factor_raw = item.split(":", 1)
            parsed = parse_component_id(component_id.strip())
            actions.append(
                {
                    "component_id": parsed.component_id,
                    "layer_idx": int(parsed.layer_idx),
                    "component_type": parsed.component_type,
                    "scale_factor": float(factor_raw),
                }
            )
        groups[name] = tuple(actions)
    return groups


def _load_head_actuator(path: Path, *, alpha: float, apply_mode: str = "") -> tuple[HeadVectorAction, ...]:
    import torch

    payload = torch.load(path, map_location="cpu")
    actions: list[HeadVectorAction] = []
    for row in payload["heads"]:
        head_id = str(row["head_id"])
        vector = payload["vectors"][head_id]
        actions.append(
            HeadVectorAction(
                layer_idx=int(row["layer_idx"]),
                head_idx=int(row["head_idx"]),
                vector=vector,
                alpha=alpha,
                apply_mode=apply_mode,
            )
        )
    return tuple(actions)


def _parse_generation_apply_mode(raw: str) -> tuple[str, int | None]:
    first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
    if first_decode_match:
        steps = int(first_decode_match.group(1) or "1")
        if steps <= 0:
            raise ValueError(f"Unsupported generation_apply_mode: {raw}")
        return raw, steps
    if raw not in {"all", "all_positions", "prefill", "decode", "decision_tokens", "boxed_decision"}:
        raise ValueError(f"Unsupported generation_apply_mode: {raw}")
    return raw, None


class JointGenerationRunner:
    def __init__(self, backend: TransformersABBackend) -> None:
        self.backend = backend
        self.model = backend._model

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
    def _should_apply(*, mode: str, first_decode_steps: int | None, seq_len: int, state: dict[str, int]) -> bool:
        if mode == "boxed_decision":
            return bool(state.get("boxed_decision_active"))
        if mode == "prefill":
            return seq_len > 1
        if mode == "decode":
            return seq_len <= 1
        if mode in {"all", "all_positions", "decision_tokens"}:
            return True
        if first_decode_steps is not None:
            if seq_len > 1:
                return False
            state["decode_steps_seen"] = state.get("decode_steps_seen", 0) + 1
            return state["decode_steps_seen"] <= first_decode_steps
        return False

    @staticmethod
    def _position_slice(*, mode: str, seq_len: int, prompt_len: int | None = None) -> slice:
        if prompt_len is None:
            if mode == "all_positions":
                return slice(0, seq_len)
            return slice(max(seq_len - 1, 0), seq_len)
        if mode in {"prefill", "prompt_last", "decision_tokens"}:
            start = max(min(prompt_len, seq_len) - 1, 0)
            return slice(start, start + 1)
        if mode == "prompt":
            return slice(0, min(prompt_len, seq_len))
        first_decode_steps = _parse_generation_apply_mode(mode)[1] if mode.startswith("first") else None
        if first_decode_steps is not None:
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + first_decode_steps, seq_len)
            return slice(start, max(stop, start))
        if mode == "all_positions":
            return slice(0, seq_len)
        return slice(max(seq_len - 1, 0), seq_len)

    @staticmethod
    def _is_active_generation_step(
        *,
        mode: str,
        first_decode_steps: int | None,
        generated_tokens_so_far: int,
        boxed_decision_active: bool,
    ) -> bool:
        if mode == "boxed_decision":
            return bool(boxed_decision_active)
        if mode in {"prefill", "prompt", "prompt_last"}:
            return True
        if mode == "decode":
            return generated_tokens_so_far >= 1
        if mode in {"all", "all_positions", "decision_tokens"}:
            return True
        if first_decode_steps is not None:
            return generated_tokens_so_far >= 1
        return False

    @staticmethod
    def _uses_boxed_decision(control: JointControl, *, apply_mode: str) -> bool:
        if apply_mode == "boxed_decision":
            return True
        for addition in control.component_additions:
            if str(addition.get("apply_mode") or apply_mode) == "boxed_decision":
                return True
        for action in [*control.head_scales, *control.head_vectors]:
            if (action.apply_mode or apply_mode) == "boxed_decision":
                return True
        for action in [*control.conditional_components, *control.conditional_heads]:
            if action.generation_apply_mode == "boxed_decision":
                return True
        return False

    @staticmethod
    def _boxed_decision_active(generated_text: str) -> bool:
        text = str(generated_text or "")
        open_box = list(re.finditer(r"\\boxed\s*\{", text))
        if not open_box:
            return False
        last = open_box[-1]
        tail = text[last.end() :]
        if "}" in tail:
            return False
        return bool(re.fullmatch(r"\s*", tail))

    @staticmethod
    def _retime_boxed_control(control: JointControl, *, apply_mode: str, active: bool) -> JointControl:
        def retime_mode(mode: str) -> str | None:
            if mode != "boxed_decision":
                return mode
            return "all" if active else None

        component_additions: list[dict[str, Any]] = []
        for addition in control.component_additions:
            local_mode = str(addition.get("apply_mode") or apply_mode)
            retimed = retime_mode(local_mode)
            if retimed is None:
                continue
            component_additions.append({**addition, "apply_mode": retimed})

        head_scales: list[HeadScaleAction] = []
        for action in control.head_scales:
            retimed = retime_mode(action.apply_mode or apply_mode)
            if retimed is None:
                continue
            head_scales.append(
                HeadScaleAction(
                    layer_idx=action.layer_idx,
                    head_idx=action.head_idx,
                    factor=action.factor,
                    apply_mode=retimed,
                )
            )

        head_vectors: list[HeadVectorAction] = []
        for action in control.head_vectors:
            retimed = retime_mode(action.apply_mode or apply_mode)
            if retimed is None:
                continue
            head_vectors.append(
                HeadVectorAction(
                    layer_idx=action.layer_idx,
                    head_idx=action.head_idx,
                    vector=action.vector,
                    alpha=action.alpha,
                    apply_mode=retimed,
                )
            )

        conditional_components: list[ConditionalComponentAction] = []
        for action in control.conditional_components:
            retimed = retime_mode(action.generation_apply_mode)
            if retimed is None:
                continue
            conditional_components.append(
                ConditionalComponentAction(
                    component_id=action.component_id,
                    layer_idx=action.layer_idx,
                    component_type=action.component_type,
                    vector=action.vector,
                    group=action.group,
                    vector_key=action.vector_key,
                    condition_vector=action.condition_vector,
                    similarity_threshold_lower=action.similarity_threshold_lower,
                    similarity_threshold_upper=action.similarity_threshold_upper,
                    similarity_temperature=action.similarity_temperature,
                    projection_threshold_lower=action.projection_threshold_lower,
                    projection_threshold_upper=action.projection_threshold_upper,
                    projection_temperature=action.projection_temperature,
                    train_apply_mode=action.train_apply_mode,
                    generation_apply_mode=retimed,
                )
            )

        conditional_heads: list[ConditionalHeadAction] = []
        for action in control.conditional_heads:
            retimed = retime_mode(action.generation_apply_mode)
            if retimed is None:
                continue
            conditional_heads.append(
                ConditionalHeadAction(
                    head_id=action.head_id,
                    layer_idx=action.layer_idx,
                    head_idx=action.head_idx,
                    vector=action.vector,
                    group=action.group,
                    vector_key=action.vector_key,
                    condition_vector=action.condition_vector,
                    similarity_threshold_lower=action.similarity_threshold_lower,
                    similarity_threshold_upper=action.similarity_threshold_upper,
                    similarity_temperature=action.similarity_temperature,
                    projection_threshold_lower=action.projection_threshold_lower,
                    projection_threshold_upper=action.projection_threshold_upper,
                    projection_temperature=action.projection_temperature,
                    train_apply_mode=action.train_apply_mode,
                    generation_apply_mode=retimed,
                )
            )

        return JointControl(
            name=control.name,
            component_additions=tuple(component_additions),
            head_scales=tuple(head_scales),
            head_vectors=tuple(head_vectors),
            conditional_components=tuple(conditional_components),
            conditional_heads=tuple(conditional_heads),
            conditional_group_specs=dict(control.conditional_group_specs),
            parts=control.parts,
        )

    def _sample_next_token(
        self,
        logits,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        if not do_sample:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(float(temperature), 1e-6)
        if top_k and top_k > 0 and top_k < int(logits.shape[-1]):
            values, _indices = self.backend._torch.topk(logits, k=int(top_k), dim=-1)
            cutoff = values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = self.backend._torch.sort(logits, descending=True, dim=-1)
            sorted_probs = self.backend._torch.nn.functional.softmax(sorted_logits, dim=-1)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative > float(top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = self.backend._torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
        probs = self.backend._torch.nn.functional.softmax(logits, dim=-1)
        return self.backend._torch.multinomial(probs, num_samples=1)

    def _generate_with_boxed_decision(
        self,
        prompt: str,
        *,
        control: JointControl,
        apply_mode: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        return self._generate_stepwise(
            prompt,
            control=control,
            apply_mode=apply_mode,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def _register_step_component_hooks(
        self,
        additions: tuple[dict[str, Any], ...],
        *,
        apply_mode: str,
        prompt_len: int,
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(addition: dict[str, Any], local_apply_mode: str):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                pos = self._position_slice(
                    mode=local_apply_mode,
                    seq_len=int(hidden.shape[1]),
                    prompt_len=prompt_len,
                )
                if "scale_factor" in addition:
                    hidden_new[:, pos, :] = hidden_new[:, pos, :] * float(addition["scale_factor"])
                else:
                    delta = addition["direction"].to(device=hidden_new.device, dtype=hidden_new.dtype)
                    hidden_new[:, pos, :] = hidden_new[:, pos, :] + float(addition["alpha"]) * delta
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for addition in additions:
            module = self.backend._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            local_apply_mode = str(addition.get("apply_mode") or apply_mode)
            handles.append(
                module.register_forward_hook(
                    make_hook(addition, local_apply_mode)
                )
            )
        return handles

    def _register_step_conditional_component_hooks(
        self,
        actions: tuple[ConditionalComponentAction, ...],
        *,
        group_specs: dict[str, ConditionalGroupSpec],
        group_gates: dict[str, Any],
        prompt_len: int,
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(action: ConditionalComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                local_mode = self._conditional_generation_mode(action)
                pos = self._position_slice(
                    mode=local_mode,
                    seq_len=int(hidden.shape[1]),
                    prompt_len=prompt_len,
                )
                spec = group_specs[action.group]
                alpha = spec.alpha_max * group_gates[action.group]
                delta = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + alpha.to(hidden_new.dtype).view(-1, 1, 1) * delta.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for action in actions:
            module = self.backend._component_module(
                layer_idx=int(action.layer_idx),
                component_type=str(action.component_type),
            )
            handles.append(module.register_forward_hook(make_hook(action)))
        return handles

    def _register_step_head_hooks(
        self,
        head_scales: tuple[HeadScaleAction, ...],
        head_vectors: tuple[HeadVectorAction, ...],
        *,
        apply_mode: str,
        prompt_len: int,
    ) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted(
            {
                (action.layer_idx, action.apply_mode or apply_mode)
                for action in [*head_scales, *head_vectors]
            }
        )
        for layer_idx, local_apply_mode in layer_modes:
            scales = [
                action
                for action in head_scales
                if action.layer_idx == layer_idx and (action.apply_mode or apply_mode) == local_apply_mode
            ]
            vectors = [
                action
                for action in head_vectors
                if action.layer_idx == layer_idx and (action.apply_mode or apply_mode) == local_apply_mode
            ]
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for action in [*scales, *vectors]:
                if action.head_idx < 0 or action.head_idx >= num_heads:
                    raise ValueError(f"Invalid {action.head_id}; L{layer_idx}.attn has {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(
                local_scales: list[HeadScaleAction],
                local_vectors: list[HeadVectorAction],
                local_head_dim: int,
                hook_apply_mode: str,
            ):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    pos = self._position_slice(
                        mode=hook_apply_mode,
                        seq_len=int(hidden.shape[1]),
                        prompt_len=prompt_len,
                    )
                    for action in local_scales:
                        start = action.head_idx * local_head_dim
                        stop = start + local_head_dim
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] * float(action.factor)
                    for action in local_vectors:
                        start = action.head_idx * local_head_dim
                        stop = start + local_head_dim
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] + float(action.alpha) * vector
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(
                module.register_forward_pre_hook(
                    make_hook(scales, vectors, head_dim, local_apply_mode)
                )
            )
        return handles

    def _register_step_conditional_head_hooks(
        self,
        actions: tuple[ConditionalHeadAction, ...],
        *,
        group_specs: dict[str, ConditionalGroupSpec],
        group_gates: dict[str, Any],
        prompt_len: int,
    ) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted({(action.layer_idx, self._conditional_generation_mode(action)) for action in actions})
        for layer_idx, local_apply_mode in layer_modes:
            local_actions = [
                action
                for action in actions
                if action.layer_idx == layer_idx and self._conditional_generation_mode(action) == local_apply_mode
            ]
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for action in local_actions:
                if action.head_idx < 0 or action.head_idx >= num_heads:
                    raise ValueError(f"Invalid {action.head_id}; L{layer_idx}.attn has {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(
                actions_for_layer: list[ConditionalHeadAction],
                local_head_dim: int,
                hook_apply_mode: str,
            ):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    pos = self._position_slice(
                        mode=hook_apply_mode,
                        seq_len=int(hidden.shape[1]),
                        prompt_len=prompt_len,
                    )
                    for action in actions_for_layer:
                        start = int(action.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        spec = group_specs[action.group]
                        alpha = spec.alpha_max * group_gates[action.group]
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = (
                            hidden_new[:, pos, start:stop]
                            + alpha.to(hidden_new.dtype).view(-1, 1, 1) * vector.view(1, 1, -1)
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(
                module.register_forward_pre_hook(
                    make_hook(local_actions, head_dim, local_apply_mode)
                )
            )
        return handles

    def _select_active_component_additions(
        self,
        additions: tuple[dict[str, Any], ...],
        *,
        apply_mode: str,
        generated_tokens_so_far: int,
        boxed_decision_active: bool,
    ) -> tuple[dict[str, Any], ...]:
        active: list[dict[str, Any]] = []
        for addition in additions:
            local_apply_mode = str(addition.get("apply_mode") or apply_mode)
            _mode, first_decode_steps = _parse_generation_apply_mode(local_apply_mode)
            if self._is_active_generation_step(
                mode=local_apply_mode,
                first_decode_steps=first_decode_steps,
                generated_tokens_so_far=generated_tokens_so_far,
                boxed_decision_active=boxed_decision_active,
            ):
                active.append(addition)
        return tuple(active)

    def _select_active_head_actions(
        self,
        actions: tuple[HeadScaleAction, ...] | tuple[HeadVectorAction, ...],
        *,
        apply_mode: str,
        generated_tokens_so_far: int,
        boxed_decision_active: bool,
    ) -> tuple[Any, ...]:
        active: list[Any] = []
        for action in actions:
            local_apply_mode = action.apply_mode or apply_mode
            _mode, first_decode_steps = _parse_generation_apply_mode(local_apply_mode)
            if self._is_active_generation_step(
                mode=local_apply_mode,
                first_decode_steps=first_decode_steps,
                generated_tokens_so_far=generated_tokens_so_far,
                boxed_decision_active=boxed_decision_active,
            ):
                active.append(action)
        return tuple(active)

    def _select_active_conditional_actions(
        self,
        component_actions: tuple[ConditionalComponentAction, ...],
        head_actions: tuple[ConditionalHeadAction, ...],
        *,
        generated_tokens_so_far: int,
        boxed_decision_active: bool,
    ) -> tuple[tuple[ConditionalComponentAction, ...], tuple[ConditionalHeadAction, ...]]:
        group_flags: dict[str, list[bool]] = {}
        for action in [*component_actions, *head_actions]:
            local_mode = self._conditional_generation_mode(action)
            _mode, first_decode_steps = _parse_generation_apply_mode(local_mode)
            group_flags.setdefault(action.group, []).append(
                self._is_active_generation_step(
                    mode=local_mode,
                    first_decode_steps=first_decode_steps,
                    generated_tokens_so_far=generated_tokens_so_far,
                    boxed_decision_active=boxed_decision_active,
                )
            )
        active_groups = {
            group
            for group, flags in group_flags.items()
            if flags and all(bool(flag) for flag in flags)
        }
        return (
            tuple(action for action in component_actions if action.group in active_groups),
            tuple(action for action in head_actions if action.group in active_groups),
        )

    def _generate_stepwise(
        self,
        prompt: str,
        *,
        control: JointControl,
        apply_mode: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        tokenizer = self.backend._tokenizer
        torch = self.backend._torch
        inputs = self.backend._encode_prompt(prompt)
        input_ids = inputs["input_ids"]
        prompt_len = int(input_ids.shape[-1])
        generated_ids: list[int] = []
        context = getattr(torch, "inference_mode", torch.no_grad)
        with context():
            for _step in range(int(max_new_tokens)):
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                active = self._boxed_decision_active(generated_text)
                generated_tokens_so_far = len(generated_ids)
                active_component_additions = self._select_active_component_additions(
                    control.component_additions,
                    apply_mode=apply_mode,
                    generated_tokens_so_far=generated_tokens_so_far,
                    boxed_decision_active=active,
                )
                active_head_scales = self._select_active_head_actions(
                    control.head_scales,
                    apply_mode=apply_mode,
                    generated_tokens_so_far=generated_tokens_so_far,
                    boxed_decision_active=active,
                )
                active_head_vectors = self._select_active_head_actions(
                    control.head_vectors,
                    apply_mode=apply_mode,
                    generated_tokens_so_far=generated_tokens_so_far,
                    boxed_decision_active=active,
                )
                active_conditional_components, active_conditional_heads = self._select_active_conditional_actions(
                    control.conditional_components,
                    control.conditional_heads,
                    generated_tokens_so_far=generated_tokens_so_far,
                    boxed_decision_active=active,
                )
                attention_mask = torch.ones_like(input_ids, device=input_ids.device)
                conditional_group_gates = self._probe_conditional_group_gates(
                    inputs={"input_ids": input_ids, "attention_mask": attention_mask},
                    component_actions=active_conditional_components,
                    head_actions=active_conditional_heads,
                    group_specs=control.conditional_group_specs,
                    prompt_len=prompt_len,
                ) if (active_conditional_components or active_conditional_heads) else {}
                handles: list[Any] = []
                handles.extend(
                    self._register_step_component_hooks(
                        active_component_additions,
                        apply_mode=apply_mode,
                        prompt_len=prompt_len,
                    )
                )
                handles.extend(
                    self._register_step_head_hooks(
                        active_head_scales,
                        active_head_vectors,
                        apply_mode=apply_mode,
                        prompt_len=prompt_len,
                    )
                )
                handles.extend(
                    self._register_step_conditional_component_hooks(
                        active_conditional_components,
                        group_specs=control.conditional_group_specs,
                        group_gates=conditional_group_gates,
                        prompt_len=prompt_len,
                    )
                )
                handles.extend(
                    self._register_step_conditional_head_hooks(
                        active_conditional_heads,
                        group_specs=control.conditional_group_specs,
                        group_gates=conditional_group_gates,
                        prompt_len=prompt_len,
                    )
                )
                try:
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
                    next_token = self._sample_next_token(
                        logits[:, -1, :],
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    )
                finally:
                    for handle in handles:
                        handle.remove()
                token_id = int(next_token[0, 0].detach().cpu().item())
                generated_ids.append(token_id)
                input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
                if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                    break
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                if stop_strings and any(stop and stop in text for stop in stop_strings):
                    break
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self.backend._cut_stop_strings(text, stop_strings).strip()

    def _register_component_hooks(
        self,
        additions: tuple[dict[str, Any], ...],
        *,
        apply_mode: str,
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(addition: dict[str, Any], local_apply_mode: str):
            _mode, first_decode_steps = _parse_generation_apply_mode(local_apply_mode)
            state: dict[str, int] = {}

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = int(hidden.shape[1])
                if not self._should_apply(
                    mode=local_apply_mode,
                    first_decode_steps=first_decode_steps,
                    seq_len=seq_len,
                    state=state,
                ):
                    return output
                hidden_new = hidden.clone()
                pos = self._position_slice(mode=local_apply_mode, seq_len=seq_len)
                if "scale_factor" in addition:
                    hidden_new[:, pos, :] = hidden_new[:, pos, :] * float(addition["scale_factor"])
                else:
                    delta = addition["direction"].to(device=hidden_new.device, dtype=hidden_new.dtype)
                    hidden_new[:, pos, :] = hidden_new[:, pos, :] + float(addition["alpha"]) * delta
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for addition in additions:
            module = self.backend._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            local_apply_mode = str(addition.get("apply_mode") or apply_mode)
            handles.append(
                module.register_forward_hook(
                    make_hook(addition, local_apply_mode)
                )
            )
        return handles

    def _register_conditional_component_hooks(
        self,
        actions: tuple[ConditionalComponentAction, ...],
        *,
        group_specs: dict[str, ConditionalGroupSpec],
        group_gates: dict[str, Any],
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(action: ConditionalComponentAction):
            _mode, first_decode_steps = _parse_generation_apply_mode(action.generation_apply_mode)
            state: dict[str, int] = {}

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = int(hidden.shape[1])
                if not self._should_apply(
                    mode=action.generation_apply_mode,
                    first_decode_steps=first_decode_steps,
                    seq_len=seq_len,
                    state=state,
                ):
                    return output
                hidden_new = hidden.clone()
                pos = self._position_slice(mode=action.generation_apply_mode, seq_len=seq_len)
                spec = group_specs[action.group]
                alpha = spec.alpha_max * group_gates[action.group]
                delta = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                hidden_new[:, pos, :] = hidden_new[:, pos, :] + alpha.to(hidden_new.dtype).view(-1, 1, 1) * delta.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (hidden_new, *output[1:])
                return hidden_new

            return hook

        for action in actions:
            module = self.backend._component_module(
                layer_idx=int(action.layer_idx),
                component_type=str(action.component_type),
            )
            handles.append(module.register_forward_hook(make_hook(action)))
        return handles

    def _register_head_hooks(
        self,
        head_scales: tuple[HeadScaleAction, ...],
        head_vectors: tuple[HeadVectorAction, ...],
        *,
        apply_mode: str,
    ) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted(
            {
                (action.layer_idx, action.apply_mode or apply_mode)
                for action in [*head_scales, *head_vectors]
            }
        )
        for layer_idx, local_apply_mode in layer_modes:
            _mode, first_decode_steps = _parse_generation_apply_mode(local_apply_mode)
            scales = [
                action
                for action in head_scales
                if action.layer_idx == layer_idx and (action.apply_mode or apply_mode) == local_apply_mode
            ]
            vectors = [
                action
                for action in head_vectors
                if action.layer_idx == layer_idx and (action.apply_mode or apply_mode) == local_apply_mode
            ]
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for action in [*scales, *vectors]:
                if action.head_idx < 0 or action.head_idx >= num_heads:
                    raise ValueError(f"Invalid {action.head_id}; L{layer_idx}.attn has {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(
                local_scales: list[HeadScaleAction],
                local_vectors: list[HeadVectorAction],
                local_head_dim: int,
                hook_apply_mode: str,
                hook_first_decode_steps: int | None,
            ):
                state: dict[str, int] = {}

                def hook(_module, inputs):
                    hidden = inputs[0]
                    seq_len = int(hidden.shape[1])
                    if not self._should_apply(
                        mode=hook_apply_mode,
                        first_decode_steps=hook_first_decode_steps,
                        seq_len=seq_len,
                        state=state,
                    ):
                        return inputs
                    hidden_new = hidden.clone()
                    pos = self._position_slice(mode=hook_apply_mode, seq_len=seq_len)
                    for action in local_scales:
                        start = action.head_idx * local_head_dim
                        stop = start + local_head_dim
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] * float(action.factor)
                    for action in local_vectors:
                        start = action.head_idx * local_head_dim
                        stop = start + local_head_dim
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] + float(action.alpha) * vector
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(
                module.register_forward_pre_hook(
                    make_hook(scales, vectors, head_dim, local_apply_mode, first_decode_steps)
                )
            )
        return handles

    def _register_conditional_head_hooks(
        self,
        actions: tuple[ConditionalHeadAction, ...],
        *,
        group_specs: dict[str, ConditionalGroupSpec],
        group_gates: dict[str, Any],
    ) -> list[Any]:
        handles: list[Any] = []
        layer_modes = sorted({(action.layer_idx, action.generation_apply_mode) for action in actions})
        for layer_idx, local_apply_mode in layer_modes:
            _mode, first_decode_steps = _parse_generation_apply_mode(local_apply_mode)
            local_actions = [
                action
                for action in actions
                if action.layer_idx == layer_idx and action.generation_apply_mode == local_apply_mode
            ]
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for action in local_actions:
                if action.head_idx < 0 or action.head_idx >= num_heads:
                    raise ValueError(f"Invalid {action.head_id}; L{layer_idx}.attn has {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(
                actions_for_layer: list[ConditionalHeadAction],
                local_head_dim: int,
                hook_apply_mode: str,
                hook_first_decode_steps: int | None,
            ):
                state: dict[str, int] = {}

                def hook(_module, inputs):
                    hidden = inputs[0]
                    seq_len = int(hidden.shape[1])
                    if not self._should_apply(
                        mode=hook_apply_mode,
                        first_decode_steps=hook_first_decode_steps,
                        seq_len=seq_len,
                        state=state,
                    ):
                        return inputs
                    hidden_new = hidden.clone()
                    pos = self._position_slice(mode=hook_apply_mode, seq_len=seq_len)
                    for action in actions_for_layer:
                        start = int(action.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        spec = group_specs[action.group]
                        alpha = spec.alpha_max * group_gates[action.group]
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype)
                        hidden_new[:, pos, start:stop] = (
                            hidden_new[:, pos, start:stop]
                            + alpha.to(hidden_new.dtype).view(-1, 1, 1) * vector.view(1, 1, -1)
                        )
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(
                module.register_forward_pre_hook(
                    make_hook(local_actions, head_dim, local_apply_mode, first_decode_steps)
                )
            )
        return handles

    @staticmethod
    def _conditional_action_key(action: ConditionalComponentAction | ConditionalHeadAction) -> str:
        if action.vector_key:
            return str(action.vector_key)
        if isinstance(action, ConditionalComponentAction):
            return f"{action.group}:{action.component_id}"
        return f"{action.group}:{action.head_id}"

    @staticmethod
    def _conditional_generation_mode(action: ConditionalComponentAction | ConditionalHeadAction) -> str:
        if action.train_apply_mode == "all" and action.generation_apply_mode == "all":
            return "all_positions"
        return action.generation_apply_mode

    def _probe_conditional_group_gates(
        self,
        *,
        inputs: dict[str, Any],
        component_actions: tuple[ConditionalComponentAction, ...],
        head_actions: tuple[ConditionalHeadAction, ...],
        group_specs: dict[str, ConditionalGroupSpec],
        prompt_len: int,
    ) -> dict[str, Any]:
        torch = self.backend._torch
        action_gates: dict[str, Any] = {}
        handles: list[Any] = []

        def make_component_probe(action: ConditionalComponentAction):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                local_mode = self._conditional_generation_mode(action)
                pos = self._position_slice(
                    mode=local_mode,
                    seq_len=int(hidden.shape[1]),
                    prompt_len=prompt_len,
                )
                action_gates[self._conditional_action_key(action)] = conditional_gate_values(
                    torch,
                    hidden_slice=hidden[:, pos, :],
                    condition_vector=action.condition_vector,
                    projection_vector=action.vector,
                    similarity_threshold_lower=action.similarity_threshold_lower,
                    similarity_threshold_upper=action.similarity_threshold_upper,
                    similarity_temperature=action.similarity_temperature,
                    projection_threshold_lower=action.projection_threshold_lower,
                    projection_threshold_upper=action.projection_threshold_upper,
                    projection_temperature=action.projection_temperature,
                    gate_mode=group_specs[action.group].gate_mode,
                )["gate"]
                return output

            return hook

        def make_head_probe(actions_for_layer: list[ConditionalHeadAction], local_head_dim: int, local_mode: str):
            def hook(_module, inputs):
                hidden = inputs[0]
                pos = self._position_slice(
                    mode=local_mode,
                    seq_len=int(hidden.shape[1]),
                    prompt_len=prompt_len,
                )
                for action in actions_for_layer:
                    start = int(action.head_idx) * local_head_dim
                    stop = start + local_head_dim
                    action_gates[self._conditional_action_key(action)] = conditional_gate_values(
                        torch,
                        hidden_slice=hidden[:, pos, start:stop],
                        condition_vector=action.condition_vector,
                        projection_vector=action.vector,
                        similarity_threshold_lower=action.similarity_threshold_lower,
                        similarity_threshold_upper=action.similarity_threshold_upper,
                        similarity_temperature=action.similarity_temperature,
                        projection_threshold_lower=action.projection_threshold_lower,
                        projection_threshold_upper=action.projection_threshold_upper,
                        projection_temperature=action.projection_temperature,
                        gate_mode=group_specs[action.group].gate_mode,
                    )["gate"]
                return inputs

            return hook

        for action in component_actions:
            module = self.backend._component_module(
                layer_idx=int(action.layer_idx),
                component_type=str(action.component_type),
            )
            handles.append(module.register_forward_hook(make_component_probe(action)))
        layer_modes = sorted({(action.layer_idx, self._conditional_generation_mode(action)) for action in head_actions})
        for layer_idx, local_mode in layer_modes:
            local_actions = [
                action
                for action in head_actions
                if action.layer_idx == layer_idx and self._conditional_generation_mode(action) == local_mode
            ]
            module = self._o_proj_module(layer_idx)
            _hidden, _num_heads, head_dim = self._head_geometry(layer_idx)
            handles.append(module.register_forward_pre_hook(make_head_probe(local_actions, head_dim, local_mode)))
        try:
            _ = self.model(**inputs, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()

        group_gates: dict[str, Any] = {}
        for group in sorted(group_specs):
            gates = [
                action_gates[self._conditional_action_key(action)]
                for action in [*component_actions, *head_actions]
                if action.group == group
            ]
            if gates:
                group_gates[group] = torch.stack(gates, dim=0).prod(dim=0)
        return group_gates

    def generate(
        self,
        prompt: str,
        *,
        control: JointControl,
        apply_mode: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        force_stepwise: bool = False,
    ) -> str:
        if force_stepwise:
            return self._generate_stepwise(
                prompt,
                control=control,
                apply_mode=apply_mode,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        if self._uses_boxed_decision(control, apply_mode=apply_mode):
            return self._generate_with_boxed_decision(
                prompt,
                control=control,
                apply_mode=apply_mode,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        if control.conditional_components or control.conditional_heads:
            return self._generate_stepwise(
                prompt,
                control=control,
                apply_mode=apply_mode,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        handles: list[Any] = []
        handles.extend(self._register_component_hooks(control.component_additions, apply_mode=apply_mode))
        handles.extend(self._register_head_hooks(control.head_scales, control.head_vectors, apply_mode=apply_mode))
        inputs = self.backend._encode_prompt(prompt)
        try:
            return self.backend._generate_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            for handle in handles:
                handle.remove()

    def generate_many(
        self,
        prompts: list[str],
        *,
        control: JointControl,
        apply_mode: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        force_stepwise: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        if force_stepwise or self._uses_boxed_decision(control, apply_mode=apply_mode):
            return [
                self.generate(
                    prompt,
                    control=control,
                    apply_mode=apply_mode,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    force_stepwise=force_stepwise,
                )
                for prompt in prompts
            ]
        if control.conditional_components or control.conditional_heads:
            return [
                self.generate(
                    prompt,
                    control=control,
                    apply_mode=apply_mode,
                    max_new_tokens=max_new_tokens,
                    stop_strings=stop_strings,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    force_stepwise=True,
                )
                for prompt in prompts
            ]
        handles: list[Any] = []
        handles.extend(self._register_component_hooks(control.component_additions, apply_mode=apply_mode))
        handles.extend(self._register_head_hooks(control.head_scales, control.head_vectors, apply_mode=apply_mode))
        inputs = self.backend._encode_prompts(prompts)
        try:
            return self.backend._generate_many_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            for handle in handles:
                handle.remove()


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _component_ids(additions: tuple[dict[str, Any], ...]) -> str:
    return ",".join(str(item.get("component_id", "")) for item in additions if item.get("component_id"))


def _head_scale_text(actions: tuple[HeadScaleAction, ...]) -> str:
    return ",".join(
        f"{action.head_id}:{action.factor:g}{('@' + action.apply_mode) if action.apply_mode else ''}"
        for action in actions
    )


def _head_vector_text(actions: tuple[HeadVectorAction, ...]) -> str:
    return ",".join(
        f"{action.head_id}:{action.alpha:g}{('@' + action.apply_mode) if action.apply_mode else ''}"
        for action in actions
    )


def _component_apply_modes(additions: tuple[dict[str, Any], ...]) -> str:
    values = []
    for addition in additions:
        component_id = str(addition.get("component_id", ""))
        mode = str(addition.get("apply_mode", ""))
        if component_id and mode:
            values.append(f"{component_id}@{mode}")
    return ",".join(values)


def _conditional_group_text(group_specs: dict[str, ConditionalGroupSpec]) -> str:
    return ",".join(sorted(group_specs))


def _conditional_apply_modes(
    component_actions: tuple[ConditionalComponentAction, ...],
    head_actions: tuple[ConditionalHeadAction, ...],
) -> str:
    values = []
    for action in component_actions:
        values.append(f"{action.group}:{action.component_id}@{action.generation_apply_mode}")
    for action in head_actions:
        values.append(f"{action.group}:{action.head_id}@{action.generation_apply_mode}")
    return ",".join(values)


def _load_streamed_generation_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f"[cecm-joint] ignoring malformed streamed row line={line_number} path={path}", flush=True)
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _generation_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row.get("control_name", "")), str(row.get("sample_id", ""))


def _write_generation_summaries(
    *,
    out_dir: Path,
    eval_rows: int,
    control_count: int,
    generation_rows: list[dict[str, object]],
    scoring_kind: str = "source",
) -> None:
    if scoring_kind == "gsm8k":
        summary_rows = summarize_gsm8k_rows(generation_rows)
    else:
        summary_rows = summarize_generation_rows(generation_rows)
    dump_csv(out_dir / "generation_summary.csv", summary_rows)
    dump_csv(
        out_dir / "run_summary.csv",
        [
            {"metric": "eval_rows", "value": eval_rows},
            {"metric": "control_specs", "value": control_count},
            {"metric": "generation_rows", "value": len(generation_rows)},
        ],
    )


def _build_controls(
    *,
    raw_controls: str,
    component_paths: dict[str, Path],
    component_scalings: dict[str, tuple[dict[str, Any], ...]],
    head_paths: dict[str, Path],
    conditional_paths: dict[str, Path],
    head_scalings: dict[str, tuple[HeadScaleAction, ...]],
    apply_mode: str,
) -> list[JointControl]:
    controls: list[JointControl] = []
    for control_name, raw_parts in _parse_semicolon_map(raw_controls).items():
        component_additions: list[dict[str, Any]] = []
        head_vectors: list[HeadVectorAction] = []
        head_scales: list[HeadScaleAction] = []
        conditional_components: list[ConditionalComponentAction] = []
        conditional_heads: list[ConditionalHeadAction] = []
        conditional_group_specs: dict[str, ConditionalGroupSpec] = {}
        parts = [part.strip() for part in raw_parts.split("+") if part.strip()]
        for part in parts:
            fields = part.split(":")
            if fields[0] == "comp":
                if len(fields) not in {3, 4}:
                    raise ValueError(f"Component part must be comp:<name>:<alpha>[:apply_mode], got {part!r}")
                name = fields[1]
                alpha = float(fields[2])
                local_apply_mode = fields[3] if len(fields) == 4 else apply_mode
                if name not in component_paths:
                    raise ValueError(f"Unknown component actuator: {name}")
                component_additions.extend(
                    load_fixed_actuator_additions(component_paths[name], alpha=alpha, apply_mode=local_apply_mode)
                )
            elif fields[0] == "comp_scale":
                if len(fields) not in {2, 3}:
                    raise ValueError(f"Component scaling part must be comp_scale:<name>[:apply_mode], got {part!r}")
                name = fields[1]
                local_apply_mode = fields[2] if len(fields) == 3 else apply_mode
                if name not in component_scalings:
                    raise ValueError(f"Unknown component scaling: {name}")
                component_additions.extend(
                    {**action, "apply_mode": local_apply_mode}
                    for action in component_scalings[name]
                )
            elif fields[0] == "head_act":
                if len(fields) not in {3, 4}:
                    raise ValueError(f"Head actuator part must be head_act:<name>:<alpha>[:apply_mode], got {part!r}")
                name = fields[1]
                alpha = float(fields[2])
                local_apply_mode = fields[3] if len(fields) == 4 else apply_mode
                if name not in head_paths:
                    raise ValueError(f"Unknown head actuator: {name}")
                head_vectors.extend(_load_head_actuator(head_paths[name], alpha=alpha, apply_mode=local_apply_mode))
            elif fields[0] == "head_scale":
                if len(fields) not in {2, 3}:
                    raise ValueError(f"Head scaling part must be head_scale:<name>[:apply_mode], got {part!r}")
                name = fields[1]
                local_apply_mode = fields[2] if len(fields) == 3 else apply_mode
                if name not in head_scalings:
                    raise ValueError(f"Unknown head scaling: {name}")
                head_scales.extend(
                    HeadScaleAction(
                        layer_idx=action.layer_idx,
                        head_idx=action.head_idx,
                        factor=action.factor,
                        apply_mode=local_apply_mode,
                    )
                    for action in head_scalings[name]
                )
            elif fields[0] == "cond":
                if len(fields) != 2:
                    raise ValueError(f"Conditional part must be cond:<name>, got {part!r}")
                name = fields[1]
                if name not in conditional_paths:
                    raise ValueError(f"Unknown conditional actuator: {name}")
                payload_components, payload_heads, payload_group_specs = load_conditional_actuator_payload(conditional_paths[name])
                duplicate_groups = set(conditional_group_specs) & set(payload_group_specs)
                if duplicate_groups:
                    raise ValueError(
                        f"Conditional control {control_name!r} reuses group names {sorted(duplicate_groups)}; "
                        "merge them into one payload or rename the groups."
                    )
                conditional_components.extend(payload_components)
                conditional_heads.extend(payload_heads)
                conditional_group_specs.update(payload_group_specs)
            else:
                raise ValueError(f"Unsupported control part: {part!r}")
        controls.append(
            JointControl(
                name=control_name,
                component_additions=tuple(component_additions),
                head_scales=tuple(head_scales),
                head_vectors=tuple(head_vectors),
                conditional_components=tuple(conditional_components),
                conditional_heads=tuple(conditional_heads),
                conditional_group_specs=dict(conditional_group_specs),
                parts=tuple(parts),
            )
        )
    return controls


def _split_for_index(row_index: int, val_mod: int) -> str:
    if val_mod <= 1:
        return "train"
    return "val" if row_index % val_mod == 0 else "train"


def _gsm8k_question(row: dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt"):
        text = " ".join(str(row.get(key, "")).strip().split())
        if text:
            return text
    return ""


def _gsm8k_prompt(row: dict[str, Any], prompt_key: str) -> tuple[str, str]:
    prompt, source = prompt_from_row(row, prompt_key)
    if prompt:
        return prompt, source
    question = _gsm8k_question(row)
    if not question:
        return "", ""
    prompts = {
        "base_math": base_math_prompt(question),
        "cot_math": cot_math_prompt(question),
        "deepseek_math_cot": deepseek_math_cot_prompt(question),
        "deepseek_math": deepseek_math_prompt(question),
        "strong_cot": strong_cot_prompt(question),
    }
    return prompts.get(prompt_key, deepseek_math_cot_prompt(question)), f"virtual.{prompt_key}"


def _load_gsm8k_eval_rows(
    path: Path,
    *,
    prompt_key: str,
    split: str | None = None,
    start: int = 0,
    max_rows: int | None = None,
    val_mod: int = 5,
) -> list[Gsm8kEvalRow]:
    selected: list[Gsm8kEvalRow] = []
    for row_index, row in enumerate(load_jsonl(path)):
        row_split = _split_for_index(row_index, val_mod)
        if split and row_split != split:
            continue
        prompt, _prompt_source = _gsm8k_prompt(row, prompt_key)
        if not prompt:
            continue
        gold = (
            normalize_numeric_answer(row.get("gold_answer", ""))
            or normalize_numeric_answer(row.get("cf_answer", ""))
            or extract_final_number(row.get("gold_solution", ""))
            or extract_final_number(row.get("answer", ""))
        )
        if not gold:
            continue
        wrong_answers = unique_numbers(
            [
                *(row.get("wrong_answers", []) or []),
                *(row.get("model_prior_answers", []) or []),
                row.get("model_prior_answer", ""),
            ]
        )
        wrong_answers = [answer for answer in wrong_answers if answer != gold]
        selected.append(
            Gsm8kEvalRow(
                sample_id=sample_id_from_row(row, row_index),
                split=row_split,
                prompt=prompt,
                gold_answer=gold,
                wrong_answers=tuple(wrong_answers),
                source_path=str(path),
            )
        )
    selected = selected[start:]
    if max_rows is not None and max_rows > 0:
        selected = selected[:max_rows]
    return selected


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stop_strings = _parse_stop_strings(args.stop_strings)
    component_paths = {
        name: _resolve_payload_path(path, filename="fixed_actuator.pt")
        for name, path in _parse_semicolon_map(args.component_actuators).items()
    }
    head_paths = {
        name: _resolve_payload_path(path, filename="head_actuator.pt")
        for name, path in _parse_semicolon_map(args.head_actuators).items()
    }
    conditional_paths = {
        name: _resolve_payload_path(path, filename="conditional_actuator.pt")
        for name, path in _parse_semicolon_map(args.conditional_actuators).items()
    }
    component_scalings = _parse_component_scalings(args.component_scalings)
    head_scalings = _parse_head_scalings(args.head_scalings)
    controls = _build_controls(
        raw_controls=args.controls,
        component_paths=component_paths,
        component_scalings=component_scalings,
        head_paths=head_paths,
        conditional_paths=conditional_paths,
        head_scalings=head_scalings,
        apply_mode=args.generation_apply_mode,
    )

    print(f"[cecm-joint] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    runner = JointGenerationRunner(backend)
    if args.scoring_kind == "gsm8k":
        rows = _load_gsm8k_eval_rows(
            args.eval_open_rows,
            prompt_key=args.generation_prompt_key,
            split=None if args.split == "all" else args.split,
            start=args.start,
            max_rows=args.max_rows,
            val_mod=args.val_mod,
        )
    else:
        rows = load_open_eval_rows(
            args.eval_open_rows,
            prompt_key=args.generation_prompt_key,
            prior_source=args.prior_source,
            split=None if args.split == "all" else args.split,
            start=args.start,
            max_rows=args.max_rows,
            val_mod=args.val_mod,
        )
    if not rows:
        raise SystemExit("No eval rows selected.")

    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "eval_open_rows": str(args.eval_open_rows),
            "generation_prompt_key": args.generation_prompt_key,
            "prior_source": args.prior_source,
            "scoring_kind": args.scoring_kind,
            "eval_rows": len(rows),
            "generation_apply_mode": args.generation_apply_mode,
            "force_stepwise": args.force_stepwise,
            "component_actuators": {name: str(path) for name, path in component_paths.items()},
            "component_scalings": {
                name: _component_ids(actions)
                for name, actions in component_scalings.items()
            },
            "head_actuators": {name: str(path) for name, path in head_paths.items()},
            "conditional_actuators": {name: str(path) for name, path in conditional_paths.items()},
            "head_scalings": {name: _head_scale_text(actions) for name, actions in head_scalings.items()},
            "controls": {control.name: list(control.parts) for control in controls},
        },
    )
    dump_csv(
        args.out_dir / "control_plan.csv",
        [
            {
                "control_name": control.name,
                "parts": "+".join(control.parts),
                "components": _component_ids(control.component_additions),
                "component_apply_modes": _component_apply_modes(control.component_additions),
                "head_scalings": _head_scale_text(control.head_scales),
                "head_vectors": _head_vector_text(control.head_vectors),
                "conditional_groups": _conditional_group_text(control.conditional_group_specs),
                "conditional_apply_modes": _conditional_apply_modes(
                    control.conditional_components,
                    control.conditional_heads,
                ),
            }
            for control in controls
        ],
    )

    rows_path = args.out_dir / "generation_rows.jsonl"
    if args.overwrite and rows_path.exists():
        rows_path.unlink()

    generation_rows = _load_streamed_generation_rows(rows_path)
    completed_keys = {_generation_key(row) for row in generation_rows}
    if generation_rows:
        print(f"[cecm-joint] resuming streamed rows: n={len(generation_rows)} path={rows_path}", flush=True)

    generated_count = 0
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("a", encoding="utf-8") as stream:
        for control in controls:
            control_new_rows = 0
            for row in tqdm(rows, desc=f"generate {control.name}"):
                key = (control.name, row.sample_id)
                if key in completed_keys:
                    continue
                if (
                    control.component_additions
                    or control.head_scales
                    or control.head_vectors
                    or control.conditional_components
                    or control.conditional_heads
                ):
                    prediction = runner.generate(
                        row.prompt,
                        control=control,
                        apply_mode=args.generation_apply_mode,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        force_stepwise=args.force_stepwise,
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
                if args.scoring_kind == "gsm8k":
                    metrics = classify_gsm8k_prediction(
                        prediction,
                        gold_answer=row.gold_answer,
                        wrong_answers=row.wrong_answers,
                    )
                    answer_fields = {
                        "gold_answer": row.gold_answer,
                        "wrong_answers_json": list(row.wrong_answers),
                    }
                else:
                    metrics = classify_source_generation(
                        prediction,
                        context_answers=row.context_answers,
                        prior_answers=row.prior_answers,
                        verbose_char_threshold=args.verbose_char_threshold,
                    )
                    answer_fields = {
                        "context_answers_json": list(row.context_answers),
                        "prior_answers_json": list(row.prior_answers),
                    }
                generation_row = {
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "control_name": control.name,
                    "baseline_kind": "joint_actuator",
                    "env_kind": "context",
                    "alpha": "fixed",
                    "generation_apply_mode": args.generation_apply_mode,
                    "prediction": prediction,
                    "generation_prompt_key": args.generation_prompt_key,
                    "prior_source": args.prior_source,
                    "scoring_kind": args.scoring_kind,
                    "control_parts": "+".join(control.parts),
                    "components": _component_ids(control.component_additions),
                    "component_apply_modes": _component_apply_modes(control.component_additions),
                    "head_scalings": _head_scale_text(control.head_scales),
                    "head_vectors": _head_vector_text(control.head_vectors),
                    "conditional_groups": _conditional_group_text(control.conditional_group_specs),
                    "conditional_apply_modes": _conditional_apply_modes(
                        control.conditional_components,
                        control.conditional_heads,
                    ),
                    **answer_fields,
                    **metrics,
                }
                generation_rows.append(generation_row)
                completed_keys.add(key)
                stream.write(json.dumps(generation_row, ensure_ascii=False) + "\n")
                generated_count += 1
                control_new_rows += 1

                if args.flush_every > 0 and generated_count % args.flush_every == 0:
                    stream.flush()
                    os.fsync(stream.fileno())
                if args.summary_every > 0 and generated_count % args.summary_every == 0:
                    _write_generation_summaries(
                        out_dir=args.out_dir,
                        eval_rows=len(rows),
                        control_count=len(controls),
                        generation_rows=generation_rows,
                        scoring_kind=args.scoring_kind,
                    )
                if args.empty_cache_every > 0 and generated_count % args.empty_cache_every == 0:
                    gc.collect()
                    if backend._torch.cuda.is_available():
                        backend._torch.cuda.empty_cache()

            stream.flush()
            os.fsync(stream.fileno())
            _write_generation_summaries(
                out_dir=args.out_dir,
                eval_rows=len(rows),
                control_count=len(controls),
                generation_rows=generation_rows,
                scoring_kind=args.scoring_kind,
            )
            print(
                f"[cecm-joint] control done name={control.name} new_rows={control_new_rows} "
                f"total_rows={len(generation_rows)}",
                flush=True,
            )

    _write_generation_summaries(
        out_dir=args.out_dir,
        eval_rows=len(rows),
        control_count=len(controls),
        generation_rows=generation_rows,
        scoring_kind=args.scoring_kind,
    )
    print(f"[cecm-joint] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
