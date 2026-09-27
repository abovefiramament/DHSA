from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from screscomp.cecm.actuator import (
    ActuatorPair,
    is_constant_zero_y_plus,
    is_dynamic_y_minus,
    require_dynamic_answer_rest_margin,
)
from screscomp.cecm.components import ComponentSpec
from screscomp.cecm.objective import (
    MODEL_MAX_OPTION_SELECTION,
    limit_options,
    score_text_start_token_index,
    score_text_token_slice,
    select_option_score,
)


@dataclass(frozen=True, slots=True)
class ContributionScanConfig:
    event: str
    apply_mode: str = "decision_tokens"
    max_aliases_per_side: int = 0
    score_mode: str = "avglogp"
    option_selection_mode: str = MODEL_MAX_OPTION_SELECTION


def all_component_specs(num_layers: int, component_types: Iterable[str]) -> list[ComponentSpec]:
    specs: list[ComponentSpec] = []
    for layer_idx in range(num_layers):
        for component_type in component_types:
            component_type = component_type.strip()
            if not component_type:
                continue
            specs.append(
                ComponentSpec(
                    component_id=f"L{layer_idx}.{component_type}",
                    layer_idx=layer_idx,
                    component_type=component_type,
                )
            )
    return specs


def limit_aliases(options: tuple[str, ...], max_aliases_per_side: int) -> tuple[str, ...]:
    if max_aliases_per_side <= 0:
        return options
    return options[:max_aliases_per_side]


class ContributionScorer:
    def __init__(self, *, backend: Any, config: ContributionScanConfig) -> None:
        self.backend = backend
        self.config = config
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.model.eval()

    def _encode_prompt_and_continuation(self, prompt: str, continuation: str) -> tuple[Any, int, int]:
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
        if mode == "prompt":
            return slice(0, prompt_len)
        if mode == "all":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported apply_mode: {mode}")

    def _register_zero_hook(
        self,
        component: ComponentSpec,
        *,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ):
        module = self.backend._component_module(
            layer_idx=int(component.layer_idx),
            component_type=str(component.component_type),
        )

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
            hidden_new[:, pos, :] = 0
            if isinstance(output, tuple):
                return (hidden_new, *output[1:])
            return hidden_new

        return module.register_forward_hook(hook)

    def candidate_score(
        self,
        prompt: str,
        continuation: str,
        *,
        zero_component: ComponentSpec | None = None,
        score_text: str = "",
    ):
        inputs, prompt_len, continuation_len = self._encode_prompt_and_continuation(prompt, continuation)
        handle = None
        if zero_component is not None:
            handle = self._register_zero_hook(
                zero_component,
                prompt_len=prompt_len,
                continuation_len=continuation_len,
                continuation=continuation,
                score_text=score_text,
            )
        try:
            logits = self.model(**inputs, use_cache=False).logits
        finally:
            if handle is not None:
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

    def endpoint_score(
        self,
        prompt: str,
        continuations: tuple[str, ...],
        *,
        zero_component: ComponentSpec | None = None,
        score_text: str = "",
    ):
        options = limit_options(
            continuations,
            max_aliases_per_side=self.config.max_aliases_per_side,
        )
        if not options:
            raise ValueError("empty endpoint continuation options")
        scores = [
            self.candidate_score(prompt, option, zero_component=zero_component, score_text=score_text)
            for option in options
        ]
        return select_option_score(
            self.torch,
            scores,
            selection_mode=self.config.option_selection_mode,
        )

    def margin(self, pair: ActuatorPair, *, zero_component: ComponentSpec | None = None):
        require_dynamic_answer_rest_margin(self.config.score_mode, pair=pair)
        if is_constant_zero_y_plus(pair):
            plus = self.torch.zeros((), device=self.device)
        else:
            plus = self.endpoint_score(
                pair.prompt,
                pair.y_plus_options,
                zero_component=zero_component,
                score_text=pair.y_plus_score_text,
            )
        if is_dynamic_y_minus(pair) or not pair.y_minus_options:
            return plus
        minus = self.endpoint_score(
            pair.prompt,
            pair.y_minus_options,
            zero_component=zero_component,
            score_text=pair.y_minus_score_text,
        )
        return plus - minus

    def scan_one(self, pair: ActuatorPair, component: ComponentSpec, *, full_margin: float) -> dict[str, object]:
        with self.torch.no_grad():
            ablated_margin = float(self.margin(pair, zero_component=component).detach().cpu().item())
        delta = full_margin - ablated_margin
        return {
            "sample_id": pair.sample_id,
            "split": pair.split,
            "event": self.config.event,
            "component_id": component.component_id,
            "layer_idx": component.layer_idx,
            "component_type": component.component_type,
            "full_margin": full_margin,
            "ablated_margin": ablated_margin,
            "delta": delta,
            "delta_sign": "positive" if delta > 0 else ("negative" if delta < 0 else "zero"),
            "apply_mode": self.config.apply_mode,
        }


def summarize_component_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_component: dict[str, list[float]] = {}
    meta: dict[str, dict[str, object]] = {}
    for row in rows:
        component_id = str(row["component_id"])
        delta = float(row["delta"])
        by_component.setdefault(component_id, []).append(delta)
        meta[component_id] = {
            "component_id": component_id,
            "layer_idx": row["layer_idx"],
            "component_type": row["component_type"],
            "event": row["event"],
            "apply_mode": row["apply_mode"],
        }

    summary: list[dict[str, object]] = []
    for component_id, deltas in by_component.items():
        n = len(deltas)
        mean_delta = sum(deltas) / n if n else math.nan
        centered = [(value - mean_delta) ** 2 for value in deltas]
        std_delta = math.sqrt(sum(centered) / (n - 1)) if n > 1 else 0.0
        stderr = std_delta / math.sqrt(n) if n else math.nan
        positive_rate = sum(1 for value in deltas if value > 0) / n if n else math.nan
        negative_rate = sum(1 for value in deltas if value < 0) / n if n else math.nan
        sign_consistency = max(positive_rate, negative_rate)
        ci_low = mean_delta - 1.96 * stderr if n else math.nan
        ci_high = mean_delta + 1.96 * stderr if n else math.nan
        edge_direction = "positive" if mean_delta >= 0 else "negative"
        summary.append(
            {
                **meta[component_id],
                "n": n,
                "mean_delta": mean_delta,
                "abs_mean_delta": abs(mean_delta),
                "std_delta": std_delta,
                "stderr_delta": stderr,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "positive_rate": positive_rate,
                "negative_rate": negative_rate,
                "sign_consistency": sign_consistency,
                "edge_direction": edge_direction,
                "ci_excludes_zero": int((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            -int(row["ci_excludes_zero"]),
            -float(row["abs_mean_delta"]),
            -float(row["sign_consistency"]),
            str(row["component_id"]),
        ),
    )


def select_core_edges(
    summary_rows: list[dict[str, object]],
    *,
    min_abs_delta: float,
    min_sign_consistency: float,
    require_ci_excludes_zero: bool,
    max_components_per_direction: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for direction in ("positive", "negative"):
        candidates = [
            row
            for row in summary_rows
            if row["edge_direction"] == direction
            and float(row["abs_mean_delta"]) >= min_abs_delta
            and float(row["sign_consistency"]) >= min_sign_consistency
            and (not require_ci_excludes_zero or int(row["ci_excludes_zero"]) == 1)
        ]
        candidates = sorted(
            candidates,
            key=lambda row: (
                -float(row["abs_mean_delta"]),
                -float(row["sign_consistency"]),
                str(row["component_id"]),
            ),
        )
        for core_index, row in enumerate(candidates[:max_components_per_direction], start=1):
            selected.append(
                {
                    **row,
                    "axis": row["event"],
                    "micro_event": row["event"],
                    "edge_sign": "support_positive" if direction == "positive" else "support_negative",
                    "core_index": core_index,
                    "selection_rule": (
                        "abs(mean_delta)>=min_abs_delta; "
                        "sign_consistency>=min_sign_consistency; "
                        "optional ci95 excludes zero"
                    ),
                }
            )
    return selected
