from __future__ import annotations

import csv
import json
import math
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from screscomp.cecm.actuator import (
    ActuatorPair,
    is_constant_zero_y_plus,
    is_dynamic_y_minus,
    require_dynamic_answer_rest_margin,
)
from screscomp.cecm.components import ComponentSpec, parse_component_id
from screscomp.cecm.objective import (
    MODEL_MAX_OPTION_SELECTION,
    limit_options,
    score_text_start_token_index,
    score_text_token_slice,
    select_option_score,
)
from screscomp.cecm.pairs import (
    CONTEXT_ANSWER_KEYS,
    DATASET_PRIOR_ANSWER_KEYS,
    MODEL_PRIOR_ANSWER_KEYS,
    all_texts,
    continuation_texts,
    normalize_answer,
    prompt_from_row,
    sample_id_from_row,
)


@dataclass(frozen=True, slots=True)
class ScalingConfig:
    score_mode: str = "answer_rest_margin"
    score_apply_mode: str = "decision_tokens"
    generation_apply_mode: str = "prefill"
    max_aliases_per_side: int = 0
    option_selection_mode: str = MODEL_MAX_OPTION_SELECTION
    up_factor: float = 1.0
    down_floor: float = 0.0
    shared_alpha_scale: float = 0.5


@dataclass(frozen=True, slots=True)
class ScalingAction:
    component: ComponentSpec
    factor: float
    role: str


@dataclass(frozen=True, slots=True)
class SourceControlSpec:
    name: str
    env_kind: str
    shared_policy: str
    up_components: tuple[ComponentSpec, ...]
    down_components: tuple[ComponentSpec, ...]
    shared_components: tuple[ComponentSpec, ...] = tuple()
    baseline_kind: str = "cecm"
    target_direction: str = ""

    def actions(self, *, alpha: float, config: ScalingConfig) -> list[ScalingAction]:
        if alpha == 0:
            return []
        up_factor = 1.0 + float(config.up_factor) * float(alpha)
        down_factor = max(float(config.down_floor), 1.0 - float(alpha))
        actions = [
            ScalingAction(component=component, factor=up_factor, role="up")
            for component in self.up_components
        ]
        actions.extend(
            ScalingAction(component=component, factor=down_factor, role="down")
            for component in self.down_components
        )
        if self.shared_policy == "on":
            shared_factor = 1.0 + float(config.shared_alpha_scale) * float(alpha)
            actions.extend(
                ScalingAction(component=component, factor=shared_factor, role="shared_up")
                for component in self.shared_components
            )
        return actions


@dataclass(frozen=True, slots=True)
class OpenEvalRow:
    sample_id: str
    split: str
    prompt: str
    context_answers: tuple[str, ...]
    prior_answers: tuple[str, ...]
    source_path: str = ""


def _as_text_tuple(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = " ".join(str(item or "").strip().split())
        norm = normalize_answer(text)
        if not text or not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return tuple(out)


def _answers_from_row(row: dict[str, Any], *, main_key: str, list_key: str) -> tuple[str, ...]:
    answers = [*_as_text_tuple(row.get(main_key)), *_as_text_tuple(row.get(list_key))]
    out: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        norm = normalize_answer(answer)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(answer)
    return tuple(out)


def _prior_answers_from_row(row: dict[str, Any], *, prior_source: str) -> tuple[str, ...]:
    if prior_source == "model_prior":
        answers, _source = all_texts(row, MODEL_PRIOR_ANSWER_KEYS)
        return answers
    if prior_source == "dataset_orig":
        answers, _source = all_texts(row, DATASET_PRIOR_ANSWER_KEYS)
        return answers
    if prior_source != "auto":
        raise ValueError(f"Unsupported prior_source: {prior_source}")
    answers, _source = all_texts(row, MODEL_PRIOR_ANSWER_KEYS)
    if answers:
        return answers
    answers, _source = all_texts(row, DATASET_PRIOR_ANSWER_KEYS)
    return answers


def load_open_eval_rows(
    path: Path,
    *,
    prompt_key: str,
    prior_source: str = "auto",
    split: str | None = None,
    start: int = 0,
    max_rows: int | None = None,
    continuation_prefix: str = " ",
    val_mod: int = 5,
) -> list[OpenEvalRow]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    selected: list[OpenEvalRow] = []
    for row_index, row in enumerate(rows):
        row_split = "val" if row_index % int(val_mod) == 0 else "train"
        if split and row_split != split:
            continue
        prompt, _prompt_source = prompt_from_row(row, prompt_key)
        if not prompt:
            continue
        context_answers, _context_source = all_texts(row, CONTEXT_ANSWER_KEYS)
        prior_answers = _prior_answers_from_row(row, prior_source=prior_source)
        if not context_answers or not prior_answers:
            continue
        context_norms = {normalize_answer(answer) for answer in context_answers}
        prior_norms = {normalize_answer(answer) for answer in prior_answers}
        if context_norms & prior_norms:
            continue
        selected.append(
            OpenEvalRow(
                sample_id=sample_id_from_row(row, row_index),
                split=row_split,
                prompt=prompt,
                context_answers=continuation_texts(context_answers, continuation_prefix),
                prior_answers=continuation_texts(prior_answers, continuation_prefix),
                source_path=str(path),
            )
        )
    selected = selected[start:]
    if max_rows is not None and max_rows > 0:
        selected = selected[:max_rows]
    return selected


def parse_component_list(raw: str) -> tuple[ComponentSpec, ...]:
    specs: list[ComponentSpec] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        specs.append(parse_component_id(item))
    return tuple(specs)


def component_universe(num_layers: int, component_types: Iterable[str] = ("attn", "mlp")) -> tuple[ComponentSpec, ...]:
    specs: list[ComponentSpec] = []
    for layer_idx in range(num_layers):
        for component_type in component_types:
            specs.append(
                ComponentSpec(
                    component_id=f"L{layer_idx}.{component_type}",
                    layer_idx=layer_idx,
                    component_type=component_type,
                )
            )
    return tuple(specs)


def _sample_components(
    *,
    rng: random.Random,
    universe: tuple[ComponentSpec, ...],
    count: int,
    exclude: set[str],
    component_type: str | None = None,
) -> tuple[ComponentSpec, ...]:
    candidates = [
        component
        for component in universe
        if component.component_id not in exclude
        and (component_type is None or component.component_type == component_type)
    ]
    if len(candidates) < count:
        raise ValueError(
            f"Not enough random components: need={count} available={len(candidates)} type={component_type}"
        )
    return tuple(rng.sample(candidates, count))


def _type_matched_sample(
    *,
    rng: random.Random,
    universe: tuple[ComponentSpec, ...],
    originals: tuple[ComponentSpec, ...],
    exclude: set[str],
) -> tuple[ComponentSpec, ...]:
    selected: list[ComponentSpec] = []
    local_exclude = set(exclude)
    for component_type in sorted({component.component_type for component in originals}):
        count = sum(1 for component in originals if component.component_type == component_type)
        sampled = _sample_components(
            rng=rng,
            universe=universe,
            count=count,
            exclude=local_exclude,
            component_type=component_type,
        )
        selected.extend(sampled)
        local_exclude.update(component.component_id for component in sampled)
    return tuple(selected)


def make_randomized_spec(
    base: SourceControlSpec,
    *,
    universe: tuple[ComponentSpec, ...],
    rng: random.Random,
    random_kind: str,
    trial_idx: int,
) -> SourceControlSpec:
    used = {
        component.component_id
        for group in (base.up_components, base.down_components, base.shared_components)
        for component in group
    }
    if random_kind == "random_any":
        up = _sample_components(rng=rng, universe=universe, count=len(base.up_components), exclude=used)
        used.update(component.component_id for component in up)
        down = _sample_components(rng=rng, universe=universe, count=len(base.down_components), exclude=used)
        used.update(component.component_id for component in down)
        shared = _sample_components(
            rng=rng,
            universe=universe,
            count=len(base.shared_components),
            exclude=used,
        )
    elif random_kind == "random_type_matched":
        up = _type_matched_sample(rng=rng, universe=universe, originals=base.up_components, exclude=used)
        used.update(component.component_id for component in up)
        down = _type_matched_sample(rng=rng, universe=universe, originals=base.down_components, exclude=used)
        used.update(component.component_id for component in down)
        shared = _type_matched_sample(rng=rng, universe=universe, originals=base.shared_components, exclude=used)
    else:
        raise ValueError(f"Unsupported random_kind: {random_kind}")
    return SourceControlSpec(
        name=f"{random_kind}_{base.name}_s{trial_idx}",
        env_kind=base.env_kind,
        shared_policy=base.shared_policy,
        up_components=up,
        down_components=down,
        shared_components=shared,
        baseline_kind=random_kind,
        target_direction=base.target_direction,
    )


def make_wrong_side_spec(base: SourceControlSpec, *, opposite: SourceControlSpec) -> SourceControlSpec:
    return SourceControlSpec(
        name=f"wrong_side_for_{base.name}",
        env_kind=base.env_kind,
        shared_policy=opposite.shared_policy,
        up_components=opposite.up_components,
        down_components=opposite.down_components,
        shared_components=opposite.shared_components,
        baseline_kind="wrong_side",
        target_direction=base.target_direction,
    )


def build_source_control_specs(
    *,
    controls: Iterable[str],
    env_kinds: Iterable[str],
    shared_policies: Iterable[str],
    context_components: tuple[ComponentSpec, ...],
    prior_components: tuple[ComponentSpec, ...],
    shared_components: tuple[ComponentSpec, ...],
) -> list[SourceControlSpec]:
    specs: list[SourceControlSpec] = []
    controls_set = [item.strip() for item in controls if item.strip()]
    env_set = [item.strip() for item in env_kinds if item.strip()]
    shared_set = [item.strip() for item in shared_policies if item.strip()]
    for control in controls_set:
        if control not in {"force_context", "force_prior", "conditional_source"}:
            raise ValueError(f"Unsupported control: {control}")
        for env_kind in env_set:
            if env_kind not in {"context", "prior"}:
                raise ValueError(f"Unsupported env_kind: {env_kind}")
            for shared_policy in shared_set:
                if shared_policy not in {"off", "on"}:
                    raise ValueError(f"Unsupported shared_policy: {shared_policy}")
                if control == "force_context":
                    up, down, target = context_components, prior_components, "context"
                elif control == "force_prior":
                    up, down, target = prior_components, context_components, "prior"
                else:
                    if env_kind == "context":
                        up, down, target = context_components, prior_components, "context"
                    else:
                        up, down, target = prior_components, context_components, "prior"
                suffix = "" if shared_policy == "off" else "_shared"
                specs.append(
                    SourceControlSpec(
                        name=f"{control}{suffix}",
                        env_kind=env_kind,
                        shared_policy=shared_policy,
                        up_components=up,
                        down_components=down,
                        shared_components=shared_components,
                        baseline_kind="cecm",
                        target_direction=target,
                    )
                )
    return specs


class ComponentScalingRunner:
    def __init__(self, *, backend: Any, config: ScalingConfig) -> None:
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

    def _slice_for_score_apply_mode(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        seq_len: int,
        continuation: str,
        score_text: str,
    ) -> slice:
        mode = self.config.score_apply_mode
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
        raise ValueError(f"Unsupported score_apply_mode: {mode}")

    @staticmethod
    def _scale_output(output, *, factor: float, pos: slice | None = None):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_new = hidden.clone()
        if pos is None:
            hidden_new[:, -1, :] = hidden_new[:, -1, :] * float(factor)
        else:
            hidden_new[:, pos, :] = hidden_new[:, pos, :] * float(factor)
        if isinstance(output, tuple):
            return (hidden_new, *output[1:])
        return hidden_new

    def _register_score_hooks(
        self,
        actions: list[ScalingAction],
        *,
        prompt_len: int,
        continuation_len: int,
        continuation: str,
        score_text: str,
    ) -> list[Any]:
        handles = []
        for action in actions:
            module = self.backend._component_module(
                layer_idx=int(action.component.layer_idx),
                component_type=str(action.component.component_type),
            )

            def make_hook(factor: float):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    pos = self._slice_for_score_apply_mode(
                        prompt_len=prompt_len,
                        continuation_len=continuation_len,
                        seq_len=int(hidden.shape[1]),
                        continuation=continuation,
                        score_text=score_text,
                    )
                    return self._scale_output(output, factor=factor, pos=pos)

                return hook

            handles.append(module.register_forward_hook(make_hook(action.factor)))
        return handles

    def _register_generation_hooks(self, actions: list[ScalingAction]) -> list[Any]:
        handles = []
        mode = self.config.generation_apply_mode
        first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", mode)
        first_decode_steps = int(first_decode_match.group(1) or "1") if first_decode_match else None
        if first_decode_steps is not None and first_decode_steps <= 0:
            raise ValueError(f"Unsupported generation_apply_mode: {mode}")
        if first_decode_steps is None and mode not in {"all", "prefill", "decode"}:
            raise ValueError(f"Unsupported generation_apply_mode: {mode}")

        for action in actions:
            module = self.backend._component_module(
                layer_idx=int(action.component.layer_idx),
                component_type=str(action.component.component_type),
            )

            def make_hook(factor: float):
                decode_steps_seen = 0

                def hook(_module, _inputs, output):
                    nonlocal decode_steps_seen
                    hidden = output[0] if isinstance(output, tuple) else output
                    seq_len = int(hidden.shape[1])
                    if mode == "prefill" and seq_len <= 1:
                        return output
                    if mode == "decode" and seq_len > 1:
                        return output
                    if first_decode_steps is not None:
                        if seq_len > 1:
                            return output
                        decode_steps_seen += 1
                        if decode_steps_seen > first_decode_steps:
                            return output
                    return self._scale_output(output, factor=factor, pos=None)

                return hook

            handles.append(module.register_forward_hook(make_hook(action.factor)))
        return handles

    def candidate_score(
        self,
        prompt: str,
        continuation: str,
        *,
        actions: list[ScalingAction] | None = None,
        score_text: str = "",
    ):
        inputs, prompt_len, continuation_len = self._encode_prompt_and_continuation(prompt, continuation)
        handles = self._register_score_hooks(
            actions or [],
            prompt_len=prompt_len,
            continuation_len=continuation_len,
            continuation=continuation,
            score_text=score_text,
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

    def endpoint_score(
        self,
        prompt: str,
        continuations: tuple[str, ...],
        *,
        actions: list[ScalingAction] | None = None,
        score_text: str = "",
    ):
        options = limit_options(
            continuations,
            max_aliases_per_side=self.config.max_aliases_per_side,
        )
        if not options:
            raise ValueError("empty endpoint continuation options")
        scores = [
            self.candidate_score(prompt, option, actions=actions, score_text=score_text)
            for option in options
        ]
        return select_option_score(
            self.torch,
            scores,
            selection_mode=self.config.option_selection_mode,
        )

    def margin(self, pair: ActuatorPair, *, actions: list[ScalingAction] | None = None):
        require_dynamic_answer_rest_margin(self.config.score_mode, pair=pair)
        if is_constant_zero_y_plus(pair):
            plus = self.torch.zeros((), device=self.device)
        else:
            plus = self.endpoint_score(
                pair.prompt,
                pair.y_plus_options,
                actions=actions,
                score_text=pair.y_plus_score_text,
            )
        if is_dynamic_y_minus(pair) or not pair.y_minus_options:
            return plus
        minus = self.endpoint_score(
            pair.prompt,
            pair.y_minus_options,
            actions=actions,
            score_text=pair.y_minus_score_text,
        )
        return plus - minus

    def generate(
        self,
        prompt: str,
        *,
        actions: list[ScalingAction] | None = None,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> str:
        handles = self._register_generation_hooks(actions or [])
        try:
            inputs = self.backend._encode_prompt(prompt)
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


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _recall_hit(prediction: str, answers: Iterable[str]) -> bool:
    pred = _normalize_answer(prediction)
    if not pred:
        return False
    return any(_normalize_answer(answer) in pred for answer in answers if _normalize_answer(answer))


def _exact_hit(prediction: str, answers: Iterable[str]) -> bool:
    pred = _normalize_answer(prediction)
    return any(pred == _normalize_answer(answer) for answer in answers if _normalize_answer(answer))


def classify_source_generation(
    prediction: str,
    *,
    context_answers: tuple[str, ...],
    prior_answers: tuple[str, ...],
    verbose_char_threshold: int = 48,
) -> dict[str, object]:
    context_plain = tuple(str(answer).strip() for answer in context_answers)
    prior_plain = tuple(str(answer).strip() for answer in prior_answers)
    context_hit = _recall_hit(prediction, context_plain)
    prior_hit = _recall_hit(prediction, prior_plain)
    context_em = _exact_hit(prediction, context_plain)
    prior_em = _exact_hit(prediction, prior_plain)
    output_chars = len(str(prediction))
    if context_hit and prior_hit:
        outcome = "both"
    elif context_hit:
        outcome = "context_only"
    elif prior_hit:
        outcome = "prior_only"
    else:
        outcome = "neither"
    return {
        "outcome": outcome,
        "context_hit": context_hit,
        "prior_hit": prior_hit,
        "context_em": context_em,
        "prior_em": prior_em,
        "both": outcome == "both",
        "neither": outcome == "neither",
        "output_chars": output_chars,
        "short_output": output_chars <= verbose_char_threshold,
        "verbose": output_chars > verbose_char_threshold,
        "short_exact_context": context_em and output_chars <= verbose_char_threshold,
    }


def summarize_generation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    baselines: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        baseline_key = (
            str(row.get("baseline_kind", "")),
            str(row.get("env_kind", "")),
            str(row.get("alpha", "")),
        )
        if str(row.get("control_name", "")) == "baseline":
            baselines.setdefault(baseline_key, {})[str(row.get("sample_id", ""))] = str(row.get("outcome", ""))
        key = (
            str(row.get("control_name", "")),
            str(row.get("baseline_kind", "")),
            str(row.get("env_kind", "")),
            str(row.get("alpha", "")),
        )
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for (control_name, baseline_kind, env_kind, alpha), group in sorted(groups.items()):
        n = len(group)
        if not n:
            continue
        pc = mean(float(bool(row["context_hit"])) for row in group)
        po = mean(float(bool(row["prior_hit"])) for row in group)
        denom = pc + po
        em = mean(float(bool(row["context_em"])) for row in group)
        baseline_outcomes = baselines.get((baseline_kind, env_kind, alpha), {})
        paired = [
            (baseline_outcomes[str(row.get("sample_id", ""))], str(row.get("outcome", "")))
            for row in group
            if str(row.get("sample_id", "")) in baseline_outcomes
        ]
        base_context = [(base, current) for base, current in paired if base == "context_only"]
        base_non_context = [(base, current) for base, current in paired if base != "context_only"]
        transition_count = sum(1 for _base, current in base_non_context if current == "context_only")
        harm_count = sum(1 for _base, current in base_context if current != "context_only")
        paired_n = len(paired)
        summary.append(
            {
                "control_name": control_name,
                "baseline_kind": baseline_kind,
                "env_kind": env_kind,
                "alpha": alpha,
                "n": n,
                "pc": pc,
                "po": po,
                "mr": po / denom if denom > 0 else 0.0,
                "em": em,
                "context_only_rate": mean(float(row["outcome"] == "context_only") for row in group),
                "prior_only_rate": mean(float(row["outcome"] == "prior_only") for row in group),
                "both_rate": mean(float(bool(row["both"])) for row in group),
                "neither_rate": mean(float(bool(row["neither"])) for row in group),
                "context_hit_rate": pc,
                "prior_hit_rate": po,
                "context_em_rate": em,
                "prior_em_rate": mean(float(bool(row["prior_em"])) for row in group),
                "short_exact_context_rate": mean(float(bool(row["short_exact_context"])) for row in group),
                "short_output_rate": mean(float(bool(row["short_output"])) for row in group),
                "verbose_rate": mean(float(bool(row["verbose"])) for row in group),
                "mean_chars": mean(float(row["output_chars"]) for row in group),
                "paired_baseline_n": paired_n,
                "base_context_only_n": len(base_context),
                "base_non_context_only_n": len(base_non_context),
                "transition_to_context_only_count": transition_count,
                "transition_to_context_only_rate": transition_count / len(base_non_context) if base_non_context else 0.0,
                "harm_from_context_only_count": harm_count,
                "harm_from_context_only_rate": harm_count / len(base_context) if base_context else 0.0,
                "net_transition_rate": (transition_count - harm_count) / paired_n if paired_n else 0.0,
            }
        )
    return summary


def summarize_endpoint_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("control_name", "")),
            str(row.get("baseline_kind", "")),
            str(row.get("env_kind", "")),
            str(row.get("alpha", "")),
        )
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for (control_name, baseline_kind, env_kind, alpha), group in sorted(groups.items()):
        n = len(group)
        if not n:
            continue
        base_margins = [float(row["base_margin"]) for row in group]
        margins = [float(row["margin"]) for row in group]
        gains = [float(row["margin_gain"]) for row in group]
        summary.append(
            {
                "control_name": control_name,
                "baseline_kind": baseline_kind,
                "env_kind": env_kind,
                "alpha": alpha,
                "n": n,
                "base_mean_margin": mean(base_margins),
                "mean_margin": mean(margins),
                "mean_margin_gain": mean(gains),
                "base_pref_rate": mean(float(value > 0) for value in base_margins),
                "pref_rate": mean(float(value > 0) for value in margins),
                "gain_positive_rate": mean(float(value > 0) for value in gains),
            }
        )
    return summary


def load_external_generations(path: Path, *, method_name: str, env_kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                rows.append(dict(value))
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    normalized: list[dict[str, object]] = []
    for row in rows:
        prediction = None
        for key in ("prediction", "output", "text", "generation"):
            if key in row and row.get(key) is not None:
                prediction = row.get(key)
                break
        sample_id = row.get("sample_id") if row.get("sample_id") is not None else row.get("id")
        if prediction is None or sample_id is None:
            continue
        normalized.append(
            {
                **row,
                "sample_id": str(sample_id),
                "prediction": str(prediction),
                "control_name": method_name,
                "baseline_kind": "ckplug",
                "env_kind": str(row.get("env_kind") or env_kind),
                "alpha": str(row.get("alpha") or "external"),
            }
        )
    return normalized
