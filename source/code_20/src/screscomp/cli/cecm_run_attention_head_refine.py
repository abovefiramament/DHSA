from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

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
    score_text_start_token_index,
    score_text_token_slice,
    select_option_score,
)
from screscomp.cecm.scaling import classify_source_generation, load_open_eval_rows, summarize_generation_rows
from screscomp.data import dump_csv, dump_json, dump_jsonl
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class HeadAction:
    layer_idx: int
    head_idx: int
    factor: float
    role: str

    @property
    def head_id(self) -> str:
        return f"L{self.layer_idx}.attn.h{self.head_idx}"


@dataclass(frozen=True, slots=True)
class HeadControl:
    name: str
    kind: str
    actions: tuple[HeadAction, ...]


class AttentionHeadScalingRunner:
    def __init__(
        self,
        *,
        backend: TransformersABBackend,
        score_mode: str,
        score_apply_mode: str,
        max_aliases_per_side: int,
        option_selection_mode: str = MODEL_MAX_OPTION_SELECTION,
    ) -> None:
        self.backend = backend
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.score_mode = score_mode
        self.score_apply_mode = score_apply_mode
        self.max_aliases_per_side = max_aliases_per_side
        self.option_selection_mode = option_selection_mode
        self.model.eval()

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

    def num_heads(self, layer_idx: int) -> int:
        _hidden, num_heads, _head_dim = self._head_geometry(layer_idx)
        return num_heads

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

    def _score_pos(
        self,
        *,
        prompt_len: int,
        continuation_len: int,
        seq_len: int,
        continuation: str,
        score_text: str,
    ) -> slice:
        mode = self.score_apply_mode
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
        first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", str(mode or ""))
        if first_decode_match:
            steps = int(first_decode_match.group(1) or "1")
            if steps <= 0:
                raise ValueError(f"Unsupported score_apply_mode: {mode}")
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + steps, seq_len)
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
        raise ValueError(f"Unsupported score_apply_mode: {mode}")

    @staticmethod
    def _parse_generation_apply_mode(raw: str) -> tuple[str, int | None]:
        first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
        if first_decode_match:
            steps = int(first_decode_match.group(1) or "1")
            if steps <= 0:
                raise ValueError(f"Unsupported generation_apply_mode: {raw}")
            return raw, steps
        if raw not in {"all", "prefill", "decode"}:
            raise ValueError(f"Unsupported generation_apply_mode: {raw}")
        return raw, None

    def _register_head_hooks(
        self,
        actions: tuple[HeadAction, ...],
        *,
        mode: str,
        prompt_len: int | None = None,
        continuation_len: int | None = None,
        continuation: str = "",
        score_text: str = "",
    ) -> list[Any]:
        actions_by_layer: dict[int, list[HeadAction]] = {}
        for action in actions:
            actions_by_layer.setdefault(action.layer_idx, []).append(action)
        handles: list[Any] = []
        first_decode_steps: int | None = None
        if mode != "score":
            _mode, first_decode_steps = self._parse_generation_apply_mode(mode)

        for layer_idx, layer_actions in actions_by_layer.items():
            _hidden, num_heads, head_dim = self._head_geometry(layer_idx)
            for action in layer_actions:
                if action.head_idx < 0 or action.head_idx >= num_heads:
                    raise ValueError(f"Invalid head {action.head_idx} for L{layer_idx}.attn with {num_heads} heads")
            module = self._o_proj_module(layer_idx)

            def make_hook(local_actions: list[HeadAction], local_head_dim: int):
                decode_steps_seen = 0

                def hook(_module, inputs):
                    nonlocal decode_steps_seen
                    hidden = inputs[0]
                    seq_len = int(hidden.shape[1])
                    if mode == "score":
                        if prompt_len is None or continuation_len is None:
                            raise ValueError("score hooks require prompt_len and continuation_len")
                        pos = self._score_pos(
                            prompt_len=prompt_len,
                            continuation_len=continuation_len,
                            seq_len=seq_len,
                            continuation=continuation,
                            score_text=score_text,
                        )
                    else:
                        if mode == "prefill" and seq_len <= 1:
                            return inputs
                        if mode == "decode" and seq_len > 1:
                            return inputs
                        if first_decode_steps is not None:
                            if seq_len > 1:
                                return inputs
                            decode_steps_seen += 1
                            if decode_steps_seen > first_decode_steps:
                                return inputs
                        pos = slice(max(seq_len - 1, 0), seq_len)

                    hidden_new = hidden.clone()
                    for action in local_actions:
                        start = int(action.head_idx) * local_head_dim
                        stop = start + local_head_dim
                        hidden_new[:, pos, start:stop] = hidden_new[:, pos, start:stop] * float(action.factor)
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_hook(layer_actions, head_dim)))
        return handles

    def candidate_score(
        self,
        prompt: str,
        continuation: str,
        *,
        actions: tuple[HeadAction, ...] = tuple(),
        score_text: str = "",
    ):
        inputs, prompt_len, continuation_len = self._encode_prompt_and_continuation(prompt, continuation)
        handles = self._register_head_hooks(
            actions,
            mode="score",
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
        actions: tuple[HeadAction, ...] = tuple(),
        score_text: str = "",
    ):
        options = limit_options(
            continuations,
            max_aliases_per_side=self.max_aliases_per_side,
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
            selection_mode=self.option_selection_mode,
        )

    def margin(self, pair: ActuatorPair, *, actions: tuple[HeadAction, ...] = tuple()):
        require_dynamic_answer_rest_margin(self.score_mode, pair=pair)
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
        actions: tuple[HeadAction, ...],
        generation_apply_mode: str,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        handles = self._register_head_hooks(actions, mode=generation_apply_mode)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refine selected attention components into head-level scaling controls.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", type=str, default="source_context_over_prior")
    p.add_argument("--eval-open-rows", type=Path, required=True)
    p.add_argument("--attn-layers", type=str, default="31,27,9,25")
    p.add_argument("--scan-factors", type=str, default="0.5,0.0,1.5")
    p.add_argument("--topks", type=str, default="2,4")
    p.add_argument("--generation-kinds", type=str, default="suppress,boost,mixed")
    p.add_argument("--generation-apply-modes", type=str, default="prefill,first_decode")
    p.add_argument("--generation-prompt-key", type=str, default="base_rag")
    p.add_argument("--prior-source", choices=["auto", "model_prior", "dataset_orig"], default="dataset_orig")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--scan-start", type=int, default=0)
    p.add_argument("--scan-max-rows", type=int, default=24)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=20)
    p.add_argument("--val-mod", type=int, default=5)
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
    p.add_argument("--max-aliases-per-side", type=int, default=1)
    p.add_argument(
        "--option-selection-mode",
        type=str,
        default="model_max",
        choices=OPTION_SELECTION_MODES,
        help="How to choose among multiple continuation options for each endpoint.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--verbose-char-threshold", type=int, default=48)
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


def _parse_ints(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv(raw)]


def _parse_floats(raw: str) -> list[float]:
    return [float(item) for item in _parse_csv(raw)]


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _load_scan_pairs(path: Path, *, event: str, split: str, start: int, max_rows: int) -> list[ActuatorPair]:
    pairs = [pair for pair in load_actuator_pairs(path, event=event) if pair.split == split]
    if start > 0:
        pairs = pairs[start:]
    pairs = limit_rows(pairs, max_rows)
    if not pairs:
        raise SystemExit(f"No scan pairs selected from {path} split={split!r} event={event!r}")
    return pairs


def _cache_baselines(runner: AttentionHeadScalingRunner, pairs: list[ActuatorPair]) -> list[float]:
    baselines: list[float] = []
    with runner.torch.no_grad():
        for pair in tqdm(pairs, desc="baseline margins"):
            baselines.append(float(runner.margin(pair).detach().cpu().item()))
    return baselines


def _evaluate_margin_gain(
    runner: AttentionHeadScalingRunner,
    pairs: list[ActuatorPair],
    baselines: list[float],
    *,
    actions: tuple[HeadAction, ...],
) -> dict[str, float]:
    margins: list[float] = []
    gains: list[float] = []
    with runner.torch.no_grad():
        for pair, base in zip(pairs, baselines, strict=True):
            value = float(runner.margin(pair, actions=actions).detach().cpu().item())
            margins.append(value)
            gains.append(value - base)
    n = len(pairs)
    if len(gains) <= 1:
        gain_ci95_low = gains[0] if gains else math.nan
        gain_ci95_high = gains[0] if gains else math.nan
    else:
        gain_half = 1.96 * stdev(gains) / math.sqrt(len(gains))
        gain_center = mean(gains)
        gain_ci95_low = gain_center - gain_half
        gain_ci95_high = gain_center + gain_half
    return {
        "n": float(n),
        "base_mean_margin": mean(baselines) if baselines else math.nan,
        "mean_margin": mean(margins) if margins else math.nan,
        "mean_margin_gain": mean(gains) if gains else math.nan,
        "gain_ci95_low": gain_ci95_low,
        "gain_ci95_high": gain_ci95_high,
        "base_pref_rate": mean(float(value > 0) for value in baselines) if baselines else math.nan,
        "pref_rate": mean(float(value > 0) for value in margins) if margins else math.nan,
        "gain_positive_rate": mean(float(value > 0) for value in gains) if gains else math.nan,
    }


def _scan_heads(
    runner: AttentionHeadScalingRunner,
    pairs: list[ActuatorPair],
    baselines: list[float],
    *,
    attn_layers: list[int],
    factors: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_idx in attn_layers:
        num_heads = runner.num_heads(layer_idx)
        for head_idx in tqdm(range(num_heads), desc=f"scan L{layer_idx}.attn heads"):
            for factor in factors:
                if factor == 1.0:
                    continue
                role = "suppress" if factor < 1.0 else "boost"
                action = HeadAction(layer_idx=layer_idx, head_idx=head_idx, factor=factor, role=role)
                metrics = _evaluate_margin_gain(runner, pairs, baselines, actions=(action,))
                rows.append(
                    {
                        "head_id": action.head_id,
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "factor": factor,
                        "role": role,
                        **metrics,
                    }
                )
    return rows


def _best_rows_by_head(scan_rows: Iterable[dict[str, object]], *, role: str) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in scan_rows:
        if str(row.get("role")) != role:
            continue
        head_id = str(row["head_id"])
        previous = best.get(head_id)
        if previous is None or float(row["mean_margin_gain"]) > float(previous["mean_margin_gain"]):
            best[head_id] = row
    return sorted(best.values(), key=lambda row: float(row["mean_margin_gain"]), reverse=True)


def _action_from_scan_row(row: dict[str, object]) -> HeadAction:
    return HeadAction(
        layer_idx=int(row["layer_idx"]),
        head_idx=int(row["head_idx"]),
        factor=float(row["factor"]),
        role=str(row["role"]),
    )


def _dedupe_actions(actions: Iterable[HeadAction]) -> tuple[HeadAction, ...]:
    out: list[HeadAction] = []
    seen: set[tuple[int, int]] = set()
    for action in actions:
        key = (action.layer_idx, action.head_idx)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
    return tuple(out)


def _build_controls(
    *,
    scan_rows: list[dict[str, object]],
    topks: list[int],
    kinds: list[str],
) -> list[HeadControl]:
    suppress_rows = _best_rows_by_head(scan_rows, role="suppress")
    boost_rows = _best_rows_by_head(scan_rows, role="boost")
    controls: list[HeadControl] = [HeadControl(name="baseline", kind="baseline", actions=tuple())]
    for topk in topks:
        if "suppress" in kinds and suppress_rows:
            controls.append(
                HeadControl(
                    name=f"suppress_top{topk}",
                    kind="suppress",
                    actions=tuple(_action_from_scan_row(row) for row in suppress_rows[:topk]),
                )
            )
        if "boost" in kinds and boost_rows:
            controls.append(
                HeadControl(
                    name=f"boost_top{topk}",
                    kind="boost",
                    actions=tuple(_action_from_scan_row(row) for row in boost_rows[:topk]),
                )
            )
        if "mixed" in kinds and (suppress_rows or boost_rows):
            controls.append(
                HeadControl(
                    name=f"mixed_top{topk}",
                    kind="mixed",
                    actions=_dedupe_actions(
                        [_action_from_scan_row(row) for row in suppress_rows[:topk]]
                        + [_action_from_scan_row(row) for row in boost_rows[:topk]]
                    ),
                )
            )
    return controls


def _actions_text(actions: tuple[HeadAction, ...]) -> str:
    return ",".join(f"{action.head_id}:{action.factor:g}" for action in actions)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attn_layers = _parse_ints(args.attn_layers)
    scan_factors = _parse_floats(args.scan_factors)
    topks = _parse_ints(args.topks)
    generation_kinds = _parse_csv(args.generation_kinds)
    generation_apply_modes = _parse_csv(args.generation_apply_modes)
    stop_strings = _parse_stop_strings(args.stop_strings)

    print(f"[cecm-attn-head-refine] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    runner = AttentionHeadScalingRunner(
        backend=backend,
        score_mode=args.score_mode,
        score_apply_mode=args.score_apply_mode,
        max_aliases_per_side=args.max_aliases_per_side,
        option_selection_mode=args.option_selection_mode,
    )
    scan_pairs = _load_scan_pairs(
        args.pairs_csv,
        event=args.event,
        split=args.split,
        start=args.scan_start,
        max_rows=args.scan_max_rows,
    )
    gen_rows = load_open_eval_rows(
        args.eval_open_rows,
        prompt_key=args.generation_prompt_key,
        prior_source=args.prior_source,
        split=None if args.split == "all" else args.split,
        start=args.start,
        max_rows=args.max_rows,
        val_mod=args.val_mod,
    )
    if not gen_rows:
        raise SystemExit("No generation rows selected.")

    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "pairs_csv": str(args.pairs_csv),
            "eval_open_rows": str(args.eval_open_rows),
            "event": args.event,
            "attn_layers": attn_layers,
            "scan_factors": scan_factors,
            "topks": topks,
            "generation_kinds": generation_kinds,
            "generation_apply_modes": generation_apply_modes,
            "scan_pairs": len(scan_pairs),
            "generation_rows": len(gen_rows),
            "score_mode": args.score_mode,
            "score_apply_mode": args.score_apply_mode,
            "option_selection_mode": args.option_selection_mode,
            "option_selection": option_selection_description(args.option_selection_mode),
            "semantics": (
                "Head refinement is restricted to previously selected attention layers. "
                "Head scaling acts on attention head outputs before the layer output projection."
            ),
        },
    )

    baselines = _cache_baselines(runner, scan_pairs)
    scan_rows = _scan_heads(
        runner,
        scan_pairs,
        baselines,
        attn_layers=attn_layers,
        factors=scan_factors,
    )
    dump_csv(args.out_dir / "head_scan.csv", scan_rows)

    controls = _build_controls(scan_rows=scan_rows, topks=topks, kinds=generation_kinds)
    dump_csv(
        args.out_dir / "head_group_plan.csv",
        [
            {
                "control_name": control.name,
                "kind": control.kind,
                "num_actions": len(control.actions),
                "actions": _actions_text(control.actions),
            }
            for control in controls
        ],
    )

    generation_rows: list[dict[str, object]] = []
    for apply_mode in generation_apply_modes:
        for control in controls:
            control_name = f"{apply_mode}__{control.name}"
            for row in tqdm(gen_rows, desc=f"generate {control_name}"):
                if control.actions:
                    prediction = runner.generate(
                        row.prompt,
                        actions=control.actions,
                        generation_apply_mode=apply_mode,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
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
                metrics = classify_source_generation(
                    prediction,
                    context_answers=row.context_answers,
                    prior_answers=row.prior_answers,
                    verbose_char_threshold=args.verbose_char_threshold,
                )
                generation_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "control_name": control_name,
                        "baseline_kind": "attn_head_scaling",
                        "env_kind": "context",
                        "alpha": "fixed",
                        "head_control_name": control.name,
                        "head_control_kind": control.kind,
                        "generation_apply_mode": apply_mode,
                        "prediction": prediction,
                        "generation_prompt_key": args.generation_prompt_key,
                        "prior_source": args.prior_source,
                        "context_answers_json": list(row.context_answers),
                        "prior_answers_json": list(row.prior_answers),
                        "head_actions": _actions_text(control.actions),
                        **metrics,
                    }
                )

    dump_jsonl(args.out_dir / "generation_rows.jsonl", generation_rows)
    dump_csv(args.out_dir / "generation_summary.csv", summarize_generation_rows(generation_rows))
    dump_csv(
        args.out_dir / "run_summary.csv",
        [
            {"metric": "scan_pairs", "value": len(scan_pairs)},
            {"metric": "scan_rows", "value": len(scan_rows)},
            {"metric": "head_controls", "value": len(controls)},
            {"metric": "generation_eval_rows", "value": len(gen_rows)},
            {"metric": "generation_rows", "value": len(generation_rows)},
        ],
    )
    print(f"[cecm-attn-head-refine] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
