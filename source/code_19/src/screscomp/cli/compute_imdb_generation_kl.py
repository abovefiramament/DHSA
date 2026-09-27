from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screscomp.cecm.actuator import load_fixed_actuator_additions
from screscomp.cli.cecm_run_joint_actuator_generation import (
    JointControl,
    JointGenerationRunner,
    _load_head_actuator,
    _parse_generation_apply_mode,
)
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable


@dataclass(frozen=True, slots=True)
class ControlSpec:
    actuator_path: str
    head_actuator_path: str
    alpha: float
    sign: float
    generation_apply_mode: str
    component_apply_mode: str
    head_apply_mode: str
    include_components: tuple[str, ...]
    exclude_components: tuple[str, ...]
    include_heads: tuple[str, ...]
    exclude_heads: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc KL scoring for IMDb actuator generations.")
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--progress-json", type=Path, default=None)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--stream-summary-every-rows", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL rows must be objects: {path}")
                rows.append(value)
    return rows


def _parse_string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    text = str(value or "").strip()
    if not text:
        return tuple()
    try:
        raw = json.loads(text)
        if isinstance(raw, list):
            return tuple(str(item) for item in raw if str(item).strip())
    except Exception:
        pass
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _control_spec_from_row(row: dict[str, Any]) -> ControlSpec:
    return ControlSpec(
        actuator_path=str(row.get("actuator_path", "")).strip(),
        head_actuator_path=str(row.get("head_actuator_path", "")).strip(),
        alpha=float(row.get("alpha", 0.0) or 0.0),
        sign=float(row.get("actuator_sign", 1.0) or 1.0),
        generation_apply_mode=str(row.get("generation_apply_mode", "") or "all"),
        component_apply_mode=str(row.get("component_apply_mode", "") or str(row.get("generation_apply_mode", "") or "all")),
        head_apply_mode=str(row.get("head_apply_mode", "") or str(row.get("generation_apply_mode", "") or "all")),
        include_components=tuple(sorted(_parse_string_list(row.get("included_components", [])))),
        exclude_components=tuple(sorted(_parse_string_list(row.get("excluded_components", [])))),
        include_heads=tuple(sorted(_parse_string_list(row.get("included_heads", [])))),
        exclude_heads=tuple(sorted(_parse_string_list(row.get("excluded_heads", [])))),
    )


def _filter_component_additions(
    additions: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    include_components: tuple[str, ...],
    exclude_components: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    include = set(include_components)
    exclude = set(exclude_components)
    out = []
    for addition in additions:
        component_id = str(addition.get("component_id", ""))
        if include and component_id not in include:
            continue
        if exclude and component_id in exclude:
            continue
        out.append(addition)
    return tuple(out)


def _filter_head_actions(
    actions: tuple[Any, ...],
    *,
    include_heads: tuple[str, ...],
    exclude_heads: tuple[str, ...],
) -> tuple[Any, ...]:
    include = set(include_heads)
    exclude = set(exclude_heads)
    out = []
    for action in actions:
        head_id = str(getattr(action, "head_id", ""))
        if include and head_id not in include:
            continue
        if exclude and head_id in exclude:
            continue
        out.append(action)
    return tuple(out)


def _build_joint_control(spec: ControlSpec) -> JointControl | None:
    if abs(spec.alpha) < 1e-12:
        return None
    component_additions: tuple[dict[str, Any], ...] = tuple()
    if spec.actuator_path:
        component_additions = _filter_component_additions(
            load_fixed_actuator_additions(
                Path(spec.actuator_path),
                alpha=float(spec.sign) * float(spec.alpha),
                apply_mode=spec.component_apply_mode,
            ),
            include_components=spec.include_components,
            exclude_components=spec.exclude_components,
        )
    head_vectors: tuple[Any, ...] = tuple()
    if spec.head_actuator_path:
        head_vectors = _filter_head_actions(
            _load_head_actuator(
                Path(spec.head_actuator_path),
                alpha=float(spec.sign) * float(spec.alpha),
                apply_mode=spec.head_apply_mode,
            ),
            include_heads=spec.include_heads,
            exclude_heads=spec.exclude_heads,
        )
    if not component_additions and not head_vectors:
        return None
    return JointControl(
        name="kl_eval",
        component_additions=component_additions,
        head_scales=tuple(),
        head_vectors=head_vectors,
        conditional_components=tuple(),
        conditional_heads=tuple(),
        conditional_group_specs={},
        parts=tuple(),
    )


def _summary_key(row: dict[str, object]) -> tuple[str, str, str]:
    split = str(row.get("split") or "unknown")
    control = str(row.get("control_name") or "generated")
    alpha = str(row.get("alpha", ""))
    return split, control, alpha


def _new_group_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "positive_sum": 0.0,
        "positive_ge_0p5_sum": 0.0,
        "label_score_sum": 0.0,
        "completion_chars_sum": 0.0,
        "positive_min": float("inf"),
        "positive_max": float("-inf"),
        "sequence_kl_sum": 0.0,
        "token_kl_sum": 0.0,
        "completion_tokens_sum": 0.0,
    }


def _f(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else default
    except Exception:
        return default


def _update_group_stats(stats: dict[str, float], row: dict[str, object]) -> None:
    positive = _f(row.get("positive_sentiment_score"), 0.0)
    label_score = _f(row.get("sentiment_label_score"), 0.0)
    completion_chars = float(len(str(row.get("completion", ""))))
    sequence_kl = _f(row.get("sequence_kl"), 0.0)
    mean_token_kl = _f(row.get("mean_token_kl"), 0.0)
    completion_tokens = _f(row.get("completion_token_count"), 0.0)
    stats["n"] += 1.0
    stats["positive_sum"] += positive
    stats["positive_ge_0p5_sum"] += 1.0 if positive >= 0.5 else 0.0
    stats["label_score_sum"] += label_score
    stats["completion_chars_sum"] += completion_chars
    stats["positive_min"] = min(stats["positive_min"], positive)
    stats["positive_max"] = max(stats["positive_max"], positive)
    stats["sequence_kl_sum"] += sequence_kl
    stats["token_kl_sum"] += mean_token_kl
    stats["completion_tokens_sum"] += completion_tokens


def _summary_rows_from_stats(grouped: dict[tuple[str, str, str], dict[str, float]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for (split, control, alpha), stats in sorted(grouped.items(), key=lambda item: item[0]):
        n = int(stats["n"])
        if n <= 0:
            continue
        n_float = float(n)
        out.append(
            {
                "split": split,
                "control_name": control,
                "alpha": alpha,
                "n": n,
                "mean_positive_sentiment_score": stats["positive_sum"] / n_float,
                "positive_rate_ge_0p5": stats["positive_ge_0p5_sum"] / n_float,
                "mean_sentiment_label_score": stats["label_score_sum"] / n_float,
                "mean_completion_chars": stats["completion_chars_sum"] / n_float,
                "min_positive_sentiment_score": stats["positive_min"],
                "max_positive_sentiment_score": stats["positive_max"],
                "mean_sequence_kl": stats["sequence_kl_sum"] / n_float,
                "mean_token_kl": stats["token_kl_sum"] / n_float,
                "mean_completion_token_count": stats["completion_tokens_sum"] / n_float,
            }
        )
    return out


def _append_jsonl_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_csv_rows(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _csv_header(path: Path) -> list[str] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
    return [str(item) for item in header]


def _ordered_field_union(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text in seen:
                continue
            seen.add(key_text)
            ordered.append(key_text)
    return ordered


def _batched(iterable: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return [iterable[idx : idx + batch_size] for idx in range(0, len(iterable), batch_size)]


def _alpha_key(value: Any) -> str:
    try:
        return f"{float(value):.8g}"
    except Exception:
        return str(value or "")


def _hash_text(value: Any) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def _list_key(value: Any) -> str:
    return json.dumps(sorted(_parse_string_list(value)), ensure_ascii=False, sort_keys=True)


def _row_resume_key(row: dict[str, Any]) -> tuple[str, ...]:
    generation_apply_mode = str(row.get("generation_apply_mode", "") or "all")
    component_apply_mode = str(row.get("component_apply_mode", "") or generation_apply_mode)
    head_apply_mode = str(row.get("head_apply_mode", "") or generation_apply_mode)
    return (
        str(row.get("model", "")),
        str(row.get("split", "")),
        str(row.get("event", "")),
        str(row.get("control_name", "")),
        _alpha_key(row.get("alpha", "")),
        str(row.get("sample_id", "")),
        str(row.get("prompt_id", "")),
        str(row.get("source_dataset", "")),
        str(row.get("source_split", "")),
        str(row.get("source_row_index", "")),
        str(row.get("sample_index", "")),
        _hash_text(row.get("prompt", "")),
        _hash_text(row.get("completion", "")),
        str(row.get("actuator_path", "")),
        str(row.get("head_actuator_path", "")),
        _alpha_key(row.get("actuator_sign", 1.0)),
        generation_apply_mode,
        component_apply_mode,
        head_apply_mode,
        _list_key(row.get("included_components", [])),
        _list_key(row.get("excluded_components", [])),
        _list_key(row.get("included_heads", [])),
        _list_key(row.get("excluded_heads", [])),
    )


def _load_existing_kl_rows(path: Path, input_keys: set[tuple[str, ...]]) -> dict[tuple[str, ...], dict[str, Any]]:
    existing: dict[tuple[str, ...], dict[str, Any]] = {}
    if not path.exists() or path.stat().st_size == 0:
        return existing
    for row in _load_jsonl(path):
        if "sequence_kl" not in row or "mean_token_kl" not in row:
            continue
        key = _row_resume_key(row)
        if key in input_keys:
            existing[key] = row
    return existing


def _sync_full_csv(out_jsonl: Path, out_csv: Path | None, input_keys: set[tuple[str, ...]]) -> None:
    if out_csv is None or not out_jsonl.exists():
        return
    rows = [
        row
        for row in _load_jsonl(out_jsonl)
        if _row_resume_key(row) in input_keys and "sequence_kl" in row and "mean_token_kl" in row
    ]
    dump_csv(out_csv, rows)


def _write_progress(
    *,
    summary_csv: Path,
    progress_json: Path | None,
    grouped: dict[tuple[str, str, str], dict[str, float]],
    rows_done: int,
    rows_total: int,
) -> None:
    dump_csv(summary_csv, _summary_rows_from_stats(grouped))
    if progress_json is not None:
        dump_json(
            progress_json,
            {
                "rows_scored": rows_done,
                "rows_total": rows_total,
                "rows_remaining": max(0, rows_total - rows_done),
                "progress_fraction": (rows_done / rows_total) if rows_total > 0 else 0.0,
                "summary_csv": str(summary_csv),
            },
        )


class KLEvaluator:
    def __init__(self, backend: TransformersABBackend) -> None:
        self.backend = backend
        self.runner = JointGenerationRunner(backend)
        self.model = backend._model
        self.tokenizer = backend._tokenizer
        self.torch = backend._torch

    def _pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        return int(pad_id)

    def _tokenize_prompt_and_continuation(self, prompt: str, continuation: str):
        formatted_prompt = self.backend._format_prompt(prompt)
        prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.backend.device)
        cont_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.backend.device)
        if int(cont_ids.shape[-1]) <= 0:
            raise ValueError(f"empty continuation after tokenization: {continuation!r}")
        return prompt_ids, cont_ids

    def _tokenize_examples(self, prompts: list[str], continuations: list[str]) -> list[tuple[Any, Any, int, int]]:
        examples: list[tuple[Any, Any, int, int]] = []
        for prompt, continuation in zip(prompts, continuations):
            prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
            examples.append((prompt_ids, cont_ids, int(prompt_ids.shape[-1]), int(cont_ids.shape[-1])))
        return examples

    def _build_prefix_inputs(self, prompt_ids: Any, cont_ids: Any, *, visible_continuation_len: int) -> dict[str, Any]:
        prefix = cont_ids[:, :visible_continuation_len]
        input_ids = self.torch.cat([prompt_ids, prefix], dim=-1)
        return {
            "input_ids": input_ids,
            "attention_mask": self.torch.ones_like(input_ids, device=input_ids.device),
        }

    def _build_left_padded_batch_inputs(
        self,
        examples: list[tuple[Any, Any, int, int]],
        *,
        visible_continuation_len: int,
    ) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        sequences: list[Any] = []
        lengths: list[int] = []
        for prompt_ids, cont_ids, _prompt_len, _cont_len in examples:
            prefix = cont_ids[:, :visible_continuation_len]
            seq = self.torch.cat([prompt_ids, prefix], dim=-1).squeeze(0)
            sequences.append(seq)
            lengths.append(int(seq.shape[0]))
        max_len = max(lengths)
        input_ids = self.torch.full(
            (len(sequences), max_len),
            int(pad_id),
            dtype=sequences[0].dtype,
            device=self.backend.device,
        )
        attention_mask = self.torch.zeros((len(sequences), max_len), dtype=self.torch.long, device=self.backend.device)
        for row_idx, seq in enumerate(sequences):
            seq_len = int(seq.shape[0])
            input_ids[row_idx, max_len - seq_len :] = seq
            attention_mask[row_idx, max_len - seq_len :] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _encode_full_batch_inputs(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> tuple[Any, Any, list[int], list[int], list[int], list[str], list[str]]:
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must have the same length")
        prompt_lens: list[int] = []
        continuation_lens: list[int] = []
        seq_lens: list[int] = []
        prompt_rows: list[Any] = []
        continuation_rows: list[Any] = []
        for prompt, continuation in zip(prompts, continuations):
            prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
            prompt_lens.append(int(prompt_ids.shape[-1]))
            continuation_lens.append(int(cont_ids.shape[-1]))
            seq_lens.append(int(prompt_ids.shape[-1] + cont_ids.shape[-1]))
            prompt_rows.append(prompt_ids.squeeze(0))
            continuation_rows.append(cont_ids.squeeze(0))
        batch_size = len(prompts)
        max_seq_len = max(seq_lens)
        pad_id = self._pad_token_id()
        input_ids = self.torch.full(
            (batch_size, max_seq_len),
            pad_id,
            dtype=prompt_rows[0].dtype,
            device=self.backend.device,
        )
        attention_mask = self.torch.zeros(
            (batch_size, max_seq_len),
            dtype=prompt_rows[0].dtype,
            device=self.backend.device,
        )
        for row_idx, (prompt_row, continuation_row) in enumerate(zip(prompt_rows, continuation_rows)):
            seq = self.torch.cat([prompt_row, continuation_row], dim=0)
            seq_len = int(seq.shape[0])
            input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = 1
        return input_ids, attention_mask, prompt_lens, continuation_lens, seq_lens, prompts, continuations

    @staticmethod
    def _full_sequence_slice_for_apply_mode(
        *,
        mode: str,
        prompt_len: int,
        continuation_len: int,
        seq_len: int,
    ) -> slice:
        if mode in {"prefill", "prompt_last"}:
            start = max(min(prompt_len, seq_len) - 1, 0)
            return slice(start, min(start + 1, seq_len))
        if mode == "prompt":
            return slice(0, min(prompt_len, seq_len))
        first_decode_steps = _parse_generation_apply_mode(mode)[1] if mode.startswith("first") else None
        if first_decode_steps is not None:
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + first_decode_steps, seq_len)
            return slice(start, max(stop, start))
        if mode == "decode":
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + continuation_len, seq_len)
            return slice(start, max(stop, start))
        if mode == "all":
            start = max(min(prompt_len, seq_len) - 1, 0)
            stop = min(prompt_len + continuation_len, seq_len)
            return slice(start, max(stop, start + 1))
        if mode == "all_positions":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported apply_mode for batched KL: {mode}")

    def _batch_position_mask_for_mode(
        self,
        *,
        mode: str,
        prompt_lens: list[int],
        continuation_lens: list[int],
        seq_lens: list[int],
    ) -> Any:
        batch_size = len(prompt_lens)
        max_seq_len = max(seq_lens)
        mask = self.torch.zeros((batch_size, max_seq_len), dtype=self.torch.bool, device=self.backend.device)
        for row_idx in range(batch_size):
            pos = self._full_sequence_slice_for_apply_mode(
                mode=mode,
                prompt_len=prompt_lens[row_idx],
                continuation_len=continuation_lens[row_idx],
                seq_len=seq_lens[row_idx],
            )
            mask[row_idx, pos] = True
        return mask

    def _register_batch_component_hooks(
        self,
        additions: tuple[dict[str, Any], ...],
        *,
        apply_mode: str,
        prompt_lens: list[int],
        continuation_lens: list[int],
        seq_lens: list[int],
    ) -> list[Any]:
        handles: list[Any] = []

        def make_hook(direction, alpha: float, local_apply_mode: str):
            base_mask = self._batch_position_mask_for_mode(
                mode=local_apply_mode,
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
            )

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                delta = direction.to(device=hidden_new.device, dtype=hidden_new.dtype).view(1, 1, -1)
                local_mask = base_mask[:, : int(hidden_new.shape[1])].to(device=hidden_new.device, dtype=hidden_new.dtype).unsqueeze(-1)
                hidden_new = hidden_new + float(alpha) * local_mask * delta
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
                    make_hook(addition["direction"], float(addition["alpha"]), local_apply_mode)
                )
            )
        return handles

    def _register_batch_head_hooks(
        self,
        head_scales: tuple[Any, ...],
        head_vectors: tuple[Any, ...],
        *,
        apply_mode: str,
        prompt_lens: list[int],
        continuation_lens: list[int],
        seq_lens: list[int],
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
            _hidden, num_heads, head_dim = self.runner._head_geometry(layer_idx)
            for action in [*scales, *vectors]:
                if action.head_idx < 0 or action.head_idx >= num_heads:
                    raise ValueError(f"Invalid {action.head_id}; L{layer_idx}.attn has {num_heads} heads")
            module = self.runner._o_proj_module(layer_idx)
            base_mask = self._batch_position_mask_for_mode(
                mode=local_apply_mode,
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
            )

            def make_hook(
                local_scales: list[Any],
                local_vectors: list[Any],
                local_head_dim: int,
                local_mask: Any,
            ):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden.clone()
                    row_mask = local_mask[:, : int(hidden_new.shape[1])].to(device=hidden_new.device, dtype=hidden_new.dtype).unsqueeze(-1)
                    for action in local_scales:
                        start = action.head_idx * local_head_dim
                        stop = start + local_head_dim
                        hidden_new[:, :, start:stop] = hidden_new[:, :, start:stop] * (
                            1.0 + row_mask * (float(action.factor) - 1.0)
                        )
                    for action in local_vectors:
                        start = action.head_idx * local_head_dim
                        stop = start + local_head_dim
                        vector = action.vector.to(device=hidden_new.device, dtype=hidden_new.dtype).view(1, 1, -1)
                        hidden_new[:, :, start:stop] = hidden_new[:, :, start:stop] + float(action.alpha) * row_mask * vector
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(
                module.register_forward_pre_hook(
                    make_hook(scales, vectors, head_dim, base_mask)
                )
            )
        return handles

    def _full_logits_with_control(
        self,
        input_ids: Any,
        attention_mask: Any,
        *,
        control: JointControl | None,
        apply_mode: str,
        prompt_lens: list[int],
        continuation_lens: list[int],
        seq_lens: list[int],
    ) -> Any:
        if control is None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        handles: list[Any] = []
        handles.extend(
            self._register_batch_component_hooks(
                control.component_additions,
                apply_mode=apply_mode,
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
            )
        )
        handles.extend(
            self._register_batch_head_hooks(
                control.head_scales,
                control.head_vectors,
                apply_mode=apply_mode,
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
            )
        )
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()

    def _sequence_kl_many_fast(
        self,
        prompts: list[str],
        continuations: list[str],
        *,
        control: JointControl | None,
        apply_mode: str,
    ) -> list[tuple[float, float, int]]:
        input_ids, attention_mask, prompt_lens, continuation_lens, seq_lens, _prompts, _continuations = self._encode_full_batch_inputs(
            prompts,
            continuations,
        )
        with self.torch.no_grad():
            ref_logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            steered_logits = self._full_logits_with_control(
                input_ids,
                attention_mask,
                control=control,
                apply_mode=apply_mode,
                prompt_lens=prompt_lens,
                continuation_lens=continuation_lens,
                seq_lens=seq_lens,
            )
        outputs: list[tuple[float, float, int]] = []
        for row_idx, continuation_len in enumerate(continuation_lens):
            prompt_len = prompt_lens[row_idx]
            start = max(prompt_len - 1, 0)
            stop = max(start + continuation_len, start)
            ref_slice = ref_logits[row_idx, start:stop, :].float()
            steered_slice = steered_logits[row_idx, start:stop, :].float()
            ref_log_probs = self.torch.nn.functional.log_softmax(ref_slice, dim=-1)
            steered_log_probs = self.torch.nn.functional.log_softmax(steered_slice, dim=-1)
            steered_probs = steered_log_probs.exp()
            step_kls = (steered_probs * (steered_log_probs - ref_log_probs)).sum(dim=-1)
            sequence_kl = float(step_kls.sum().detach().cpu().item())
            mean_token_kl = sequence_kl / max(continuation_len, 1)
            outputs.append((sequence_kl, mean_token_kl, continuation_len))
        return outputs

    def _last_logits_with_control(
        self,
        inputs: dict[str, Any],
        *,
        control: JointControl | None,
        apply_mode: str,
        prompt_len: int,
        generated_tokens_so_far: int,
    ):
        if control is None:
            return self.model(**inputs, use_cache=False).logits[:, -1, :].float()
        active_component_additions = self.runner._select_active_component_additions(
            control.component_additions,
            apply_mode=apply_mode,
            generated_tokens_so_far=generated_tokens_so_far,
            boxed_decision_active=False,
        )
        active_head_scales = self.runner._select_active_head_actions(
            control.head_scales,
            apply_mode=apply_mode,
            generated_tokens_so_far=generated_tokens_so_far,
            boxed_decision_active=False,
        )
        active_head_vectors = self.runner._select_active_head_actions(
            control.head_vectors,
            apply_mode=apply_mode,
            generated_tokens_so_far=generated_tokens_so_far,
            boxed_decision_active=False,
        )
        handles: list[Any] = []
        handles.extend(
            self.runner._register_step_component_hooks(
                active_component_additions,
                apply_mode=apply_mode,
                prompt_len=prompt_len,
            )
        )
        handles.extend(
            self.runner._register_step_head_hooks(
                active_head_scales,
                active_head_vectors,
                apply_mode=apply_mode,
                prompt_len=prompt_len,
            )
        )
        try:
            return self.model(**inputs, use_cache=False).logits[:, -1, :].float()
        finally:
            for handle in handles:
                handle.remove()

    def sequence_kl(self, prompt: str, continuation: str, *, control: JointControl | None, apply_mode: str) -> tuple[float, float, int]:
        if control is None:
            prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
            token_count = int(cont_ids.shape[-1])
            return 0.0, 0.0, token_count

        prompt_ids, cont_ids = self._tokenize_prompt_and_continuation(prompt, continuation)
        prompt_len = int(prompt_ids.shape[-1])
        continuation_len = int(cont_ids.shape[-1])
        step_kls: list[Any] = []
        with self.torch.no_grad():
            for target_token_index in range(continuation_len):
                inputs = self._build_prefix_inputs(
                    prompt_ids,
                    cont_ids,
                    visible_continuation_len=target_token_index,
                )
                ref_logits = self.model(**inputs, use_cache=False).logits[:, -1, :].float()
                steered_logits = self._last_logits_with_control(
                    inputs,
                    control=control,
                    apply_mode=apply_mode,
                    prompt_len=prompt_len,
                    generated_tokens_so_far=target_token_index,
                )
                ref_log_probs = self.torch.nn.functional.log_softmax(ref_logits, dim=-1)
                steered_log_probs = self.torch.nn.functional.log_softmax(steered_logits, dim=-1)
                steered_probs = steered_log_probs.exp()
                kl = (steered_probs * (steered_log_probs - ref_log_probs)).sum(dim=-1)
                step_kls.append(kl.squeeze(0))
        sequence_kl = self.torch.stack(step_kls, dim=0).sum()
        mean_token_kl = sequence_kl / max(continuation_len, 1)
        return float(sequence_kl.detach().cpu().item()), float(mean_token_kl.detach().cpu().item()), continuation_len

    def supports_batched_sequence_kl(self, *, control: JointControl | None, apply_mode: str) -> bool:
        if control is None:
            return True
        if control.conditional_components or control.conditional_heads:
            return False
        if self.runner._uses_boxed_decision(control, apply_mode=apply_mode):
            return False
        modes = [apply_mode]
        for addition in control.component_additions:
            modes.append(str(addition.get("apply_mode") or apply_mode))
        for action in [*control.head_scales, *control.head_vectors]:
            modes.append(str(action.apply_mode or apply_mode))
        for mode in modes:
            parsed_mode, _first_decode_steps = _parse_generation_apply_mode(mode)
            if parsed_mode not in {"prefill", "prompt_last", "prompt", "decode", "all", "all_positions", "first_decode"}:
                return False
        return True

    def sequence_kl_many(
        self,
        prompts: list[str],
        continuations: list[str],
        *,
        control: JointControl | None,
        apply_mode: str,
    ) -> list[tuple[float, float, int]]:
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must have the same length")
        if not prompts:
            return []
        if control is None:
            examples = self._tokenize_examples(prompts, continuations)
            return [(0.0, 0.0, cont_len) for _prompt_ids, _cont_ids, _prompt_len, cont_len in examples]
        if self.supports_batched_sequence_kl(control=control, apply_mode=apply_mode):
            return self._sequence_kl_many_fast(
                prompts,
                continuations,
                control=control,
                apply_mode=apply_mode,
            )
        if not self.supports_batched_sequence_kl(control=control, apply_mode=apply_mode):
            return [
                self.sequence_kl(prompt, continuation, control=control, apply_mode=apply_mode)
                for prompt, continuation in zip(prompts, continuations)
            ]
        raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are mutually exclusive")
    rows = _load_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit(f"No rows found: {args.input_jsonl}")

    summary_csv = args.summary_csv or (args.out_jsonl.parent / "score_summary.csv")
    progress_json = args.progress_json
    if args.overwrite:
        for path in [args.out_jsonl, args.out_csv, summary_csv, progress_json]:
            if path is not None and path.exists():
                path.unlink()
    elif not args.resume:
        existing_paths = [
            str(path)
            for path in [args.out_jsonl, args.out_csv, summary_csv, progress_json]
            if path is not None and path.exists()
        ]
        if existing_paths:
            raise SystemExit(
                "Refusing to append to existing KL outputs without --resume or --overwrite: "
                + ", ".join(existing_paths)
            )

    model_name = args.model or str(rows[0].get("model", "")).strip()
    if not model_name:
        raise SystemExit("--model is required when rows do not carry a model field")

    row_specs = [_control_spec_from_row(row) for row in rows]
    row_keys = [_row_resume_key(row) for row in rows]
    input_key_set = set(row_keys)
    existing_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    if args.resume:
        existing_by_key = _load_existing_kl_rows(args.out_jsonl, input_key_set)

    grouped_stats: dict[tuple[str, str, str], dict[str, float]] = {}
    for out_row in existing_by_key.values():
        key = _summary_key(out_row)
        stats = grouped_stats.setdefault(key, _new_group_stats())
        _update_group_stats(stats, out_row)

    input_fields = _ordered_field_union(rows)
    csv_fields: list[str] | None = None
    if args.out_csv is not None and args.resume:
        csv_fields = _csv_header(args.out_csv)
    rows_done = len(existing_by_key)
    work_items = [
        (row, spec)
        for row, spec, key in zip(rows, row_specs, row_keys)
        if key not in existing_by_key
    ]

    if not work_items:
        _write_progress(
            summary_csv=summary_csv,
            progress_json=progress_json,
            grouped=grouped_stats,
            rows_done=rows_done,
            rows_total=len(rows),
        )
        _sync_full_csv(args.out_jsonl, args.out_csv, input_key_set)
        existing_model = ""
        if existing_by_key:
            existing_model = str(next(iter(existing_by_key.values())).get("kl_reference_model", ""))
        dump_json(
            args.out_jsonl.parent / "kl_manifest.json",
            {
                "input_jsonl": str(args.input_jsonl),
                "out_jsonl": str(args.out_jsonl),
                "out_csv": str(args.out_csv) if args.out_csv else "",
                "summary_csv": str(summary_csv),
                "progress_json": str(progress_json) if progress_json else "",
                "rows": rows_done,
                "rows_reused": rows_done,
                "rows_computed": 0,
                "model": existing_model or model_name,
                "batch_size": args.batch_size,
                "resume": args.resume,
                "kl_definition": "sequence-level KL(pi_steered || pi_ref) summed over continuation decode steps; mean_token_kl is the per-token average",
            },
        )
        print(
            f"[imdb-generation-kl] resume complete rows={rows_done} reused={rows_done} path={args.out_jsonl}",
            flush=True,
        )
        return

    backend = TransformersABBackend(model_name_or_path=model_name, device=args.device, torch_dtype=args.torch_dtype)
    evaluator = KLEvaluator(backend)

    control_cache: dict[ControlSpec, JointControl | None] = {}
    cursor = 0
    rows_computed = 0
    with tqdm(total=len(rows), initial=rows_done, desc="compute imdb KL") as progress:
        while cursor < len(work_items):
            row, spec = work_items[cursor]
            batch_rows = [row]
            cursor += 1
            while cursor < len(work_items) and work_items[cursor][1] == spec and len(batch_rows) < args.batch_size:
                batch_rows.append(work_items[cursor][0])
                cursor += 1

            control = control_cache.get(spec)
            if control is None and spec not in control_cache:
                control = _build_joint_control(spec)
                control_cache[spec] = control
            else:
                control = control_cache.get(spec)

            results = evaluator.sequence_kl_many(
                [str(row.get("prompt", "")) for row in batch_rows],
                [str(row.get("completion", "")) for row in batch_rows],
                control=control,
                apply_mode=spec.generation_apply_mode,
            )

            out_rows: list[dict[str, object]] = []
            for row, (sequence_kl, mean_token_kl, token_count) in zip(batch_rows, results):
                out_row = {
                    **row,
                    "sequence_kl": sequence_kl,
                    "mean_token_kl": mean_token_kl,
                    "completion_token_count": token_count,
                    "kl_reference_model": backend.model_id,
                    "kl_apply_mode": spec.generation_apply_mode,
                }
                out_rows.append(out_row)
                key = _summary_key(out_row)
                stats = grouped_stats.setdefault(key, _new_group_stats())
                _update_group_stats(stats, out_row)

            _append_jsonl_rows(args.out_jsonl, out_rows)
            if args.out_csv is not None and out_rows:
                if csv_fields is None:
                    csv_fields = input_fields + [key for key in out_rows[0].keys() if key not in input_fields]
                _append_csv_rows(args.out_csv, out_rows, fieldnames=csv_fields)

            rows_done += len(out_rows)
            rows_computed += len(out_rows)
            progress.update(len(out_rows))
            if args.stream_summary_every_rows > 0 and rows_done % args.stream_summary_every_rows == 0:
                _write_progress(
                    summary_csv=summary_csv,
                    progress_json=progress_json,
                    grouped=grouped_stats,
                    rows_done=rows_done,
                    rows_total=len(rows),
                )

    _write_progress(
        summary_csv=summary_csv,
        progress_json=progress_json,
        grouped=grouped_stats,
        rows_done=rows_done,
        rows_total=len(rows),
    )
    _sync_full_csv(args.out_jsonl, args.out_csv, input_key_set)
    dump_json(
        args.out_jsonl.parent / "kl_manifest.json",
        {
            "input_jsonl": str(args.input_jsonl),
            "out_jsonl": str(args.out_jsonl),
            "out_csv": str(args.out_csv) if args.out_csv else "",
            "summary_csv": str(summary_csv),
            "progress_json": str(progress_json) if progress_json else "",
            "rows": rows_done,
            "rows_reused": len(existing_by_key),
            "rows_computed": rows_computed,
            "model": backend.model_id,
            "batch_size": args.batch_size,
            "resume": args.resume,
            "kl_definition": "sequence-level KL(pi_steered || pi_ref) summed over continuation decode steps; mean_token_kl is the per-token average",
        },
    )
    print(
        f"[imdb-generation-kl] wrote rows={rows_done} reused={len(existing_by_key)} computed={rows_computed} path={args.out_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
