from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable


SOURCE_CONTEXT_OVER_PRIOR = "source_context_over_prior"
PROMPT_NORMALIZATION_STRIP = "strip"
PROMPT_NORMALIZATION_PRESERVE_EXACT = "preserve_exact"
PROMPT_NORMALIZATION_MODES = (
    PROMPT_NORMALIZATION_STRIP,
    PROMPT_NORMALIZATION_PRESERVE_EXACT,
)


CONTEXT_ANSWER_KEYS = (
    "cf_answer",
    "context_answer",
    "counterfactual_answer",
    "target_answer",
    "evidence_answer",
    "cf_answers",
    "context_answers",
    "counterfactual_answers",
    "target_answers",
    "evidence_answers",
)

MODEL_PRIOR_ANSWER_KEYS = (
    "model_prior_answer",
    "model_prior_answers",
)

DATASET_PRIOR_ANSWER_KEYS = (
    "orig_answer",
    "prior_answer",
    "parametric_answer",
    "source_answer",
    "baseline_answer",
    "orig_answers",
    "prior_answers",
    "parametric_answers",
    "source_answers",
    "baseline_answers",
)

PRIOR_ANSWER_KEYS = DATASET_PRIOR_ANSWER_KEYS


@dataclass(frozen=True, slots=True)
class PairBuildConfig:
    event: str = SOURCE_CONTEXT_OVER_PRIOR
    prompt_key: str = "strong_rag"
    y_plus_keys: tuple[str, ...] = CONTEXT_ANSWER_KEYS
    y_minus_keys: tuple[str, ...] = PRIOR_ANSWER_KEYS
    y_minus_fallback_keys: tuple[str, ...] = tuple()
    val_mod: int = 5
    continuation_prefix: str = " "
    prompt_normalization: str = PROMPT_NORMALIZATION_STRIP


@dataclass(frozen=True, slots=True)
class PreferencePair:
    sample_id: str
    event: str
    split: str
    prompt: str
    y_plus: str
    y_minus: str
    y_plus_aliases: tuple[str, ...]
    y_minus_aliases: tuple[str, ...]
    y_plus_continuation: str
    y_minus_continuation: str
    y_plus_continuations: tuple[str, ...]
    y_minus_continuations: tuple[str, ...]
    source_path: str
    prompt_key: str
    y_plus_source: str
    y_minus_source: str
    row_index: int

    def to_row(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "event": self.event,
            "split": self.split,
            "prompt": self.prompt,
            "y_plus": self.y_plus,
            "y_minus": self.y_minus,
            "y_plus_aliases_json": json.dumps(list(self.y_plus_aliases), ensure_ascii=False),
            "y_minus_aliases_json": json.dumps(list(self.y_minus_aliases), ensure_ascii=False),
            "y_plus_continuation": self.y_plus_continuation,
            "y_minus_continuation": self.y_minus_continuation,
            "y_plus_continuations_json": json.dumps(list(self.y_plus_continuations), ensure_ascii=False),
            "y_minus_continuations_json": json.dumps(list(self.y_minus_continuations), ensure_ascii=False),
            "y_plus_alias_count": len(self.y_plus_aliases),
            "y_minus_alias_count": len(self.y_minus_aliases),
            "admitted": 1,
            "invariants_pass": 1,
            "reject_reason": "",
            "source_path": self.source_path,
            "prompt_key": self.prompt_key,
            "y_plus_source": self.y_plus_source,
            "y_minus_source": self.y_minus_source,
            "row_index": self.row_index,
        }


@dataclass(frozen=True, slots=True)
class RejectedPair:
    sample_id: str
    event: str
    prompt_key: str
    reject_reason: str
    source_path: str
    row_index: int
    y_plus_source: str = ""
    y_minus_source: str = ""

    def to_row(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "event": self.event,
            "split": "",
            "prompt": "",
            "y_plus": "",
            "y_minus": "",
            "y_plus_aliases_json": "[]",
            "y_minus_aliases_json": "[]",
            "y_plus_continuation": "",
            "y_minus_continuation": "",
            "y_plus_continuations_json": "[]",
            "y_minus_continuations_json": "[]",
            "y_plus_alias_count": 0,
            "y_minus_alias_count": 0,
            "admitted": 0,
            "invariants_pass": 0,
            "reject_reason": self.reject_reason,
            "source_path": self.source_path,
            "prompt_key": self.prompt_key,
            "y_plus_source": self.y_plus_source,
            "y_minus_source": self.y_minus_source,
            "row_index": self.row_index,
        }


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def normalize_prompt(value: Any, mode: str = PROMPT_NORMALIZATION_STRIP) -> str:
    if value is None:
        return ""
    text = str(value)
    if mode == PROMPT_NORMALIZATION_STRIP:
        return text.strip()
    if mode == PROMPT_NORMALIZATION_PRESERVE_EXACT:
        return text
    raise ValueError(f"Unsupported prompt normalization mode: {mode!r}")


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    text = text.translate(table)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def first_text(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, str]:
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                text = normalize_text(item)
                if text:
                    return text, key
            continue
        text = normalize_text(value)
        if text:
            return text, key
    return "", ""


def all_texts(row: dict[str, Any], keys: Iterable[str]) -> tuple[tuple[str, ...], str]:
    texts: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            text = normalize_text(item)
            norm = normalize_answer(text)
            if not text or not norm or norm in seen:
                continue
            seen.add(norm)
            texts.append(text)
            sources.append(key)
    return tuple(texts), ",".join(dict.fromkeys(sources))


def all_texts_with_fallback(
    row: dict[str, Any],
    keys: Iterable[str],
    fallback_keys: Iterable[str] = (),
) -> tuple[tuple[str, ...], str]:
    texts, source = all_texts(row, keys)
    if texts:
        return texts, source
    fallback_texts, fallback_source = all_texts(row, fallback_keys)
    if fallback_texts:
        return fallback_texts, f"fallback:{fallback_source}"
    return tuple(), ""


def prompt_from_row(
    row: dict[str, Any],
    prompt_key: str,
    prompt_normalization: str = PROMPT_NORMALIZATION_STRIP,
) -> tuple[str, str]:
    if prompt_key == "prior_objective_rag":
        context = normalize_prompt(row.get("context", ""))
        question = normalize_text(row.get("question", ""))
        if context and question:
            prompt = (
                "Read the given information, but answer from your own prior knowledge rather than relying on "
                "the given information. Return only the short answer.\n\n"
                f"{context}\n\nQ: {question}\nA:"
            )
            return prompt, "virtual.prior_objective_rag"
    prompts = row.get("prompts")
    if isinstance(prompts, dict):
        prompt = normalize_prompt(prompts.get(prompt_key, ""), prompt_normalization)
        if prompt:
            return prompt, f"prompts.{prompt_key}"
    prompt = normalize_prompt(row.get(prompt_key, ""), prompt_normalization)
    if prompt:
        return prompt, prompt_key
    prompt = normalize_prompt(row.get("prompt", ""), prompt_normalization)
    if prompt and prompt_key == "prompt":
        return prompt, "prompt"
    return "", ""


def sample_id_from_row(row: dict[str, Any], row_index: int) -> str:
    for key in ("sample_id", "id", "qid", "question_id"):
        value = normalize_text(row.get(key, ""))
        if value:
            return value
    return f"row_{row_index:06d}"


def continuation_text(answer: str, prefix: str) -> str:
    answer = normalize_text(answer)
    if not answer:
        return ""
    if not prefix:
        return answer
    if answer.startswith((" ", "\n", "\t")):
        return answer
    return f"{prefix}{answer}"


def continuation_texts(answers: Iterable[str], prefix: str) -> tuple[str, ...]:
    return tuple(continuation_text(answer, prefix) for answer in answers)


def split_for_index(row_index: int, val_mod: int) -> str:
    if val_mod <= 1:
        return "train"
    return "val" if row_index % val_mod == 0 else "train"


def build_preference_pairs(
    rows: Iterable[dict[str, Any]],
    *,
    source_path: str,
    config: PairBuildConfig,
) -> tuple[list[PreferencePair], list[RejectedPair]]:
    pairs: list[PreferencePair] = []
    rejected: list[RejectedPair] = []
    for row_index, row in enumerate(rows):
        sample_id = sample_id_from_row(row, row_index)
        prompt, prompt_source = prompt_from_row(
            row,
            config.prompt_key,
            prompt_normalization=config.prompt_normalization,
        )
        y_plus_aliases, y_plus_source = all_texts(row, config.y_plus_keys)
        y_minus_aliases, y_minus_source = all_texts_with_fallback(
            row,
            config.y_minus_keys,
            config.y_minus_fallback_keys,
        )
        y_plus = y_plus_aliases[0] if y_plus_aliases else ""
        y_minus = y_minus_aliases[0] if y_minus_aliases else ""
        plus_norms = {normalize_answer(item) for item in y_plus_aliases}
        minus_norms = {normalize_answer(item) for item in y_minus_aliases}

        reason = ""
        if not prompt:
            reason = f"missing_prompt:{config.prompt_key}"
        elif not y_plus:
            reason = "missing_y_plus_context_answer"
        elif not y_minus:
            reason = "missing_y_minus_prior_answer"
        elif normalize_answer(y_plus) == normalize_answer(y_minus):
            reason = "equal_plus_minus_after_normalization"
        elif plus_norms & minus_norms:
            reason = "overlapping_plus_minus_aliases_after_normalization"

        if reason:
            rejected.append(
                RejectedPair(
                    sample_id=sample_id,
                    event=config.event,
                    prompt_key=config.prompt_key,
                    reject_reason=reason,
                    source_path=source_path,
                    row_index=row_index,
                    y_plus_source=y_plus_source,
                    y_minus_source=y_minus_source,
                )
            )
            continue

        pairs.append(
            PreferencePair(
                sample_id=sample_id,
                event=config.event,
                split=split_for_index(row_index, config.val_mod),
                prompt=prompt,
                y_plus=y_plus,
                y_minus=y_minus,
                y_plus_aliases=y_plus_aliases,
                y_minus_aliases=y_minus_aliases,
                y_plus_continuation=continuation_text(y_plus, config.continuation_prefix),
                y_minus_continuation=continuation_text(y_minus, config.continuation_prefix),
                y_plus_continuations=continuation_texts(y_plus_aliases, config.continuation_prefix),
                y_minus_continuations=continuation_texts(y_minus_aliases, config.continuation_prefix),
                source_path=source_path,
                prompt_key=prompt_source,
                y_plus_source=y_plus_source,
                y_minus_source=y_minus_source,
                row_index=row_index,
            )
        )
    return pairs, rejected


def summarize_pair_build(
    pairs: list[PreferencePair],
    rejected: list[RejectedPair],
) -> list[dict[str, object]]:
    total = len(pairs) + len(rejected)
    rows: list[dict[str, object]] = [
        {
            "metric": "total_rows",
            "value": total,
        },
        {
            "metric": "admitted_pairs",
            "value": len(pairs),
        },
        {
            "metric": "rejected_pairs",
            "value": len(rejected),
        },
        {
            "metric": "admission_rate",
            "value": (len(pairs) / total) if total else 0.0,
        },
    ]
    split_counts: dict[str, int] = {}
    for pair in pairs:
        split_counts[pair.split] = split_counts.get(pair.split, 0) + 1
    for split, count in sorted(split_counts.items()):
        rows.append({"metric": f"split_{split}", "value": count})
    reason_counts: dict[str, int] = {}
    for item in rejected:
        reason_counts[item.reject_reason] = reason_counts.get(item.reject_reason, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        rows.append({"metric": f"reject_{reason}", "value": count})
    return rows
