from __future__ import annotations

import re
from typing import Any, Sequence


MODEL_MAX_OPTION_SELECTION = "model_max"
FIRST_OPTION_SELECTION = "first"
OPTION_SELECTION_MODES = (
    MODEL_MAX_OPTION_SELECTION,
    FIRST_OPTION_SELECTION,
)


def limit_options(
    options: tuple[str, ...],
    *,
    max_aliases_per_side: int,
) -> tuple[str, ...]:
    if max_aliases_per_side <= 0:
        return options
    return options[:max_aliases_per_side]


def select_option_score(
    torch_module: Any,
    scores: Sequence[Any],
    *,
    selection_mode: str = MODEL_MAX_OPTION_SELECTION,
) -> Any:
    if not scores:
        raise ValueError("empty endpoint continuation options")
    if selection_mode not in OPTION_SELECTION_MODES:
        raise ValueError(f"Unsupported option_selection_mode={selection_mode!r}")
    if len(scores) == 1:
        return scores[0]
    if selection_mode == FIRST_OPTION_SELECTION:
        return scores[0]
    if selection_mode == MODEL_MAX_OPTION_SELECTION:
        return torch_module.stack(list(scores)).max()
    raise ValueError(f"Unsupported option_selection_mode={selection_mode!r}")


def option_selection_description(selection_mode: str) -> str:
    if selection_mode == MODEL_MAX_OPTION_SELECTION:
        return "Endpoint aliases are selected by the model score max over continuation options."
    if selection_mode == FIRST_OPTION_SELECTION:
        return "The first endpoint option is used directly."
    raise ValueError(f"Unsupported option_selection_mode={selection_mode!r}")


def score_text_token_slice(tokenizer: Any, continuation: str, score_text: str) -> slice | None:
    """Return a continuation-relative token slice for the final score_text occurrence."""
    text = str(score_text or "").strip()
    if not text:
        return None
    continuation = str(continuation or "")
    start = continuation.rfind(text)
    if start < 0:
        compact = text.replace(",", "")
        if compact != text:
            start = continuation.replace(",", "").rfind(compact)
        if start < 0:
            return None
    end = start + len(text)
    before_ids = tokenizer(continuation[:start], return_tensors="pt", add_special_tokens=False)["input_ids"]
    upto_ids = tokenizer(continuation[:end], return_tensors="pt", add_special_tokens=False)["input_ids"]
    token_start = int(before_ids.shape[-1])
    token_stop = int(upto_ids.shape[-1])
    if token_stop <= token_start:
        return None
    return slice(token_start, token_stop)


def _compact_span(raw: str, compact_text: str) -> tuple[int, int] | None:
    compact_chars: list[str] = []
    index_map: list[int] = []
    for idx, char in enumerate(raw):
        if char == ",":
            continue
        compact_chars.append(char)
        index_map.append(idx)
    start = "".join(compact_chars).rfind(compact_text)
    if start < 0 or not compact_text:
        return None
    stop = start + len(compact_text) - 1
    if stop >= len(index_map):
        return None
    return index_map[start], index_map[stop] + 1


def _final_text_span(raw: str, text: str) -> tuple[int, int] | None:
    start = raw.rfind(text)
    if start >= 0:
        return start, start + len(text)
    compact = text.replace(",", "")
    if compact != text:
        return _compact_span(raw, compact)
    return None


def _boxed_score_text_span(continuation: str, score_text: str) -> tuple[int, int] | None:
    for match in reversed(list(re.finditer(r"\\boxed\s*\{([^{}]+)\}", continuation))):
        inner = match.group(1)
        local = _final_text_span(inner, score_text)
        if local is None:
            continue
        return match.start(1) + local[0], match.start(1) + local[1]
    return None


def score_text_start_token_index(
    tokenizer: Any,
    continuation: str,
    score_text: str,
    *,
    prefer_boxed: bool = False,
) -> int | None:
    """Return the continuation-relative token index where score_text starts."""
    text = str(score_text or "").strip()
    if not text:
        return None
    continuation = str(continuation or "")
    if prefer_boxed:
        span = _boxed_score_text_span(continuation, text)
    else:
        span = _final_text_span(continuation, text)
    if span is None:
        return None
    before_ids = tokenizer(continuation[: span[0]], return_tensors="pt", add_special_tokens=False)["input_ids"]
    return int(before_ids.shape[-1])
