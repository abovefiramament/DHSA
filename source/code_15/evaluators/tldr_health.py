"""Pure, protocol-defined text-health metrics for TL;DR evaluation."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Mapping, Sequence


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_TERMINAL = re.compile(r"""[.!?][)"']*$""")
_STOPWORDS = frozenset(
    {
        "a", "about", "after", "all", "am", "an", "and", "any", "are", "as",
        "at", "be", "because", "been", "before", "being", "but", "by", "can",
        "could", "did", "do", "does", "doing", "dont", "for", "from", "had",
        "has", "have", "having", "he", "her", "hers", "him", "his", "how",
        "i", "if", "im", "in", "into", "is", "it", "its", "ive", "just",
        "me", "my", "not", "now", "of", "on", "or", "our", "out", "she",
        "should", "so", "that", "the", "their", "them", "then", "there",
        "they", "this", "to", "too", "up", "was", "we", "were", "what",
        "when", "where", "who", "why", "will", "with", "would", "you",
    }
)
_TRAILING = frozenset({"and", "or", "but", "because", "to", "with", "of", "the", "a", "an"})


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower().replace("'", "") for match in _TOKEN.finditer(text)]


def _ngrams(items: Sequence[str], width: int) -> list[tuple[str, ...]]:
    return [tuple(items[index : index + width]) for index in range(max(0, len(items) - width + 1))]


def _rouge_l_f1(prediction: Sequence[str], reference: Sequence[str]) -> float:
    if not prediction or not reference:
        return 0.0
    prior = [0] * (len(reference) + 1)
    for token in prediction:
        current = [0]
        for index, reference_token in enumerate(reference, start=1):
            current.append(prior[index - 1] + 1 if token == reference_token else max(prior[index], current[-1]))
        prior = current
    lcs = prior[-1]
    precision, recall = lcs / len(prediction), lcs / len(reference)
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _termination(text: str, *, words: int, max_new_tokens: int | None) -> tuple[bool, bool]:
    stripped = text.strip()
    if not stripped:
        return True, True
    trailing = bool((tail := _tokens(stripped[-80:])) and tail[-1] in _TRAILING)
    terminal_bad = not _TERMINAL.search(stripped) or stripped.endswith((",", ":", ";", "-", "("))
    cap_pressure = bool(max_new_tokens and max_new_tokens >= 100 and words >= int(max_new_tokens * 0.90))
    return False, terminal_bad or trailing or cap_pressure


def _skeleton(tokens: Sequence[str]) -> str:
    output: list[str] = []
    for token in tokens[:36]:
        value = token if token in _STOPWORDS else "<X>"
        if value != "<X>" or not output or output[-1] != "<X>":
            output.append(value)
    return " ".join(output)


def _mean(values: Sequence[float | int]) -> float:
    return float(mean(values)) if values else 0.0


def summarize_tldr_health(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute the locked health metrics for one evaluated alpha group."""

    word_counts: list[int] = []
    rouge: list[float] = []
    repeats2: list[float] = []
    repeats3: list[float] = []
    echo = empty = unfinished = 0
    exact: Counter[str] = Counter()
    skeletons: Counter[str] = Counter()
    totals: dict[int, int] = defaultdict(int)
    unique: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    bigram_sets: list[set[tuple[str, ...]]] = []

    for prediction, reference in rows:
        text = str(prediction.get("generated_text", "") or "").strip()
        tokens = _tokens(text)
        bigrams, trigrams = _ngrams(tokens, 2), _ngrams(tokens, 3)
        raw_cap = prediction.get("max_new_tokens")
        cap = int(raw_cap) if isinstance(raw_cap, (int, float)) and not isinstance(raw_cap, bool) else None
        is_empty, is_unfinished = _termination(text, words=len(tokens), max_new_tokens=cap)

        word_counts.append(len(tokens))
        rouge.append(_rouge_l_f1(tokens, _tokens(str(reference.get("reference_summary", "") or ""))))
        repeats2.append((len(bigrams) - len(set(bigrams))) / len(bigrams) if bigrams else 0.0)
        repeats3.append((len(trigrams) - len(set(trigrams))) / len(trigrams) if trigrams else 0.0)
        echo += int("tl;dr" in text.lower() or "tldr" in text.lower())
        empty += int(is_empty)
        unfinished += int(is_unfinished)
        exact[" ".join(tokens)] += 1
        skeletons[_skeleton(tokens)] += 1
        for width in (1, 2, 3):
            grams = _ngrams(tokens, width)
            totals[width] += len(grams)
            unique[width].update(grams)
        bigram_sets.append(set(bigrams))

    count = len(word_counts)
    if not count:
        raise ValueError("TLDR health requires at least one prediction")

    nearest = []
    for index, current in enumerate(bigram_sets):
        candidates = [
            len(current & other) / len(current | other)
            for other_index, other in enumerate(bigram_sets)
            if index != other_index and current and other and current | other
        ]
        nearest.append(max(candidates, default=0.0))

    top_skeleton = skeletons.most_common(1)[0][1] if skeletons else 0
    metrics = {
        "avg_words": _mean(word_counts),
        "rouge_l_f1_vs_human": _mean(rouge),
        "repeat_bigram_rate": _mean(repeats2),
        "repeat_trigram_rate": _mean(repeats3),
        "tldr_echo_rate": echo / count,
        "empty_output_rate": empty / count,
        "likely_unfinished_rate": unfinished / count,
        "unique_summary_ratio": len(exact) / count,
        "distinct_1": len(unique[1]) / totals[1] if totals[1] else 0.0,
        "distinct_2": len(unique[2]) / totals[2] if totals[2] else 0.0,
        "distinct_3": len(unique[3]) / totals[3] if totals[3] else 0.0,
        "top_skeleton_share": top_skeleton / count,
        "mean_nn_bigram_jaccard": _mean(nearest),
    }
    details = {
        "rows": count,
        "top_skeletons": [
            {"skeleton": value, "count": frequency}
            for value, frequency in skeletons.most_common(10)
        ],
    }
    return metrics, details
