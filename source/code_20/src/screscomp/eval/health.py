from __future__ import annotations

import re
import string
from typing import Any


TAGGING_INSTRUCTION = (
    "\n\nIf your response mentions any possible answer candidate itself, wrap only that candidate span "
    "as <cand>...</cand>. Do not introduce extra candidates only for tagging. At the end, confirm your "
    "final answer as <final>...</final>."
)


_TAG_RE = re.compile(r"<(?P<closing>/)?(?P<tag>cand|final)>", flags=re.IGNORECASE)
_CAND_RE = re.compile(r"<cand>(?P<text>.*?)</cand>", flags=re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"<final>(?P<text>.*?)</final>", flags=re.IGNORECASE | re.DOTALL)
def add_candidate_tagging_instruction(prompt: str) -> str:
    if "If your response mentions any possible answer candidate itself" in prompt:
        return prompt
    return prompt.rstrip() + TAGGING_INSTRUCTION


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def strip_markup(text: str) -> str:
    return _TAG_RE.sub("", str(text)).strip()


def _recall_hit(text: str, answers: list[str]) -> bool:
    norm = normalize_answer(text)
    if not norm:
        return False
    return any(normalize_answer(answer) in norm for answer in answers if normalize_answer(answer))


def _source_for_text(text: str, orig_answers: list[str], cf_answers: list[str]) -> str:
    context = _recall_hit(text, cf_answers)
    prior = _recall_hit(text, orig_answers)
    if context and prior:
        return "both"
    if context:
        return "context"
    if prior:
        return "prior"
    return "neither"


def _combine_sources(sources: list[str]) -> str:
    concrete = {source for source in sources if source not in {"", "neither"}}
    if "both" in concrete or ("context" in concrete and "prior" in concrete):
        return "both"
    if "context" in concrete:
        return "context"
    if "prior" in concrete:
        return "prior"
    if sources:
        return "neither"
    return "missing"


def _alias_spans(text: str, answers: list[str], source: str) -> list[dict[str, Any]]:
    norm_to_answer: dict[str, str] = {}
    for answer in answers:
        norm = normalize_answer(answer)
        if norm and norm not in norm_to_answer:
            norm_to_answer[norm] = str(answer)
    normalized_text = normalize_answer(text)
    spans: list[dict[str, Any]] = []
    for norm, answer in norm_to_answer.items():
        pattern = re.compile(rf"(?<!\w){re.escape(norm)}(?!\w)")
        for match in pattern.finditer(normalized_text):
            spans.append(
                {
                    "source": source,
                    "start": match.start(),
                    "end": match.end(),
                    "text": answer,
                    "norm": norm,
                }
            )
    return spans


def analyze_open_timeline(
    prediction: str,
    *,
    orig_answers: list[str],
    cf_answers: list[str],
) -> dict[str, Any]:
    text = strip_markup(str(prediction))
    spans = [
        *_alias_spans(text, cf_answers, "context"),
        *_alias_spans(text, orig_answers, "prior"),
    ]
    spans.sort(key=lambda item: (item["start"], item["end"], item["source"]))
    # Merge exact same normalized span if an alias overlap survived upstream.
    merged: list[dict[str, Any]] = []
    for span in spans:
        if merged and span["start"] == merged[-1]["start"] and span["end"] == merged[-1]["end"]:
            if span["source"] != merged[-1]["source"]:
                merged[-1]["source"] = "both"
            continue
        merged.append(span)

    sources = [span["source"] for span in merged]
    first = sources[0] if sources else "missing"
    last = sources[-1] if sources else "missing"
    context_count = sum(1 for source in sources if source in {"context", "both"})
    prior_count = sum(1 for source in sources if source in {"prior", "both"})
    prior_then_context = False
    context_then_prior = False
    seen_prior = False
    seen_context = False
    for source in sources:
        if source in {"prior", "both"}:
            seen_prior = True
            if seen_context:
                context_then_prior = True
        if source in {"context", "both"}:
            seen_context = True
            if seen_prior:
                prior_then_context = True

    answer_flip = (
        (first == "prior" and last == "context")
        or (first == "context" and last == "prior")
        or prior_then_context
        or context_then_prior
    )
    return {
        "open_candidate_count": len(merged),
        "open_candidate_context_count": context_count,
        "open_candidate_prior_count": prior_count,
        "open_first_candidate_source": first,
        "open_last_candidate_source": last,
        "open_candidate_source_sequence": ">".join(sources),
        "open_prior_then_context": prior_then_context,
        "open_context_then_prior": context_then_prior,
        "open_answer_flip": answer_flip,
        "open_mixed_candidate_sources": context_count > 0 and prior_count > 0,
    }


def _tag_balance_valid(text: str) -> bool:
    counts = {
        "cand_open": len(re.findall(r"<cand>", text, flags=re.IGNORECASE)),
        "cand_close": len(re.findall(r"</cand>", text, flags=re.IGNORECASE)),
        "final_open": len(re.findall(r"<final>", text, flags=re.IGNORECASE)),
        "final_close": len(re.findall(r"</final>", text, flags=re.IGNORECASE)),
    }
    return counts["cand_open"] == counts["cand_close"] and counts["final_open"] == counts["final_close"]


def analyze_candidate_health(
    prediction: str,
    *,
    orig_answers: list[str],
    cf_answers: list[str],
) -> dict[str, Any]:
    text = str(prediction)
    cand_matches = list(_CAND_RE.finditer(text))
    final_matches = list(_FINAL_RE.finditer(text))
    candidate_sources = [
        _source_for_text(match.group("text"), orig_answers=orig_answers, cf_answers=cf_answers)
        for match in cand_matches
    ]
    context_candidates = sum(1 for source in candidate_sources if source in {"context", "both"})
    prior_candidates = sum(1 for source in candidate_sources if source in {"prior", "both"})
    neither_candidates = sum(1 for source in candidate_sources if source == "neither")

    first_candidate_source = candidate_sources[0] if candidate_sources else "missing"
    candidate_sequence = ">".join(candidate_sources)
    prior_then_context = False
    context_then_prior = False
    seen_prior = False
    seen_context = False
    for source in candidate_sources:
        if source in {"prior", "both"}:
            seen_prior = True
            if seen_context:
                context_then_prior = True
        if source in {"context", "both"}:
            seen_context = True
            if seen_prior:
                prior_then_context = True

    final_texts = [match.group("text") for match in final_matches]
    final_text = final_texts[0] if final_texts else ""
    final_candidate_matches = list(_CAND_RE.finditer(final_text))
    final_candidate_sources = [
        _source_for_text(match.group("text"), orig_answers=orig_answers, cf_answers=cf_answers)
        for match in final_candidate_matches
    ]
    if final_candidate_sources:
        final_source = _combine_sources(final_candidate_sources)
    elif final_text:
        final_source = _source_for_text(strip_markup(final_text), orig_answers=orig_answers, cf_answers=cf_answers)
    else:
        final_source = "missing"

    final_span = final_matches[0].span() if final_matches else None
    extra_candidate_count = 0
    candidates_after_final = 0
    if final_span is not None:
        for match in cand_matches:
            if not (final_span[0] <= match.start() and match.end() <= final_span[1]):
                extra_candidate_count += 1
            if match.start() > final_span[1]:
                candidates_after_final += 1
    else:
        extra_candidate_count = len(cand_matches)

    final_for_scoring = strip_markup(final_text) if final_text else strip_markup(text)
    tag_format_valid = _tag_balance_valid(text) and len(final_matches) == 1 and len(cand_matches) > 0
    healthy_context_final = (
        tag_format_valid
        and final_source == "context"
        and prior_candidates == 0
    )
    open_timeline = analyze_open_timeline(
        prediction=text,
        orig_answers=orig_answers,
        cf_answers=cf_answers,
    )

    return {
        **open_timeline,
        "tag_format_valid": tag_format_valid,
        "final_tag_count": len(final_matches),
        "candidate_tag_count": len(cand_matches),
        "final_candidate_count": len(final_candidate_matches),
        "extra_candidate_count": extra_candidate_count,
        "candidates_after_final": candidates_after_final,
        "candidate_context_count": context_candidates,
        "candidate_prior_count": prior_candidates,
        "candidate_neither_count": neither_candidates,
        "first_candidate_source": first_candidate_source,
        "final_answer_source": final_source,
        "candidate_source_sequence": candidate_sequence,
        "prior_then_context": prior_then_context,
        "context_then_prior": context_then_prior,
        "mixed_candidate_sources": context_candidates > 0 and prior_candidates > 0,
        "healthy_context_final": healthy_context_final,
        "final_text_for_scoring": final_for_scoring,
        "stripped_prediction": strip_markup(text),
    }
