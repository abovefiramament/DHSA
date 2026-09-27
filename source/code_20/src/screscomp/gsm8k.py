from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from statistics import mean
from typing import Any, Iterable


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_CERTIFIED_PARSE_SOURCES = {"boxed", "hash", "answer_marker"}
_COT_CUE_RE = re.compile(
    r"\b("
    r"first|second|next|then|therefore|so|because|since|"
    r"we need|we can|let's|let us|calculate|total|altogether|"
    r"step\s*\d+"
    r")\b",
    re.IGNORECASE,
)


def normalize_numeric_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = text.replace(",", "")
    if text.endswith("."):
        text = text[:-1]
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if not number.is_finite():
        return text
    if number == number.to_integral_value():
        try:
            normalized = format(number.to_integral_value(), "f")
        except (InvalidOperation, ValueError):
            return text
        normalized = normalized.split(".", 1)[0]
        return "0" if normalized in {"-0", "+0"} else normalized
    try:
        normalized = format(number.normalize(), "f")
    except (InvalidOperation, ValueError):
        return text
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", "+0", ""}:
        return "0"
    return normalized


def extract_final_number(text: Any) -> str:
    parsed, _source = extract_final_number_with_source(text)
    return parsed


def extract_boxed_span(text: Any) -> str:
    raw = str(text or "")
    matches = re.findall(r"(\\boxed\s*\{[^{}]+\})", raw)
    return matches[-1] if matches else ""

def extract_boxed_answer_text(text: Any) -> str:
    raw = str(text or "")
    matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    return matches[-1].strip() if matches else ""

def extract_final_number_with_source(text: Any) -> tuple[str, str]:
    raw = str(text or "")
    if not raw.strip():
        return "", "empty"
    boxed_matches = re.findall(r"\\boxed\s*\{([^{}]+)\}", raw)
    if boxed_matches:
        match = _NUMBER_RE.search(boxed_matches[-1])
        if match:
            return normalize_numeric_answer(match.group(0)), "boxed"
    if "####" in raw:
        tail = raw.rsplit("####", 1)[-1]
        match = _NUMBER_RE.search(tail)
        return (normalize_numeric_answer(match.group(0)), "hash") if match else ("", "hash_empty")

    answer_markers = (
        "final answer is",
        "answer is",
        "answer:",
        "therefore,",
        "therefore",
    )
    lowered = raw.lower()
    for marker in answer_markers:
        idx = lowered.rfind(marker)
        if idx >= 0:
            matches = list(_NUMBER_RE.finditer(raw[idx:]))
            if matches:
                return normalize_numeric_answer(matches[-1].group(0)), "answer_marker"

    matches = list(_NUMBER_RE.finditer(raw))
    if not matches:
        return "", "no_number"
    return normalize_numeric_answer(matches[-1].group(0)), "last_number_fallback"


def unique_numbers(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        number = extract_final_number(value) if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value or "").strip()) else normalize_numeric_answer(value)
        if number and number not in seen:
            seen.add(number)
            out.append(number)
    return out


def base_math_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def answer_only_math_prompt(question: str) -> str:
    return (
        "Answer the math word problem. Return only the final number.\n\n"
        f"Q: {question}\nA:"
    )


def cot_math_prompt(question: str) -> str:
    return (
        "Solve the math word problem step by step. End with the final answer as #### <number>.\n\n"
        f"Q: {question}\nA:"
    )


def deepseek_math_cot_prompt(question: str) -> str:
    return (
        "User: Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        f"{question}\n\n"
        "Assistant:"
    )


def deepseek_math_prompt(question: str) -> str:
    return (
        f"User: {question}\n\n"
        "Put your final answer within \\boxed{}.\n\n"
        "Assistant:"
    )


def strong_cot_prompt(question: str) -> str:
    return (
        "Carefully solve the grade-school math problem. Track every quantity, avoid shortcuts, "
        "and end with exactly one final line in the form #### <number>.\n\n"
        f"Q: {question}\nA:"
    )


def reasoning_step_count(prediction: Any) -> int:
    text = str(prediction or "").strip()
    if not text:
        return 0
    explicit_steps = re.findall(r"(?im)^\s*(?:step\s*\d+\s*[:.)-]|\d+\s*[.)]\s+)", text)
    if explicit_steps:
        return len(explicit_steps)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    reasoning_lines = [
        line
        for line in lines
        if _COT_CUE_RE.search(line) or re.search(r"\d+\s*[-+*/=]\s*\d+", line)
    ]
    if len(reasoning_lines) >= 2:
        return len(reasoning_lines)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    reasoning_sentences = [
        sent
        for sent in sentences
        if _COT_CUE_RE.search(sent) or re.search(r"\d+\s*[-+*/=]\s*\d+", sent)
    ]
    return len(reasoning_sentences)


def classify_cot_behavior(prediction: Any, *, min_tokens: int = 24, min_chars: int = 80) -> dict[str, object]:
    text = str(prediction or "")
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    steps = reasoning_step_count(text)
    cue_hits = len(_COT_CUE_RE.findall(text))
    equation_hits = len(re.findall(r"\d+\s*[-+*/=]\s*\d+", text))
    has_final_marker = bool(re.search(r"(####|\\boxed|final answer|answer is)", text, flags=re.IGNORECASE))
    cot_like = bool(
        len(tokens) >= min_tokens
        and len(text.strip()) >= min_chars
        and (steps >= 2 or cue_hits >= 2 or equation_hits >= 2)
    )
    return {
        "cot_like": cot_like,
        "reasoning_steps": steps,
        "cot_cue_hits": cue_hits,
        "equation_hits": equation_hits,
        "has_final_marker": has_final_marker,
    }


def classify_gsm8k_prediction(
    prediction: Any,
    *,
    gold_answer: Any,
    wrong_answers: Iterable[Any] = (),
) -> dict[str, object]:
    parsed, parse_source = extract_final_number_with_source(prediction)
    gold = normalize_numeric_answer(gold_answer)
    wrongs = [normalize_numeric_answer(item) for item in wrong_answers]
    wrongs = [item for item in wrongs if item]
    exact = bool(parsed and gold and parsed == gold)
    wrong_hit = bool(parsed and parsed in set(wrongs))
    valid = bool(parsed)
    if exact and wrong_hit:
        outcome = "both"
    elif exact:
        outcome = "gold"
    elif wrong_hit:
        outcome = "known_wrong"
    elif valid:
        outcome = "other_wrong"
    else:
        outcome = "invalid"
    parse_certified = parse_source in _CERTIFIED_PARSE_SOURCES
    return {
        "parsed_answer": parsed,
        "parse_source": parse_source,
        "parse_certified": parse_certified,
        "gold_answer": gold,
        "wrong_answers_json": list(wrongs),
        "final_exact": exact,
        "strict_final_exact": bool(exact and parse_certified),
        "fallback_exact": bool(exact and parse_source == "last_number_fallback"),
        "numeric_valid": valid,
        "known_wrong_hit": wrong_hit,
        "other_wrong": valid and not exact and not wrong_hit,
        "invalid": not valid,
        "gsm8k_outcome": outcome,
        "output_chars": len(str(prediction or "")),
        **classify_cot_behavior(prediction),
    }


def summarize_gsm8k_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("control_name", "")), []).append(row)
    summary: list[dict[str, object]] = []
    for control_name, group in sorted(groups.items()):
        n = len(group)
        if not n:
            continue
        summary.append(
            {
                "control_name": control_name,
                "n": n,
                "final_exact": mean(float(bool(row.get("final_exact"))) for row in group),
                "strict_final_exact": mean(float(bool(row.get("strict_final_exact"))) for row in group),
                "fallback_exact": mean(float(bool(row.get("fallback_exact"))) for row in group),
                "numeric_valid": mean(float(bool(row.get("numeric_valid"))) for row in group),
                "parse_certified": mean(float(bool(row.get("parse_certified"))) for row in group),
                "known_wrong_hit": mean(float(bool(row.get("known_wrong_hit"))) for row in group),
                "other_wrong": mean(float(bool(row.get("other_wrong"))) for row in group),
                "invalid": mean(float(bool(row.get("invalid"))) for row in group),
                "cot_like": mean(float(bool(row.get("cot_like"))) for row in group),
                "has_final_marker": mean(float(bool(row.get("has_final_marker"))) for row in group),
                "fallback_parse": mean(
                    float(str(row.get("parse_source", "")) == "last_number_fallback") for row in group
                ),
                "boxed_parse": mean(float(str(row.get("parse_source", "")) == "boxed") for row in group),
                "hash_parse": mean(float(str(row.get("parse_source", "")) == "hash") for row in group),
                "answer_marker_parse": mean(
                    float(str(row.get("parse_source", "")) == "answer_marker") for row in group
                ),
                "mean_reasoning_steps": mean(float(row.get("reasoning_steps", 0)) for row in group),
                "mean_chars": mean(float(row.get("output_chars", 0)) for row in group),
            }
        )
    return summary
