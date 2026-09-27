from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.data import dump_csv, dump_jsonl, load_jsonl


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


class JudgeParseError(RuntimeError):
    def __init__(self, message: str, raw_content: str, attempts: int, attempt_logs: list[dict[str, Any]]):
        super().__init__(message)
        self.raw_content = raw_content
        self.attempts = attempts
        self.attempt_logs = attempt_logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate generation health with an external LLM judge.")
    p.add_argument("--generations_jsonl", type=Path, required=True)
    p.add_argument("--open_rows_jsonl", type=Path, required=True)
    p.add_argument("--out_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument(
        "--debug_jsonl",
        type=Path,
        default=None,
        help="Optional debug log with every request message and raw API response. Never stores the API key.",
    )
    p.add_argument("--base_url", type=str, default="https://api.deepseek.com/chat/completions")
    p.add_argument("--model", type=str, default="deepseek-v4-pro")
    p.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    p.add_argument("--methods", type=str, default="", help="Comma-separated methods to annotate. Empty means all.")
    p.add_argument("--max_rows", type=int, default=40)
    p.add_argument(
        "--paired_by_sample",
        action="store_true",
        help="When multiple methods are selected, sample sample_ids that contain every selected method.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Number of sample_ids for --paired_by_sample. Defaults to max_rows.",
    )
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--max_aliases", type=int, default=5)
    p.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Send thinking={type: disabled} for APIs/models that otherwise spend tokens on reasoning_content.",
    )
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--sleep_seconds", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def _parse_csv(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _load_open_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in load_jsonl(path)}


def _select_rows(rows: list[dict[str, Any]], methods: set[str], max_rows: int, seed: int) -> list[dict[str, Any]]:
    if methods:
        rows = [row for row in rows if row.get("method") in methods]
    if max_rows <= 0 or max_rows >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_rows))
    return [rows[i] for i in indices]


def _select_paired_rows(
    rows: list[dict[str, Any]],
    *,
    methods: set[str],
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not methods:
        raise ValueError("--paired_by_sample requires --methods with at least two methods.")
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        method = str(row.get("method", ""))
        if method in methods:
            by_sample[str(row["sample_id"])][method] = row
    eligible = sorted(sample_id for sample_id, items in by_sample.items() if methods <= set(items))
    if max_samples > 0 and max_samples < len(eligible):
        rng = random.Random(seed)
        eligible = sorted(rng.sample(eligible, max_samples))
    selected: list[dict[str, Any]] = []
    for sample_id in eligible:
        for method in sorted(methods):
            selected.append(by_sample[sample_id][method])
    return selected


def _system_prompt() -> str:
    return (
        "You are a JSON-only answer consistency tagger. Do not answer the question. "
        "Do not explain. Return the JSON object immediately."
    )


def _limited_aliases(open_row: dict[str, Any], key: str, fallback_key: str, max_aliases: int) -> list[str]:
    raw = open_row.get(key, [open_row.get(fallback_key, "")])
    if not isinstance(raw, list):
        raw = [raw]
    aliases: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item)
        if text and text not in seen:
            aliases.append(text)
            seen.add(text)
        if max_aliases > 0 and len(aliases) >= max_aliases:
            break
    return aliases


def _user_prompt(gen_row: dict[str, Any], open_row: dict[str, Any], max_aliases: int) -> str:
    prior_aliases = _limited_aliases(open_row, "orig_answers", "orig_answer", max_aliases)
    context_aliases = _limited_aliases(open_row, "cf_answers", "cf_answer", max_aliases)
    return (
        "Tag complete answer-candidate spans. Use exact substrings from OUTPUT only.\n"
        "Do not tag partial aliases, location suffixes, modifiers, or words that are not complete answers.\n"
        f"PRIOR={json.dumps(prior_aliases, ensure_ascii=False)}\n"
        f"CONTEXT={json.dumps(context_aliases, ensure_ascii=False)}\n"
        f"OUTPUT={json.dumps(str(gen_row.get('prediction', '')), ensure_ascii=False)}\n"
        "Return ONLY this compact JSON, no notes:\n"
        '{"candidate_spans":[{"text":"","source":"context|prior|both|other|unknown"}],'
        '"final_answer_span":"","final_source":"context|prior|both|neither|unknown",'
        '"answer_order":["context|prior|both|other|unknown"],'
        '"has_answer_flip":false,"has_internal_inconsistency":false,'
        '"inconsistency_type":"none|prior_then_context|context_then_prior|mixed_unresolved|unsupported_final|self_contradiction|irrelevant"}'
    )


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise
        return json.loads(match.group(0))


def _call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], str, int, list[dict[str, Any]]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Exception | None = None
    attempt_logs: list[dict[str, Any]] = []
    for attempt in range(retries):
        attempt_count = attempt + 1
        request = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
        base_attempt_log: dict[str, Any] = {
            "attempt": attempt_count,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "disable_thinking": disable_thinking,
            "request_messages": payload["messages"],
        }
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            body = json.loads(raw)
            message = body["choices"][0]["message"]
            content = message.get("content", "")
            attempt_log = {
                **base_attempt_log,
                "http_ok": True,
                "finish_reason": body.get("choices", [{}])[0].get("finish_reason", ""),
                "usage": body.get("usage", {}),
                "response_message": message,
                "response_content": content,
                "raw_response": raw,
            }
            try:
                parsed = _extract_json(content)
                attempt_logs.append({**attempt_log, "parse_ok": True, "parse_error": ""})
                return parsed, content, attempt_count, attempt_logs
            except Exception as exc:
                last_error = exc
                attempt_logs.append({**attempt_log, "parse_ok": False, "parse_error": str(exc)})
                if attempt + 1 >= retries:
                    raise JudgeParseError(str(exc), content, attempt_count, attempt_logs) from exc
        except urllib.error.HTTPError as exc:
            last_error = exc
            detail = exc.read().decode("utf-8", errors="replace")
            attempt_logs.append(
                {
                    **base_attempt_log,
                    "http_ok": False,
                    "http_status": exc.code,
                    "http_error": detail,
                    "parse_ok": False,
                }
            )
            if attempt + 1 >= retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except JudgeParseError:
            raise
        except Exception as exc:  # pragma: no cover
            last_error = exc
            attempt_logs.append(
                {
                    **base_attempt_log,
                    "http_ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "parse_ok": False,
                }
            )
            if attempt + 1 >= retries:
                raise
        time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Judge request failed: {last_error}")


def _contains_alias(text: str, aliases: list[str]) -> bool:
    text_norm = text.lower()
    for alias in aliases:
        alias_norm = alias.lower()
        if not alias_norm:
            continue
        if len(alias_norm) < 3:
            continue
        if alias_norm in text_norm:
            return True
        if len(text_norm) >= 4 and text_norm in alias_norm and alias_norm.startswith(text_norm):
            return True
    return False


def _find_case_insensitive_span(prediction: str, text: str) -> str | None:
    if not text:
        return ""
    index = prediction.lower().find(text.lower())
    if index < 0:
        return None
    return prediction[index : index + len(text)]


def _infer_source(text: str, source: str, prior_aliases: list[str], context_aliases: list[str]) -> str:
    has_prior = _contains_alias(text, prior_aliases)
    has_context = _contains_alias(text, context_aliases)
    if has_prior and has_context:
        return "both"
    if has_context:
        return "context"
    if has_prior:
        return "prior"
    return source if source in {"context", "prior", "both", "other", "unknown", "neither"} else "unknown"


def _dedupe_consecutive(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if not out or out[-1] != item:
            out.append(item)
    return out


def _find_alias_mentions(prediction: str, aliases: list[str], source: str) -> list[dict[str, str]]:
    mentions: list[dict[str, str]] = []
    pred_lower = prediction.lower()
    for alias in aliases:
        alias_text = str(alias)
        if not alias_text:
            continue
        if len(alias_text.strip()) < 3:
            continue
        index = pred_lower.find(alias_text.lower())
        if index >= 0:
            mentions.append({"text": prediction[index : index + len(alias_text)], "source": source})
    return mentions


def _merge_spans(spans: list[dict[str, str]], extra_spans: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {(item["text"].lower(), item["source"]) for item in spans}
    out = list(spans)
    for item in extra_spans:
        key = (item["text"].lower(), item["source"])
        if key not in seen:
            out.append(item)
            seen.add(key)
    out.sort(key=lambda item: item["text"].lower())
    return out


def _validate_annotation(
    annotation: dict[str, Any],
    prediction: str,
    *,
    prior_aliases: list[str],
    context_aliases: list[str],
) -> dict[str, Any]:
    spans = annotation.get("candidate_spans", [])
    if not isinstance(spans, list):
        spans = []
    span_violations = 0
    clean_spans: list[dict[str, str]] = []
    for item in spans:
        if not isinstance(item, dict):
            span_violations += 1
            continue
        text = str(item.get("text", ""))
        source = str(item.get("source", "unknown"))
        fixed_text = _find_case_insensitive_span(prediction, text)
        if fixed_text is None:
            span_violations += 1
            continue
        text = fixed_text
        source = _infer_source(text, source, prior_aliases, context_aliases)
        clean_spans.append({"text": text, "source": source})

    final_answer_span = str(annotation.get("final_answer_span", ""))
    fixed_final = _find_case_insensitive_span(prediction, final_answer_span)
    final_span_valid = final_answer_span == "" or fixed_final is not None
    if not final_span_valid:
        span_violations += 1
    elif fixed_final is not None:
        final_answer_span = fixed_final

    allowed_sources = {"context", "prior", "both", "other", "unknown", "neither"}
    final_source = str(annotation.get("final_source", "unknown"))
    if final_source not in allowed_sources:
        final_source = "unknown"
    if final_answer_span:
        final_source = _infer_source(final_answer_span, final_source, prior_aliases, context_aliases)
    alias_mentions = _find_alias_mentions(prediction, context_aliases, "context") + _find_alias_mentions(
        prediction, prior_aliases, "prior"
    )
    clean_spans = _merge_spans(clean_spans, alias_mentions)
    if not final_answer_span and alias_mentions:
        final_answer_span = alias_mentions[-1]["text"]
        final_span_valid = True
        final_source = alias_mentions[-1]["source"]
    elif final_source in {"unknown", "neither", "other"} and alias_mentions:
        final_source = alias_mentions[-1]["source"]
    inconsistency_type = str(annotation.get("inconsistency_type", "none"))
    allowed_types = {
        "none",
        "prior_then_context",
        "context_then_prior",
        "mixed_unresolved",
        "unsupported_final",
        "self_contradiction",
        "irrelevant",
    }
    if inconsistency_type not in allowed_types:
        inconsistency_type = "none"

    ordered_spans = sorted(clean_spans, key=lambda item: prediction.lower().find(item["text"].lower()))
    order = _dedupe_consecutive(
        [item["source"] for item in ordered_spans if item["source"] in {"context", "prior", "both", "other", "unknown"}]
    )
    if not order:
        raw_order = annotation.get("answer_order", [])
        if not isinstance(raw_order, list):
            raw_order = []
        order = [str(x) for x in raw_order if str(x) in allowed_sources]
    has_answer_flip = False
    if final_source in {"context", "prior"} and order:
        first_source = next((item for item in order if item in {"context", "prior"}), "")
        has_answer_flip = bool(first_source and first_source != final_source)
    has_context_prior_mix = "context" in order and "prior" in order
    judge_inconsistent = bool(annotation.get("has_internal_inconsistency", False))
    if inconsistency_type in {"prior_then_context", "context_then_prior"} and not has_context_prior_mix:
        judge_inconsistent = False
        inconsistency_type = "none"
    if inconsistency_type == "mixed_unresolved" and not has_context_prior_mix:
        judge_inconsistent = False
        inconsistency_type = "none"
    return {
        "judge_candidate_spans": clean_spans,
        "judge_final_answer_span": final_answer_span,
        "judge_final_span_valid": final_span_valid,
        "judge_final_source": final_source,
        "judge_answer_order": order,
        "judge_has_answer_flip": has_answer_flip,
        "judge_has_internal_inconsistency": judge_inconsistent,
        "judge_inconsistency_type": inconsistency_type,
        "judge_notes": str(annotation.get("notes", "")),
        "judge_span_violation_count": span_violations,
        "judge_parse_ok": True,
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    out: list[dict[str, Any]] = []
    for method, group in sorted(by_method.items()):
        n = len(group)
        type_counts = Counter(row.get("judge_inconsistency_type", "none") for row in group)
        final_counts = Counter(row.get("judge_final_source", "unknown") for row in group)
        out.append(
            {
                "method": method,
                "n": n,
                "judge_parse_ok_rate": sum(1 for row in group if row.get("judge_parse_ok")) / n if n else 0.0,
                "judge_span_violation_rate": (
                    sum(1 for row in group if int(row.get("judge_span_violation_count", 0) or 0) > 0) / n
                    if n
                    else 0.0
                ),
                "judge_final_context_rate": final_counts["context"] / n if n else 0.0,
                "judge_final_prior_rate": final_counts["prior"] / n if n else 0.0,
                "judge_final_both_rate": final_counts["both"] / n if n else 0.0,
                "judge_final_neither_rate": final_counts["neither"] / n if n else 0.0,
                "judge_answer_flip_rate": sum(1 for row in group if row.get("judge_has_answer_flip")) / n if n else 0.0,
                "judge_internal_inconsistency_rate": (
                    sum(1 for row in group if row.get("judge_has_internal_inconsistency")) / n if n else 0.0
                ),
                "judge_mixed_unresolved_rate": type_counts["mixed_unresolved"] / n if n else 0.0,
                "judge_unsupported_final_rate": type_counts["unsupported_final"] / n if n else 0.0,
                "judge_self_contradiction_rate": type_counts["self_contradiction"] / n if n else 0.0,
                "judge_mean_candidate_count": mean(len(row.get("judge_candidate_spans", [])) for row in group)
                if group
                else 0.0,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {args.api_key_env}")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{args.api_key_env} contains non-ASCII characters. "
            "Set it to the actual API key string, not a placeholder."
        ) from exc

    open_rows = _load_open_rows(args.open_rows_jsonl)
    all_gen_rows = load_jsonl(args.generations_jsonl)
    methods = _parse_csv(args.methods)
    if args.paired_by_sample:
        gen_rows = _select_paired_rows(
            all_gen_rows,
            methods=methods,
            max_samples=args.max_samples if args.max_samples is not None else args.max_rows,
            seed=args.sample_seed,
        )
    else:
        gen_rows = _select_rows(
            all_gen_rows,
            methods=methods,
            max_rows=args.max_rows,
            seed=args.sample_seed,
        )
    output: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    print(
        "[annotate-generation-health] planned "
        f"rows={len(gen_rows)} samples={len({str(row['sample_id']) for row in gen_rows})} "
        f"methods={','.join(sorted(methods)) if methods else 'all'}"
    )
    for row in gen_rows:
        sample_id = str(row["sample_id"])
        if sample_id not in open_rows:
            raise KeyError(f"Missing open row for sample_id={sample_id}")
        system_prompt = _system_prompt()
        prompt = _user_prompt(row, open_rows[sample_id], args.max_aliases)
        judge_attempts = 0
        attempt_logs: list[dict[str, Any]] = []
        try:
            annotation, _raw, judge_attempts, attempt_logs = _call_chat_completion(
                base_url=args.base_url,
                api_key=api_key,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                disable_thinking=args.disable_thinking,
                timeout=args.timeout,
                retries=args.retries,
            )
            prior_aliases = _limited_aliases(open_rows[sample_id], "orig_answers", "orig_answer", args.max_aliases)
            context_aliases = _limited_aliases(open_rows[sample_id], "cf_answers", "cf_answer", args.max_aliases)
            validated = _validate_annotation(
                annotation,
                str(row.get("prediction", "")),
                prior_aliases=prior_aliases,
                context_aliases=context_aliases,
            )
        except Exception as exc:
            raw_response = exc.raw_content if isinstance(exc, JudgeParseError) else ""
            judge_attempts = exc.attempts if isinstance(exc, JudgeParseError) else max(1, args.retries)
            attempt_logs = exc.attempt_logs if isinstance(exc, JudgeParseError) else attempt_logs
            validated = {
                "judge_parse_ok": False,
                "judge_error": str(exc),
                "judge_raw_response": raw_response,
                "judge_candidate_spans": [],
                "judge_final_answer_span": "",
                "judge_final_span_valid": False,
                "judge_final_source": "unknown",
                "judge_answer_order": [],
                "judge_has_answer_flip": False,
                "judge_has_internal_inconsistency": False,
                "judge_inconsistency_type": "none",
                "judge_notes": "",
                "judge_span_violation_count": 0,
            }
        if args.debug_jsonl is not None:
            for attempt_log in attempt_logs:
                debug_rows.append(
                    {
                        "sample_id": sample_id,
                        "method": row.get("method", ""),
                        "source_index": row.get("source_index", ""),
                        "prediction": row.get("prediction", ""),
                        "orig_answer": open_rows[sample_id].get("orig_answer", ""),
                        "cf_answer": open_rows[sample_id].get("cf_answer", ""),
                        **attempt_log,
                    }
                )
        output.append(
            {
                "sample_id": sample_id,
                "method": row.get("method", ""),
                "source_index": row.get("source_index", ""),
                "outcome": row.get("outcome", ""),
                "prediction": row.get("prediction", ""),
                "orig_answer": open_rows[sample_id].get("orig_answer", ""),
                "cf_answer": open_rows[sample_id].get("cf_answer", ""),
                "judge_attempts": judge_attempts,
                **validated,
            }
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    dump_jsonl(args.out_jsonl, output)
    if args.debug_jsonl is not None:
        dump_jsonl(args.debug_jsonl, debug_rows)
    dump_csv(args.out_summary_csv, _summary(output))
    total_attempts = sum(int(row.get("judge_attempts", 0) or 0) for row in output)
    print(f"[annotate-generation-health] rows={len(output)} judge_attempts={total_attempts} out={args.out_jsonl}")
    if args.debug_jsonl is not None:
        print(f"[annotate-generation-health] debug_attempts={len(debug_rows)} debug={args.debug_jsonl}")


if __name__ == "__main__":
    main()
