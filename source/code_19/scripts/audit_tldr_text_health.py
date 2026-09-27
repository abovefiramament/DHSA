from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TERMINAL_RE = re.compile(r"""[.!?][)"'\]]*$""")
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "dont",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "im",
    "in",
    "into",
    "is",
    "it",
    "its",
    "ive",
    "just",
    "me",
    "my",
    "not",
    "now",
    "of",
    "on",
    "or",
    "our",
    "out",
    "she",
    "should",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "too",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TL;DR generation text health and template-like collapse metrics.")
    parser.add_argument("--out-root", type=Path, default=Path("runs/tldr_gptj_dpo_20260608_163146"))
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument(
        "--prompts-jsonl",
        type=Path,
        default=Path("data_tldr/tldr_gptj_prepared/prompts/test_prompts.jsonl"),
    )
    parser.add_argument("--temps", nargs="+", default=[f"{i / 10:.1f}" for i in range(11)])
    parser.add_argument("--methods", nargs="+", default=["old_four", "ppo"], choices=["old_four", "ppo", "sft"])
    parser.add_argument("--expected-rows", type=int, default=320)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--templates-json", type=Path, required=True)
    parser.add_argument("--include-incomplete", action="store_true")
    return parser.parse_args()


def temp_tag(temp_text: str) -> str:
    temp = float(temp_text)
    if abs(temp) < 1e-9:
        return "temp0"
    if abs(temp - round(temp)) < 1e-9:
        return f"temp{int(round(temp))}"
    return ("temp%.1f" % temp).replace(".", "p")


def generation_path(out_root: Path, sweep_root: Path, method: str, temp_text: str) -> Path:
    tag = temp_tag(temp_text)
    if method == "old_four":
        if tag == "temp0":
            return out_root / "open_test" / "old_four_temp0" / "generations.jsonl"
        if tag == "temp0p7":
            return out_root / "open_test" / "head_unified_sweetspot_formal_dpo_aligned" / "generations.jsonl"
        return sweep_root / f"old_four_{tag}" / "generations.jsonl"
    if method == "ppo":
        if tag == "temp0":
            return out_root / "open_test" / "ppo_temp0" / "generations.jsonl"
        if tag == "temp0p7":
            return out_root / "open_test" / "ppo" / "generations.jsonl"
        return sweep_root / f"ppo_{tag}" / "generations.jsonl"
    if method == "sft":
        if tag == "temp0p7":
            return out_root / "open_test" / "base" / "generations.jsonl"
        return sweep_root / f"sft_{tag}" / "generations.jsonl"
    raise ValueError(f"Unknown method: {method}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokens(text: str) -> list[str]:
    return [match.group(0).lower().replace("'", "") for match in TOKEN_RE.finditer(text)]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    if len(items) < n:
        return []
    return [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]


def rouge_l_f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for tok in pred:
        cur = [0]
        for j, ref_tok in enumerate(ref, start=1):
            if tok == ref_tok:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def source_text(prompt: dict[str, Any]) -> str:
    raw = "\n".join(str(prompt.get(key, "") or "") for key in ("raw_title", "raw_post")).strip()
    if raw:
        return raw
    text = str(prompt.get("prompt", ""))
    return text.split("TL;DR:", 1)[0]


def termination_flags(text: str, word_count: int, max_new_tokens: int | None) -> dict[str, bool]:
    stripped = text.strip()
    if not stripped:
        return {
            "empty_output": True,
            "bad_terminal_punctuation": True,
            "trailing_function_word": False,
            "hit_length_cap_proxy": False,
            "likely_unfinished": True,
        }
    last_tokens = tokens(stripped[-80:])
    trailing_bad = bool(last_tokens and last_tokens[-1] in {"and", "or", "but", "because", "to", "with", "of", "the", "a", "an"})
    terminal_bad = (not TERMINAL_RE.search(stripped)) or stripped.endswith((",", ":", ";", "-", "("))
    # We do not currently store EOS/finish_reason, so this is a conservative text-level proxy.
    # GPT-J tokens are usually fewer than words; >=90 words for a 100-token cap almost always indicates truncation pressure.
    length_cap_proxy = bool(max_new_tokens and max_new_tokens >= 100 and word_count >= int(max_new_tokens * 0.90))
    return {
        "empty_output": False,
        "bad_terminal_punctuation": terminal_bad,
        "trailing_function_word": trailing_bad,
        "hit_length_cap_proxy": length_cap_proxy,
        "likely_unfinished": terminal_bad or trailing_bad or length_cap_proxy,
    }


def skeleton_key(tok: list[str], max_tokens: int = 36) -> str:
    out: list[str] = []
    for token in tok[:max_tokens]:
        value = token if token in STOPWORDS else "<X>"
        if value == "<X>" and out and out[-1] == "<X>":
            continue
        out.append(value)
    return " ".join(out)


def normalized_text(text: str) -> str:
    return " ".join(tokens(text))


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def entropy(counter: Counter[tuple[str, ...] | str], total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log(p)
    return value


def audit_one(
    *,
    method: str,
    temp_text: str,
    path: Path,
    prompts_by_id: dict[str, dict[str, Any]],
    expected_rows: int,
    include_incomplete: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    rows = load_jsonl(path)
    if not rows or (len(rows) != expected_rows and not include_incomplete):
        return None
    rows = rows[:expected_rows]

    word_counts: list[int] = []
    char_counts: list[int] = []
    source_token_support: list[float] = []
    source_bigram_support: list[float] = []
    rouge_l: list[float] = []
    repeat_bigram_rates: list[float] = []
    repeat_trigram_rates: list[float] = []
    tldr_echo = 0
    empty_output = 0
    bad_terminal_punctuation = 0
    trailing_function_word = 0
    hit_length_cap_proxy = 0
    likely_unfinished = 0
    exact_texts: Counter[str] = Counter()
    prefix3: Counter[tuple[str, ...]] = Counter()
    prefix5: Counter[tuple[str, ...]] = Counter()
    skeletons: Counter[str] = Counter()
    corpus_ngram_totals: dict[int, int] = defaultdict(int)
    corpus_ngram_sets: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    doc_freq_4: Counter[tuple[str, ...]] = Counter()
    doc_freq_5: Counter[tuple[str, ...]] = Counter()
    bigram_sets: list[set[tuple[str, ...]]] = []
    texts_by_sample: dict[str, str] = {}

    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        prompt = prompts_by_id.get(sample_id)
        if prompt is None:
            continue
        text = str(row.get("completion", "") or "").strip()
        tok = tokens(text)
        src_tok = tokens(source_text(prompt))
        ref_tok = tokens(str(prompt.get("reference_summary", "") or ""))
        src_set = set(src_tok)
        src_bigrams = set(ngrams(src_tok, 2))
        pred_bigrams = ngrams(tok, 2)
        pred_trigrams = ngrams(tok, 3)

        word_counts.append(len(tok))
        char_counts.append(len(text))
        source_token_support.append(sum(1 for token in tok if token in src_set) / len(tok) if tok else 0.0)
        source_bigram_support.append(
            sum(1 for gram in pred_bigrams if gram in src_bigrams) / len(pred_bigrams) if pred_bigrams else 0.0
        )
        rouge_l.append(rouge_l_f1(tok, ref_tok))
        repeat_bigram_rates.append((len(pred_bigrams) - len(set(pred_bigrams))) / len(pred_bigrams) if pred_bigrams else 0.0)
        repeat_trigram_rates.append(
            (len(pred_trigrams) - len(set(pred_trigrams))) / len(pred_trigrams) if pred_trigrams else 0.0
        )
        tldr_echo += int("tl;dr" in text.lower() or "tldr" in text.lower())
        flags = termination_flags(text, len(tok), int(row.get("max_new_tokens", 0) or 0))
        empty_output += int(flags["empty_output"])
        bad_terminal_punctuation += int(flags["bad_terminal_punctuation"])
        trailing_function_word += int(flags["trailing_function_word"])
        hit_length_cap_proxy += int(flags["hit_length_cap_proxy"])
        likely_unfinished += int(flags["likely_unfinished"])
        exact_texts[normalized_text(text)] += 1
        if len(tok) >= 3:
            prefix3[tuple(tok[:3])] += 1
        if len(tok) >= 5:
            prefix5[tuple(tok[:5])] += 1
        skeletons[skeleton_key(tok)] += 1
        for n in (1, 2, 3):
            grams = ngrams(tok, n)
            corpus_ngram_totals[n] += len(grams)
            corpus_ngram_sets[n].update(grams)
        doc_freq_4.update(set(ngrams(tok, 4)))
        doc_freq_5.update(set(ngrams(tok, 5)))
        bigram_sets.append(set(pred_bigrams))
        texts_by_sample[sample_id] = text

    n = len(word_counts)
    if n == 0:
        return None

    nn_jaccards: list[float] = []
    for i, grams in enumerate(bigram_sets):
        best = 0.0
        if grams:
            for j, other in enumerate(bigram_sets):
                if i == j or not other:
                    continue
                union = len(grams | other)
                if union:
                    best = max(best, len(grams & other) / union)
        nn_jaccards.append(best)

    def top_share(counter: Counter[Any]) -> float:
        return counter.most_common(1)[0][1] / n if counter else 0.0

    top10_4 = [gram for gram, _ in doc_freq_4.most_common(10)]
    top10_4_coverage = 0
    if top10_4:
        top10_set = set(top10_4)
        for text in texts_by_sample.values():
            if any(gram in top10_set for gram in set(ngrams(tokens(text), 4))):
                top10_4_coverage += 1

    summary = {
        "method": method,
        "temperature": temp_text,
        "tag": temp_tag(temp_text),
        "path": str(path),
        "n": n,
        "avg_words": safe_mean(word_counts),
        "median_words": safe_median(word_counts),
        "p95_words": sorted(word_counts)[int(0.95 * (n - 1))],
        "avg_chars": safe_mean(char_counts),
        "source_token_support": safe_mean(source_token_support),
        "source_bigram_support": safe_mean(source_bigram_support),
        "rouge_l_f1_vs_human": safe_mean(rouge_l),
        "repeat_bigram_rate": safe_mean(repeat_bigram_rates),
        "repeat_trigram_rate": safe_mean(repeat_trigram_rates),
        "tldr_echo_rate": tldr_echo / n,
        "empty_output_rate": empty_output / n,
        "bad_terminal_punctuation_rate": bad_terminal_punctuation / n,
        "trailing_function_word_rate": trailing_function_word / n,
        "hit_length_cap_proxy_rate": hit_length_cap_proxy / n,
        "likely_unfinished_rate": likely_unfinished / n,
        "exact_duplicate_rate": sum(count for count in exact_texts.values() if count > 1) / n,
        "unique_summary_ratio": len(exact_texts) / n,
        "distinct_1": len(corpus_ngram_sets[1]) / corpus_ngram_totals[1] if corpus_ngram_totals[1] else 0.0,
        "distinct_2": len(corpus_ngram_sets[2]) / corpus_ngram_totals[2] if corpus_ngram_totals[2] else 0.0,
        "distinct_3": len(corpus_ngram_sets[3]) / corpus_ngram_totals[3] if corpus_ngram_totals[3] else 0.0,
        "top_prefix3_share": top_share(prefix3),
        "top_prefix5_share": top_share(prefix5),
        "prefix5_entropy_norm": entropy(prefix5, sum(prefix5.values())) / math.log(max(len(prefix5), 2)),
        "top_skeleton_share": top_share(skeletons),
        "unique_skeleton_ratio": len(skeletons) / n,
        "top4gram_doc_share": (doc_freq_4.most_common(1)[0][1] / n) if doc_freq_4 else 0.0,
        "top5gram_doc_share": (doc_freq_5.most_common(1)[0][1] / n) if doc_freq_5 else 0.0,
        "top10_4gram_doc_coverage": top10_4_coverage / n,
        "mean_nn_bigram_jaccard": safe_mean(nn_jaccards),
        "p95_nn_bigram_jaccard": sorted(nn_jaccards)[int(0.95 * (n - 1))],
        "high_nn_bigram_jaccard_rate_ge_0p30": sum(1 for value in nn_jaccards if value >= 0.30) / n,
    }

    details = {
        "method": method,
        "temperature": temp_text,
        "tag": temp_tag(temp_text),
        "top_prefix5": [{"prefix": " ".join(key), "count": value} for key, value in prefix5.most_common(10)],
        "top_4grams": [{"ngram": " ".join(key), "doc_count": value} for key, value in doc_freq_4.most_common(20)],
        "top_5grams": [{"ngram": " ".join(key), "doc_count": value} for key, value in doc_freq_5.most_common(20)],
        "top_skeletons": [{"skeleton": key, "count": value} for key, value in skeletons.most_common(10)],
    }
    return summary, details


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root or (args.out_root / "open_test" / "temp_sweep")
    prompts = load_jsonl(args.prompts_jsonl)
    prompts_by_id = {str(row["sample_id"]): row for row in prompts}

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for temp_text in args.temps:
        for method in args.methods:
            path = generation_path(args.out_root, sweep_root, method, temp_text)
            result = audit_one(
                method=method,
                temp_text=temp_text,
                path=path,
                prompts_by_id=prompts_by_id,
                expected_rows=args.expected_rows,
                include_incomplete=args.include_incomplete,
            )
            if result is None:
                summaries.append(
                    {
                        "method": method,
                        "temperature": temp_text,
                        "tag": temp_tag(temp_text),
                        "path": str(path),
                        "n": 0,
                        "status": "missing_or_incomplete",
                    }
                )
                continue
            summary, detail = result
            summary["status"] = "ok"
            summaries.append(summary)
            details.append(detail)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.templates_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in summaries for key in row})
    preferred = ["method", "temperature", "tag", "status", "n", "avg_words", "source_token_support"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with args.out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    args.templates_json.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"[tldr-text-health] wrote {args.out_csv}")
    print(f"[tldr-text-health] wrote {args.templates_json}")


if __name__ == "__main__":
    main()
