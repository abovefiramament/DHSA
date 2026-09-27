from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import itertools
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl


LABELS = ("A", "B")
SYSTEM_PROMPT = (
    "You are an impartial evaluator. Follow the requested comparison criteria exactly. "
    "The summaries are anonymous. Return only a valid JSON object."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO-aligned blind pairwise TL;DR win-rate evaluation using DS4-Pro.")
    p.add_argument("--prompts-jsonl", type=Path, required=True)
    p.add_argument("--candidate", action="append", required=True, metavar="NAME=GENERATIONS_JSONL")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--expected-rows", type=int, default=320)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--random-seed", type=int, default=20260613)
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--max-output-tokens", type=int, default=800)
    p.add_argument("--thinking-mode", choices=["disabled", "enabled"], default="disabled")
    p.add_argument("--reasoning-effort", choices=["high", "max"], default="high")
    p.add_argument("--review-fraction", type=float, default=0.10)
    p.add_argument(
        "--human-only",
        action="store_true",
        help="Only compare each candidate against the human reference, skipping candidate-vs-candidate pairs.",
    )
    p.add_argument(
        "--allow-batch-seed-mismatch",
        action="store_true",
        help="Allow comparisons whose generation seeds differ, for example deterministic decoding runs.",
    )
    p.add_argument("--skip-review", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--input-price-per-million", type=float, default=0.435)
    p.add_argument("--output-price-per-million", type=float, default=0.87)
    return p.parse_args(argv)


def _parse_candidates(values: list[str], *, human_only: bool = False) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"Invalid --candidate {value!r}; expected NAME=GENERATIONS_JSONL")
        if name.strip() == "human":
            raise ValueError("Candidate name 'human' is reserved")
        candidates.append((name.strip(), Path(path.strip())))
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    if human_only:
        if not candidates:
            raise ValueError("At least one --candidate is required in --human-only mode")
        return candidates
    if len(candidates) != 2:
        raise ValueError("Exactly two uniquely named --candidate arguments are required")
    return candidates


def _indexed(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in indexed:
            raise ValueError(f"Missing or duplicate sample_id={sample_id!r} in {source}")
        indexed[sample_id] = row
    return indexed


def _seed_for(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _mapping(left: str, right: str, sample_id: str, seed: int, pass_name: str) -> dict[str, str]:
    systems = [left, right]
    random.Random(_seed_for(seed, pass_name, sample_id, left, right)).shuffle(systems)
    return dict(zip(LABELS, systems, strict=True))


def _review_mapping(first_mapping: dict[str, str]) -> dict[str, str]:
    return {"A": first_mapping["B"], "B": first_mapping["A"]}


def _request_text(packet: dict[str, Any]) -> str:
    return f"""Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

Post:
{packet["source"]}

Summary A:
{packet["summaries"]["A"]}

Summary B:
{packet["summaries"]["B"]}

First provide a one-sentence comparison of the two summaries, explaining which you prefer and why. Then choose only A or B.
Return JSON exactly as:
{{"comparison": "one-sentence comparison and explanation", "preferred": "A or B"}}"""


def build_packets(
    args: argparse.Namespace,
    candidates: list[tuple[str, Path]],
    *,
    pass_name: str = "first",
    first_keys: list[dict[str, Any]] | None = None,
    selected_packet_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = load_jsonl(args.prompts_jsonl)
    prompt_by_id = _indexed(prompts, args.prompts_jsonl)
    candidate_rows = [(name, _indexed(load_jsonl(path), path)) for name, path in candidates]
    common = set(prompt_by_id)
    for _, rows in candidate_rows:
        common &= set(rows)
    sample_ids = [str(row["sample_id"]) for row in prompts if str(row["sample_id"]) in common]
    expected = args.max_rows if args.max_rows is not None else args.expected_rows
    if len(sample_ids) < expected:
        raise ValueError(f"Only {len(sample_ids)} aligned rows are available; expected at least {expected}")
    sample_ids = sample_ids[:expected]
    summaries_by_sample: dict[str, dict[str, str]] = {}
    for sample_id in sample_ids:
        prompt = prompt_by_id[sample_id]
        records = {name: rows[sample_id] for name, rows in candidate_rows}
        source_indices = {int(prompt.get("source_row_index", -1)), *(int(row.get("source_row_index", -2)) for row in records.values())}
        if len(source_indices) != 1:
            raise ValueError(f"source_row_index mismatch for sample_id={sample_id}")
        batch_seeds = {str(row.get("batch_seed", "")) for row in records.values()}
        if len(batch_seeds) != 1 and not args.allow_batch_seed_mismatch:
            raise ValueError(f"batch_seed mismatch for sample_id={sample_id}: {sorted(batch_seeds)}")
        if any(str(row.get("split", "")) != "test" for row in [prompt, *records.values()]):
            raise ValueError(f"Non-test row found for sample_id={sample_id}")
        summaries = {name: str(row.get("completion", "")).strip() for name, row in records.items()}
        summaries["human"] = str(prompt.get("reference_summary", "")).strip()
        if any(not value for value in summaries.values()):
            raise ValueError(f"Empty summary for sample_id={sample_id}")
        summaries_by_sample[sample_id] = summaries

    systems = [name for name, _ in candidates] + ["human"]
    first_key_by_packet = {str(row["packet_id"]): row for row in first_keys or []}
    packets: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    judge_index = 0
    for sample_id in sample_ids:
        prompt = prompt_by_id[sample_id]
        for left, right in itertools.combinations(systems, 2):
            if args.human_only and "human" not in {left, right}:
                continue
            comparison = f"{left}_vs_{right}"
            packet_id = f"{sample_id}::{comparison}"
            if selected_packet_ids is not None and packet_id not in selected_packet_ids:
                continue
            if pass_name == "review":
                mapping = _review_mapping(first_key_by_packet[packet_id]["label_to_system"])
            else:
                mapping = _mapping(left, right, sample_id, args.random_seed, pass_name)
            packet = {
                "judge_index": judge_index,
                "packet_id": packet_id,
                "sample_id": sample_id,
                "comparison": comparison,
                "source_row_index": prompt.get("source_row_index"),
                "source": str(prompt.get("prompt", "")).strip(),
                "summaries": {label: summaries_by_sample[sample_id][system] for label, system in mapping.items()},
            }
            packets.append(packet)
            keys.append(
                {
                    "judge_index": judge_index,
                    "packet_id": packet_id,
                    "sample_id": sample_id,
                    "comparison": comparison,
                    "label_to_system": mapping,
                }
            )
            judge_index += 1
    return packets, keys


def _validate_judgment(value: dict[str, Any]) -> dict[str, str]:
    preferred = str(value.get("preferred", "")).strip().upper()
    comparison = str(value.get("comparison", "")).strip()
    if preferred not in LABELS:
        raise ValueError(f"Invalid preferred value: {preferred!r}")
    if not comparison:
        raise ValueError("Missing comparison")
    return {"preferred": preferred, "comparison": comparison}


def _api_request(args: argparse.Namespace, api_key: str, packet: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _request_text(packet)}],
        "thinking": {"type": args.thinking_mode},
        "response_format": {"type": "json_object"},
        "max_tokens": args.max_output_tokens,
        "stream": False,
    }
    if args.thinking_mode == "enabled":
        body["reasoning_effort"] = args.reasoning_effort
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    message = payload["choices"][0]["message"]
    judgment = _validate_judgment(json.loads(str(message.get("content", "")).strip()))
    return judgment, {
        "model": payload.get("model", args.model),
        "usage": payload.get("usage", {}),
        "reasoning_chars": len(str(message.get("reasoning_content", ""))),
    }


def _judge_one(args: argparse.Namespace, api_key: str, packet: dict[str, Any], key: dict[str, Any], pass_name: str) -> dict[str, Any]:
    last_error = ""
    for attempt in range(args.max_retries):
        try:
            judgment, metadata = _api_request(args, api_key, packet)
            preferred = judgment["preferred"]
            return {
                "judge_index": packet["judge_index"],
                "packet_id": packet["packet_id"],
                "sample_id": packet["sample_id"],
                "comparison": packet["comparison"],
                "pass": pass_name,
                "status": "ok",
                "winner": key["label_to_system"][preferred],
                "anonymous_judgment": judgment,
                **metadata,
            }
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < args.max_retries:
                time.sleep(min(2**attempt, 16))
    return {
        "judge_index": packet["judge_index"],
        "packet_id": packet["packet_id"],
        "sample_id": packet["sample_id"],
        "comparison": packet["comparison"],
        "pass": pass_name,
        "status": "error",
        "error": last_error,
    }


def _latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = {str(row["packet_id"]): row for row in rows}
    return sorted(latest.values(), key=lambda row: (str(row["sample_id"]), str(row["comparison"])))


def _run_pass(
    args: argparse.Namespace,
    packets: list[dict[str, Any]],
    keys: list[dict[str, Any]],
    output_path: Path,
    pass_name: str,
) -> list[dict[str, Any]]:
    existing = load_jsonl(output_path) if output_path.exists() else []
    done = {str(row["packet_id"]) for row in existing if row.get("status") == "ok"}
    todo = [(packet, key) for packet, key in zip(packets, keys, strict=True) if packet["packet_id"] not in done]
    if not todo:
        return _latest_rows(existing)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key in environment variable {args.api_key_env}")
    with output_path.open("a", encoding="utf-8") as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(_judge_one, args, api_key, packet, key, pass_name) for packet, key in todo]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[tldr-dpo-judge] pass={pass_name} completed={index}/{len(todo)} packet={row['packet_id']} status={row['status']}")
    return _latest_rows(load_jsonl(output_path))


def _review_packet_ids(first_rows: list[dict[str, Any]], args: argparse.Namespace) -> set[str]:
    threshold = int(max(0.0, min(1.0, args.review_fraction)) * 10_000)
    return {
        str(row["packet_id"])
        for row in first_rows
        if row.get("status") != "ok" or _seed_for(args.random_seed, "review", row["packet_id"]) % 10_000 < threshold
    }


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    rate = wins / n
    denominator = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denominator
    return center - radius, center + radius


def summarize(first_rows: list[dict[str, Any]], review_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    first_ok = [row for row in first_rows if row.get("status") == "ok"]
    review_by_packet = {str(row["packet_id"]): row for row in review_rows if row.get("status") == "ok"}
    pairwise: list[dict[str, Any]] = []
    for comparison in sorted({str(row["comparison"]) for row in first_ok}):
        rows = [row for row in first_ok if row["comparison"] == comparison]
        left, right = comparison.split("_vs_", 1)
        left_wins = sum(row["winner"] == left for row in rows)
        right_wins = sum(row["winner"] == right for row in rows)
        reviewed = [row for row in rows if row["packet_id"] in review_by_packet]
        agreements = sum(review_by_packet[row["packet_id"]]["winner"] == row["winner"] for row in reviewed)
        review_left_wins = sum(review_by_packet[row["packet_id"]]["winner"] == left for row in reviewed)
        preferred_a = sum(row.get("anonymous_judgment", {}).get("preferred") == "A" for row in rows)
        wilson_low, wilson_high = _wilson_interval(left_wins, len(rows))
        pairwise.append(
            {
                "pair": comparison,
                "left": left,
                "right": right,
                "n": len(rows),
                "left_wins": left_wins,
                "right_wins": right_wins,
                "left_win_rate": left_wins / len(rows) if rows else 0.0,
                "left_win_rate_wilson95_low": wilson_low,
                "left_win_rate_wilson95_high": wilson_high,
                "anonymous_preferred_a": preferred_a,
                "anonymous_preferred_a_rate": preferred_a / len(rows) if rows else 0.0,
                "reviewed": len(reviewed),
                "review_left_wins": review_left_wins,
                "review_left_win_rate": review_left_wins / len(reviewed) if reviewed else None,
                "order_swap_agreements": agreements,
                "order_swap_agreement_rate": agreements / len(reviewed) if reviewed else None,
            }
        )
    usage_rows = [row for row in [*first_rows, *review_rows] if row.get("status") == "ok"]
    input_tokens = sum(int(row.get("usage", {}).get("prompt_tokens", 0)) for row in usage_rows)
    output_tokens = sum(int(row.get("usage", {}).get("completion_tokens", 0)) for row in usage_rows)
    estimated_cost = input_tokens / 1_000_000 * args.input_price_per_million + output_tokens / 1_000_000 * args.output_price_per_million
    return {
        "protocol": "DPO Appendix C.2 summarization GPT-4 win-rate prompt (C), JSON-adapted for DS4-Pro",
        "primary_score": "first-pass randomly ordered pairwise win rate; review never changes primary score",
        "first_pass_ok": len(first_ok),
        "first_pass_errors": sum(row.get("status") != "ok" for row in first_rows),
        "review_ok": sum(row.get("status") == "ok" for row in review_rows),
        "pairwise": pairwise,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "estimated_cost_usd": estimated_cost},
    }


def _write_manifest(args: argparse.Namespace, candidates: list[tuple[str, Path]], sample_rows: int, packets: int) -> None:
    dump_json(
        args.out_dir / "judge_manifest.json",
        {
            "protocol": "tldr_dpo_appendix_c2_concise_pairwise_v1",
            "source": "Direct Preference Optimization paper, Appendix C.2, Summarization GPT-4 win rate prompt (C)",
            "primary_score": "first-pass pairwise win rate",
            "review_semantics": "deterministic subset with A/B order swapped; audit only",
            "prompts_jsonl": str(args.prompts_jsonl),
            "candidates": [{"name": name, "path": str(path)} for name, path in candidates],
            "sample_rows": sample_rows,
            "judge_packets": packets,
            "model": args.model,
            "thinking_mode": args.thinking_mode,
            "random_seed": args.random_seed,
            "review_fraction": args.review_fraction,
            "human_only": args.human_only,
            "api_key_env": args.api_key_env,
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidates = _parse_candidates(args.candidate, human_only=args.human_only)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        first_rows = _latest_rows(load_jsonl(args.out_dir / "judge_first_pass.jsonl"))
        review_path = args.out_dir / "judge_order_swap_review.jsonl"
        review_rows = _latest_rows(load_jsonl(review_path)) if review_path.exists() else []
    else:
        packets, keys = build_packets(args, candidates)
        dump_jsonl(args.out_dir / "judge_packets.jsonl", packets)
        dump_jsonl(args.out_dir / "judge_keys_private.jsonl", keys)
        sample_rows = len({str(packet["sample_id"]) for packet in packets})
        _write_manifest(args, candidates, sample_rows, len(packets))
        print(f"[tldr-dpo-judge] prepared_samples={sample_rows} prepared_pairwise_packets={len(packets)}")
        if args.prepare_only:
            return
        first_rows = _run_pass(args, packets, keys, args.out_dir / "judge_first_pass.jsonl", "first")
        if args.skip_review:
            review_rows = []
        else:
            review_ids = _review_packet_ids(first_rows, args)
            review_packets, review_keys = build_packets(
                args, candidates, pass_name="review", first_keys=keys, selected_packet_ids=review_ids
            )
            dump_jsonl(args.out_dir / "judge_order_swap_packets.jsonl", review_packets)
            review_rows = _run_pass(
                args, review_packets, review_keys, args.out_dir / "judge_order_swap_review.jsonl", "review"
            )
    summary = summarize(first_rows, review_rows, args)
    dump_json(args.out_dir / "pairwise_summary.json", summary)
    dump_csv(args.out_dir / "pairwise_summary.csv", summary["pairwise"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
