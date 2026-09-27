from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from screscomp.cli.score_tldr_openai_rm4 import (
    QUERY_LENGTH,
    _batches,
    _encode_query,
    _encode_response,
    _load_rm4,
)
from screscomp.data import dump_json, dump_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate OpenAI TL;DR rm4 on preference pairs.")
    parser.add_argument("--rm4-dir", type=Path, required=True)
    parser.add_argument("--openai-code-dir", type=Path, required=True)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--act-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args(argv)


def _sample_pairs(path: Path, max_rows: int, seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    admitted = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            pair = raw.get("pair", raw)
            if not pair.get("admitted", 1):
                continue
            if not all(pair.get(key) for key in ("raw_post", "raw_title", "raw_subreddit", "y_plus", "y_minus")):
                continue
            admitted += 1
            if len(reservoir) < max_rows:
                reservoir.append(pair)
                continue
            index = rng.randrange(admitted)
            if index < max_rows:
                reservoir[index] = pair
    return reservoir, admitted


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pairs, admitted = _sample_pairs(args.pairs_jsonl, args.max_rows, args.seed)
    print(f"[tldr-rm4-pref] sampled={len(pairs)} admitted_source_rows={admitted}", flush=True)
    encoder, model, reward_head, act_dtype = _load_rm4(args)

    rows_to_score: list[dict[str, Any]] = []
    for pair in pairs:
        query_tokens = _encode_query(encoder, pair)
        for side, field in (("chosen", "y_plus"), ("rejected", "y_minus")):
            completion = str(pair[field])
            response_tokens, last_response_index = _encode_response(encoder, completion)
            rows_to_score.append(
                {
                    "sample_id": str(pair["sample_id"]),
                    "side": side,
                    "completion": completion,
                    "tokens": query_tokens + response_tokens,
                    "reward_index": QUERY_LENGTH + last_response_index,
                }
            )

    scored: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(_batches(rows_to_score, args.batch_size), start=1):
        tokens = torch.tensor([row["tokens"] for row in batch], dtype=torch.long, device=args.device)
        reward_indices = torch.tensor(
            [row["reward_index"] for row in batch], dtype=torch.long, device=args.device
        )
        with torch.inference_mode():
            acts = model(tokens, act_dtype=act_dtype)["acts"]
            all_rewards = reward_head(acts.type(reward_head.weight.dtype)).squeeze(-1)
            rewards = all_rewards.gather(1, reward_indices[:, None]).squeeze(1).float().cpu().tolist()
        for row, reward in zip(batch, rewards, strict=True):
            scored.append(
                {
                    "sample_id": row["sample_id"],
                    "side": row["side"],
                    "reward": float(reward),
                    "completion": row["completion"],
                }
            )
        if batch_index % 10 == 0:
            print(f"[tldr-rm4-pref] scored={len(scored)}/{len(rows_to_score)}", flush=True)

    by_id: dict[str, dict[str, float]] = {}
    for row in scored:
        by_id.setdefault(str(row["sample_id"]), {})[str(row["side"])] = float(row["reward"])
    deltas = [values["chosen"] - values["rejected"] for values in by_id.values()]
    summary = {
        "sampled_pairs": len(deltas),
        "admitted_source_rows": admitted,
        "seed": args.seed,
        "chosen_mean_reward": mean(values["chosen"] for values in by_id.values()),
        "rejected_mean_reward": mean(values["rejected"] for values in by_id.values()),
        "mean_reward_delta": mean(deltas),
        "preference_accuracy": sum(delta > 0 for delta in deltas) / len(deltas),
        "tie_rate": sum(delta == 0 for delta in deltas) / len(deltas),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.out_dir / "scores.jsonl", scored)
    dump_json(args.out_dir / "summary.json", summary)
    if args.device.startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(
            f"[tldr-rm4-pref] cuda_peak_allocated_gib={peak_allocated:.3f} "
            f"cuda_peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    print(f"[tldr-rm4-pref] complete summary={summary}", flush=True)


if __name__ == "__main__":
    main()
