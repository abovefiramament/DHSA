from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.data import (
    build_prior_ready_pairs,
    dump_csv,
    dump_jsonl,
    filter_stable_triples,
    generate_fact_pair_candidates,
    load_csv,
    normalize_triples,
    parse_seed_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build fact-pair candidates and prior-only prompts.")
    p.add_argument("--seed_csv", type=Path, required=True, help="Input curated seed facts CSV.")
    p.add_argument("--out_dir", type=Path, default=Path("data"), help="Output data root directory.")
    p.add_argument("--num_counters", type=int, default=1, help="Counter candidates per fact.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--template_family", type=str, default="main_v1", help="Prompt template family.")
    p.add_argument(
        "--out_funnel_csv",
        type=Path,
        default=None,
        help="Optional CSV path for build-stage sample funnel.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_rows = load_csv(args.seed_csv)
    raw_triples = parse_seed_rows(raw_rows)
    norm_triples = normalize_triples(raw_triples)
    stable_triples, excluded_triples = filter_stable_triples(norm_triples)
    pair_candidates = generate_fact_pair_candidates(
        triples=stable_triples,
        num_counters=args.num_counters,
        seed=args.seed,
    )
    prompt_ready = build_prior_ready_pairs(pair_candidates, template_family=args.template_family)

    inter = args.out_dir / "01_intermediate"
    dump_jsonl(inter / "01_raw_triples.jsonl", [x.to_dict() for x in raw_triples])
    dump_jsonl(inter / "02_norm_triples.jsonl", [x.to_dict() for x in norm_triples])
    dump_jsonl(inter / "02_excluded_triples.jsonl", [x.to_dict() for x in excluded_triples])
    dump_jsonl(inter / "02_stable_triples.jsonl", [x.to_dict() for x in stable_triples])
    dump_jsonl(inter / "03_fact_pairs_candidates.jsonl", [x.to_dict() for x in pair_candidates])
    dump_jsonl(inter / "04_prompt_ready_pairs.jsonl", [x.to_dict() for x in prompt_ready])

    if args.out_funnel_csv is not None:
        dump_csv(
            args.out_funnel_csv,
            [
                {
                    "raw_count": len(raw_triples),
                    "norm_count": len(norm_triples),
                    "stable_count": len(stable_triples),
                    "pair_count": len(pair_candidates),
                    "prompt_ready_count": len(prompt_ready),
                }
            ],
        )

    print(
        f"[build] raw={len(raw_triples)} norm={len(norm_triples)} stable={len(stable_triples)} "
        f"excluded={len(excluded_triples)} pairs={len(pair_candidates)}"
    )


if __name__ == "__main__":
    main()
