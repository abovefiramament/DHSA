from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.data import dump_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two CAST/CECM actuator vector payloads.")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    return p.parse_args()


def _load_vectors(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    vectors = payload.get("vectors")
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError(f"No vectors found in {path}")
    return vectors


def _cosine(left: Any, right: Any) -> float:
    import torch

    left_vec = left.float().flatten()
    right_vec = right.float().flatten()
    denom = left_vec.norm() * right_vec.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float(torch.dot(left_vec, right_vec).div(denom).item())


def main() -> None:
    args = parse_args()
    reference = _load_vectors(args.reference)
    candidate = _load_vectors(args.candidate)
    common = sorted(set(reference) & set(candidate))
    if not common:
        raise ValueError(f"No shared vector ids between {args.reference} and {args.candidate}")

    rows = []
    for vector_id in common:
        ref_vec = reference[vector_id]
        cand_vec = candidate[vector_id]
        rows.append(
            {
                "name": args.name,
                "vector_id": vector_id,
                "cosine": _cosine(ref_vec, cand_vec),
                "reference_norm": float(ref_vec.float().norm().item()),
                "candidate_norm": float(cand_vec.float().norm().item()),
                "dim": int(ref_vec.numel()),
            }
        )
    rows.append(
        {
            "name": args.name,
            "vector_id": "__mean__",
            "cosine": mean(float(row["cosine"]) for row in rows),
            "reference_norm": mean(float(row["reference_norm"]) for row in rows),
            "candidate_norm": mean(float(row["candidate_norm"]) for row in rows),
            "dim": sum(int(row["dim"]) for row in rows),
        }
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_csv, rows)
    for row in rows:
        print(
            f"{row['name']}\t{row['vector_id']}\tcos={float(row['cosine']):.6f}\t"
            f"ref_norm={float(row['reference_norm']):.4f}\tcand_norm={float(row['candidate_norm']):.4f}"
        )


if __name__ == "__main__":
    main()
