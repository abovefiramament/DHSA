from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank CK-PLUG alpha sweep rows by a fixed arbitration objective.")
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    return p.parse_args()


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def _alpha_from_method(method: str) -> str:
    match = re.search(r"_alpha([0-9.]+)$", method)
    return match.group(1) if match else ""


def main() -> None:
    args = parse_args()
    with args.summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    output: list[dict[str, Any]] = []
    for row in rows:
        score = (
            _as_float(row, "context_only_rate")
            + _as_float(row, "cf_em_rate")
            - _as_float(row, "po")
            - _as_float(row, "neither_rate")
        )
        output.append(
            {
                **row,
                "alpha_from_method": _alpha_from_method(row.get("method", "")),
                "arbitration_score": score,
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["arbitration_score"]),
            -float(row.get("context_only_rate", 0.0)),
            float(row.get("mr", 0.0)),
            str(row.get("method", "")),
        )
    )
    dump_csv(args.out_csv, output)
    if output:
        best = output[0]
        print(
            "[select-ckplug-alpha] best="
            f"{best['method']} alpha={best['alpha_from_method']} score={best['arbitration_score']}"
        )


if __name__ == "__main__":
    main()
