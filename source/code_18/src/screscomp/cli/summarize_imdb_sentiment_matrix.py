from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect IMDb sentiment-control score summaries under one run root.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _format_alpha_sweep(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(f"{float(item):.8g}" for item in value)
    return str(value or "")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")
    out_csv = args.out_csv or (root / "matrix_score_summary.csv")

    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("**/score_summary.csv")):
        if summary_path == out_csv:
            continue
        eval_dir = summary_path.parent
        if eval_dir.name == "shared_base":
            continue
        manifest = _read_json(eval_dir / "generation_manifest.json")
        for row in _read_csv(summary_path):
            rows.append(
                {
                    "eval_dir": eval_dir.name,
                    "relative_dir": str(eval_dir.relative_to(root)),
                    "manifest_control_name": manifest.get("control_name", ""),
                    "generation_apply_mode": manifest.get("generation_apply_mode", ""),
                    "component_apply_mode": manifest.get("component_apply_mode", ""),
                    "head_apply_mode": manifest.get("head_apply_mode", ""),
                    "actuator_path": manifest.get("actuator_path", ""),
                    "head_actuator_path": manifest.get("head_actuator_path", ""),
                    "samples_per_prompt": manifest.get("samples_per_prompt", ""),
                    "alpha_sweep": _format_alpha_sweep(manifest.get("alpha_sweep", "")),
                    **row,
                }
            )

    dump_csv(out_csv, rows)
    print(f"[imdb-matrix-summary] wrote rows={len(rows)} path={out_csv}", flush=True)


if __name__ == "__main__":
    main()
