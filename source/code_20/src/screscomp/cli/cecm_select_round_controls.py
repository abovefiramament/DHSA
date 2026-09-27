from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select the best open-generation controls from one or more CECM round eval directories. "
            "The selector is intentionally post-hoc: it does not train anything, it only ranks already "
            "logged mlp/att/full controls and exports collision-free joint-control specs."
        )
    )
    p.add_argument("--round-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--eval-name", type=str, default="")
    p.add_argument("--top-per-category", type=int, default=2)
    p.add_argument("--metric", type=str, default="final_exact")
    p.add_argument("--avoid-boundary", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "control"


def _eval_dir(round_dir: Path, eval_name: str) -> Path | None:
    if eval_name:
        path = round_dir / "eval" / eval_name
        return path if (path / "generation_summary.csv").exists() else None
    eval_root = round_dir / "eval"
    if not eval_root.exists():
        return None
    candidates = sorted(
        (path for path in eval_root.iterdir() if (path / "generation_summary.csv").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _load_control_parts(eval_dir: Path) -> dict[str, str]:
    path = eval_dir / "control_plan.csv"
    if not path.exists():
        return {}
    return {row.get("control_name", ""): row.get("parts", "") for row in load_csv(path)}


def _control_category(control_name: str, parts: str) -> str:
    if control_name == "base":
        return "base"
    has_comp = "comp:" in parts
    has_head = "head_act:" in parts or "head_scale:" in parts
    lower = control_name.lower()
    if has_comp and has_head:
        return "full"
    if has_comp or "mlp" in lower:
        return "mlp"
    if has_head or "att" in lower or "head" in lower:
        return "att"
    return "other"


def _part_alphas(parts: str, prefix: str) -> list[float]:
    values: list[float] = []
    for part in [item.strip() for item in parts.split("+") if item.strip()]:
        fields = part.split(":")
        if fields and fields[0] == prefix and len(fields) >= 3:
            try:
                values.append(float(fields[2]))
            except ValueError:
                pass
    return values


def _alpha_boundary(parts: str, mlp_grid: set[float], head_grid: set[float]) -> bool:
    boundary = False
    if len(mlp_grid) >= 2:
        lo, hi = min(mlp_grid), max(mlp_grid)
        boundary = boundary or any(alpha in {lo, hi} for alpha in _part_alphas(parts, "comp"))
    if len(head_grid) >= 2:
        lo, hi = min(head_grid), max(head_grid)
        boundary = boundary or any(alpha in {lo, hi} for alpha in _part_alphas(parts, "head_act"))
    return boundary


def _remap_parts(
    *,
    parts: str,
    prefix: str,
    component_map: dict[str, str],
    head_map: dict[str, str],
    out_components: dict[str, str],
    out_heads: dict[str, str],
) -> str:
    remapped: list[str] = []
    for part in [item.strip() for item in parts.split("+") if item.strip()]:
        fields = part.split(":")
        if fields[0] == "comp" and len(fields) >= 3:
            old_name = fields[1]
            new_name = f"{prefix}_{old_name}"
            if old_name not in component_map:
                raise ValueError(f"Missing component actuator {old_name!r} for selected part {part!r}")
            out_components[new_name] = component_map[old_name]
            fields[1] = new_name
            remapped.append(":".join(fields))
        elif fields[0] == "head_act" and len(fields) >= 3:
            old_name = fields[1]
            new_name = f"{prefix}_{old_name}"
            if old_name not in head_map:
                raise ValueError(f"Missing head actuator {old_name!r} for selected part {part!r}")
            out_heads[new_name] = head_map[old_name]
            fields[1] = new_name
            remapped.append(":".join(fields))
        else:
            remapped.append(part)
    return "+".join(remapped)


def _semicolon_map(values: dict[str, str]) -> str:
    return ";".join(f"{name}={value}" for name, value in values.items()) + (";" if values else "")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    base_by_eval: dict[str, float] = {}

    for round_dir in args.round_dirs:
        eval_dir = _eval_dir(round_dir, args.eval_name)
        if eval_dir is None:
            continue
        summary_path = eval_dir / "generation_summary.csv"
        run_config_path = eval_dir / "run_config.json"
        control_parts = _load_control_parts(eval_dir)
        run_config = _read_json(run_config_path) if run_config_path.exists() else {}
        rows = load_csv(summary_path)
        base_row = next((row for row in rows if row.get("control_name") == "base"), None)
        base_score = _float(base_row or {}, args.metric)
        base_by_eval[str(eval_dir)] = base_score
        for row in rows:
            control_name = row.get("control_name", "")
            parts = control_parts.get(control_name) or "+".join(run_config.get("controls", {}).get(control_name, []))
            category = _control_category(control_name, parts)
            if category not in {"mlp", "att", "full"}:
                continue
            score = _float(row, args.metric)
            candidates.append(
                {
                    "round_dir": str(round_dir),
                    "eval_dir": str(eval_dir),
                    "control_name": control_name,
                    "category": category,
                    "parts": parts,
                    "score": score,
                    "delta_vs_base": score - base_score,
                    "n": row.get("n", ""),
                    "parse_certified": row.get("parse_certified", ""),
                    "run_config": run_config,
                }
            )

    mlp_grid = {alpha for row in candidates for alpha in _part_alphas(str(row["parts"]), "comp")}
    head_grid = {alpha for row in candidates for alpha in _part_alphas(str(row["parts"]), "head_act")}
    for row in candidates:
        row["alpha_boundary"] = int(_alpha_boundary(str(row["parts"]), mlp_grid, head_grid))

    selected: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_category[str(row["category"])].append(row)
    for category in ("mlp", "att", "full"):
        rows = by_category.get(category, [])
        if args.avoid_boundary:
            preferred = [row for row in rows if not row["alpha_boundary"]]
            if len(preferred) >= args.top_per_category:
                rows = preferred
        rows = sorted(
            rows,
            key=lambda row: (
                float(row["score"]),
                float(row["delta_vs_base"]),
                _float({"x": str(row.get("parse_certified", ""))}, "x"),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows[: args.top_per_category], start=1):
            selected.append({**row, "rank": rank})

    component_actuators: dict[str, str] = {}
    head_actuators: dict[str, str] = {}
    controls: dict[str, str] = {"base": ""}
    selected_rows: list[dict[str, object]] = []
    for row in selected:
        prefix = _safe_name(f"{Path(str(row['round_dir'])).name}_{row['control_name']}")
        run_config = dict(row["run_config"])
        joint_name = _safe_name(f"{row['category']}_r{row['rank']}_{Path(str(row['round_dir'])).name}_{row['control_name']}")
        joint_parts = _remap_parts(
            parts=str(row["parts"]),
            prefix=prefix,
            component_map={str(k): str(v) for k, v in dict(run_config.get("component_actuators", {})).items()},
            head_map={str(k): str(v) for k, v in dict(run_config.get("head_actuators", {})).items()},
            out_components=component_actuators,
            out_heads=head_actuators,
        )
        controls[joint_name] = joint_parts
        selected_rows.append(
            {
                "category": row["category"],
                "rank": row["rank"],
                "control_name": row["control_name"],
                "joint_control_name": joint_name,
                "score": row["score"],
                "delta_vs_base": row["delta_vs_base"],
                "n": row["n"],
                "alpha_boundary": row["alpha_boundary"],
                "round_dir": row["round_dir"],
                "eval_dir": row["eval_dir"],
                "parts": row["parts"],
                "joint_parts": joint_parts,
            }
        )

    dump_csv(
        args.out_dir / "all_candidates.csv",
        [
            {
                "category": row["category"],
                "control_name": row["control_name"],
                "score": row["score"],
                "delta_vs_base": row["delta_vs_base"],
                "n": row["n"],
                "alpha_boundary": row["alpha_boundary"],
                "round_dir": row["round_dir"],
                "eval_dir": row["eval_dir"],
                "parts": row["parts"],
            }
            for row in sorted(candidates, key=lambda item: (str(item["category"]), -float(item["score"])))
        ],
    )
    dump_csv(args.out_dir / "selected_controls.csv", selected_rows)
    joint_specs = {
        "component_actuators": component_actuators,
        "head_actuators": head_actuators,
        "controls": controls,
        "component_actuators_text": _semicolon_map(component_actuators),
        "head_actuators_text": _semicolon_map(head_actuators),
        "controls_text": _semicolon_map(controls),
        "metric": args.metric,
        "avoid_boundary": bool(args.avoid_boundary),
        "base_by_eval": base_by_eval,
    }
    dump_json(args.out_dir / "joint_specs.json", joint_specs)
    print(f"[cecm-select-round-controls] selected={len(selected_rows)} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
