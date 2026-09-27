from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from screscomp.cecm.control import join_control_parts, max_abs_alpha, parse_control_parts
from screscomp.cecm.round_executor import _round_background, execute_round
from screscomp.cecm.rounds import load_round_config, materialize_round_pairs, round_plan
from screscomp.data import dump_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Execute a family-repair GSM8K round: train/repair MLP-only and ATT-only branches first, "
            "then run a small full scan over the repaired vectors."
        )
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--family-alpha-grid", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--repair-passes", type=int, default=2)
    p.add_argument("--full-local-radius", type=int, default=1)
    p.add_argument("--rollout-max-rows", type=int, default=0)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--flush-every", type=int, default=1)
    p.add_argument("--summary-every", type=int, default=200)
    p.add_argument("--empty-cache-every", type=int, default=25)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_grid(raw: str) -> list[float]:
    out = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise ValueError("family alpha grid is empty")
    return out


def _write_effective_config(path: Path, config: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        existing = _read_json(path)
        if existing != config:
            raise SystemExit(
                f"Refusing to reuse dirty round directory with different config: {path.parent} "
                f"(use a new out-dirLOCAL_HOME or pass --force after cleaning target outputs)"
            )
    dump_json(path, config)


def _infer_rollout_rows(path: Path) -> int:
    max_index = -1
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            sample_id = str(row.get("sample_id", ""))
            if sample_id.startswith("row_"):
                try:
                    idx = int(sample_id.split("_", 1)[1])
                except Exception:
                    continue
                if idx > max_index:
                    max_index = idx
    return max_index + 1


def _stage_selected_path(stage_dir: Path) -> Path:
    path = stage_dir / "selection" / "best_small_tune_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected control under {stage_dir}")
    return path


def _run_round_stage(
    *,
    config: dict[str, Any],
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_effective_config(out_dir / "effective_round_config.json", config, force=force)
    pairs_result = None if force else _reuse_pairs_if_present(out_dir)
    if pairs_result is None:
        pairs_result = materialize_round_pairs(config, out_dir=out_dir)
    pairs_csv = Path(str(pairs_result["pairs_csv"]))
    plan = round_plan(config, out_dir=out_dir, pairs_csv=str(pairs_csv))
    plan["pair_materialization"] = pairs_result
    dump_json(out_dir / "round_plan.json", plan)
    return execute_round(
        config=config,
        out_dir=out_dir,
        pairs_csv=str(pairs_csv),
        force=force,
        skip_eval=False,
    )


def _reuse_pairs_if_present(out_dir: Path) -> dict[str, Any] | None:
    pairs_csv = out_dir / "pairs" / "pairs.csv"
    manifest = out_dir / "pairs" / "pair_build_manifest.json"
    if not pairs_csv.exists():
        return None
    payload: dict[str, Any] = {"pairs_csv": str(pairs_csv)}
    if manifest.exists():
        payload.update(_read_json(manifest))
    return payload


def _family_only_config(base: dict[str, Any], family: str, alpha_grid: list[float]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    training = cfg.setdefault("training", {})
    mlp_cfg = training.setdefault("mlp", {})
    att_cfg = training.setdefault("attention", {})
    full_cfg = training.setdefault("full", {})
    discovery = cfg.setdefault("discovery", {})
    selection = cfg.setdefault("selection", {})
    alpha = cfg.setdefault("alpha", {})

    full_cfg["enabled"] = False
    selection["exclude_controls"] = "base,prev"
    selection.pop("reference_from_eval", None)
    selection.pop("reference_config", None)
    selection.pop("reference_parts", None)
    selection.pop("reference_controls", None)
    selection.pop("reference_control_name", None)
    selection["skip_single_families"] = []

    mlp_cfg.pop("reuse_actuators", None)
    mlp_cfg.pop("init_payloads", None)
    att_cfg.pop("reuse_actuators", None)
    att_cfg.pop("init_payloads", None)
    att_cfg.pop("reuse_head_dir", None)

    if family == "mlp":
        mlp_cfg["enabled"] = True
        att_cfg["enabled"] = False
        discovery["component_types"] = ["mlp"]
        discovery["component_type_apply_modes"] = {"mlp": "prompt_last"}
        discovery["attn_layer_topk"] = 0
        discovery["allow_empty_attn"] = True
        alpha["mlp"] = list(alpha_grid)
    elif family == "att":
        mlp_cfg["enabled"] = False
        att_cfg["enabled"] = True
        discovery["component_types"] = ["attn"]
        discovery["component_type_apply_modes"] = {"attn": "all"}
        discovery["mlp_topk"] = 0
        discovery["mlp_negative_topk"] = 0
        alpha["att"] = list(alpha_grid)
    else:
        raise ValueError(f"Unsupported family: {family}")
    return cfg


def _relative_delta_parts(*, control_parts: str, background_parts: str) -> str:
    if not background_parts:
        return control_parts
    background_remaining = list(parse_control_parts(background_parts))
    kept: list[str] = []
    for part in parse_control_parts(control_parts):
        try:
            index = background_remaining.index(part)
        except ValueError:
            kept.append(part.to_text())
            continue
        background_remaining.pop(index)
    return "+".join(kept)


def _selected_alpha(selected_payload: dict[str, Any], background_parts: str) -> float:
    relative = _relative_delta_parts(
        control_parts=str(selected_payload.get("best_control_parts", "")),
        background_parts=background_parts,
    )
    return max_abs_alpha(relative)


def _local_grid(grid: list[float], center: float, radius: int) -> list[float]:
    index = min(range(len(grid)), key=lambda idx: abs(float(grid[idx]) - float(center)))
    start = max(0, index - max(0, int(radius)))
    end = min(len(grid), index + max(0, int(radius)) + 1)
    return list(grid[start:end])


def _family_repair_config(
    *,
    stage_config: dict[str, Any],
    stage_dir: Path,
    repair_rollout_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(stage_config)
    selected = _read_json(_stage_selected_path(stage_dir))
    execution = _read_json(stage_dir / "round_execution.json")

    task = cfg.setdefault("task", {})
    task["rollouts_jsonl"] = str(repair_rollout_dir / "generation_rows.jsonl")

    discovery = cfg.setdefault("discovery", {})
    discovery["component_source"] = "reuse"
    discovery["reselect_components"] = False
    discovery["reuse_selected_dir"] = str(stage_dir / "discovery" / "selected")
    discovery.pop("reuse_previous_round", None)
    discovery.pop("reuse_previous_round_dir", None)

    training = cfg.setdefault("training", {})
    mlp_cfg = training.setdefault("mlp", {})
    att_cfg = training.setdefault("attention", {})
    mlp_cfg.pop("reuse_actuators", None)
    att_cfg.pop("reuse_actuators", None)

    selected_parts = str(selected.get("best_control_parts", "") or "")
    active_component_names: set[str] = set()
    active_head_names: set[str] = set()
    for part in parse_control_parts(selected_parts):
        if part.kind == "comp":
            active_component_names.add(part.name)
        elif part.kind in {"head_act", "head_scale"}:
            active_head_names.add(part.name)

    draft_mlp_names = {str(name) for name in dict(execution.get("mlp_actuators") or {}).keys()}
    draft_head_names = {str(name) for name in dict(execution.get("head_actuators") or {}).keys()}
    active_component_names = {name for name in active_component_names if name in draft_mlp_names}
    active_head_names = {name for name in active_head_names if name in draft_head_names}
    active_mlp_suffixes = {name.split("__", 1)[1] for name in active_component_names if "__" in name}

    mlp_enabled = bool(active_component_names)
    att_enabled = bool(active_head_names)
    mlp_cfg["enabled"] = mlp_enabled
    att_cfg["enabled"] = att_enabled

    timing_grid = mlp_cfg.get("timing_grid")
    if mlp_enabled and isinstance(timing_grid, list) and active_mlp_suffixes:
        filtered = []
        for item in timing_grid:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", mlp_cfg.get("generation_apply_mode", "prefill")))
            slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "default"
            if slug in active_mlp_suffixes:
                filtered.append(item)
        if filtered:
            mlp_cfg["timing_grid"] = filtered

    mlp_init_payloads: dict[str, str] = {}
    for actuator_name, path in dict(execution.get("mlp_actuators") or {}).items():
        actuator_name = str(actuator_name)
        if actuator_name not in active_component_names:
            continue
        if "__" in actuator_name:
            role, suffix = actuator_name.split("__", 1)
            train_name = f"train_{role}_{suffix}"
        else:
            train_name = f"train_{actuator_name}"
        mlp_init_payloads[train_name] = str(path)
    if mlp_init_payloads:
        mlp_cfg["init_payloads"] = mlp_init_payloads
    else:
        mlp_cfg.pop("init_payloads", None)

    head_init_payloads: dict[str, str] = {}
    for actuator_name, path in dict(execution.get("head_actuators") or {}).items():
        actuator_name = str(actuator_name)
        if actuator_name not in active_head_names:
            continue
        head_init_payloads[f"train_{actuator_name}"] = str(path)
    if head_init_payloads:
        att_cfg["init_payloads"] = head_init_payloads
        att_cfg["reuse_head_dir"] = str(stage_dir / "discovery" / "head_scan")
    else:
        att_cfg.pop("init_payloads", None)
        att_cfg.pop("reuse_head_dir", None)
    return cfg


def _selected_actuator_args(selected_payload: dict[str, Any]) -> tuple[str, str, str]:
    component_arg = str(selected_payload.get("component_actuators_text", "") or "")
    head_arg = str(selected_payload.get("head_actuators_text", "") or "")
    control_parts = str(selected_payload.get("best_control_parts", "") or "")
    return component_arg, head_arg, control_parts


def _sample_controls(parts: str, k: int) -> str:
    return ";".join(f"sample{i:02d}={parts}" for i in range(k)) + ";"


def _run_generation(
    *,
    model_path: str,
    source_rows: str,
    prompt_key: str,
    scoring_kind: str,
    val_mod: int,
    max_rows: int,
    k: int,
    component_arg: str,
    head_arg: str,
    control_parts: str,
    out_dir: Path,
    temperature: float,
    top_p: float,
    top_k: int,
    flush_every: int,
    summary_every: int,
    empty_cache_every: int,
) -> None:
    controls = _sample_controls(control_parts, k)
    command = [
        sys.executable,
        "-m",
        "screscomp.cli.cecm_run_joint_actuator_generation",
        "--model",
        model_path,
        "--eval-open-rows",
        source_rows,
        "--component-actuators",
        component_arg,
        "--head-actuators",
        head_arg,
        "--controls",
        controls,
        "--generation-prompt-key",
        prompt_key,
        "--scoring-kind",
        scoring_kind,
        "--split",
        "all",
        "--start",
        "0",
        "--max-rows",
        str(max_rows),
        "--val-mod",
        str(val_mod),
        "--generation-apply-mode",
        "prefill",
        "--max-new-tokens",
        "512",
        "--stop-strings",
        "\nUser:",
        "--do-sample",
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--flush-every",
        str(flush_every),
        "--summary-every",
        str(summary_every),
        "--empty-cache-every",
        str(empty_cache_every),
        "--out-dir",
        str(out_dir),
        "--overwrite",
    ]
    subprocess.run(command, check=True)


def _full_compare_config(
    *,
    base_config: dict[str, Any],
    mlp_stage_dir: Path,
    att_stage_dir: Path,
    alpha_grid: list[float],
    radius: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    background_parts = str(_round_background(base_config).get("control_parts", ""))

    mlp_selected = _read_json(_stage_selected_path(mlp_stage_dir))
    att_selected = _read_json(_stage_selected_path(att_stage_dir))
    mlp_execution = _read_json(mlp_stage_dir / "round_execution.json")
    att_execution = _read_json(att_stage_dir / "round_execution.json")

    mlp_alpha = _selected_alpha(mlp_selected, background_parts)
    att_alpha = _selected_alpha(att_selected, background_parts)
    mlp_local = _local_grid(alpha_grid, mlp_alpha, radius)
    att_local = _local_grid(alpha_grid, att_alpha, radius)

    training = cfg.setdefault("training", {})
    mlp_cfg = training.setdefault("mlp", {})
    att_cfg = training.setdefault("attention", {})
    full_cfg = training.setdefault("full", {})
    discovery = cfg.setdefault("discovery", {})
    alpha = cfg.setdefault("alpha", {})
    selection = cfg.setdefault("selection", {})

    mlp_cfg["enabled"] = False
    att_cfg["enabled"] = False
    full_cfg["enabled"] = True
    mlp_cfg.pop("init_payloads", None)
    att_cfg.pop("init_payloads", None)
    att_cfg.pop("reuse_head_dir", None)
    mlp_cfg["reuse_actuators"] = {
        str(name): str(path) for name, path in dict(mlp_execution.get("mlp_actuators") or {}).items()
    }
    att_cfg["reuse_actuators"] = {
        str(name): str(path) for name, path in dict(att_execution.get("head_actuators") or {}).items()
    }

    discovery["component_source"] = "reuse"
    discovery["reselect_components"] = False
    reuse_selected_dir = mlp_stage_dir / "discovery" / "selected"
    if not reuse_selected_dir.exists():
        reuse_selected_dir = att_stage_dir / "discovery" / "selected"
    discovery["reuse_selected_dir"] = str(reuse_selected_dir)
    discovery.pop("reuse_previous_round", None)
    discovery.pop("reuse_previous_round_dir", None)

    alpha["mlp"] = mlp_local
    alpha["att"] = att_local
    alpha.pop("full", None)

    selection.pop("reference_from_eval", None)
    selection.pop("reference_config", None)
    selection.pop("reference_parts", None)
    selection.pop("reference_control_name", None)
    selection["skip_single_families"] = ["mlp", "att"]
    selection["reference_controls"] = [
        {
            "name": "repair_mlp",
            "parts": _relative_delta_parts(
                control_parts=str(mlp_selected.get("best_control_parts", "")),
                background_parts=background_parts,
            ),
            "join_background": True,
        },
        {
            "name": "repair_att",
            "parts": _relative_delta_parts(
                control_parts=str(att_selected.get("best_control_parts", "")),
                background_parts=background_parts,
            ),
            "join_background": True,
        },
    ]
    selection["exclude_controls"] = "base"
    return cfg


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    started = time.time()
    base_config = load_round_config(args.config)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_effective_config(out_dir / "effective_round_config.json", base_config, force=args.force)

    pairs_result = None if args.force else _reuse_pairs_if_present(out_dir)
    if pairs_result is None:
        pairs_result = materialize_round_pairs(base_config, out_dir=out_dir)
    plan = round_plan(base_config, out_dir=out_dir, pairs_csv=str(pairs_result["pairs_csv"]))
    plan["config_path"] = str(args.config)
    plan["pair_materialization"] = pairs_result
    dump_json(out_dir / "round_plan.json", plan)
    pairs_csv = Path(str(pairs_result["pairs_csv"]))

    alpha_grid = _float_grid(args.family_alpha_grid)
    task = base_config.get("task", {})
    model = base_config.get("model", {})
    rollout_rows = int(args.rollout_max_rows or 0)
    if rollout_rows <= 0:
        rollout_rows = _infer_rollout_rows(Path(str(task.get("rollouts_jsonl", ""))))
    if rollout_rows <= 0:
        raise SystemExit("Could not infer rollout_max_rows; pass --rollout-max-rows explicitly")
    k = int(args.k or 0)
    if k <= 0:
        k = len(task.get("rollout_control_names", []) or [])
    if k <= 0:
        raise SystemExit("Could not infer K from config; pass --k explicitly")

    family_results: dict[str, dict[str, Any]] = {}
    for family in ("mlp", "att"):
        family_root = out_dir / f"{family}_branch"
        draft_dir = family_root / "draft"
        family_cfg = _family_only_config(base_config, family, alpha_grid)
        _run_round_stage(config=family_cfg, out_dir=draft_dir, force=args.force)
        stage_dir = draft_dir
        stage_cfg = family_cfg
        for repair_idx in range(1, max(0, int(args.repair_passes)) + 1):
            selected = _read_json(_stage_selected_path(stage_dir))
            component_arg, head_arg, control_parts = _selected_actuator_args(selected)
            repair_rollout_dir = out_dir / "rollouts" / f"{family}_repair_{repair_idx:02d}_pool"
            _run_generation(
                model_path=str(model.get("path", "")),
                source_rows=str(task.get("source_jsonl", "")),
                prompt_key=str(task.get("prompt_key", "deepseek_math")),
                scoring_kind="gsm8k",
                val_mod=int(task.get("val_mod", 5)),
                max_rows=rollout_rows,
                k=k,
                component_arg=component_arg,
                head_arg=head_arg,
                control_parts=control_parts,
                out_dir=repair_rollout_dir,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                flush_every=int(args.flush_every),
                summary_every=int(args.summary_every),
                empty_cache_every=int(args.empty_cache_every),
            )
            repair_cfg = _family_repair_config(
                stage_config=stage_cfg,
                stage_dir=stage_dir,
                repair_rollout_dir=repair_rollout_dir,
            )
            repair_dir = family_root / f"repair_{repair_idx:02d}"
            _run_round_stage(config=repair_cfg, out_dir=repair_dir, force=args.force)
            stage_dir = repair_dir
            stage_cfg = repair_cfg
        family_results[family] = {
            "final_dir": str(stage_dir),
            "selected_json": str(_stage_selected_path(stage_dir)),
            "execution_json": str(stage_dir / "round_execution.json"),
        }

    full_dir = out_dir / "full_compare"
    full_cfg = _full_compare_config(
        base_config=base_config,
        mlp_stage_dir=Path(family_results["mlp"]["final_dir"]),
        att_stage_dir=Path(family_results["att"]["final_dir"]),
        alpha_grid=alpha_grid,
        radius=int(args.full_local_radius),
    )
    full_execution = _run_round_stage(config=full_cfg, out_dir=full_dir, force=args.force)

    _copy_tree_if_exists(full_dir / "selection", out_dir / "selection")
    _copy_tree_if_exists(full_dir / "eval", out_dir / "eval")
    final_selected = _stage_selected_path(full_dir)
    payload = {
        "status": "done",
        "elapsed_sec": round(time.time() - started, 3),
        "round_index": int(base_config.get("round", {}).get("index", 0)),
        "flow": "family_repair_then_full_local",
        "repair_passes": int(args.repair_passes),
        "family_alpha_grid": alpha_grid,
        "full_local_radius": int(args.full_local_radius),
        "rollout_max_rows": rollout_rows,
        "k": k,
        "background": _round_background(base_config),
        "pair_materialization": pairs_result,
        "family_results": family_results,
        "full_compare_dir": str(full_dir),
        "full_compare_execution": full_execution,
        "final_selected_json": str(final_selected),
    }
    dump_json(out_dir / "round_execution.json", payload)
    print(
        f"[cecm-family-repair] out={out_dir} selected={final_selected} "
        f"pairs={pairs_result.get('admitted_pairs', '')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
