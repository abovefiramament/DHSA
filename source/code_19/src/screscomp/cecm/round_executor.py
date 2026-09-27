from __future__ import annotations

import shlex
import subprocess
import sys
import time
import copy
from pathlib import Path
from typing import Any

from screscomp.cecm.control import (
    join_control_parts,
    max_abs_alpha,
    parse_control_parts,
    remap_control_parts,
    scale_control_parts,
)
from screscomp.data import dump_csv, dump_json, load_csv


COMPONENT_SOURCE_RESELECT = "reselect"
COMPONENT_SOURCE_REUSE = "reuse"
COMPONENT_SOURCE_REUSE_PREVIOUS = "reuse_previous"
COMPONENT_SOURCE_MODES = (
    COMPONENT_SOURCE_RESELECT,
    COMPONENT_SOURCE_REUSE,
    COMPONENT_SOURCE_REUSE_PREVIOUS,
)


def _cmd_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _python_module(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def _string_list(values: Any, *, default: list[str]) -> list[str]:
    if values in (None, ""):
        return list(default)
    if isinstance(values, str):
        return [item.strip() for item in values.split(",") if item.strip()]
    return [str(item) for item in values]


def _float_list(values: Any, *, default: list[float]) -> list[float]:
    if values in (None, ""):
        return list(default)
    if isinstance(values, str):
        return [float(item.strip()) for item in values.split(",") if item.strip()]
    return [float(item) for item in values]


def _join(values: Any, *, default: str) -> str:
    if values in (None, ""):
        return default
    if isinstance(values, str):
        return values
    return ",".join(str(item) for item in values)


def _alpha_name(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_") or "default"


def _mlp_timing_modes(config: dict[str, Any]) -> list[tuple[str, str, str]]:
    training = config.get("training", {})
    mlp_config = training.get("mlp", {})
    timing_grid = mlp_config.get("timing_grid")
    if not timing_grid:
        name = str(mlp_config.get("generation_apply_mode", "prefill"))
        train_mode = str(mlp_config.get("train_apply_mode", "prompt_last"))
        generation_mode = str(mlp_config.get("generation_apply_mode", "prefill"))
        return [(name, train_mode, generation_mode)]
    modes: list[tuple[str, str, str]] = []
    for item in timing_grid:
        if not isinstance(item, dict):
            continue
        generation_mode = str(item.get("generation_apply_mode", mlp_config.get("generation_apply_mode", "prefill")))
        train_mode = str(item.get("train_apply_mode", mlp_config.get("train_apply_mode", "prompt_last")))
        name = str(item.get("name", generation_mode))
        modes.append((name, train_mode, generation_mode))
    if not modes:
        name = str(mlp_config.get("generation_apply_mode", "prefill"))
        modes.append((name, str(mlp_config.get("train_apply_mode", "prompt_last")), name))
    return modes


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_has_rows(path: Path) -> bool:
    return path.exists() and len(load_csv(path)) > 0


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _semicolon_path_map(values: dict[str, Any]) -> str:
    return ";".join(f"{name}={path}" for name, path in sorted(values.items()))


def _actuators_used_by_control(
    control_parts: str,
    *,
    component_actuators: dict[str, Any],
    head_actuators: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    component_names: set[str] = set()
    head_names: set[str] = set()
    for part in parse_control_parts(control_parts):
        if part.kind == "comp":
            component_names.add(part.name)
        elif part.kind in {"head_act", "head_scale"}:
            head_names.add(part.name)
    return (
        {name: path for name, path in component_actuators.items() if name in component_names},
        {name: path for name, path in head_actuators.items() if name in head_names},
    )


def _round_index(config: dict[str, Any]) -> int:
    return int(config.get("round", {}).get("index", 0))


def _selection_json_candidates(round_path: Path, config: dict[str, Any] | None = None) -> list[Path]:
    candidates: list[Path] = []
    if config is not None:
        write_best = config.get("selection", {}).get("write_best_config")
        if write_best:
            candidates.append(round_path / str(write_best))
    candidates.extend(
        [
            round_path / "selection" / "best_small_tune_config.json",
            round_path / "selection" / "best_config.json",
            round_path / "selection" / "best_config" / "best_config.json",
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _load_selected_control(round_path: Path, config: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
    for path in _selection_json_candidates(round_path, config):
        if path.exists():
            payload = _read_json(path)
            if str(payload.get("best_control_parts", "")).strip():
                return path, payload
    raise FileNotFoundError(f"No selected control JSON with best_control_parts under {round_path}")


def _round_background(config: dict[str, Any]) -> dict[str, object]:
    background = dict(config.get("background", {}) or {})
    index = _round_index(config)
    enabled = _truthy(
        background.get("enabled"),
        default=index > 0 and _truthy(config.get("round", {}).get("auto_previous_control"), default=False),
    )
    if not enabled:
        return {
            "enabled": False,
            "control_parts": "",
            "component_actuators": {},
            "head_actuators": {},
            "source": "",
            "prev_scale": 0.0,
        }

    prev_scale = float(background.get("prev_scale", config.get("round", {}).get("prev_scale", 1.0)))
    explicit_parts = str(background.get("control_parts", "") or "")
    explicit_components = dict(background.get("component_actuators", {}) or {})
    explicit_heads = dict(background.get("head_actuators", {}) or {})
    if explicit_parts:
        return {
            "enabled": True,
            "control_parts": scale_control_parts(explicit_parts, prev_scale),
            "component_actuators": {str(k): str(v) for k, v in explicit_components.items()},
            "head_actuators": {str(k): str(v) for k, v in explicit_heads.items()},
            "source": "explicit",
            "prev_scale": prev_scale,
        }
    if explicit_components or explicit_heads:
        return {
            "enabled": True,
            "control_parts": "",
            "component_actuators": {str(k): str(v) for k, v in explicit_components.items()},
            "head_actuators": {str(k): str(v) for k, v in explicit_heads.items()},
            "source": "explicit_actuators_only",
            "prev_scale": prev_scale,
        }

    previous_dir = Path(str(background.get("previous_round_dir") or _previous_round_dir(config)))
    selected_path, selected = _load_selected_control(previous_dir, config)
    execution_path = previous_dir / "round_execution.json"
    execution = _read_json(execution_path) if execution_path.exists() else {}
    component_actuators = {
        str(k): str(v)
        for k, v in dict(selected.get("component_actuators") or execution.get("mlp_actuators") or {}).items()
    }
    head_actuators = {
        str(k): str(v)
        for k, v in dict(selected.get("head_actuators") or execution.get("head_actuators") or {}).items()
    }
    prefix = str(background.get("prefix") or f"prev_r{max(index - 1, 0):02d}_")
    component_name_map = {name: f"{prefix}{name}" for name in component_actuators}
    head_name_map = {name: f"{prefix}{name}" for name in head_actuators}
    return {
        "enabled": True,
        "control_parts": remap_control_parts(
            str(selected["best_control_parts"]),
            component_names=component_name_map,
            head_names=head_name_map,
            scale=prev_scale,
        ),
        "component_actuators": {component_name_map[name]: path for name, path in component_actuators.items()},
        "head_actuators": {head_name_map[name]: path for name, path in head_actuators.items()},
        "source": str(selected_path),
        "prev_scale": prev_scale,
        "previous_round_dir": str(previous_dir),
        "previous_control_name": str(selected.get("best_control_name", "")),
    }


def _append_background_args(command: list[str], config: dict[str, Any]) -> list[str]:
    background = _round_background(config)
    control_parts = str(background.get("control_parts", ""))
    if not control_parts:
        return command
    component_actuators = _semicolon_path_map(dict(background.get("component_actuators", {}) or {}))
    head_actuators = _semicolon_path_map(dict(background.get("head_actuators", {}) or {}))
    return [
        *command,
        "--background-component-actuators",
        component_actuators,
        "--background-head-actuators",
        head_actuators,
        "--background-control-parts",
        control_parts,
    ]


def _append_model_args(command: list[str], model: dict[str, Any]) -> list[str]:
    out = [
        *command,
        "--torch-dtype",
        str(model.get("torch_dtype", "auto")),
        "--device",
        str(model.get("device", "auto")),
    ]
    if _truthy(model.get("use_chat_template"), default=False):
        out.append("--use-chat-template")
    return out


def _run_command(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    expected: Path | None,
    force: bool,
) -> dict[str, object]:
    if expected is not None and expected.exists() and not force:
        return {
            "name": name,
            "status": "skipped_existing",
            "expected": str(expected),
            "log": str(log_path),
            "command": command,
        }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%S')}] $ {_cmd_text(command)}\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] exit={completed.returncode}\n")
    elapsed = round(time.time() - started, 3)
    if completed.returncode != 0:
        raise RuntimeError(f"Round phase failed: {name}; see {log_path}")
    if expected is not None and not expected.exists():
        raise RuntimeError(f"Round phase {name} completed but did not create expected file: {expected}")
    return {
        "name": name,
        "status": "done",
        "elapsed_sec": elapsed,
        "expected": str(expected) if expected is not None else "",
        "log": str(log_path),
        "command": command,
    }


def _component_source_mode(discovery: dict[str, Any]) -> str:
    mode = str(discovery.get("component_source") or discovery.get("component_mode") or "").strip()
    if not mode:
        if discovery.get("reuse_selected_dir") or discovery.get("reuse_component_screen_csv") or discovery.get("reuse_components"):
            mode = COMPONENT_SOURCE_REUSE
        elif _truthy(discovery.get("reuse_previous_round"), default=False):
            mode = COMPONENT_SOURCE_REUSE_PREVIOUS
        elif _truthy(discovery.get("reselect_components"), default=True):
            mode = COMPONENT_SOURCE_RESELECT
        else:
            mode = COMPONENT_SOURCE_REUSE
    if mode not in COMPONENT_SOURCE_MODES:
        raise ValueError(f"Unsupported discovery.component_source={mode!r}; expected one of {COMPONENT_SOURCE_MODES}")
    return mode


def _previous_round_dir(config: dict[str, Any]) -> Path:
    discovery = config.get("discovery", {})
    explicit = discovery.get("previous_round_dir") or discovery.get("reuse_previous_round_dir")
    if explicit:
        return Path(str(explicit))
    index = int(config.get("round", {}).get("index", 0))
    if index <= 0:
        raise ValueError("reuse_previous requires discovery.previous_round_dir when round.index <= 0")
    root = Path(str(config.get("output", {}).get("root", "data/rounds/default")))
    return root / f"round_{index - 1:02d}"


def _resolve_reuse_object(config: dict[str, Any]) -> tuple[str, Path]:
    discovery = config.get("discovery", {})
    selected = discovery.get("reuse_selected_dir") or discovery.get("reuse_components")
    screen = discovery.get("reuse_component_screen_csv")
    if selected:
        path = Path(str(selected))
        if path.is_dir() and (path / "component_screen.csv").exists() and not (path / "components.csv").exists():
            return "component_screen_csv", path / "component_screen.csv"
        if path.name == "component_screen.csv":
            return "component_screen_csv", path
        return "selected_dir", path
    if screen:
        return "component_screen_csv", Path(str(screen))
    raise ValueError("component_source='reuse' requires reuse_selected_dir, reuse_component_screen_csv, or reuse_components")


def _resolve_previous_object(config: dict[str, Any]) -> tuple[str, Path]:
    previous = _previous_round_dir(config)
    selected = previous / "discovery" / "selected"
    if (selected / "component_selection_manifest.json").exists():
        return "selected_dir", selected
    screen = previous / "discovery" / "component_screen.csv"
    if screen.exists():
        return "component_screen_csv", screen
    raise ValueError(f"Could not reuse previous round components from {previous}")


def _scan_components(
    *,
    config: dict[str, Any],
    out_dir: Path,
    pairs_csv: str,
    force: bool,
) -> tuple[Path, list[dict[str, object]]]:
    model = config.get("model", {})
    task = config.get("task", {})
    training = config.get("training", {})
    discovery = config.get("discovery", {})
    event = str(task.get("event", "source_context_over_prior"))
    scan_mode = str(discovery.get("scan_mode", "margin"))
    score_mode = str(training.get("score_mode", "avglogp"))
    option_selection_mode = str(training.get("option_selection_mode", "model_max"))
    max_aliases = str(training.get("max_aliases_per_side", 1))
    component_types = _string_list(discovery.get("component_types"), default=["attn", "mlp"])
    type_apply_modes = {str(k): str(v) for k, v in dict(discovery.get("component_type_apply_modes", {})).items()}
    rollout_cfg = dict(discovery.get("rollout", {}) or {})
    scorer_cfg = dict(discovery.get("scorer", {}) or {})

    def rollout_value(key: str, default: Any) -> Any:
        return rollout_cfg.get(key, discovery.get(key, default))

    def scorer_value(key: str, default: Any) -> Any:
        return scorer_cfg.get(key, discovery.get(key, default))

    default_apply_modes = {
        "attn": str(training.get("attention", {}).get("train_apply_mode", "all")),
        "mlp": str(training.get("mlp", {}).get("train_apply_mode", "prompt_last")),
    }
    results: list[dict[str, object]] = []
    screen_paths: list[Path] = []
    for component_type in component_types:
        apply_mode = type_apply_modes.get(component_type, default_apply_modes.get(component_type, "decision_tokens"))
        scan_dir = out_dir / "discovery" / f"components_{component_type}"
        command = [
            *_python_module("screscomp.cli.cecm_scan_component_contributions"),
            "--model",
            str(model.get("path", "")),
            "--event",
            event,
            "--scan-mode",
            scan_mode,
            "--split",
            str(discovery.get("split", "train")),
            "--start",
            str(discovery.get("start", 0)),
            "--max-rows",
            str(discovery.get("rows", 60)),
            "--component-types",
            component_type,
            "--apply-mode",
            apply_mode,
            "--min-abs-delta",
            str(discovery.get("min_abs_delta", 0.0)),
            "--min-sign-consistency",
            str(discovery.get("scan_min_sign_consistency", 0.50)),
            "--max-components-per-direction",
            str(discovery.get("max_components_per_direction", 16)),
            "--out-dir",
            str(scan_dir),
        ]
        if scan_mode == "rollout":
            prompts_jsonl = (
                discovery.get("prompts_jsonl")
                or rollout_cfg.get("prompts_jsonl")
                or task.get("prompts_jsonl")
                or task.get("discovery_prompts_jsonl")
            )
            if not prompts_jsonl:
                raise ValueError("discovery.scan_mode='rollout' requires discovery.prompts_jsonl or task.prompts_jsonl")
            command.extend(
                [
                    "--prompts-jsonl",
                    str(prompts_jsonl),
                    "--prompt-field",
                    str(rollout_value("prompt_field", "prompt")),
                    "--sample-id-field",
                    str(rollout_value("sample_id_field", "sample_id")),
                    "--samples-per-prompt",
                    str(rollout_value("samples_per_prompt", 1)),
                    "--generation-batch-size",
                    str(rollout_value("generation_batch_size", 1)),
                    "--max-new-tokens",
                    str(rollout_value("max_new_tokens", 64)),
                    "--stop-strings",
                    str(rollout_value("stop_strings", "")),
                    "--temperature",
                    str(rollout_value("temperature", 1.0)),
                    "--top-p",
                    str(rollout_value("top_p", 1.0)),
                    "--top-k",
                    str(rollout_value("top_k", 50)),
                    "--seed",
                    str(rollout_value("seed", 42)),
                    "--scorer-model",
                    str(scorer_value("model", "siebert/sentiment-roberta-large-english")),
                    "--target-label",
                    str(scorer_value("target_label", "POSITIVE")),
                    "--source-label",
                    str(scorer_value("source_label", "NEGATIVE")),
                    "--score-text",
                    str(scorer_value("score_text", "completion")),
                    "--scorer-batch-size",
                    str(scorer_value("batch_size", 16)),
                    "--scorer-max-length",
                    str(scorer_value("max_length", 512)),
                    "--scorer-device",
                    str(scorer_value("device", -1)),
                ]
            )
            command.append("--do-sample" if _truthy(rollout_value("do_sample", True), default=True) else "--no-do-sample")
        else:
            command.extend(
                [
                    "--pairs-csv",
                    str(pairs_csv),
                    "--score-mode",
                    score_mode,
                    "--option-selection-mode",
                    option_selection_mode,
                    "--max-aliases-per-side",
                    max_aliases,
                ]
            )
        if _truthy(discovery.get("allow_ci_cross_zero"), default=True):
            command.append("--allow-ci-cross-zero")
        command = _append_model_args(command, model)
        expected = scan_dir / "component_screen.csv"
        results.append(
            _run_command(
                name=f"scan_{component_type}",
                command=command,
                log_path=out_dir / "logs" / f"scan_{component_type}.log",
                expected=expected,
                force=force,
            )
        )
        screen_paths.append(expected)

    merged_rows: list[dict[str, str]] = []
    for path in screen_paths:
        merged_rows.extend(load_csv(path))
    merged_screen = out_dir / "discovery" / "component_screen.csv"
    dump_csv(merged_screen, [dict(row) for row in merged_rows])
    return merged_screen, results


def _select_components(
    *,
    config: dict[str, Any],
    out_dir: Path,
    component_screen_csv: Path,
    force: bool,
) -> tuple[Path, dict[str, object]]:
    task = config.get("task", {})
    discovery = config.get("discovery", {})
    selected_dir = out_dir / "discovery" / "selected"
    command = [
        *_python_module("screscomp.cli.cecm_select_margin_components"),
        "--component-screen-csv",
        str(component_screen_csv),
        "--event",
        str(task.get("event", "source_context_over_prior")),
        "--topk-mlp",
        str(discovery.get("mlp_topk", 4)),
        "--topk-negative-mlp",
        str(discovery.get("mlp_negative_topk", discovery.get("mlp_topk", 4))),
        "--topk-attn-layers",
        str(discovery.get("attn_layer_topk", 4)),
        "--mlp-policy",
        str(discovery.get("mlp_policy", "positive")),
        "--min-sign-consistency",
        str(discovery.get("min_sign_consistency", 0.0)),
        "--mlp-min-sign-consistency",
        str(discovery.get("mlp_min_sign_consistency", discovery.get("min_sign_consistency", 0.0))),
        "--attn-min-sign-consistency",
        str(discovery.get("attn_min_sign_consistency", discovery.get("min_sign_consistency", 0.0))),
        "--min-mlp-abs-mean-delta",
        str(discovery.get("min_mlp_abs_mean_delta", 0.0)),
        "--min-attn-abs-mean-delta",
        str(discovery.get("min_attn_abs_mean_delta", 0.0)),
        "--min-bidirectional-rate",
        str(discovery.get("min_bidirectional_rate", 0.0)),
        "--min-directional-transition-rate",
        str(discovery.get("min_directional_transition_rate", discovery.get("min_bidirectional_rate", 0.0))),
        "--min-ablated-format-ok",
        str(discovery.get("min_ablated_format_ok", 0.0)),
        "--max-format-collapse-rate",
        str(discovery.get("max_format_collapse_rate", 1.0)),
        "--min-abs-mean-target-drop",
        str(discovery.get("min_abs_mean_target_drop", 0.0)),
        "--min-abs-mean-source-rise",
        str(discovery.get("min_abs_mean_source_rise", 0.0)),
        "--exclude-mlp-layers",
        str(discovery.get("exclude_mlp_layers", "")),
        "--exclude-attn-layers",
        str(discovery.get("exclude_attn_layers", "")),
        "--out-dir",
        str(selected_dir),
    ]
    if _truthy(discovery.get("require_directional_ci"), default=False):
        command.append("--require-directional-ci")
    if _truthy(discovery.get("mlp_require_directional_ci"), default=False):
        command.append("--mlp-require-directional-ci")
    if _truthy(discovery.get("mlp_allow_ci_cross_zero"), default=False):
        command.append("--mlp-allow-ci-cross-zero")
    if _truthy(discovery.get("attn_require_directional_ci"), default=False):
        command.append("--attn-require-directional-ci")
    if _truthy(discovery.get("attn_allow_ci_cross_zero"), default=False):
        command.append("--attn-allow-ci-cross-zero")
    if _truthy(discovery.get("allow_empty_attn"), default=False):
        command.append("--allow-empty-attn")
    result = _run_command(
        name="select_components",
        command=command,
        log_path=out_dir / "logs" / "select_components.log",
        expected=selected_dir / "component_selection_manifest.json",
        force=force,
    )
    return selected_dir, result


def _component_source(
    *,
    config: dict[str, Any],
    out_dir: Path,
    pairs_csv: str,
    force: bool,
) -> tuple[Path, list[dict[str, object]], dict[str, object]]:
    discovery = config.get("discovery", {})
    mode = _component_source_mode(discovery)
    phase_results: list[dict[str, object]] = []
    source_record: dict[str, object] = {"mode": mode}
    if mode == COMPONENT_SOURCE_RESELECT:
        screen, scan_results = _scan_components(config=config, out_dir=out_dir, pairs_csv=pairs_csv, force=force)
        phase_results.extend(scan_results)
        selected_dir, select_result = _select_components(config=config, out_dir=out_dir, component_screen_csv=screen, force=force)
        phase_results.append(select_result)
        source_record.update({"component_screen_csv": str(screen), "selected_dir": str(selected_dir)})
        return selected_dir, phase_results, source_record

    kind, path = _resolve_reuse_object(config) if mode == COMPONENT_SOURCE_REUSE else _resolve_previous_object(config)
    source_record.update({"reuse_kind": kind, "reuse_path": str(path)})
    if kind == "selected_dir":
        if not (path / "component_selection_manifest.json").exists():
            raise ValueError(f"Selected component directory is missing manifest: {path}")
        source_record["selected_dir"] = str(path)
        return path, phase_results, source_record
    selected_dir, select_result = _select_components(config=config, out_dir=out_dir, component_screen_csv=path, force=force)
    phase_results.append(select_result)
    source_record.update({"component_screen_csv": str(path), "selected_dir": str(selected_dir)})
    return selected_dir, phase_results, source_record


def _train_fixed_actuator(
    *,
    config: dict[str, Any],
    out_dir: Path,
    pairs_csv: str,
    components_csv: Path,
    name: str,
    apply_mode: str | None = None,
    force: bool,
) -> tuple[Path | None, dict[str, object]]:
    if not _csv_has_rows(components_csv):
        return None, {"name": name, "status": "skipped_empty_components", "components_csv": str(components_csv)}
    model = config.get("model", {})
    task = config.get("task", {})
    training = config.get("training", {})
    mlp = training.get("mlp", {})
    alpha = config.get("alpha", {})
    init_payload = str(mlp.get("init_payloads", {}).get(name, "") or "")
    train_dir = out_dir / "train" / name
    command = [
        *_python_module("screscomp.cli.cecm_train_fixed_actuator"),
        "--model",
        str(model.get("path", "")),
        "--pairs-csv",
        str(pairs_csv),
        "--components-csv",
        str(components_csv),
        "--event",
        str(task.get("event", "source_context_over_prior")),
        "--train-split",
        str(training.get("train_split", "train")),
        "--val-split",
        str(training.get("val_split", "val")),
        "--max-train-rows",
        str(training.get("max_train_rows", training.get("train_rows", 240))),
        "--max-val-rows",
        str(training.get("max_val_rows", training.get("val_rows", 60))),
        "--epochs",
        str(mlp.get("epochs", training.get("epochs", 2))),
        "--lr",
        str(mlp.get("lr", training.get("lr", 0.05))),
        "--lambda-norm",
        str(training.get("lambda_norm", 1e-4)),
        "--alpha-train",
        str(training.get("alpha_train", 1.0)),
        "--state-margin-weight",
        str(training.get("state_margin_weight", 1.0)),
        "--gain-weight",
        str(training.get("gain_weight", 0.0)),
        "--target-margin",
        str(training.get("target_margin", 0.0)),
        "--target-gain",
        str(training.get("target_gain", 0.0)),
        "--apply-mode",
        str(apply_mode or mlp.get("train_apply_mode", "prompt_last")),
        "--score-mode",
        str(training.get("score_mode", "avglogp")),
        "--option-selection-mode",
        str(training.get("option_selection_mode", "model_max")),
        "--alpha-sweep",
        _join(alpha.get("train_sweep"), default="0,1"),
        "--max-aliases-per-side",
        str(training.get("max_aliases_per_side", 1)),
        "--empty-cache-every",
        str(training.get("empty_cache_every", 25)),
        "--out-dir",
        str(train_dir),
    ]
    if init_payload:
        command.extend(["--init-fixed-actuator", init_payload])
    command = _append_background_args(command, config)
    command = _append_model_args(command, model)
    result = _run_command(
        name=name,
        command=command,
        log_path=out_dir / "logs" / f"{name}.log",
        expected=train_dir / "fixed_actuator.pt",
        force=force,
    )
    return train_dir / "fixed_actuator.pt", result


def _scan_heads(
    *,
    config: dict[str, Any],
    out_dir: Path,
    pairs_csv: str,
    selected_dir: Path,
    force: bool,
) -> tuple[Path | None, dict[str, object]]:
    layers = _read_text(selected_dir / "attention_layers.txt")
    if not layers:
        return None, {"name": "scan_heads", "status": "skipped_no_attention_layers", "selected_dir": str(selected_dir)}
    reuse_head_dir = config.get("training", {}).get("attention", {}).get("reuse_head_dir")
    if reuse_head_dir:
        head_dir = Path(str(reuse_head_dir))
        manifest = head_dir / "head_selection_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing reused head selection manifest: {manifest}")
        return head_dir, {
            "name": "scan_heads",
            "status": "skipped_reuse_head_dir",
            "head_dir": str(head_dir),
            "manifest": str(manifest),
        }
    model = config.get("model", {})
    task = config.get("task", {})
    training = config.get("training", {})
    attention = training.get("attention", {})
    discovery = config.get("discovery", {})
    head_dir = out_dir / "discovery" / "head_scan"
    command = [
        *_python_module("screscomp.cli.cecm_scan_attention_heads"),
        "--model",
        str(model.get("path", "")),
        "--pairs-csv",
        str(pairs_csv),
        "--event",
        str(task.get("event", "source_context_over_prior")),
        "--attn-layers",
        layers,
        "--scan-factors",
        _join(discovery.get("head_scan_factors"), default="0.0,1.5"),
        "--topk-heads",
        str(discovery.get("head_topk", 4)),
        "--split",
        "train",
        "--scan-start",
        "0",
        "--scan-max-rows",
        str(discovery.get("head_scan_rows", 24)),
        "--score-mode",
        str(training.get("score_mode", "avglogp")),
        "--score-apply-mode",
        str(attention.get("train_apply_mode", "all")),
        "--option-selection-mode",
        str(training.get("option_selection_mode", "model_max")),
        "--max-aliases-per-side",
        str(training.get("max_aliases_per_side", 1)),
        "--min-mean-margin-gain",
        str(discovery.get("head_min_mean_margin_gain", 0.0)),
        "--out-dir",
        str(head_dir),
    ]
    if _truthy(discovery.get("head_require_directional_ci"), default=False):
        command.append("--require-directional-ci")
    if _truthy(discovery.get("allow_nonpositive_heads"), default=False):
        command.append("--allow-nonpositive-selection")
    command = _append_model_args(command, model)
    result = _run_command(
        name="scan_heads",
        command=command,
        log_path=out_dir / "logs" / "scan_heads.log",
        expected=head_dir / "head_selection_manifest.json",
        force=force,
    )
    return head_dir, result


def _train_head_actuator(
    *,
    config: dict[str, Any],
    out_dir: Path,
    pairs_csv: str,
    heads_txt: Path,
    name: str,
    force: bool,
) -> tuple[Path | None, dict[str, object]]:
    heads = _read_text(heads_txt)
    if not heads:
        return None, {"name": name, "status": "skipped_empty_heads", "heads_txt": str(heads_txt)}
    model = config.get("model", {})
    task = config.get("task", {})
    training = config.get("training", {})
    attention = training.get("attention", {})
    alpha = config.get("alpha", {})
    init_payload = str(attention.get("init_payloads", {}).get(name, "") or "")
    train_dir = out_dir / "train" / name
    command = [
        *_python_module("screscomp.cli.cecm_train_attention_head_actuator"),
        "--model",
        str(model.get("path", "")),
        "--pairs-csv",
        str(pairs_csv),
        "--event",
        str(task.get("event", "source_context_over_prior")),
        "--heads",
        heads,
        "--train-split",
        str(training.get("train_split", "train")),
        "--val-split",
        str(training.get("val_split", "val")),
        "--max-train-rows",
        str(training.get("max_train_rows", training.get("train_rows", 240))),
        "--max-val-rows",
        str(training.get("max_val_rows", training.get("val_rows", 60))),
        "--epochs",
        str(attention.get("epochs", training.get("epochs", 2))),
        "--lr",
        str(attention.get("lr", training.get("lr", 0.05))),
        "--lambda-norm",
        str(training.get("lambda_norm", 1e-4)),
        "--alpha-train",
        str(training.get("alpha_train", 1.0)),
        "--state-margin-weight",
        str(training.get("state_margin_weight", 1.0)),
        "--gain-weight",
        str(training.get("gain_weight", 0.0)),
        "--target-margin",
        str(training.get("target_margin", 0.0)),
        "--target-gain",
        str(training.get("target_gain", 0.0)),
        "--apply-mode",
        str(attention.get("train_apply_mode", "all")),
        "--score-mode",
        str(training.get("score_mode", "avglogp")),
        "--option-selection-mode",
        str(training.get("option_selection_mode", "model_max")),
        "--alpha-sweep",
        _join(alpha.get("train_sweep"), default="0,1"),
        "--max-aliases-per-side",
        str(training.get("max_aliases_per_side", 1)),
        "--empty-cache-every",
        str(training.get("empty_cache_every", 25)),
        "--out-dir",
        str(train_dir),
    ]
    if init_payload:
        command.extend(["--init-head-actuator", init_payload])
    command = _append_background_args(command, config)
    command = _append_model_args(command, model)
    result = _run_command(
        name=name,
        command=command,
        log_path=out_dir / "logs" / f"{name}.log",
        expected=train_dir / "head_actuator.pt",
        force=force,
    )
    return train_dir / "head_actuator.pt", result


def _control_parts(
    *,
    config: dict[str, Any],
    mlp_paths: dict[str, Path],
    head_paths: dict[str, Path],
    include_full: bool = True,
    full_pairs_override: list[tuple[float, float]] | None = None,
) -> tuple[str, str, str]:
    training = config.get("training", {})
    mlp_config = training.get("mlp", {})
    head_mode = str(training.get("attention", {}).get("generation_apply_mode", "all"))
    alpha_config = config.get("alpha", {})
    background = _round_background(config)
    background_parts = str(background.get("control_parts", ""))
    all_mlp_paths = {
        **{str(k): Path(str(v)) for k, v in dict(background.get("component_actuators", {}) or {}).items()},
        **mlp_paths,
    }
    all_head_paths = {
        **{str(k): Path(str(v)) for k, v in dict(background.get("head_actuators", {}) or {}).items()},
        **head_paths,
    }
    comp_text = _semicolon_path_map(all_mlp_paths)
    head_text = _semicolon_path_map(all_head_paths)
    controls: dict[str, str] = {"base": ""}
    if background_parts:
        controls["prev"] = background_parts
    selection_cfg = config.get("selection", {}) if isinstance(config.get("selection", {}), dict) else {}
    reference_controls = selection_cfg.get("reference_controls", [])
    if isinstance(reference_controls, list):
        for item in reference_controls:
            if not isinstance(item, dict):
                continue
            reference_name = str(item.get("name", "") or "").strip()
            reference_parts = str(item.get("parts", "") or "").strip()
            if not reference_name or not reference_parts:
                continue
            join_background = _truthy(item.get("join_background"), default=True)
            controls[reference_name] = (
                join_control_parts(background_parts, reference_parts) if join_background else reference_parts
            )
    reference = _selection_reference(config)
    if reference is not None:
        reference_name, reference_parts = reference
        controls[reference_name] = reference_parts
    selection_source = (
        config.get("selection", {}).get("source")
        or training.get("full", {}).get("selected_config")
        or (alpha_config.get("full", {}) if isinstance(alpha_config.get("full", {}), dict) else {}).get("source")
    )
    if selection_source and Path(str(selection_source)).exists():
        selected = _read_json(Path(str(selection_source)))
        name = str(selected.get("best_control_name", "cast_selected"))
        parts = str(selected.get("best_control_parts", ""))
        if parts:
            controls[name] = join_control_parts(background_parts, parts)
            controls_text = ";".join(f"{control_name}={control_parts}" for control_name, control_parts in controls.items()) + ";"
            return controls_text, comp_text, head_text
    mlp_alphas = _float_list(alpha_config.get("mlp"), default=[0.05])
    head_alphas = _float_list(alpha_config.get("att", alpha_config.get("head")), default=[0.5])
    mlp_modes = [(name, generation_mode) for name, _train_mode, generation_mode in _mlp_timing_modes(config)]
    timing_suffixes = {f"__{_slug(name)}" for name, _mode in mlp_modes}
    skip_single_families = {
        str(item).strip().lower()
        for item in selection_cfg.get("skip_single_families", [])
        if str(item).strip()
    }

    def mlp_paths_for_mode(mode_name: str) -> dict[str, Path]:
        suffix = f"__{_slug(mode_name)}"
        mode_paths = {name: path for name, path in mlp_paths.items() if name.endswith(suffix)}
        if mode_paths:
            return mode_paths
        legacy_paths = {
            name: path
            for name, path in mlp_paths.items()
            if not any(name.endswith(existing_suffix) for existing_suffix in timing_suffixes)
        }
        return legacy_paths or mlp_paths
    full_config = alpha_config.get("full", {})
    paired_full = []
    full_fixed_parts = ""
    if isinstance(full_config, dict) and str(full_config.get("grid_mode", "")) == "paired_centerline":
        for item in full_config.get("pairs", []):
            if isinstance(item, dict):
                paired_full.append((float(item["mlp"]), float(item["att"])))
        full_fixed_parts = str(full_config.get("fixed_parts", "") or "")
    if "mlp" not in skip_single_families:
        for mode_name, mlp_mode in mlp_modes:
            for alpha in mlp_alphas:
                mode_mlp_paths = mlp_paths_for_mode(mode_name)
                parts = [f"comp:{name}:{alpha}:{mlp_mode}" for name in sorted(mode_mlp_paths)]
                if parts:
                    controls[f"cast_mlp_{mode_name}_a{_alpha_name(alpha)}"] = join_control_parts(background_parts, "+".join(parts))
    if "att" not in skip_single_families:
        for alpha in head_alphas:
            parts = [f"head_act:{name}:{alpha}:{head_mode}" for name in sorted(head_paths)]
            if parts:
                controls[f"cast_att_a{_alpha_name(alpha)}"] = join_control_parts(background_parts, "+".join(parts))
    if include_full and _truthy(training.get("full", {}).get("enabled"), default=True):
        full_pairs = full_pairs_override or paired_full or [(mlp_alpha, head_alpha) for mlp_alpha in mlp_alphas for head_alpha in head_alphas]
        for mode_name, mlp_mode in mlp_modes:
            for mlp_alpha, head_alpha in full_pairs:
                mode_mlp_paths = mlp_paths_for_mode(mode_name)
                mlp_parts = [f"comp:{name}:{mlp_alpha}:{mlp_mode}" for name in sorted(mode_mlp_paths)]
                head_parts = [f"head_act:{name}:{head_alpha}:{head_mode}" for name in sorted(head_paths)]
                parts = ([full_fixed_parts] if full_fixed_parts else []) + [*mlp_parts, *head_parts]
                if parts:
                    controls[f"cast_full_{mode_name}_m{_alpha_name(mlp_alpha)}_h{_alpha_name(head_alpha)}"] = join_control_parts(
                        background_parts,
                        "+".join(parts),
                    )
    controls_text = ";".join(f"{name}={parts}" for name, parts in controls.items()) + ";"
    return controls_text, comp_text, head_text


def _eval_command(
    *,
    config: dict[str, Any],
    out_dir: Path,
    eval_dir: Path,
    controls: str,
    component_actuators: str,
    head_actuators: str,
    overwrite: bool,
) -> list[str]:
    evaluation = config.get("evaluation", {})
    model = config.get("model", {})
    task = config.get("task", {})
    command = [
        *_python_module("screscomp.cli.cecm_run_joint_actuator_generation"),
        "--model",
        str(model.get("path", "")),
        "--eval-open-rows",
        str(evaluation.get("eval_open_rows") or task.get("eval_open_rows") or ""),
        "--component-actuators",
        component_actuators,
        "--head-actuators",
        head_actuators,
        "--controls",
        controls,
        "--generation-prompt-key",
        str(evaluation.get("generation_prompt_key", task.get("prompt_key", "deepseek_math_cot"))),
        "--prior-source",
        str(evaluation.get("prior_source", "model_prior")),
        "--scoring-kind",
        str(
            evaluation.get(
                "scoring_kind",
                "gsm8k" if str(task.get("adapter", "")).startswith("gsm8k") else "source",
            )
        ),
        "--split",
        str(evaluation.get("split", "all")),
        "--start",
        str(evaluation.get("start", 0)),
        "--max-rows",
        str(evaluation.get("max_rows", 0)),
        "--val-mod",
        str(evaluation.get("val_mod", task.get("val_mod", 5))),
        "--generation-apply-mode",
        str(evaluation.get("generation_apply_mode", "prefill")),
        "--max-new-tokens",
        str(evaluation.get("max_new_tokens", 512)),
        "--stop-strings",
        str(evaluation.get("stop_strings", "")),
        "--flush-every",
        str(evaluation.get("flush_every", 1)),
        "--summary-every",
        str(evaluation.get("summary_every", 100)),
        "--empty-cache-every",
        str(evaluation.get("empty_cache_every", 25)),
        "--out-dir",
        str(eval_dir),
    ]
    if _truthy(evaluation.get("do_sample"), default=False):
        command.extend(
            [
                "--do-sample",
                "--temperature",
                str(evaluation.get("temperature", 1.0)),
                "--top-p",
                str(evaluation.get("top_p", 1.0)),
                "--top-k",
                str(evaluation.get("top_k", 100)),
            ]
        )
    if overwrite or _truthy(evaluation.get("overwrite"), default=False):
        command.append("--overwrite")
    return _append_model_args(command, model)


def _control_sort_key(row: dict[str, Any], *, metric: str, tie_breakers: list[str]) -> tuple[float, ...]:
    values: list[float] = [_metric_value(row, metric)]
    for tie in tie_breakers:
        if tie == "smaller_alpha":
            values.append(-max_abs_alpha(str(row.get("control_parts", ""))))
        else:
            values.append(_metric_value(row, tie))
    return tuple(values)


def _selection_reference(
    config: dict[str, Any],
) -> tuple[str, str] | None:
    selection_cfg = config.get("selection", {}) if isinstance(config.get("selection", {}), dict) else {}
    reference_eval = selection_cfg.get("reference_from_eval")
    if isinstance(reference_eval, dict):
        summary_csv = Path(str(reference_eval.get("summary_csv") or ""))
        control_plan_csv = Path(str(reference_eval.get("control_plan_csv") or ""))
        if summary_csv.exists() and control_plan_csv.exists():
            metric = str(reference_eval.get("metric") or selection_cfg.get("metric", "strict_final_exact"))
            tie_breakers = [
                str(item)
                for item in (reference_eval.get("tie_breakers") or selection_cfg.get("tie_breakers", []))
            ]
            family = str(reference_eval.get("family", "") or "").strip().lower()
            prefix = str(reference_eval.get("control_prefix", "") or "")
            if not prefix:
                prefix = {"mlp": "cast_mlp_", "att": "cast_att_", "full": "cast_full_"}.get(family, "")
            parts_by_name = {row.get("control_name", ""): row.get("parts", "") for row in load_csv(control_plan_csv)}
            candidates: list[dict[str, Any]] = []
            for row in load_csv(summary_csv):
                name = str(row.get("control_name", ""))
                if prefix and not name.startswith(prefix):
                    continue
                parts = str(parts_by_name.get(name, ""))
                if not parts:
                    continue
                candidates.append({**row, "control_parts": parts})
            if candidates:
                best = max(candidates, key=lambda row: _control_sort_key(row, metric=metric, tie_breakers=tie_breakers))
                return (
                    str(reference_eval.get("reference_control_name") or best.get("control_name") or "reference"),
                    str(best.get("control_parts", "")),
                )

    reference_parts = str(selection_cfg.get("reference_parts", "") or "")
    if reference_parts:
        return str(selection_cfg.get("reference_control_name") or "reference"), reference_parts

    reference_source = selection_cfg.get("reference_config")
    if reference_source and Path(str(reference_source)).exists():
        selected = _read_json(Path(str(reference_source)))
        name = str(
            selection_cfg.get("reference_control_name")
            or selected.get("best_control_name")
            or "reference"
        )
        parts = str(selected.get("best_control_parts", ""))
        if parts:
            return name, parts
    return None


def _around_center(values: list[float], center: float, radius: int) -> list[float]:
    if not values:
        return []
    index = min(range(len(values)), key=lambda idx: abs(float(values[idx]) - float(center)))
    start = max(0, index - max(0, int(radius)))
    end = min(len(values), index + max(0, int(radius)) + 1)
    return [float(values[idx]) for idx in range(start, end)]


def _select_single_family_winners(
    *,
    config: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    parts_by_name: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    selection = config.get("selection", {})
    metric = str(selection.get("metric", "strict_final_exact"))
    tie_breakers = [str(item) for item in selection.get("tie_breakers", [])]
    reference = _selection_reference(config)
    reference_name = str(reference[0]) if reference is not None else ""
    reference_parts = str(reference[1]) if reference is not None else ""
    reference_is_mlp = False
    if reference_name and reference_parts:
        parsed = parse_control_parts(reference_parts)
        reference_is_mlp = bool(parsed) and any(part.kind == "comp" for part in parsed) and not any(part.kind == "head_act" for part in parsed)

    mlp_rows: list[dict[str, Any]] = []
    att_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        name = str(row.get("control_name", ""))
        parts = str(parts_by_name.get(name, ""))
        if not parts:
            continue
        row_with_parts = {**row, "control_parts": parts}
        if name.startswith("cast_mlp_") or (reference_is_mlp and name == reference_name):
            mlp_rows.append(row_with_parts)
        elif name.startswith("cast_att_"):
            att_rows.append(row_with_parts)
    best_mlp = max(mlp_rows, key=lambda row: _control_sort_key(row, metric=metric, tie_breakers=tie_breakers)) if mlp_rows else None
    best_att = max(att_rows, key=lambda row: _control_sort_key(row, metric=metric, tie_breakers=tie_breakers)) if att_rows else None
    return best_mlp, best_att


def _first_part_value(parts: str, kind: str) -> tuple[float, str] | None:
    for part in parse_control_parts(parts):
        if part.kind == kind and part.value is not None:
            return float(part.value), str(part.apply_mode or "")
    return None


def _merge_eval_csvs(destination: Path, paths: list[Path]) -> None:
    merged: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            merged.extend(load_csv(path))
    dump_csv(destination, [dict(row) for row in merged])


def _evaluate(
    *,
    config: dict[str, Any],
    out_dir: Path,
    mlp_paths: dict[str, Path],
    head_paths: dict[str, Path],
    force: bool,
    skip_eval: bool,
) -> dict[str, object]:
    evaluation = config.get("evaluation", {})
    if skip_eval or not _truthy(evaluation.get("enabled"), default=True):
        return {"name": "eval", "status": "skipped_disabled"}
    eval_open_rows = evaluation.get("eval_open_rows") or config.get("task", {}).get("eval_open_rows")
    if not eval_open_rows:
        return {"name": "eval", "status": "skipped_no_eval_open_rows"}
    eval_dir = out_dir / "eval" / str(evaluation.get("name", "main"))
    full_config = config.get("alpha", {}).get("full", {})
    search_mode = str(full_config.get("search_mode", "")) if isinstance(full_config, dict) else ""
    if search_mode != "around_single_winners":
        controls, component_actuators, head_actuators = _control_parts(
            config=config,
            mlp_paths=mlp_paths,
            head_paths=head_paths,
        )
        command = _eval_command(
            config=config,
            out_dir=out_dir,
            eval_dir=eval_dir,
            controls=controls,
            component_actuators=component_actuators,
            head_actuators=head_actuators,
            overwrite=force,
        )
        return _run_command(
            name="eval",
            command=command,
            log_path=out_dir / "logs" / "eval.log",
            expected=eval_dir / "generation_summary.csv",
            force=True,
        )

    singles_dir = eval_dir / "singles"
    singles_controls, component_actuators, head_actuators = _control_parts(
        config=config,
        mlp_paths=mlp_paths,
        head_paths=head_paths,
        include_full=False,
    )
    singles_command = _eval_command(
        config=config,
        out_dir=out_dir,
        eval_dir=singles_dir,
        controls=singles_controls,
        component_actuators=component_actuators,
        head_actuators=head_actuators,
        overwrite=force,
    )
    singles_result = _run_command(
        name="eval_singles",
        command=singles_command,
        log_path=out_dir / "logs" / "eval_singles.log",
        expected=singles_dir / "generation_summary.csv",
        force=True,
    )
    summary_rows = load_csv(singles_dir / "generation_summary.csv")
    control_rows = load_csv(singles_dir / "control_plan.csv")
    parts_by_name = {row.get("control_name", ""): row.get("parts", "") for row in control_rows}
    best_mlp, best_att = _select_single_family_winners(config=config, summary_rows=summary_rows, parts_by_name=parts_by_name)
    if best_mlp is None or best_att is None:
        _merge_eval_csvs(eval_dir / "generation_summary.csv", [singles_dir / "generation_summary.csv"])
        _merge_eval_csvs(eval_dir / "control_plan.csv", [singles_dir / "control_plan.csv"])
        dump_json(
            eval_dir / "local_full_search.json",
            {
                "search_mode": search_mode,
                "status": "skipped_missing_single_winner",
                "best_mlp_control": str(best_mlp.get("control_name", "")) if best_mlp else "",
                "best_att_control": str(best_att.get("control_name", "")) if best_att else "",
            },
        )
        return singles_result

    mlp_value = _first_part_value(str(best_mlp["control_parts"]), "comp")
    att_value = _first_part_value(str(best_att["control_parts"]), "head_act")
    if mlp_value is None or att_value is None:
        _merge_eval_csvs(eval_dir / "generation_summary.csv", [singles_dir / "generation_summary.csv"])
        _merge_eval_csvs(eval_dir / "control_plan.csv", [singles_dir / "control_plan.csv"])
        dump_json(
            eval_dir / "local_full_search.json",
            {
                "search_mode": search_mode,
                "status": "skipped_unparseable_single_winner",
                "best_mlp_control": str(best_mlp.get("control_name", "")),
                "best_att_control": str(best_att.get("control_name", "")),
            },
        )
        return singles_result

    best_mlp_alpha, best_mlp_mode = mlp_value
    best_att_alpha, _head_mode = att_value
    mlp_alphas = _float_list(config.get("alpha", {}).get("mlp"), default=[best_mlp_alpha])
    att_alphas = _float_list(config.get("alpha", {}).get("att", config.get("alpha", {}).get("head")), default=[best_att_alpha])
    variant_specs: list[tuple[str, dict[str, Any]]] = []
    if isinstance(full_config, dict) and isinstance(full_config.get("variants"), list):
        for idx, item in enumerate(full_config.get("variants", []), start=1):
            if not isinstance(item, dict):
                continue
            variant_name = str(item.get("name", "") or f"full_local_{idx}")
            variant_specs.append((variant_name, item))
    if not variant_specs:
        variant_specs = [("full_local", full_config if isinstance(full_config, dict) else {})]

    merged_summary_paths = [singles_dir / "generation_summary.csv"]
    merged_control_paths = [singles_dir / "control_plan.csv"]
    variant_results: list[dict[str, Any]] = []

    for variant_name, variant_cfg in variant_specs:
        full_mlp_alphas = _float_list(
            variant_cfg.get("mlp_grid", variant_cfg.get("mlp")) if isinstance(variant_cfg, dict) else None,
            default=mlp_alphas,
        )
        full_att_alphas = _float_list(
            variant_cfg.get("att_grid", variant_cfg.get("att")) if isinstance(variant_cfg, dict) else None,
            default=att_alphas,
        )
        mlp_center_scale = float(variant_cfg.get("mlp_winner_scale", 1.0)) if isinstance(variant_cfg, dict) else 1.0
        att_center_scale = float(variant_cfg.get("att_winner_scale", 1.0)) if isinstance(variant_cfg, dict) else 1.0
        mlp_center = best_mlp_alpha * mlp_center_scale
        att_center = best_att_alpha * att_center_scale
        radius = int(variant_cfg.get("search_window", 1) or 1) if isinstance(variant_cfg, dict) else 1
        local_mlp_alphas = _around_center(full_mlp_alphas, mlp_center, radius)
        local_att_alphas = _around_center(full_att_alphas, att_center, radius)
        local_pairs = [(mlp_alpha, att_alpha) for mlp_alpha in local_mlp_alphas for att_alpha in local_att_alphas]

        variant_slug = _slug(variant_name)
        full_dir = eval_dir / variant_slug
        local_config = copy.deepcopy(config)
        local_config.setdefault("alpha", {})["mlp"] = []
        local_config.setdefault("alpha", {})["att"] = []
        mlp_cfg = local_config.setdefault("training", {}).setdefault("mlp", {})
        timing_grid = mlp_cfg.get("timing_grid")
        if isinstance(timing_grid, list) and timing_grid:
            filtered = [item for item in timing_grid if str(item.get("name", "")) == best_mlp_mode]
            if filtered:
                mlp_cfg["timing_grid"] = filtered
        full_controls, local_component_actuators, local_head_actuators = _control_parts(
            config=local_config,
            mlp_paths=mlp_paths,
            head_paths=head_paths,
            include_full=True,
            full_pairs_override=local_pairs,
        )
        full_command = _eval_command(
            config=local_config,
            out_dir=out_dir,
            eval_dir=full_dir,
            controls=full_controls,
            component_actuators=local_component_actuators,
            head_actuators=local_head_actuators,
            overwrite=force,
        )
        full_result = _run_command(
            name=f"eval_{variant_slug}",
            command=full_command,
            log_path=out_dir / "logs" / f"eval_{variant_slug}.log",
            expected=full_dir / "generation_summary.csv",
            force=True,
        )
        merged_summary_paths.append(full_dir / "generation_summary.csv")
        merged_control_paths.append(full_dir / "control_plan.csv")
        variant_results.append(
            {
                "name": variant_name,
                "slug": variant_slug,
                "result": full_result,
                "summary_csv": str(full_dir / "generation_summary.csv"),
                "control_plan_csv": str(full_dir / "control_plan.csv"),
                "mlp_center_scale": mlp_center_scale,
                "att_center_scale": att_center_scale,
                "mlp_center": mlp_center,
                "att_center": att_center,
                "search_window": radius,
                "local_mlp_alphas": local_mlp_alphas,
                "local_att_alphas": local_att_alphas,
                "local_pairs": [{"mlp": mlp_alpha, "att": att_alpha} for mlp_alpha, att_alpha in local_pairs],
            }
        )

    _merge_eval_csvs(eval_dir / "generation_summary.csv", merged_summary_paths)
    _merge_eval_csvs(eval_dir / "control_plan.csv", merged_control_paths)
    dump_json(
        eval_dir / "local_full_search.json",
        {
            "search_mode": search_mode,
            "status": "completed",
            "best_mlp_control": str(best_mlp.get("control_name", "")),
            "best_att_control": str(best_att.get("control_name", "")),
            "best_mlp_alpha": best_mlp_alpha,
            "best_mlp_mode": best_mlp_mode,
            "best_att_alpha": best_att_alpha,
            "singles_summary_csv": str(singles_dir / "generation_summary.csv"),
            "variants": variant_results,
        },
    )
    result_payload = {
        "name": "eval",
        "status": "completed_local_full_search",
        "singles": singles_result,
        "full_local_variants": variant_results,
        "eval_dir": str(eval_dir),
        "search_mode": search_mode,
    }
    if len(variant_results) == 1:
        result_payload["full_local"] = variant_results[0]["result"]
    return result_payload


def _metric_value(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        value = row.get(key, "")
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _select_best_control(
    *,
    config: dict[str, Any],
    out_dir: Path,
    mlp_paths: dict[str, Path],
    head_paths: dict[str, Path],
    skip_eval: bool,
) -> dict[str, object]:
    selection = config.get("selection", {})
    write_best = selection.get("write_best_config")
    if skip_eval or not write_best or not _truthy(selection.get("enabled"), default=True):
        return {"name": "select_best_control", "status": "skipped_disabled"}
    evaluation = config.get("evaluation", {})
    eval_dir = out_dir / "eval" / str(evaluation.get("name", "main"))
    summary_csv = eval_dir / "generation_summary.csv"
    control_plan_csv = eval_dir / "control_plan.csv"
    if not summary_csv.exists() or not control_plan_csv.exists():
        return {
            "name": "select_best_control",
            "status": "skipped_missing_eval",
            "summary_csv": str(summary_csv),
            "control_plan_csv": str(control_plan_csv),
        }

    metric = str(selection.get("metric", "strict_final_exact"))
    excluded = {
        item.strip()
        for item in str(selection.get("exclude_controls", "base")).split(",")
        if item.strip()
    }
    rows = load_csv(summary_csv)
    parts_by_name = {row.get("control_name", ""): row.get("parts", "") for row in load_csv(control_plan_csv)}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("control_name", ""))
        parts = str(parts_by_name.get(name, ""))
        if not name or name in excluded or not parts:
            continue
        candidates.append({**row, "control_parts": parts})
    if not candidates:
        return {"name": "select_best_control", "status": "skipped_no_candidates", "summary_csv": str(summary_csv)}

    tie_breakers = [str(item) for item in selection.get("tie_breakers", [])]

    def sort_key(row: dict[str, Any]) -> tuple[float, ...]:
        values: list[float] = [_metric_value(row, metric)]
        for tie in tie_breakers:
            if tie == "smaller_alpha":
                values.append(-max_abs_alpha(str(row.get("control_parts", ""))))
            else:
                values.append(_metric_value(row, tie))
        return tuple(values)

    base_row = next((row for row in rows if row.get("control_name") == "base"), None)
    prev_row = next((row for row in rows if row.get("control_name") == "prev"), None)
    ranked = sorted(candidates, key=sort_key, reverse=True)
    raw_best = ranked[0]
    best = raw_best
    residual_min_gain_over_prev = float(selection.get("residual_min_gain_over_prev", 0.0) or 0.0)
    threshold_applied = False
    threshold_rejected_control = ""
    prev_candidate = next((row for row in candidates if row.get("control_name") == "prev"), None)
    if prev_candidate is not None and residual_min_gain_over_prev > 0.0 and str(raw_best.get("control_name", "")) != "prev":
        raw_best_metric = _metric_value(raw_best, metric)
        prev_candidate_metric = _metric_value(prev_candidate, metric)
        if raw_best_metric - prev_candidate_metric < residual_min_gain_over_prev:
            best = prev_candidate
            threshold_applied = True
            threshold_rejected_control = str(raw_best.get("control_name", ""))
    background = _round_background(config)
    component_actuators = {
        **{str(k): str(v) for k, v in dict(background.get("component_actuators", {}) or {}).items()},
        **{str(k): str(v) for k, v in mlp_paths.items()},
    }
    head_actuators = {
        **{str(k): str(v) for k, v in dict(background.get("head_actuators", {}) or {}).items()},
        **{str(k): str(v) for k, v in head_paths.items()},
    }
    best_metric = _metric_value(best, metric)
    raw_best_metric = _metric_value(raw_best, metric)
    base_metric = _metric_value(base_row, metric)
    prev_metric = _metric_value(prev_row, metric, default=base_metric)
    used_component_actuators, used_head_actuators = _actuators_used_by_control(
        str(best["control_parts"]),
        component_actuators=component_actuators,
        head_actuators=head_actuators,
    )
    payload = {
        "selected_by": "round_eval_metric",
        "round_index": _round_index(config),
        "summary_csv": str(summary_csv),
        "control_plan_csv": str(control_plan_csv),
        "metric": metric,
        "raw_best_control_name": str(raw_best["control_name"]),
        "raw_best_metric": raw_best_metric,
        "best_control_name": str(best["control_name"]),
        "best_control_parts": str(best["control_parts"]),
        "best_metric": best_metric,
        "base_metric": base_metric,
        "prev_metric": prev_metric,
        "gain_over_base": best_metric - base_metric,
        "gain_over_prev": best_metric - prev_metric,
        "raw_best_gain_over_prev": raw_best_metric - prev_metric,
        "residual_min_gain_over_prev": residual_min_gain_over_prev,
        "residual_threshold_applied": threshold_applied,
        "residual_threshold_rejected_control": threshold_rejected_control,
        "minimum_material_gain": selection.get("minimum_material_gain", ""),
        "material_gain": best_metric - base_metric >= float(selection.get("minimum_material_gain", 0.0) or 0.0),
        "best_row": best,
        "background": background,
        "component_actuators": used_component_actuators,
        "head_actuators": used_head_actuators,
        "available_component_actuators": component_actuators,
        "available_head_actuators": head_actuators,
        "component_actuators_text": _semicolon_path_map(used_component_actuators),
        "head_actuators_text": _semicolon_path_map(used_head_actuators),
        "selection_semantics": (
            "Selection is post-hoc over already generated controls. It never changes the trained vectors; "
            "the exported best_control_parts and actuator maps are the accepted control state for the next round. "
            "If the prev control wins, current-round residual vectors are not carried forward."
        ),
    }
    selected_path = out_dir / str(write_best)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(selected_path, payload)
    env_path = selected_path.with_suffix(".env")
    env_path.write_text(
        "\n".join(
            [
                f"BEST_CAST_CONTROL_NAME={payload['best_control_name']}",
                f"BEST_CAST_CONTROL_PARTS={payload['best_control_parts']}",
                f"BEST_METRIC={best_metric:.8f}",
                f"BEST_GAIN_OVER_BASE={best_metric - base_metric:.8f}",
                f"BEST_GAIN_OVER_PREV={best_metric - prev_metric:.8f}",
                f"BEST_MATERIAL_GAIN={1 if payload['material_gain'] else 0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dump_csv(
        selected_path.parent / "selection_ranked.csv",
        [
            {
                "rank": rank,
                "control_name": row.get("control_name", ""),
                "metric": metric,
                "metric_value": _metric_value(row, metric),
                "control_parts": row.get("control_parts", ""),
            }
            for rank, row in enumerate(ranked, start=1)
        ],
    )
    return {
        "name": "select_best_control",
        "status": "done",
        "selected": str(selected_path),
        "best_control_name": payload["best_control_name"],
        "best_metric": best_metric,
        "gain_over_base": best_metric - base_metric,
        "gain_over_prev": best_metric - prev_metric,
    }


def execute_round(
    *,
    config: dict[str, Any],
    out_dir: Path,
    pairs_csv: str,
    force: bool = False,
    skip_eval: bool = False,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    selected_dir, component_results, component_source = _component_source(
        config=config,
        out_dir=out_dir,
        pairs_csv=pairs_csv,
        force=force,
    )
    phase_results: list[dict[str, object]] = list(component_results)
    mlp_paths: dict[str, Path] = {}
    if _truthy(config.get("training", {}).get("mlp", {}).get("enabled"), default=True):
        timing_modes = _mlp_timing_modes(config)
        per_timing_vectors = bool(config.get("training", {}).get("mlp", {}).get("timing_grid"))
        for timing_name, train_apply_mode, _generation_apply_mode in timing_modes:
            timing_suffix = f"__{_slug(timing_name)}" if per_timing_vectors else ""
            train_name_suffix = f"_{_slug(timing_name)}" if per_timing_vectors else ""
            for role, csv_name in (
                ("mlp_positive", "mlp_positive_components.csv"),
                ("mlp_negative", "mlp_negative_components.csv"),
            ):
                path, result = _train_fixed_actuator(
                    config=config,
                    out_dir=out_dir,
                    pairs_csv=pairs_csv,
                    components_csv=selected_dir / csv_name,
                    name=f"train_{role}{train_name_suffix}",
                    apply_mode=train_apply_mode,
                    force=force,
                )
                phase_results.append(result)
                if path is not None:
                    mlp_paths[f"{role}{timing_suffix}"] = path
    reuse_mlp_paths = {
        str(name): Path(str(path))
        for name, path in dict(config.get("training", {}).get("mlp", {}).get("reuse_actuators") or {}).items()
    }
    mlp_paths = {**reuse_mlp_paths, **mlp_paths}

    head_paths: dict[str, Path] = {}
    if _truthy(config.get("training", {}).get("attention", {}).get("enabled"), default=True):
        head_dir, result = _scan_heads(
            config=config,
            out_dir=out_dir,
            pairs_csv=pairs_csv,
            selected_dir=selected_dir,
            force=force,
        )
        phase_results.append(result)
        if head_dir is not None:
            for role, txt_name in (
                ("head_suppress", "selected_suppress_heads.txt"),
                ("head_boost", "selected_boost_heads.txt"),
            ):
                path, train_result = _train_head_actuator(
                    config=config,
                    out_dir=out_dir,
                    pairs_csv=pairs_csv,
                    heads_txt=head_dir / txt_name,
                    name=f"train_{role}",
                    force=force,
                )
                phase_results.append(train_result)
                if path is not None:
                    head_paths[role] = path
    reuse_head_paths = {
        str(name): Path(str(path))
        for name, path in dict(config.get("training", {}).get("attention", {}).get("reuse_actuators") or {}).items()
    }
    head_paths = {**reuse_head_paths, **head_paths}

    eval_result = _evaluate(
        config=config,
        out_dir=out_dir,
        mlp_paths=mlp_paths,
        head_paths=head_paths,
        force=force,
        skip_eval=skip_eval,
    )
    phase_results.append(eval_result)
    selection_result = _select_best_control(
        config=config,
        out_dir=out_dir,
        mlp_paths=mlp_paths,
        head_paths=head_paths,
        skip_eval=skip_eval,
    )
    phase_results.append(selection_result)
    payload = {
        "status": "done",
        "elapsed_sec": round(time.time() - started, 3),
        "round_index": _round_index(config),
        "background": _round_background(config),
        "component_source": component_source,
        "selected_dir": str(selected_dir),
        "mlp_actuators": {name: str(path) for name, path in mlp_paths.items()},
        "head_actuators": {name: str(path) for name, path in head_paths.items()},
        "phases": phase_results,
    }
    dump_json(out_dir / "round_execution.json", payload)
    return payload
