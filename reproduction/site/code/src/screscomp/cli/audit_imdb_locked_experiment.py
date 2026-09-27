from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json


GROUPS = (
    "mlp_positive",
    "mlp_negative",
    "att_positive",
    "att_negative",
    "full_positive",
    "full_negative",
)
COMPONENT_FILES = (
    "mlp_positive_components.csv",
    "mlp_negative_components.csv",
    "attn_positive_components.csv",
    "attn_negative_components.csv",
)
VAL_ALPHAS = tuple(round(idx / 10, 1) for idx in range(11))
TEST_ALPHAS = (0.0, 0.3)
REQUIRED_PAIR_FIELDS = {
    "sample_id",
    "split",
    "event",
    "admitted",
    "prompt",
    "y_plus",
    "y_minus",
    "y_plus_reward",
    "y_minus_reward",
    "reward_margin",
    "prompt_id",
    "prompt_sample_id",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the locked IMDb main-table experiment.")
    parser.add_argument("--stage", choices=["components", "pairs", "training", "eval", "final"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--component-dir", type=Path, default=None)
    parser.add_argument("--pairs-csv", type=Path, default=None)
    parser.add_argument("--train-root", type=Path, default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--expected-prompts", type=int, default=0)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if line:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON objects: {path}")
                rows.append(row)
    return rows


def _f(value: Any) -> float:
    return float(str(value).strip() or 0.0)


def _alpha(value: Any) -> float:
    return round(_f(value), 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"[locked-audit] ERROR: {message}")


def _protocol(root: Path) -> dict[str, Any]:
    path = root / "protocol.json"
    if not path.exists():
        return {}
    return _read_json(path)


def _expected_model(root: Path) -> str:
    protocol = _protocol(root)
    model = protocol.get("model")
    return str(model or "edbeeching/gpt2-large-imdb")


def _expected_head_prompts(root: Path) -> str:
    protocol = _protocol(root)
    head = protocol.get("head_localization", {})
    if isinstance(head, dict):
        prompts = head.get("rollout_prompts")
        if prompts:
            return Path(str(prompts)).as_posix()
    return Path("runs/imdb_dpo_prompt_positive_rollout_scan_20260605_115624/env/prompts.jsonl").as_posix()


def _expected_head_batch(root: Path, key: str, default: int = 16) -> int:
    protocol = _protocol(root)
    overrides = protocol.get("env_overrides", {})
    if isinstance(overrides, dict):
        value = overrides.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                return default
    return default


def _expected_training_int(root: Path, key: str, default: int) -> int:
    protocol = _protocol(root)
    training = protocol.get("training", {})
    if isinstance(training, dict):
        value = training.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
    return default


def _expected_head_health_pair_floor(root: Path, default: float = 0.9) -> float:
    protocol = _protocol(root)
    head = protocol.get("head_localization", {})
    if isinstance(head, dict):
        value = head.get("healthy_pair_rate_floor")
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
    return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_components(component_dir: Path) -> dict[str, Any]:
    manifest_path = component_dir / "selection_manifest.json"
    _require(manifest_path.exists(), f"missing component manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    _require(int(manifest.get("topk_mlp", 0)) == 4, "component topk_mlp must be 4")
    _require(int(manifest.get("topk_attn", 0)) == 4, "component topk_attn must be 4")
    selected_groups = manifest.get("selected_groups", {})
    _require(isinstance(selected_groups, dict), "selected_groups must be an object")
    environment_manifest_path = component_dir.parent.parent / "env" / "environment_manifest.json"
    _require(environment_manifest_path.exists(), f"missing discovery environment manifest: {environment_manifest_path}")
    environment_manifest = _read_json(environment_manifest_path)
    _require(str(environment_manifest.get("source_split", "")) == "train", "component discovery must not use official test")

    file_hashes: dict[str, str] = {}
    selected_ids: dict[str, list[str]] = {}
    for filename in COMPONENT_FILES:
        path = component_dir / filename
        _require(path.exists(), f"missing selected component file: {path}")
        rows = _read_csv(path)
        _require(len(rows) == 4, f"{filename} must contain exactly 4 components, got {len(rows)}")
        ids = [str(row.get("component_id", "")) for row in rows]
        _require(all(ids), f"{filename} contains an empty component_id")
        _require(len(set(ids)) == 4, f"{filename} contains duplicate component ids")
        group_name = filename.removesuffix("_components.csv").replace("attn_", "attn_")
        expected = list(selected_groups.get(group_name, []))
        _require(ids == expected, f"{filename} does not match selection_manifest.json")
        file_hashes[filename] = _sha256(path)
        selected_ids[group_name] = ids

    directionality_path = component_dir.parent / "directionality_scan" / "component_directionality.csv"
    _require(directionality_path.exists(), f"missing rollout directionality data: {directionality_path}")
    directionality = {str(row.get("component_id", "")): row for row in _read_csv(directionality_path)}
    for ids in selected_ids.values():
        for component_id in ids:
            row = directionality.get(component_id)
            _require(row is not None, f"selected component missing from directionality scan: {component_id}")
            _require(int(_f(row.get("n"))) == 300, f"{component_id} was not audited on exactly 300 rollout samples")
            _require(str(row.get("apply_mode", "")) == "all", f"{component_id} discovery apply_mode must be all")
            _require(_f(row.get("mean_ablated_format_ok")) >= 0.9, f"{component_id} fails format health floor")
            _require(_f(row.get("format_collapse_rate")) <= 0.1, f"{component_id} exceeds collapse ceiling")

    return {
        "stage": "components",
        "component_dir": str(component_dir),
        "component_file_sha256": file_hashes,
        "selected_groups": selected_ids,
        "rollout_samples_per_component": 300,
        "discovery_source_split": "train",
    }


def audit_pairs(pairs_csv: Path) -> dict[str, Any]:
    _require(pairs_csv.exists(), f"missing pairs: {pairs_csv}")
    rows = _read_csv(pairs_csv)
    _require(bool(rows), "pairs.csv is empty")
    _require(REQUIRED_PAIR_FIELDS <= set(rows[0]), f"pairs.csv missing fields: {sorted(REQUIRED_PAIR_FIELDS - set(rows[0]))}")

    counts: Counter[str] = Counter()
    prompt_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    prompt_sample_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    for row in rows:
        split = str(row.get("split", ""))
        if split not in prompt_ids:
            continue
        _require(str(row.get("admitted", "")) == "1", "all formal pairs must be admitted")
        _require(str(row.get("event", "")) == "imdb_positive_sentiment", "unexpected pair event")
        _require(str(row.get("prompt", "")).strip() != "", "empty pair prompt")
        _require(str(row.get("y_plus", "")).strip() != "", "empty y_plus")
        _require(str(row.get("y_minus", "")).strip() != "", "empty y_minus")
        _require(str(row.get("y_plus", "")) != str(row.get("y_minus", "")), "identical y_plus/y_minus")
        plus = _f(row.get("y_plus_reward"))
        minus = _f(row.get("y_minus_reward"))
        margin = _f(row.get("reward_margin"))
        _require(plus > minus, "pair reward ordering is invalid")
        _require(margin > 0.0, f"pair reward margin must be positive: {margin}")
        _require(abs((plus - minus) - margin) < 1e-5, "reward_margin does not match endpoint rewards")
        counts[split] += 1
        prompt_ids[split].add(str(row.get("prompt_id", "")))
        prompt_sample_id = str(row.get("prompt_sample_id", ""))
        _require(prompt_sample_id not in prompt_sample_ids[split], f"{split}: duplicate prompt_sample_id {prompt_sample_id}")
        prompt_sample_ids[split].add(prompt_sample_id)

    _require(counts["train"] == 512, f"generated pair artifact must contain exactly 512 train pairs, got {counts['train']}")
    _require(counts["val"] == 128, f"generated pair artifact must contain exactly 128 val pairs, got {counts['val']}")
    _require(not (prompt_ids["train"] & prompt_ids["val"]), "train and val pair prompts overlap")
    return {
        "stage": "pairs",
        "pairs_csv": str(pairs_csv),
        "pairs_sha256": _sha256(pairs_csv),
        "pair_counts": dict(counts),
        "prompt_counts": {key: len(value) for key, value in prompt_ids.items()},
        "prompt_sample_counts": {key: len(value) for key, value in prompt_sample_ids.items()},
    }


def _check_training_config(
    path: Path,
    *,
    kind: str,
    pairs_csv: Path,
    expected_model: str,
    expected_train_pairs: int,
    expected_val_pairs: int,
    expected_train_batch_size: int,
) -> dict[str, Any]:
    _require(path.exists(), f"missing training config: {path}")
    config = _read_json(path)
    _require(str(config.get("model", "")) == expected_model, f"{path}: wrong model")
    _require(Path(str(config.get("pairs_csv", ""))).as_posix() == pairs_csv.as_posix(), f"{path}: wrong pairs_csv")
    _require(int(config.get("epochs", 0)) == 2, f"{path}: epochs must be 2")
    _require(int(config.get("train_batch_size", 0)) == expected_train_batch_size, f"{path}: train_batch_size must be {expected_train_batch_size}")
    _require(int(config.get("max_train_rows", 0)) == expected_train_pairs, f"{path}: max_train_rows must be {expected_train_pairs}")
    _require(int(config.get("max_val_rows", 0)) == expected_val_pairs, f"{path}: max_val_rows must be {expected_val_pairs}")
    _require(str(config.get("preference_loss_mode", "")) == "dpo", f"{path}: loss must be dpo")
    _require(abs(float(config.get("dpo_beta", 0.0)) - 1.0) < 1e-12, f"{path}: dpo_beta must be 1")
    _require(abs(float(config.get("alpha_train", 0.0)) - 1.0) < 1e-12, f"{path}: alpha_train must be 1")
    expected_apply_mode = "prefill" if kind == "mlp" else "all"
    _require(str(config.get("apply_mode", "")) == expected_apply_mode, f"{path}: apply_mode must be {expected_apply_mode}")
    _require(bool(config.get("causal_train_mask", False)), f"{path}: causal_train_mask must be true")
    _require(str(config.get("score_mode", "")) == "avglogp", f"{path}: score_mode must be avglogp")
    _require(str(config.get("endpoint_objective", "")) == "pair_margin", f"{path}: endpoint_objective must be pair_margin")
    _require(str(config.get("option_selection_mode", "")) == "model_max", f"{path}: option_selection_mode must be model_max")
    _require(abs(float(config.get("lr", 0.0)) - 0.05) < 1e-12, f"{path}: lr must be 0.05")
    _require(abs(float(config.get("lambda_norm", 0.0)) - 1e-4) < 1e-12, f"{path}: lambda_norm must be 1e-4")
    _require(not str(config.get("background_component_actuators", "")), f"{path}: background component actuators are forbidden")
    _require(not str(config.get("background_head_actuators", "")), f"{path}: background head actuators are forbidden")
    _require(not str(config.get("background_control_parts", "")), f"{path}: background control parts are forbidden")
    if kind == "mlp":
        _require(Path(str(config.get("components_csv", ""))).exists(), f"{path}: missing components_csv")
    else:
        heads = list(config.get("heads", []))
        _require(len(heads) == 4 and len(set(heads)) == 4, f"{path}: expected exactly 4 unique heads")
    return config


def audit_training(train_root: Path, pairs_csv: Path) -> dict[str, Any]:
    root = train_root.parent
    expected_train_pairs = _expected_training_int(root, "train_pairs", 512)
    expected_val_pairs = _expected_training_int(root, "val_pairs", 128)
    expected_train_batch_size = _expected_training_int(root, "train_batch_size", 16)
    common = {
        "pairs_csv": pairs_csv,
        "expected_model": _expected_model(root),
        "expected_train_pairs": expected_train_pairs,
        "expected_val_pairs": expected_val_pairs,
        "expected_train_batch_size": expected_train_batch_size,
    }
    configs = {
        "mlp_positive": _check_training_config(train_root / "train_mlp_positive" / "run_config.json", kind="mlp", **common),
        "mlp_negative": _check_training_config(train_root / "train_mlp_negative" / "run_config.json", kind="mlp", **common),
        "att_positive": _check_training_config(train_root / "train_head_positive" / "run_config.json", kind="head", **common),
        "att_negative": _check_training_config(train_root / "train_head_negative" / "run_config.json", kind="head", **common),
    }
    payloads = (
        train_root / "train_mlp_positive" / "fixed_actuator.pt",
        train_root / "train_mlp_negative" / "fixed_actuator.pt",
        train_root / "train_head_positive" / "head_actuator.pt",
        train_root / "train_head_negative" / "head_actuator.pt",
    )
    for payload in payloads:
        _require(payload.exists() and payload.stat().st_size > 0, f"missing trained actuator: {payload}")
    head_scan_config = _read_json(train_root / "head_scan_pool" / "raw" / "scan_config.json")
    _require(str(head_scan_config.get("model", "")) == _expected_model(root), "head scan used wrong model")
    _require(str(head_scan_config.get("scan_mode", "")) == "rollout_head_zero", "head scan must be rollout head zero-ablation")
    _require(
        Path(str(head_scan_config.get("prompts_jsonl", ""))).as_posix()
        == _expected_head_prompts(root),
        "head rollout scan must reuse the configured 300 component-discovery prompts",
    )
    _require(int(head_scan_config.get("prompt_samples", 0)) == 300, "head rollout scan must use exactly 300 samples")
    _require(str(head_scan_config.get("apply_mode", "")) == "all", "head rollout scan apply mode must be all")
    head_health_pair_floor = _expected_head_health_pair_floor(root)
    _require(int(head_scan_config.get("generation_batch_size", 0)) > 0, "head rollout generation batch must be positive")
    _require(int(head_scan_config.get("scorer_batch_size", 0)) > 0, "head rollout scorer batch must be positive")
    head_full_base = _read_jsonl(train_root / "head_scan_pool" / "raw" / "full_baseline.jsonl")
    _require(len(head_full_base) == 300, "head rollout scan must persist exactly one shared full baseline per prompt")
    _require(
        len({(str(row.get("sample_id", "")), int(row.get("completion_id", 0))) for row in head_full_base}) == 300,
        "head rollout shared full baseline contains duplicate sample keys",
    )
    component_dir = train_root.parent / "components"
    expected_attn_layers = {
        int(_f(row.get("layer_idx")))
        for filename in ("attn_positive_components.csv", "attn_negative_components.csv")
        for row in _read_csv(component_dir / filename)
    }
    _require(set(int(value) for value in head_scan_config.get("attn_layers", [])) == expected_attn_layers, "head scan layer pool is not the union of selected attention layers")
    selection_manifest = _read_json(train_root / "head_scan_pool" / "head_selection_manifest.json")
    _require(bool(selection_manifest.get("shared_pool", False)), "positive/negative heads were not selected from one shared pool")
    _require(selection_manifest.get("selection", {}).get("ci_filter") is False, "head selection must not use a CI filter")
    expected_positive_heads = [
        str(row.get("head_id", "")) for row in _read_csv(train_root / "head_scan_pool" / "selected_boost_heads.csv")
    ]
    expected_negative_heads = [
        str(row.get("head_id", "")) for row in _read_csv(train_root / "head_scan_pool" / "selected_suppress_heads.csv")
    ]
    _require(len(expected_positive_heads) == 4 and len(set(expected_positive_heads)) == 4, "need exactly 4 unique positive rollout heads")
    _require(len(expected_negative_heads) == 4 and len(set(expected_negative_heads)) == 4, "need exactly 4 unique negative rollout heads")
    for head_id in [*expected_positive_heads, *expected_negative_heads]:
        layer_idx = int(head_id.removeprefix("L").split(".attn.h", 1)[0])
        _require(layer_idx in expected_attn_layers, f"selected head escaped shared attention-layer pool: {head_id}")
    directionality = {
        str(row.get("component_id", "")): row
        for row in _read_csv(train_root / "head_scan_pool" / "directionality" / "component_directionality.csv")
    }
    for head_id in [*expected_positive_heads, *expected_negative_heads]:
        row = directionality.get(head_id)
        _require(row is not None, f"selected head missing from rollout directionality: {head_id}")
        _require(int(_f(row.get("n"))) == 300, f"{head_id}: expected 300 rollout samples")
        _require(_f(row.get("mean_ablated_format_ok")) >= 0.9, f"{head_id}: failed rollout format health")
        _require(_f(row.get("healthy_pair_rate")) >= head_health_pair_floor, f"{head_id}: failed rollout healthy-pair floor {head_health_pair_floor}")
        _require(_f(row.get("format_collapse_rate")) <= 0.1, f"{head_id}: exceeded rollout collapse ceiling")
    _require(list(configs["att_positive"].get("heads", [])) == expected_positive_heads, "positive head trainer did not use selected boost heads")
    _require(list(configs["att_negative"].get("heads", [])) == expected_negative_heads, "negative head trainer did not use selected suppress heads")
    return {
        "stage": "training",
        "train_root": str(train_root),
        "pairs_csv": str(pairs_csv),
        "epochs": 2,
        "train_pairs": expected_train_pairs,
        "val_pairs": expected_val_pairs,
        "head_rollout_samples": 300,
        "actuator_sha256": {payload.name + ":" + payload.parent.name: _sha256(payload) for payload in payloads},
    }


def _health(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, float]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("control_name", "")), _alpha(row.get("alpha"))), []).append(row)
    out: dict[tuple[str, float], dict[str, float]] = {}
    repeated_char = re.compile(r"(.)\1{7,}")
    repeated_word = re.compile(r"\b(\w+(?:\s+\w+){0,2})\b(?:\s+\1\b){2,}", flags=re.IGNORECASE)
    for key, items in grouped.items():
        empty = 0
        nonprintable = 0
        repetition = 0
        case_anomaly = 0
        symbol_anomaly = 0
        unhealthy = 0
        for row in items:
            text = str(row.get("completion", ""))
            is_empty = not text.strip()
            has_nonprintable = any((not char.isprintable()) and char not in "\n\r\t" for char in text)
            has_repetition = bool(repeated_char.search(text) or repeated_word.search(text))
            alpha_chars = [char for char in text if char.isalpha()]
            printable_nonspace = [char for char in text if char.isprintable() and not char.isspace()]
            has_case_anomaly = len(alpha_chars) >= 20 and sum(char.isupper() for char in alpha_chars) / len(alpha_chars) > 0.5
            has_symbol_anomaly = (
                len(printable_nonspace) >= 20
                and sum(not char.isalnum() for char in printable_nonspace) / len(printable_nonspace) > 0.5
            )
            empty += int(is_empty)
            nonprintable += int(has_nonprintable)
            repetition += int(has_repetition)
            case_anomaly += int(has_case_anomaly)
            symbol_anomaly += int(has_symbol_anomaly)
            unhealthy += int(is_empty or has_nonprintable or has_repetition or has_case_anomaly or has_symbol_anomaly)
        n = float(len(items))
        out[key] = {
            "empty_rate": empty / n,
            "nonprintable_rate": nonprintable / n,
            "repetition_rate": repetition / n,
            "case_anomaly_rate": case_anomaly / n,
            "symbol_anomaly_rate": symbol_anomaly / n,
            "local_format_ok_rate": 1.0 - (unhealthy / n),
        }
    return out


def audit_eval(eval_root: Path, expected_prompts: int, expected_alphas: tuple[float, ...]) -> dict[str, Any]:
    _require(expected_prompts > 0, "expected prompts must be positive")
    shared_dir = eval_root / "shared_base"
    shared_base = shared_dir / "generations.jsonl"
    _require(shared_base.exists(), f"missing shared base: {shared_base}")
    _require((shared_dir / "score_manifest.json").exists(), "missing single shared-base reward score manifest")
    _require((shared_dir / "kl_manifest.json").exists(), "missing single shared-base KL manifest")
    shared_generation_manifest = _read_json(shared_dir / "generation_manifest.json")
    shared_score_manifest = _read_json(shared_dir / "score_manifest.json")
    shared_kl_manifest = _read_json(shared_dir / "kl_manifest.json")
    root = eval_root.parent.parent if eval_root.name in {"val_select", "test_eval"} else eval_root.parent
    _require(int(shared_generation_manifest.get("generation_batch_size", 0)) > 0, "shared-base generation batch must be positive")
    _require(int(shared_score_manifest.get("batch_size", 0)) > 0, "shared-base scorer batch must be positive")
    _require(int(shared_kl_manifest.get("batch_size", 0)) > 0, "shared-base KL batch must be positive")
    shared_rows = _read_jsonl(shared_base)
    _require(len(shared_rows) == expected_prompts, f"shared base rows must be {expected_prompts}, got {len(shared_rows)}")
    shared_kl_rows = _read_jsonl(shared_dir / "kl_scored_generations.jsonl")
    _require(len(shared_kl_rows) == expected_prompts, f"shared base KL rows must be {expected_prompts}")
    shared_map = {
        (str(row.get("sample_id", "")), int(row.get("sample_index", 0))): row
        for row in shared_kl_rows
    }

    group_reports: dict[str, Any] = {}
    for group in GROUPS:
        group_dir = eval_root / group
        generation_manifest = _read_json(group_dir / "generation_manifest.json")
        score_manifest = _read_json(group_dir / "score_manifest.json")
        kl_manifest = _read_json(group_dir / "kl_manifest.json")
        _require(int(generation_manifest.get("generation_batch_size", 0)) > 0, f"{group}: generation batch must be positive")
        _require(int(score_manifest.get("batch_size", 0)) > 0, f"{group}: scorer batch must be positive")
        _require(int(kl_manifest.get("batch_size", 0)) > 0, f"{group}: KL batch must be positive")
        rows = _read_jsonl(group_dir / "kl_scored_generations.jsonl")
        alpha_counts = Counter(_alpha(row.get("alpha")) for row in rows)
        _require(set(alpha_counts) == set(expected_alphas), f"{group}: wrong alpha set {sorted(alpha_counts)}")
        for alpha in expected_alphas:
            _require(alpha_counts[alpha] == expected_prompts, f"{group} alpha={alpha}: expected {expected_prompts} rows")
        base_rows = [row for row in rows if abs(_alpha(row.get("alpha"))) < 1e-12]
        nonzero_rows = [row for row in rows if abs(_alpha(row.get("alpha"))) >= 1e-12]
        for row in base_rows:
            key = (str(row.get("sample_id", "")), int(row.get("sample_index", 0)))
            shared_row = shared_map.get(key)
            _require(shared_row is not None, f"{group}: base key missing from shared base at {key}")
            _require(str(row.get("completion", "")) == str(shared_row.get("completion", "")), f"{group}: base completion drift at {key}")
            _require(str(row.get("prompt", "")) == str(shared_row.get("prompt", "")), f"{group}: base prompt drift at {key}")
            _require(str(row.get("full_text", "")) == str(shared_row.get("full_text", "")), f"{group}: base full_text drift at {key}")
            _require(
                abs(_f(row.get("positive_sentiment_score")) - _f(shared_row.get("positive_sentiment_score"))) < 1e-12,
                f"{group}: base reward drift at {key}",
            )
            _require(abs(_f(row.get("sequence_kl"))) < 1e-6, f"{group}: base sequence KL must be zero")
        for row in nonzero_rows:
            _require(str(row.get("control_name", "")) == group, f"{group}: wrong control_name")
            _require(str(row.get("generation_apply_mode", "")) == "all", f"{group}: generation apply mode must be all")
            _require(str(row.get("component_apply_mode", "")) == "prefill", f"{group}: component apply mode must be prefill")
            _require(str(row.get("head_apply_mode", "")) == "all", f"{group}: head apply mode must be all")
        group_reports[group] = {"alpha_counts": dict(alpha_counts)}

    return {
        "stage": "eval",
        "eval_root": str(eval_root),
        "expected_prompts": expected_prompts,
        "expected_alphas": expected_alphas,
        "shared_base_sha256": _sha256(shared_base),
        "groups": group_reports,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _paired_bootstrap_ci(values: list[float], *, seed: int, samples: int = 2000) -> tuple[float, float]:
    _require(bool(values), "cannot bootstrap an empty paired sample")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def build_main_table(root: Path, eval_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        rows.extend(_read_jsonl(eval_root / group / "kl_scored_generations.jsonl"))
    health = _health(rows)
    summary: dict[tuple[str, float], dict[str, float]] = {}
    for row in rows:
        key = (str(row.get("control_name", "")), _alpha(row.get("alpha")))
        stats = summary.setdefault(
            key,
            {"n": 0.0, "reward": 0.0, "sequence_kl": 0.0, "chars": 0.0, "tokens": 0.0},
        )
        stats["n"] += 1.0
        stats["reward"] += _f(row.get("positive_sentiment_score"))
        stats["sequence_kl"] += _f(row.get("sequence_kl"))
        stats["chars"] += len(str(row.get("completion", "")))
        stats["tokens"] += _f(row.get("completion_token_count"))

    table: list[dict[str, Any]] = []
    for group in GROUPS:
        base = summary[(group, 0.0)]
        steered = summary[(group, 0.3)]
        base_n = base["n"]
        n = steered["n"]
        base_reward = base["reward"] / base_n
        reward = steered["reward"] / n
        base_chars = base["chars"] / base_n
        chars = steered["chars"] / n
        base_tokens = base["tokens"] / base_n
        tokens = steered["tokens"] / n
        group_rows = [row for row in rows if str(row.get("control_name", "")) == group]
        base_by_key = {
            (str(row.get("sample_id", "")), int(row.get("sample_index", 0))): row
            for row in group_rows
            if abs(_alpha(row.get("alpha"))) < 1e-12
        }
        steered_by_key = {
            (str(row.get("sample_id", "")), int(row.get("sample_index", 0))): row
            for row in group_rows
            if _alpha(row.get("alpha")) == 0.3
        }
        _require(base_by_key.keys() == steered_by_key.keys(), f"{group}: alpha=0 and alpha=0.3 samples are not paired")
        reward_deltas: list[float] = []
        transition_values: list[float] = []
        neg_to_pos = 0
        pos_to_neg = 0
        base_positive = 0
        steered_positive = 0
        for key, base_row in base_by_key.items():
            steered_row = steered_by_key[key]
            base_score = _f(base_row.get("positive_sentiment_score"))
            steered_score = _f(steered_row.get("positive_sentiment_score"))
            base_is_positive = base_score >= 0.5
            steered_is_positive = steered_score >= 0.5
            reward_deltas.append(steered_score - base_score)
            base_positive += int(base_is_positive)
            steered_positive += int(steered_is_positive)
            moved_positive = (not base_is_positive) and steered_is_positive
            moved_negative = base_is_positive and (not steered_is_positive)
            neg_to_pos += int(moved_positive)
            pos_to_neg += int(moved_negative)
            transition_values.append(float(int(moved_positive) - int(moved_negative)))
        delta_ci_low, delta_ci_high = _paired_bootstrap_ci(reward_deltas, seed=42)
        transition_ci_low, transition_ci_high = _paired_bootstrap_ci(transition_values, seed=4242)
        base_health = health[(group, 0.0)]
        steered_health = health[(group, 0.3)]
        token_ratio = tokens / base_tokens if base_tokens else 0.0
        _require(0.75 <= token_ratio <= 1.25, f"{group}: completion token ratio fails health gate: {token_ratio}")
        _require(steered_health["local_format_ok_rate"] >= 0.97, f"{group}: local format-ok rate below 0.97")
        _require(
            steered_health["local_format_ok_rate"] >= base_health["local_format_ok_rate"] - 0.02,
            f"{group}: local format-ok rate dropped by more than 0.02",
        )
        table.append(
            {
                "method": group,
                "alpha": 0.3,
                "n": int(n),
                "base_reward": base_reward,
                "reward": reward,
                "delta_reward": reward - base_reward,
                "delta_reward_ci95_low": delta_ci_low,
                "delta_reward_ci95_high": delta_ci_high,
                "base_positive_rate": base_positive / n,
                "positive_rate": steered_positive / n,
                "delta_positive_rate": (steered_positive - base_positive) / n,
                "neg_to_pos_count": neg_to_pos,
                "neg_to_pos_rate_all": neg_to_pos / n,
                "neg_to_pos_rate_given_base_neg": neg_to_pos / (n - base_positive) if n > base_positive else 0.0,
                "pos_to_neg_count": pos_to_neg,
                "pos_to_neg_rate_all": pos_to_neg / n,
                "pos_to_neg_rate_given_base_pos": pos_to_neg / base_positive if base_positive else 0.0,
                "net_positive_transition_rate": (neg_to_pos - pos_to_neg) / n,
                "net_positive_transition_ci95_low": transition_ci_low,
                "net_positive_transition_ci95_high": transition_ci_high,
                "mean_sequence_kl": steered["sequence_kl"] / n,
                "base_mean_completion_chars": base_chars,
                "mean_completion_chars": chars,
                "completion_char_ratio": chars / base_chars if base_chars else 0.0,
                "base_mean_completion_tokens": base_tokens,
                "mean_completion_tokens": tokens,
                "completion_token_ratio": token_ratio,
                **steered_health,
            }
        )
    out_csv = root / "main_table.csv"
    dump_csv(out_csv, table)
    report_columns = (
        "method",
        "alpha",
        "reward",
        "delta_reward",
        "delta_reward_ci95_low",
        "delta_reward_ci95_high",
        "net_positive_transition_rate",
        "net_positive_transition_ci95_low",
        "net_positive_transition_ci95_high",
        "mean_sequence_kl",
        "completion_token_ratio",
        "local_format_ok_rate",
    )
    markdown_lines = [
        "# IMDb Locked Main Table",
        "",
        "This table is emitted only after the shared-base, timing, pairing, KL, and health gates pass.",
        "",
        "| " + " | ".join(report_columns) + " |",
        "|" + "|".join("---" for _ in report_columns) + "|",
    ]
    for row in table:
        values = []
        for column in report_columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        markdown_lines.append("| " + " | ".join(values) + " |")
    out_markdown = root / "main_table.md"
    out_markdown.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return {
        "stage": "final",
        "main_table_eligible": True,
        "main_table_csv": str(out_csv),
        "main_table_markdown": str(out_markdown),
        "rows": len(table),
        "health_gates": {
            "completion_token_ratio": [0.75, 1.25],
            "minimum_local_format_ok_rate": 0.97,
            "maximum_local_format_ok_drop": 0.02,
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == "components":
        _require(args.component_dir is not None, "--component-dir is required")
        report = audit_components(args.component_dir)
    elif args.stage == "pairs":
        _require(args.pairs_csv is not None, "--pairs-csv is required")
        report = audit_pairs(args.pairs_csv)
    elif args.stage == "training":
        _require(args.train_root is not None and args.pairs_csv is not None, "--train-root and --pairs-csv are required")
        report = audit_training(args.train_root, args.pairs_csv)
    elif args.stage == "eval":
        _require(args.eval_root is not None, "--eval-root is required")
        expected_alphas = VAL_ALPHAS if args.eval_root.name == "val_select" else TEST_ALPHAS
        report = audit_eval(args.eval_root, args.expected_prompts, expected_alphas)
    else:
        _require(args.eval_root is not None, "--eval-root is required")
        audit_eval(args.eval_root, args.expected_prompts, TEST_ALPHAS)
        report = build_main_table(args.root, args.eval_root)
    out_path = args.root / f"audit_{args.stage}.json"
    dump_json(out_path, report)
    print(f"[locked-audit] PASS stage={args.stage} report={out_path}", flush=True)


if __name__ == "__main__":
    main()
