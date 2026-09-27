from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl
from screscomp.tldr_site.protocol import CANDIDATE_IDS, SELECTORS, TldrSiteProtocol, require, sha256_file


RAW_LABELS = {
    "rcm_zero_signed": "RCM-zero-signed",
    "rcm_patch_signed": "RCM-patch-signed",
    "iti": "ITI",
    "random": "Random",
}
AUDIT_LABELS = {
    "rcm_zero_signed": "RCM-zero + common audit",
    "rcm_patch_signed": "RCM-patch + common audit",
    "iti": "ITI + common audit",
    "random": "Random search + common audit",
}


def build_selector_overlap(protocol: TldrSiteProtocol) -> dict[str, Any]:
    configurations: list[dict[str, Any]] = []
    source_manifests = []
    for selector in SELECTORS:
        path = protocol.paths.candidates_json(selector)
        require(path.is_file(), f"missing selector candidates for overlap audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload.get("config_sha256") == protocol.config_sha256, f"selector config drift: {selector}")
        source_manifests.append({"selector": selector, "path": str(path), "sha256": sha256_file(path)})
        for row in payload["configurations"]:
            configurations.append({
                "selector": selector,
                "candidate_id": str(row["candidate_id"]),
                "heads": [str(value) for value in row["heads"]],
            })
    require(len(configurations) == len(SELECTORS) * len(CANDIDATE_IDS), "selector overlap input count drift")
    rows = []
    for left, right in combinations(configurations, 2):
        left_heads = set(left["heads"])
        right_heads = set(right["heads"])
        intersection = sorted(left_heads & right_heads)
        union = left_heads | right_heads
        rows.append({
            "left_selector": left["selector"],
            "left_candidate_id": left["candidate_id"],
            "right_selector": right["selector"],
            "right_candidate_id": right["candidate_id"],
            "cross_selector": left["selector"] != right["selector"],
            "matched_candidate_index": left["candidate_id"] == right["candidate_id"],
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union),
            "overlap_heads": ",".join(intersection),
        })
    out_dir = protocol.paths.root / "selectors"
    table_path = out_dir / "selector_overlap.csv"
    manifest_path = out_dir / "selector_overlap_manifest.json"
    dump_csv(table_path, rows)
    payload = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "source_candidate_manifests": source_manifests,
        "configurations": len(configurations),
        "pairwise_rows": len(rows),
        "table": str(table_path),
        "table_sha256": sha256_file(table_path),
        "status": "complete",
    }
    dump_json(manifest_path, payload)
    return payload


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):.1f}".replace("-", "m").replace(".", "p")


def point_name(selector: str, candidate_id: str, alpha: float) -> str:
    return f"{selector}--{candidate_id}--a{alpha_tag(alpha)}"


def split_generation_sweep(
    protocol: TldrSiteProtocol,
    selector: str,
    candidate_id: str,
    sweep_path: Path,
) -> dict[str, Path]:
    rows = load_jsonl(sweep_path)
    expected_n = int(protocol.data["inputs"]["calibration"]["rows"])
    outputs: dict[str, Path] = {}
    manifest_rows = []
    for alpha in protocol.alpha_grid:
        selected = [row for row in rows if abs(float(row.get("alpha", -999.0)) - alpha) <= 1e-12]
        require(len(selected) == expected_n, f"{selector}/{candidate_id}/alpha={alpha} has {len(selected)} rows")
        sample_ids = [str(row.get("sample_id", "")) for row in selected]
        require(len(set(sample_ids)) == expected_n and all(sample_ids), f"duplicate calibration samples for {selector}/{candidate_id}/alpha={alpha}")
        name = point_name(selector, candidate_id, alpha)
        path = protocol.paths.calibration_alpha_dir() / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        dump_jsonl(path, selected)
        outputs[name] = path
        manifest_rows.append({
            "point_name": name,
            "selector": selector,
            "candidate_id": candidate_id,
            "alpha": alpha,
            "rows": len(selected),
            "path": str(path),
            "sha256": sha256_file(path),
        })
    dump_json(
        sweep_path.with_name("alpha_split_manifest.json"),
        {
            "protocol_id": protocol.data["protocol_id"],
            "config_sha256": protocol.config_sha256,
            "source_sweep": str(sweep_path),
            "source_sweep_sha256": sha256_file(sweep_path),
            "points": manifest_rows,
        },
    )
    return outputs


def all_calibration_points(protocol: TldrSiteProtocol) -> dict[str, Path]:
    points: dict[str, Path] = {}
    for selector in SELECTORS:
        for candidate_id in CANDIDATE_IDS:
            for alpha in protocol.alpha_grid:
                name = point_name(selector, candidate_id, alpha)
                path = protocol.paths.calibration_alpha_dir() / f"{name}.jsonl"
                require(path.is_file(), f"missing calibration point: {path}")
                points[name] = path
    require(len(points) == 72, f"calibration plan must contain 72 points, found {len(points)}")
    return points


def _load_health(path: Path) -> dict[str, dict[str, str]]:
    require(path.is_file(), f"missing health audit: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {str(row["method"]): row for row in rows}


def _pairwise_index(summary_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    require(summary_path.is_file(), f"missing judge summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {(str(row["left"]), str(row["right"])): row for row in payload["pairwise"]}


def _balanced_row(row: dict[str, Any]) -> dict[str, float]:
    first = float(row["left_win_rate"])
    swap_raw = row.get("review_left_win_rate")
    require(swap_raw is not None, f"comparison {row.get('pair')} lacks full order swap")
    swap = float(swap_raw)
    return {
        "first": first,
        "swap": swap,
        "balanced": (first + swap) / 2.0,
        "agreement": float(row.get("order_swap_agreement_rate") or 0.0),
        "n": int(row["n"]),
        "swap_n": int(row.get("reviewed", 0)),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["balanced_human_win_rate"]),
        float(row["order_swap_agreement"]),
        -float(row["alpha"]),
        -int(row["candidate_index"]),
    )


def freeze_calibration_selection(protocol: TldrSiteProtocol) -> dict[str, Any]:
    if protocol.paths.calibration_selection.exists() or protocol.paths.calibration_selection_seal.exists():
        require(
            protocol.paths.calibration_selection.is_file()
            and protocol.paths.calibration_selection_seal.is_file(),
            "partial frozen-selection seal; refusing to recompute",
        )
        return load_frozen_selection(protocol)
    judge_summary = protocol.paths.calibration_judge_dir / "pairwise_summary.json"
    pairwise = _pairwise_index(judge_summary)
    health = _load_health(protocol.paths.calibration_health_csv)
    rows: list[dict[str, Any]] = []
    for selector in SELECTORS:
        for candidate_index, candidate_id in enumerate(CANDIDATE_IDS):
            for alpha in protocol.alpha_grid:
                name = point_name(selector, candidate_id, alpha)
                key = (name, "human")
                require(key in pairwise, f"missing calibration judge comparison: {name} vs human")
                require(name in health, f"missing calibration health row: {name}")
                judge = _balanced_row(pairwise[key])
                require(judge["n"] == 320 and judge["swap_n"] == 320, f"incomplete calibration judge: {name}")
                require(int(float(health[name].get("n", 0))) == 320, f"incomplete calibration health: {name}")
                rows.append({
                    "point_name": name,
                    "selector": selector,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "alpha": alpha,
                    "first_human_win_rate": judge["first"],
                    "swap_human_win_rate": judge["swap"],
                    "balanced_human_win_rate": judge["balanced"],
                    "order_swap_agreement": judge["agreement"],
                    "actual_n": judge["n"],
                    "health_status": health[name].get("status", ""),
                    "empty_output_rate": health[name].get("empty_output_rate", ""),
                    "repeat_trigram_rate": health[name].get("repeat_trigram_rate", ""),
                    "likely_unfinished_rate": health[name].get("likely_unfinished_rate", ""),
                    "unique_summary_ratio": health[name].get("unique_summary_ratio", ""),
                })
    require(len(rows) == 72, "calibration score table is incomplete")
    dump_csv(protocol.paths.root / "calibration" / "calibration_scores.csv", rows)
    selections: dict[str, Any] = {}
    for selector in SELECTORS:
        local = [row for row in rows if row["selector"] == selector]
        raw = max((row for row in local if row["candidate_id"] == "candidate_00"), key=_selection_key)
        audited = max(local, key=_selection_key)
        selections[selector] = {
            "raw": {**raw, "row_label": RAW_LABELS[selector], "position_search_trials": 1},
            "common_audit": {**audited, "row_label": AUDIT_LABELS[selector], "position_search_trials": 3},
        }
    payload = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "calibration_split": protocol.data["inputs"]["calibration"],
        "selection_metric": protocol.data["common_position_audit"]["selection_metric"],
        "health_used_for_selection": False,
        "judge_summary": str(judge_summary),
        "judge_summary_sha256": sha256_file(judge_summary),
        "health_audit": str(protocol.paths.calibration_health_csv),
        "health_audit_sha256": sha256_file(protocol.paths.calibration_health_csv),
        "calibration_scores": str(protocol.paths.root / "calibration" / "calibration_scores.csv"),
        "selections": selections,
        "status": "frozen_before_final",
    }
    dump_json(protocol.paths.calibration_selection, payload)
    dump_json(protocol.paths.calibration_selection_seal, {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "selection": str(protocol.paths.calibration_selection),
        "selection_sha256": sha256_file(protocol.paths.calibration_selection),
        "judge_summary_sha256": payload["judge_summary_sha256"],
        "health_audit_sha256": payload["health_audit_sha256"],
        "status": "sealed_before_final",
    })
    return payload


def load_frozen_selection(protocol: TldrSiteProtocol) -> dict[str, Any]:
    path = protocol.paths.calibration_selection
    seal_path = protocol.paths.calibration_selection_seal
    require(path.is_file() and seal_path.is_file(), "final split is locked until calibration selection is sealed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    require(payload.get("config_sha256") == protocol.config_sha256, "frozen selection config drift")
    require(payload.get("status") == "frozen_before_final", "calibration selection is not frozen")
    require(seal.get("config_sha256") == protocol.config_sha256, "frozen selection seal config drift")
    require(seal.get("status") == "sealed_before_final", "calibration selection seal is incomplete")
    require(seal.get("selection_sha256") == sha256_file(path), "frozen selection hash drift")
    require(
        payload.get("judge_summary_sha256") == sha256_file(Path(payload["judge_summary"])),
        "frozen calibration judge evidence drift",
    )
    require(
        payload.get("health_audit_sha256") == sha256_file(Path(payload["health_audit"])),
        "frozen calibration health evidence drift",
    )
    require(seal.get("judge_summary_sha256") == payload["judge_summary_sha256"], "selection seal judge hash drift")
    require(seal.get("health_audit_sha256") == payload["health_audit_sha256"], "selection seal health hash drift")
    return payload


def unique_frozen_points(protocol: TldrSiteProtocol) -> dict[str, dict[str, Any]]:
    selection = load_frozen_selection(protocol)
    points: dict[str, dict[str, Any]] = {}
    for selector in SELECTORS:
        for table in ("raw", "common_audit"):
            row = selection["selections"][selector][table]
            points.setdefault(str(row["point_name"]), row)
    return points


def _oriented_pair_metrics(
    pairwise: dict[tuple[str, str], dict[str, Any]],
    target: str,
    opponent: str,
) -> dict[str, float]:
    if (target, opponent) in pairwise:
        return _balanced_row(pairwise[(target, opponent)])
    require((opponent, target) in pairwise, f"missing final comparison: {target} vs {opponent}")
    raw = _balanced_row(pairwise[(opponent, target)])
    return {
        "first": 1.0 - raw["first"],
        "swap": 1.0 - raw["swap"],
        "balanced": 1.0 - raw["balanced"],
        "agreement": raw["agreement"],
        "n": raw["n"],
        "swap_n": raw["swap_n"],
    }


def finalize_tables(protocol: TldrSiteProtocol) -> dict[str, Any]:
    selection = load_frozen_selection(protocol)
    judge_summary = protocol.paths.final_judge_dir / "pairwise_summary.json"
    pairwise = _pairwise_index(judge_summary)
    health = _load_health(protocol.paths.final_health_csv)
    points = unique_frozen_points(protocol)
    require("sft" in health, "final health audit lacks shared SFT")

    table_rows: dict[str, list[dict[str, Any]]] = {"raw": [], "common_audit": []}
    for table in table_rows:
        for selector in SELECTORS:
            selected = selection["selections"][selector][table]
            name = str(selected["point_name"])
            require(name in points and name in health, f"missing final point audit: {name}")
            human = _oriented_pair_metrics(pairwise, name, "human")
            sft = _oriented_pair_metrics(pairwise, name, "sft")
            require(human["n"] == human["swap_n"] == 320, f"incomplete final human comparison: {name}")
            require(sft["n"] == sft["swap_n"] == 320, f"incomplete final SFT comparison: {name}")
            row = {
                "row_label": selected["row_label"],
                "selector": selector,
                "point_name": name,
                "candidate_id": selected["candidate_id"],
                "alpha": selected["alpha"],
                "position_search_trials": selected["position_search_trials"],
                "first_vs_human": human["first"],
                "swap_vs_human": human["swap"],
                "balanced_vs_human": human["balanced"],
                "agreement_vs_human": human["agreement"],
                "balanced_vs_sft": sft["balanced"],
                "agreement_vs_sft": sft["agreement"],
                "actual_n": human["n"],
            }
            for field in protocol.data["evaluation"]["health"]["required_metrics"]:
                row[field] = health[name].get(field, "")
            table_rows[table].append(row)

    raw_path = protocol.paths.final_dir / "raw_selector_table.csv"
    audit_path = protocol.paths.final_dir / "common_audit_table.csv"
    dump_csv(raw_path, table_rows["raw"])
    dump_csv(audit_path, table_rows["common_audit"])
    pair_rows = []
    for (left, right), row in sorted(pairwise.items()):
        metrics = _balanced_row(row)
        pair_rows.append({
            "left": left,
            "right": right,
            "first_left_win_rate": metrics["first"],
            "swap_left_win_rate": metrics["swap"],
            "balanced_left_win_rate": metrics["balanced"],
            "order_swap_agreement": metrics["agreement"],
            "n": metrics["n"],
            "swap_n": metrics["swap_n"],
        })
    pair_path = protocol.paths.final_dir / "pairwise_matrix_long.csv"
    dump_csv(pair_path, pair_rows)
    payload = {
        "protocol_id": protocol.data["protocol_id"],
        "config_sha256": protocol.config_sha256,
        "frozen_selection": str(protocol.paths.calibration_selection),
        "frozen_selection_sha256": sha256_file(protocol.paths.calibration_selection),
        "frozen_selection_seal": str(protocol.paths.calibration_selection_seal),
        "frozen_selection_seal_sha256": sha256_file(protocol.paths.calibration_selection_seal),
        "final_split": protocol.data["inputs"]["final"],
        "judge_summary": str(judge_summary),
        "judge_summary_sha256": sha256_file(judge_summary),
        "health_audit": str(protocol.paths.final_health_csv),
        "health_audit_sha256": sha256_file(protocol.paths.final_health_csv),
        "raw_table": str(raw_path),
        "raw_table_sha256": sha256_file(raw_path),
        "common_audit_table": str(audit_path),
        "common_audit_table_sha256": sha256_file(audit_path),
        "pairwise_matrix": str(pair_path),
        "pairwise_matrix_sha256": sha256_file(pair_path),
        "selector_overlap": str(protocol.paths.root / "selectors" / "selector_overlap.csv"),
        "selector_overlap_sha256": sha256_file(protocol.paths.root / "selectors" / "selector_overlap.csv"),
        "status": "complete",
    }
    dump_json(protocol.paths.final_dir / "final_manifest.json", payload)
    return payload
