#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screscomp.data import dump_json  # noqa: E402
from screscomp.tldr_site.protocol import CANDIDATE_IDS, SELECTORS, TldrSiteProtocol, require, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and expand the locked TLDR Site + ReFT audit protocol.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _validate_pair_manifest(protocol: TldrSiteProtocol) -> dict[str, Any]:
    path = Path(protocol.data["inputs"]["pair_build_manifest"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("event") == "tldr_summary_preference", "pair manifest event drift")
    require(payload.get("prompt_format") == "openai_tldr_structured_v1", "pair prompt format drift")
    require(int(payload.get("admitted_pairs", 0)) == 176632, "pair count drift")
    require(payload.get("actual_counts") == {"val": 83778, "train": 92854}, "pair split counts drift")
    return payload


def _static_plan(protocol: TldrSiteProtocol) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selector in SELECTORS:
        rows.append({
            "stage": "selector",
            "selector": selector,
            "candidate_id": "",
            "bank": "",
            "alpha": "",
            "count": 1,
        })
    for selector in SELECTORS:
        banks = ("target_support", "competitor_support") if selector.startswith("rcm_") else ("unsigned",)
        for candidate_id in CANDIDATE_IDS:
            for bank in banks:
                rows.append({
                    "stage": "train_reft",
                    "selector": selector,
                    "candidate_id": candidate_id,
                    "bank": bank,
                    "alpha": "",
                    "count": 1,
                })
            rows.append({
                "stage": "calibration_generation_sweep",
                "selector": selector,
                "candidate_id": candidate_id,
                "bank": "+".join(banks),
                "alpha": ",".join(str(value) for value in protocol.alpha_grid),
                "count": 1,
            })
            for alpha in protocol.alpha_grid:
                rows.append({
                    "stage": "calibration_judge_point",
                    "selector": selector,
                    "candidate_id": candidate_id,
                    "bank": "+".join(banks),
                    "alpha": alpha,
                    "count": 1,
                })
    return rows


def main() -> None:
    args = parse_args()
    protocol = TldrSiteProtocol.load(args.config, validate_files=False)
    require(protocol.repo_root == REPO_ROOT, "config is not inside the remote source repository")
    entrypoint_files = (
        protocol.data["documentation"],
        protocol.data["preflight"],
        protocol.data["runner"],
        "src/screscomp/tldr_site/protocol.py",
        "src/screscomp/tldr_site/selection.py",
        "src/screscomp/tldr_site/audit.py",
        "src/screscomp/cli/tldr_site_select_heads.py",
        "src/screscomp/cli/cecm_train_cast_reft.py",
        "src/screscomp/cli/run_cast_reft_generation.py",
        "src/screscomp/cli/evaluate_tldr_dpo_ds4.py",
        "scripts/audit_tldr_text_health.py",
    )
    source_files = tuple(
        str(path.relative_to(REPO_ROOT))
        for path in sorted((REPO_ROOT / "src" / "screscomp").rglob("*.py"))
    )
    implementation_files = tuple(dict.fromkeys([*entrypoint_files, *source_files]))
    for relative in implementation_files:
        require((REPO_ROOT / relative).is_file(), f"missing protocol implementation: {relative}")

    input_audit = protocol.validate_input_files()
    pair_manifest = _validate_pair_manifest(protocol)
    iti_repo = Path(protocol.selectors["iti"]["official_repository"])
    iti_commit = _git(iti_repo, "rev-parse", "HEAD")
    require(iti_commit == protocol.selectors["iti"]["official_commit"], "honest_llama commit drift")
    plan = _static_plan(protocol)
    require(sum(row["stage"] == "selector" for row in plan) == 4, "selector plan drift")
    require(sum(row["stage"] == "train_reft" for row in plan) == 18, "ReFT training plan drift")
    require(sum(row["stage"] == "calibration_generation_sweep" for row in plan) == 12, "generation plan drift")
    require(sum(row["stage"] == "calibration_judge_point" for row in plan) == 72, "calibration point drift")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "execution_plan.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plan[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(plan)
    package_versions = {}
    for distribution in ("torch", "transformers", "numpy", "scikit-learn", "safetensors"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = "missing"
    audit = {
        "protocol_id": protocol.data["protocol_id"],
        "config": str(protocol.config_path),
        "config_sha256": protocol.config_sha256,
        "documentation": str(REPO_ROOT / protocol.data["documentation"]),
        "documentation_sha256": sha256_file(REPO_ROOT / protocol.data["documentation"]),
        "repository": str(REPO_ROOT),
        "repository_commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "dirty_tracked_files": _git(REPO_ROOT, "diff", "--name-only").splitlines(),
        "implementation_files": [
            {
                "path": relative,
                "sha256": sha256_file(REPO_ROOT / relative),
            }
            for relative in implementation_files
        ],
        "runtime_environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "packages": package_versions,
        },
        "input_audit": input_audit,
        "pair_manifest": pair_manifest,
        "honest_llama_commit": iti_commit,
        "selector_count": 4,
        "complete_configurations_per_selector": 3,
        "complete_configurations": 12,
        "reft_training_jobs": 18,
        "calibration_generation_sweeps": 12,
        "calibration_candidate_alpha_points": 72,
        "trainable_parameters_per_complete_configuration": 16384,
        "raw_table_rule": protocol.data["common_position_audit"]["raw_table"],
        "common_audit_rule": protocol.data["common_position_audit"]["audit_assisted_table"],
        "final_split_locked_until_calibration_freeze": True,
        "gpu_work_started": False,
        "status": "no_gpu_preflight_passed",
    }
    dump_json(args.out_dir / "protocol_audit.json", audit)
    print(
        "[tldr-site-preflight] passed "
        f"selectors=4 configurations=12 reft_jobs=18 calibration_points=72 out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
