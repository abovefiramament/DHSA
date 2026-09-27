#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screscomp.site.protocol import SiteJob, SiteProtocol, sha256_file  # noqa: E402


MODEL_KEY = "qwen25_14b"
SUBSETS = ("qa", "mr")
EXPECTED_JOBS = 12
PROTOCOL_DOCUMENT = REPO_ROOT / "docs" / "SITE_POSITION_FULL_FIXED_PROTOCOL_20260719_V12.md"
AMENDMENT_DOCUMENT = REPO_ROOT / "docs" / "SITE_V12_QWEN_EXACT_MEMORY_EXECUTION_AMENDMENT_20260731_V2.md"
CODE_SNAPSHOT_PATHS = (
    REPO_ROOT / "src" / "screscomp",
    REPO_ROOT / "scripts" / "paper",
    REPO_ROOT / "pyproject.toml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze completed Qwen2.5-14B Site V12 QA/MR evidence without recomputation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        shutil.copy2(source, destination)


def line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def required_job_files(job: SiteJob) -> tuple[str, ...]:
    common = (
        "selector/selector_manifest.json",
        "selector/selected_heads.csv",
        "actuator/training_manifest.json",
        "actuator/head_actuator.pt",
        "evaluation/alpha_selection.json",
        "evaluation/dev_manifest.json",
        "evaluation/dev/generation_rows.jsonl",
        "evaluation/dev/site_summary.csv",
        "evaluation/test_manifest.json",
        "evaluation/test/generation_rows.jsonl",
        "evaluation/test/generation_summary.csv",
        "evaluation/test/site_summary.csv",
    )
    if job.selector.startswith("rcm_"):
        return common + (
            "actuator/target_support/head_actuator.pt",
            "actuator/competitor_support/head_actuator.pt",
        )
    return common


def validate_job(protocol: SiteProtocol, artifact_root: Path, job: SiteJob) -> dict[str, Any]:
    job_dir = artifact_root / job.job_id
    missing = [relative for relative in required_job_files(job) if not (job_dir / relative).is_file()]
    if missing:
        raise ValueError(f"incomplete job {job.job_id}: {missing}")

    manifest = read_json(job_dir / "evaluation" / "test_manifest.json")
    expected_manifest = {
        "protocol_id": protocol.data["protocol_id"],
        "protocol_version": protocol.data["version"],
        "config_sha256": protocol.config_sha256,
        "job_id": job.job_id,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"test manifest drift job={job.job_id} key={key} "
                f"expected={expected!r} actual={manifest.get(key)!r}"
            )

    alpha = read_json(job_dir / "evaluation" / "alpha_selection.json")
    dev_rows = read_csv(job_dir / "evaluation" / "dev" / "site_summary.csv")
    test_rows = read_csv(job_dir / "evaluation" / "test" / "site_summary.csv")
    if not dev_rows or not test_rows:
        raise ValueError(f"empty evaluation summary: {job.job_id}")
    selected_alpha = float(alpha["selected_alpha"])
    final_row = test_rows[-1]
    if float(final_row["alpha"]) != selected_alpha:
        raise ValueError(f"selected/test alpha mismatch: {job.job_id}")
    actual_n = int(final_row["n"])
    generated_rows = line_count(job_dir / "evaluation" / "test" / "generation_rows.jsonl")
    if generated_rows != actual_n:
        raise ValueError(
            f"test row mismatch job={job.job_id} summary_n={actual_n} generated_rows={generated_rows}"
        )
    return {
        "job_id": job.job_id,
        "dataset": job.dataset,
        "subset": job.subset,
        "model_key": job.model_key,
        "model": job.model,
        "model_revision": job.model_revision,
        "selector": job.selector,
        "replicate": job.replicate,
        "seed": job.seed,
        "selected_alpha": selected_alpha,
        "dev_actual_n": int(alpha["actual_n"]),
        "test_actual_n": actual_n,
        "status": "complete",
    }


def collect_result_rows(artifact_root: Path, jobs: Iterable[SiteJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        summary = artifact_root / job.job_id / "evaluation" / "test" / "site_summary.csv"
        for row in read_csv(summary):
            rows.append(
                {
                    "job_id": job.job_id,
                    "subset": job.subset,
                    "selector": job.selector,
                    "replicate": job.replicate,
                    "seed": job.seed,
                    **row,
                }
            )
    return rows


def collect_selected_heads(artifact_root: Path, jobs: Iterable[SiteJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        selected = artifact_root / job.job_id / "selector" / "selected_heads.csv"
        for row in read_csv(selected):
            rows.append(
                {
                    "job_id": job.job_id,
                    "subset": job.subset,
                    "selector": job.selector,
                    "replicate": job.replicate,
                    "seed": job.seed,
                    **row,
                }
            )
    return rows


def inventory_files(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "freeze_manifest.json":
            continue
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def repository_state() -> tuple[str, list[str]]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--short"], text=True
    ).splitlines()
    return commit, dirty


def environment_snapshot() -> dict[str, Any]:
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import json, platform, torch, transformers; "
                "print(json.dumps({'python': platform.python_version(), "
                "'torch': torch.__version__, 'transformers': transformers.__version__, "
                "'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}))"
            ),
        ],
        text=True,
    )
    return {
        **json.loads(probe),
        "executable": sys.executable,
        "pip_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines(),
    }


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    artifact_root = args.artifact_root.resolve()
    locked_root = Path(protocol.site["execution"]["run_root"]).resolve()
    if artifact_root != locked_root:
        raise ValueError(f"artifact root drift: locked={locked_root} requested={artifact_root}")

    amendment = read_json(args.execution_amendment.resolve())
    if amendment.get("scientific_settings_changed") is not False:
        raise ValueError("execution amendment does not preserve scientific settings")

    jobs = [
        job
        for job in protocol.iter_jobs()
        if job.dataset == "confiqa" and job.model_key == MODEL_KEY and job.subset in SUBSETS
    ]
    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(f"Qwen QA/MR job-count drift: expected={EXPECTED_JOBS} actual={len(jobs)}")
    records = [validate_job(protocol, artifact_root, job) for job in jobs]
    print(f"[qwen-site-freeze] validated jobs={len(records)} subsets={','.join(SUBSETS)}")
    if args.dry_run:
        return

    out_dir = args.out_dir.resolve()
    staging = out_dir.with_name(f"{out_dir.name}.staging")
    if out_dir.exists() or staging.exists():
        raise FileExistsError(f"immutable destination exists: {out_dir} or {staging}")
    staging.mkdir(parents=True)

    try:
        reproduction = staging / "reproduction"
        for source in (args.config.resolve(), args.execution_amendment.resolve()):
            copy_path(source, reproduction / "configs" / source.name)
        for source in (PROTOCOL_DOCUMENT, AMENDMENT_DOCUMENT):
            copy_path(source, reproduction / "docs" / source.name)
        for source in CODE_SNAPSHOT_PATHS:
            copy_path(source, reproduction / "code" / source.relative_to(REPO_ROOT))

        run_protocol = artifact_root / "protocol"
        copy_path(run_protocol, reproduction / "run_protocol")

        for subset in SUBSETS:
            copy_path(
                artifact_root / "confiqa" / subset,
                staging / "artifacts" / "confiqa" / subset,
            )
            copy_path(
                artifact_root / "shared_selector" / "confiqa" / subset / MODEL_KEY,
                staging / "artifacts" / "shared_selector" / "confiqa" / subset / MODEL_KEY,
            )
        for job in jobs:
            copy_path(artifact_root / job.job_id, staging / "artifacts" / job.job_id)

        summaries = staging / "summaries"
        write_csv(summaries / "job_status.csv", records)
        write_csv(summaries / "qwen_qa_mr_results.csv", collect_result_rows(artifact_root, jobs))
        write_csv(summaries / "selected_heads.csv", collect_selected_heads(artifact_root, jobs))
        (reproduction / "environment_at_freeze.json").write_text(
            json.dumps(environment_snapshot(), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        commit, dirty = repository_state()
        manifest: dict[str, Any] = {
            "freeze_id": out_dir.name,
            "created_at": utc_now(),
            "status": "immutable_completed_subset_snapshot",
            "scope": {
                "dataset": "confiqa",
                "subsets": list(SUBSETS),
                "model_key": MODEL_KEY,
                "completed_jobs": EXPECTED_JOBS,
                "excluded_incomplete_subset": "mc",
            },
            "protocol_id": protocol.data["protocol_id"],
            "protocol_version": protocol.data["version"],
            "config_sha256": protocol.config_sha256,
            "execution_amendment": args.execution_amendment.name,
            "execution_amendment_sha256": sha256_file(args.execution_amendment.resolve()),
            "scientific_settings_changed_by_execution_amendment": False,
            "artifact_source": str(artifact_root),
            "completed_job_ids": [job.job_id for job in jobs],
            "repository_commit": commit,
            "dirty_tree_manifest_at_freeze": dirty,
            "files": [],
        }
        readme = [
            "# Qwen2.5-14B Site V12 Completed QA/MR Freeze",
            "",
            f"Created: {manifest['created_at']}",
            "",
            "This immutable snapshot contains the 12 completed Qwen2.5-14B-Instruct Site V12 jobs",
            "for ConFiQA QA and MR: RCM-zero, RCM-patch, ITI, and three registered Random seeds.",
            "The still-running MC subset is intentionally excluded and is not represented as evidence here.",
            "",
            "`artifacts/` preserves shared admitted samples, selector caches, selected heads, actuator",
            "payloads, alpha-dev generations, selected alphas, held-out generations, and test summaries.",
            "`reproduction/` preserves the canonical protocol, exact-memory execution amendment, run-local",
            "launch files, environment inventory, and the exact code snapshot used by the completed jobs.",
            "",
            "The exact-memory amendment changes execution storage only. It does not quantize the model,",
            "truncate samples, change batches, alter precision, or modify any scientific hyperparameter.",
            "Every preserved file is sealed by SHA-256 in `freeze_manifest.json`.",
            "",
        ]
        (staging / "README.md").write_text("\n".join(readme), encoding="utf-8")
        manifest["files"] = inventory_files(staging)
        manifest["file_count_excluding_manifest"] = len(manifest["files"])
        (staging / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(out_dir)
    except Exception:
        print(f"[qwen-site-freeze] failed; staging retained: {staging}", file=sys.stderr)
        raise

    print(
        f"[qwen-site-freeze] complete out={out_dir} "
        f"files={manifest['file_count_excluding_manifest'] + 1}"
    )


if __name__ == "__main__":
    main()
