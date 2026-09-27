#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screscomp.site.protocol import SiteJob, SiteProtocol, sha256_file  # noqa: E402


PROTOCOL_DOCUMENT = REPO_ROOT / "docs" / "SITE_POSITION_FULL_FIXED_PROTOCOL_20260719_V12.md"
PLAN_DIRECTORY = REPO_ROOT / "runs" / "site_position_full_plan_20260719_v12"
CODE_SNAPSHOT_PATHS = (
    REPO_ROOT / "src" / "screscomp",
    REPO_ROOT / "scripts" / "paper",
    REPO_ROOT / "pyproject.toml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze completed Site V12 results and exact reproduction artifacts without recomputation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default="confiqa,imdb",
        help="Comma-separated reporting scopes. This does not change scientific execution.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


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


def required_job_files(job: SiteJob) -> tuple[str, ...]:
    common = (
        "selector/selector_manifest.json",
        "selector/selected_heads.csv",
        "actuator/training_manifest.json",
        "evaluation/test_manifest.json",
        "evaluation/test/generation_rows.jsonl",
    )
    if job.dataset == "confiqa":
        return common + (
            "evaluation/test/site_summary.csv",
            "evaluation/test/generation_summary.csv",
        )
    return common + (
        "evaluation/test/generation_manifest.json",
        "evaluation/test/sentiment_scored.jsonl",
        "evaluation/test/sentiment_summary.csv",
        "evaluation/test/kl_rows.jsonl",
        "evaluation/test/kl_summary.csv",
        "evaluation/test/kl_manifest.json",
    )


def inspect_job(protocol: SiteProtocol, artifact_root: Path, job: SiteJob) -> tuple[bool, list[str]]:
    job_dir = artifact_root / job.job_id
    missing = [relative for relative in required_job_files(job) if not (job_dir / relative).is_file()]
    if missing:
        return False, [f"missing:{relative}" for relative in missing]

    manifest = read_json(job_dir / "evaluation" / "test_manifest.json")
    reasons: list[str] = []
    if manifest.get("protocol_id") != protocol.data["protocol_id"]:
        reasons.append("test_manifest_protocol_id_drift")
    if int(manifest.get("protocol_version", -1)) != int(protocol.data["version"]):
        reasons.append("test_manifest_protocol_version_drift")
    if manifest.get("config_sha256") != protocol.config_sha256:
        reasons.append("test_manifest_config_hash_drift")
    if manifest.get("job_id") != job.job_id:
        reasons.append("test_manifest_job_id_drift")

    if job.dataset == "imdb":
        downstream = job.task["downstream"]
        expected_n = int(downstream["test_prompts"]) * int(downstream["samples_per_prompt"])
        alpha_count = len(downstream["alpha_nonzero_grid"])
        expected_rows = expected_n * alpha_count
        test_dir = job_dir / "evaluation" / "test"
        for name in ("generation_rows.jsonl", "sentiment_scored.jsonl", "kl_rows.jsonl"):
            actual = count_jsonl(test_dir / name)
            if actual != expected_rows:
                reasons.append(f"{name}_rows={actual}_expected={expected_rows}")
        sentiment_rows = read_csv(test_dir / "sentiment_summary.csv")
        kl_rows = read_csv(test_dir / "kl_summary.csv")
        if len(sentiment_rows) != alpha_count:
            reasons.append(f"sentiment_summary_rows={len(sentiment_rows)}_expected={alpha_count}")
        if len(kl_rows) != alpha_count:
            reasons.append(f"kl_summary_rows={len(kl_rows)}_expected={alpha_count}")
        for row in sentiment_rows:
            if int(row["n"]) != expected_n:
                reasons.append(f"sentiment_summary_n={row['n']}_expected={expected_n}")
                break
        for row in kl_rows:
            if int(row["n"]) != expected_n:
                reasons.append(f"kl_summary_n={row['n']}_expected={expected_n}")
                break

    return not reasons, reasons


def job_record(job: SiteJob, complete: bool, reasons: list[str]) -> dict[str, Any]:
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
        "status": "complete" if complete else "incomplete",
        "detail": ";".join(reasons),
    }


def collect_confiqa_rows(artifact_root: Path, jobs: Iterable[SiteJob]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for job in jobs:
        summary = artifact_root / job.job_id / "evaluation" / "test" / "site_summary.csv"
        for row in read_csv(summary):
            output.append({
                "job_id": job.job_id,
                "dataset": job.dataset,
                "subset": job.subset,
                "model_key": job.model_key,
                "model_revision": job.model_revision,
                "selector": job.selector,
                "replicate": job.replicate,
                "seed": job.seed,
                **row,
            })
    return output


def collect_imdb_rows(
    artifact_root: Path,
    jobs: Iterable[SiteJob],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    token_audit: list[dict[str, Any]] = []
    for job in jobs:
        test_dir = artifact_root / job.job_id / "evaluation" / "test"
        sentiment = {row["alpha"]: row for row in read_csv(test_dir / "sentiment_summary.csv")}
        kl = {row["alpha"]: row for row in read_csv(test_dir / "kl_summary.csv")}
        if set(sentiment) != set(kl):
            raise ValueError(f"IMDb sentiment/KL alpha mismatch: {job.job_id}")

        token_counts: dict[str, int] = defaultdict(int)
        token_rows: dict[str, int] = defaultdict(int)
        with (test_dir / "kl_rows.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                alpha = str(float(row["alpha"]))
                token_counts[alpha] += int(row["completion_token_count"])
                token_rows[alpha] += 1

        for alpha in sorted(sentiment, key=float):
            sentiment_row = sentiment[alpha]
            kl_row = kl[alpha]
            base = {
                "job_id": job.job_id,
                "dataset": job.dataset,
                "subset": job.subset,
                "model_key": job.model_key,
                "model_revision": job.model_revision,
                "selector": job.selector,
                "replicate": job.replicate,
                "seed": job.seed,
                "alpha": alpha,
            }
            output.append({
                **base,
                **{f"sentiment_{key}": value for key, value in sentiment_row.items() if key != "alpha"},
                **{f"kl_{key}": value for key, value in kl_row.items() if key not in {"alpha", "n"}},
            })
            normalized_alpha = str(float(alpha))
            token_audit.append({
                **base,
                "n": sentiment_row["n"],
                "reported_actual_generated_tokens": sentiment_row["actual_generated_tokens"],
                "audited_actual_generated_tokens": token_counts[normalized_alpha],
                "audited_kl_rows": token_rows[normalized_alpha],
                "source": "sum(completion_token_count) over frozen kl_rows.jsonl",
                "changes_scientific_result": False,
            })
    return output, token_audit


def inventory_files(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "freeze_manifest.json":
            continue
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def repository_state() -> tuple[str, list[str]]:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--short"],
        text=True,
    ).splitlines()
    return commit, dirty


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    artifact_root = args.artifact_root.resolve()
    locked_root = Path(protocol.site["execution"]["run_root"]).resolve()
    if artifact_root != locked_root:
        raise ValueError(f"artifact root drift: locked={locked_root} requested={artifact_root}")

    datasets = tuple(value.strip() for value in args.datasets.split(",") if value.strip())
    unknown = set(datasets) - {"confiqa", "imdb"}
    if unknown:
        raise ValueError(f"unsupported reporting dataset(s): {sorted(unknown)}")

    records: list[dict[str, Any]] = []
    completed: list[SiteJob] = []
    for job in protocol.iter_jobs():
        if job.dataset not in datasets:
            continue
        is_complete, reasons = inspect_job(protocol, artifact_root, job)
        records.append(job_record(job, is_complete, reasons))
        if is_complete:
            completed.append(job)

    print(
        f"[site-freeze] expected={len(records)} complete={len(completed)} "
        f"incomplete={len(records) - len(completed)} datasets={','.join(datasets)}"
    )
    for row in records:
        if row["status"] != "complete":
            print(f"[site-freeze] incomplete job={row['job_id']} detail={row['detail']}")
    if args.dry_run:
        return

    out_dir = args.out_dir.resolve()
    staging = out_dir.with_name(f"{out_dir.name}.staging")
    if out_dir.exists() or staging.exists():
        raise FileExistsError(f"immutable freeze destination already exists: {out_dir} or {staging}")
    staging.mkdir(parents=True)

    try:
        reproduction = staging / "reproduction"
        copy_path(args.config.resolve(), reproduction / "configs" / args.config.name)
        copy_path(PROTOCOL_DOCUMENT, reproduction / "docs" / PROTOCOL_DOCUMENT.name)
        copy_path(PLAN_DIRECTORY, reproduction / "runs" / PLAN_DIRECTORY.name)
        for source in CODE_SNAPSHOT_PATHS:
            copy_path(source, reproduction / "code" / source.relative_to(REPO_ROOT))

        copied_shared: set[Path] = set()
        for job in completed:
            source_job = artifact_root / job.job_id
            copy_path(source_job, staging / "artifacts" / job.job_id)
            shared = protocol.paths(artifact_root, job.job_id).shared_dataset_dir
            if shared not in copied_shared:
                copy_path(shared, staging / "artifacts" / shared.relative_to(artifact_root))
                copied_shared.add(shared)
        for shared_name in ("shared_base", "shared_selector"):
            source = artifact_root / shared_name
            if source.exists():
                copy_path(source, staging / "artifacts" / shared_name)

        summaries = staging / "summaries"
        write_csv(summaries / "job_status.csv", records)
        confiqa_jobs = [job for job in completed if job.dataset == "confiqa"]
        imdb_jobs = [job for job in completed if job.dataset == "imdb"]
        write_csv(summaries / "confiqa_current_results.csv", collect_confiqa_rows(artifact_root, confiqa_jobs))
        imdb_rows, token_audit = collect_imdb_rows(artifact_root, imdb_jobs)
        write_csv(summaries / "imdb_current_results.csv", imdb_rows)
        write_csv(summaries / "imdb_actual_token_audit.csv", token_audit)

        commit, dirty = repository_state()
        file_inventory = inventory_files(staging)
        manifest = {
            "freeze_id": out_dir.name,
            "created_at": utc_now(),
            "status": "immutable_current_result_snapshot",
            "protocol_id": protocol.data["protocol_id"],
            "protocol_version": protocol.data["version"],
            "config_source": str(args.config.resolve()),
            "config_sha256": protocol.config_sha256,
            "artifact_source": str(artifact_root),
            "datasets": datasets,
            "expected_jobs_in_scope": len(records),
            "completed_jobs": len(completed),
            "incomplete_jobs": len(records) - len(completed),
            "completed_job_ids": [job.job_id for job in completed],
            "repository_commit": commit,
            "dirty_tree_manifest_at_freeze": dirty,
            "known_issues": [
                {
                    "scope": "IMDb sentiment_summary.csv",
                    "field": "actual_generated_tokens",
                    "observed": "reported as zero because generation rows did not emit that field",
                    "scientific_effect": "none on generation, sentiment reward, positive rate, or KL",
                    "repair": "summaries/imdb_actual_token_audit.csv sums completion_token_count from frozen KL rows",
                }
            ],
            "file_count_excluding_manifest": len(file_inventory),
            "files": file_inventory,
        }
        (staging / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        readme = [
            "# Site V12 Current Result Freeze",
            "",
            f"Created: {manifest['created_at']}",
            "",
            f"Protocol: `{manifest['protocol_id']}` version {manifest['protocol_version']}",
            "",
            f"Completed jobs in ConFiQA/IMDb scope: {len(completed)} / {len(records)}.",
            "",
            "This directory preserves only jobs whose V12 test manifests and required raw outputs passed",
            "the freeze audit. Incomplete model/task rows remain explicit in `summaries/job_status.csv`.",
            "",
            "Reproduction inputs, canonical protocol/config, launch plan, and an exact source snapshot are",
            "under `reproduction/`. Raw prepared inputs, selectors, actuator payloads, generations, scores,",
            "KL rows, and test manifests are under `artifacts/`.",
            "",
            "IMDb `actual_generated_tokens=0` is a bookkeeping defect in the original sentiment summary.",
            "The raw outputs are unchanged; `summaries/imdb_actual_token_audit.csv` records the corrected",
            "token totals derived from the frozen KL rows.",
            "",
            "Every preserved file is sealed by SHA-256 in `freeze_manifest.json`.",
            "",
        ]
        (staging / "README.md").write_text("\n".join(readme), encoding="utf-8")
        file_inventory = inventory_files(staging)
        manifest["file_count_excluding_manifest"] = len(file_inventory)
        manifest["files"] = file_inventory
        (staging / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(out_dir)
    except Exception:
        print(f"[site-freeze] failed; staging preserved for audit: {staging}", file=sys.stderr)
        raise

    print(f"[site-freeze] complete out={out_dir} files={len(file_inventory) + 2}")


if __name__ == "__main__":
    main()
