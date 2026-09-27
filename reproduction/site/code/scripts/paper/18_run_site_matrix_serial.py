#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

from screscomp.site.protocol import SiteProtocol


REPO_ROOT = Path(__file__).resolve().parents[2]
JOB_RUNNER = REPO_ROOT / "scripts" / "paper" / "17_run_site_job.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the emitted Site plan serially on one empty GPU.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stages", default="prepare,imdb_pairs,selector,train,dev,test")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    locked_root = Path(protocol.site["execution"]["run_root"]).resolve()
    if args.artifact_root.resolve() != locked_root:
        raise RuntimeError(
            f"artifact root drift: locked={locked_root} requested={args.artifact_root.resolve()}"
        )
    with args.plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty Site job plan: {args.plan}")
    expected_ids = [job.job_id for job in protocol.iter_jobs()]
    actual_ids = [row["job_id"] for row in rows]
    if actual_ids != expected_ids:
        raise RuntimeError("Site plan job order/content disagrees with the locked config")
    hash_path = args.plan.parent / "config.sha256"
    snapshot = args.plan.parent / "config.snapshot.json"
    if not hash_path.is_file() or not snapshot.is_file():
        raise RuntimeError("Site plan lacks its config hash or snapshot")
    registered_hash = hash_path.read_text(encoding="ascii").strip()
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if registered_hash != protocol.config_sha256 or snapshot_hash != protocol.config_sha256:
        raise RuntimeError("Site plan config snapshot/hash drift")
    for row, job in zip(rows, protocol.iter_jobs(), strict=True):
        if Path(row["output_dir"]).resolve() != job.output_dir.resolve():
            raise RuntimeError(f"Site plan output drift for {job.job_id}")
    if args.validate_only:
        print(
            f"[site-matrix] validated-only jobs={len(rows)} config_sha256={protocol.config_sha256}",
            flush=True,
        )
        return
    for index, row in enumerate(rows, start=1):
        job_id = row["job_id"]
        print(f"[site-matrix] job={index}/{len(rows)} id={job_id}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(JOB_RUNNER),
                "--config", str(args.config),
                "--job-id", job_id,
                "--artifact-root", str(args.artifact_root),
                "--device", args.device,
                "--stages", args.stages,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    print(f"[site-matrix] complete jobs={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
