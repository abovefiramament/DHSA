#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from screscomp.site.protocol import SiteProtocol


STAGES = ("prepare", "imdb_pairs", "selector", "train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered stages for one locked Site job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    return parser.parse_args()


def _run(module: str, common: list[str], extra: list[str] | None = None) -> None:
    subprocess.run([sys.executable, "-m", module, *common, *(extra or [])], check=True)


def _registered(
    manifest: Path,
    config_sha256: str,
    *,
    required: tuple[Path, ...] = (),
) -> bool:
    if not manifest.is_file():
        partial = [str(path) for path in required if path.is_file()]
        if partial:
            raise RuntimeError(
                f"unregistered partial Site artifact(s) beside missing {manifest}: {partial}"
            )
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != config_sha256:
        raise RuntimeError(
            f"Site artifact config drift in {manifest}: "
            f"registered={payload.get('config_sha256')} locked={config_sha256}"
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"registered Site stage is incomplete: {missing}")
    return True


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    job = protocol.job(args.job_id)
    paths = protocol.paths(args.artifact_root, args.job_id)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown Site stages: {unknown}")
    common = [
        "--config", str(protocol.config_path),
        "--job-id", job.job_id,
        "--artifact-root", str(paths.root),
    ]
    for stage in stages:
        if stage == "prepare":
            if not _registered(
                paths.shared_dataset_dir / "input_manifest.json",
                protocol.config_sha256,
            ):
                _run("screscomp.cli.site_prepare_inputs", common)
        elif stage == "imdb_pairs":
            if job.dataset == "imdb" and not _registered(
                paths.pair_dir / "site_pair_manifest.json",
                protocol.config_sha256,
                required=(paths.pairs_csv,),
            ):
                _run("screscomp.cli.site_prepare_imdb_pairs", common, ["--device", args.device])
        elif stage == "selector":
            if not _registered(
                paths.selector_dir / "selector_manifest.json",
                protocol.config_sha256,
                required=(paths.selected_heads,),
            ):
                _run("screscomp.cli.site_select_heads", common, ["--device", args.device])
        elif stage == "train":
            if not _registered(
                paths.actuator_dir / "training_manifest.json",
                protocol.config_sha256,
                required=(paths.actuator_dir / "head_actuator.pt",),
            ):
                _run("screscomp.cli.site_train_actuator", common, ["--device", args.device])
        elif stage == "dev":
            if job.dataset == "confiqa" and not _registered(
                paths.evaluation_dir / "dev_manifest.json",
                protocol.config_sha256,
            ):
                _run("screscomp.cli.site_evaluate", common, ["--stage", "dev", "--device", args.device])
        elif stage == "test":
            if not _registered(
                paths.evaluation_dir / "test_manifest.json",
                protocol.config_sha256,
            ):
                _run("screscomp.cli.site_evaluate", common, ["--stage", "test", "--device", args.device])
    print(f"[site-job] complete job={job.job_id} stages={','.join(stages)}", flush=True)


if __name__ == "__main__":
    main()
