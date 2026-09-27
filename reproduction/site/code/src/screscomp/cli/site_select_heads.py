from __future__ import annotations

import argparse
from pathlib import Path

from screscomp.data import dump_json
from screscomp.modeling import TransformersABBackend
from screscomp.site.baselines import run_iti, run_random
from screscomp.site.model import PreOInterface
from screscomp.site.protocol import SiteProtocol
from screscomp.site.rcm import run_confiqa_rcm, run_imdb_rcm
from screscomp.site.runtime import configure_gpu, resolve_pinned_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one locked Site position-selector job.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", required=True, help="Physical CUDA device, for example cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    job = protocol.job(args.job_id)
    paths = protocol.paths(args.artifact_root, args.job_id)
    gpu = configure_gpu(args.device, protocol.site["execution"])
    paths.selector_dir.mkdir(parents=True, exist_ok=True)
    dump_json(paths.selector_dir / "preflight_manifest.json", {**protocol.snapshot_manifest(job), "gpu": gpu})
    pinned_model = resolve_pinned_model(job.model, job.model_revision)
    backend = TransformersABBackend(
        model_name_or_path=pinned_model,
        tokenizer_name_or_path=pinned_model,
        device="cuda:0",
        use_chat_template=job.use_chat_template,
        torch_dtype="auto",
    )
    interface = PreOInterface(backend)
    if job.selector == "random":
        run_random(protocol, job, paths, interface)
    elif job.selector == "iti":
        run_iti(protocol, job, paths, interface)
    elif job.dataset == "confiqa":
        run_confiqa_rcm(protocol, job, paths, interface)
    else:
        run_imdb_rcm(protocol, job, paths, interface)
    protocol.run_canonical_validator()
    print(f"[site-selector] complete job={job.job_id} output={paths.selected_heads}", flush=True)


if __name__ == "__main__":
    main()
