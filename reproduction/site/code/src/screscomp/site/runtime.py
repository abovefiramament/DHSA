from __future__ import annotations

import os
import subprocess
from pathlib import Path


def resolve_pinned_model(model: str, revision: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        resolved = path.resolve()
        if "snapshots" in resolved.parts and revision not in resolved.parts:
            raise RuntimeError(
                f"local model snapshot does not match locked revision {revision}: {resolved}"
            )
        return str(resolved)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pinned remote models require huggingface_hub") from exc
    return snapshot_download(repo_id=model, revision=revision)


def cuda_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Site GPU stages require an explicit CUDA device, got {device!r}")


def configure_gpu(device: str, execution: dict[str, object]) -> dict[str, object]:
    index = cuda_index(device)
    allowed = {int(value) for value in execution["preferred_physical_gpu_indices"]}
    if index not in allowed:
        raise RuntimeError(f"Site GPU {index} is outside the locked physical set {sorted(allowed)}")
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    parsed = [tuple(part.strip() for part in row.split(",", 5)) for row in gpu_rows if row.strip()]
    matches = [row for row in parsed if int(row[0]) == index]
    if len(matches) != 1:
        raise RuntimeError(f"could not resolve CUDA device {device} in nvidia-smi")
    gpu_index, gpu_uuid, gpu_name, memory_used, memory_total, utilization = matches[0]
    process_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    occupied = [
        row for row in process_output.splitlines()
        if row.strip() and row.split(",", 1)[0].strip() == gpu_uuid
    ]
    memory_used_mib = int(memory_used)
    utilization_percent = int(utilization)
    co_tenancy_observed = bool(occupied) or memory_used_mib > 512 or utilization_percent > 5
    if bool(execution["empty_gpu_required"]) and co_tenancy_observed:
        raise RuntimeError(f"locked Site runner refuses occupied {device} ({gpu_name})")
    if not bool(execution["gpu_co_tenancy_allowed"]) and co_tenancy_observed:
        raise RuntimeError(f"locked Site runner forbids GPU co-tenancy on {device} ({gpu_name})")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    return {
        "requested_device": device,
        "physical_index": gpu_index,
        "uuid": gpu_uuid,
        "name": gpu_name,
        "memory_used_mib": memory_used_mib,
        "memory_total_mib": int(memory_total),
        "utilization_percent": utilization_percent,
        "visible_compute_processes": occupied,
        "co_tenancy_observed": co_tenancy_observed,
        "co_tenancy_allowed": bool(execution["gpu_co_tenancy_allowed"]),
    }
