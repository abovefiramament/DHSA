"""Read-only numerical execution provenance for RCM and CAST diagnostics."""
from __future__ import annotations
import importlib.metadata
import os
import platform
import subprocess
import sys
from collections import Counter
from typing import Any

def capture_numerical_environment(model=None) -> dict[str, Any]:
    import torch
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    result = {
        "record_kind": "observed_current_process_not_historical_attestation",
        "python": platform.python_version(), "python_build": sys.version,
        "platform": platform.platform(), "packages": packages,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "sdpa_allowed": {n: getattr(torch.backends.cuda, n+"_sdp_enabled")()
                         for n in ["flash", "math", "mem_efficient", "cudnn"]
                         if hasattr(torch.backends.cuda, n+"_sdp_enabled")},
        "sdpa_actual_kernel": "not_profiled; allowed flags are not selected-kernel evidence",
        "environment": {k: os.environ.get(k) for k in [
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED"]},
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result["devices"].append({"logical_index": i, "name": p.name,
                "capability": [p.major, p.minor], "memory_bytes": p.total_memory})
    try:
        result["nvidia_smi"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi_unavailable"] = type(exc).__name__
    if model is not None:
        result["parameter_dtypes"] = dict(Counter(str(p.dtype) for p in model.parameters()))
        result["buffer_dtypes"] = {n: str(b.dtype) for n,b in model.named_buffers()}
        result["attention_classes"] = sorted({type(m).__name__ for m in model.modules()
                                             if "Attention" in type(m).__name__})
        result["attention_implementation"] = getattr(model.config, "_attn_implementation", None)
    return result
