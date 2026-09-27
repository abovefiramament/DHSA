"""Enumerate the model's attention-head geometry for Random selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..loaders import load_model, resolve_model_id
from ..position_scanning import PositionScanResult


@dataclass(frozen=True)
class RandomScanRequest:
    model_config: Mapping[str, Any]
    scan_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


class ModelGeometryScanner:
    """Return every valid attention head; sampling remains in Random selector."""

    def __init__(
        self,
        *,
        model_provider: Any,
        layer_count: Callable[[Any], int],
        head_count: Callable[[Any, int], int],
    ) -> None:
        self.model_provider = model_provider
        self.layer_count = layer_count
        self.head_count = head_count

    def scan(self, request: RandomScanRequest) -> PositionScanResult:
        model = load_model(self.model_provider, request.model_config, mode="inference")
        layers = int(self.layer_count(model))
        if layers <= 0:
            raise ValueError("Random geometry scanner reported no layers")
        candidates: list[dict[str, Any]] = []
        for layer in range(layers):
            heads = int(self.head_count(model, layer))
            if heads <= 0:
                raise ValueError(
                    f"Random geometry scanner reported no heads for layer {layer}"
                )
            candidates.extend(
                {
                    "component_id": f"L{layer}.attn.h{head}",
                    "layer_idx": layer,
                    "head_idx": head,
                }
                for head in range(heads)
            )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = request.output_dir / "candidate_scores.jsonl"
        with candidate_path.open("w", encoding="utf-8") as handle:
            for row in candidates:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        execution_path = request.output_dir / "scan_execution.json"
        execution_path.write_text(
            json.dumps(
                {
                    "method": "random",
                    "model_id": resolve_model_id(request.model_config, mode="inference"),
                    "layers": layers,
                    "candidate_heads": len(candidates),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return PositionScanResult(
            candidate_scores=candidate_path,
            execution_manifest=execution_path,
        )
