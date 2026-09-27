"""Official grouped two-fold ITI probe orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..loaders import load_model, load_rows, resolve_model_id
from ..position_scanning import PositionScanResult


@dataclass(frozen=True)
class ITIScanRequest:
    data_role_manifest: Path
    model_config: Mapping[str, Any]
    scan_config: Mapping[str, Any]
    trajectory_config: Mapping[str, Any]
    evidence_root: Path
    output_dir: Path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


class GroupedTwoFoldITIScanner:
    """Capture paired states and call the registered official probe trainer."""

    def __init__(
        self,
        *,
        model_provider: Any,
        capture_pair: Callable[
            [Any, Mapping[str, Any], Mapping[str, Any]],
            Any,
        ],
        train_probes: Callable[..., Any],
    ) -> None:
        self.model_provider = model_provider
        self.capture_pair = capture_pair
        self.train_probes = train_probes

    def scan(self, request: ITIScanRequest) -> PositionScanResult:
        import numpy as np

        random_state = request.scan_config.get("random_state")
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise ValueError("ITI scan requires an integer random_state")
        model = load_model(self.model_provider, request.model_config, mode="inference")
        rows = load_rows(request.data_role_manifest, root=request.evidence_root)
        if len(rows) < 2:
            raise ValueError("ITI grouped two-fold scan requires at least two groups")
        separated_activations: list[Any] = []
        separated_labels: list[Any] = []
        trajectory_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            raw_id = row.get("sample_id", row.get("id", index))
            group_id = raw_id if isinstance(raw_id, str) and raw_id else str(raw_id)
            captured = self.capture_pair(model, row, request.scan_config)
            if isinstance(captured, Mapping):
                positive = captured.get("positive_activation")
                negative = captured.get("negative_activation")
                positive_text = captured.get("positive_text")
                negative_text = captured.get("negative_text")
            elif isinstance(captured, Sequence) and not isinstance(captured, (str, bytes)) and len(captured) == 2:
                positive, negative = captured
                positive_text = negative_text = None
            else:
                raise ValueError("ITI capture_pair must return two activations or a named object")
            positive_array = np.asarray(positive)
            negative_array = np.asarray(negative)
            if positive_array.shape != negative_array.shape or positive_array.ndim != 3:
                raise ValueError("ITI activations must share [layers, heads, head_dim] shape")
            separated_activations.append(
                np.stack([positive_array, negative_array], axis=0)
            )
            separated_labels.append(np.asarray([1, 0], dtype=np.int64))
            if request.trajectory_config.get("enabled", True) and isinstance(positive_text, str):
                trajectory_rows.append(
                    {
                        "sample_id": f"{group_id}::target",
                        "generated_text": positive_text,
                    }
                )
            if request.trajectory_config.get("enabled", True) and isinstance(negative_text, str):
                trajectory_rows.append(
                    {
                        "sample_id": f"{group_id}::reference",
                        "generated_text": negative_text,
                    }
                )
        num_layers, num_heads, _head_dim = separated_activations[0].shape[1:]
        for activation in separated_activations[1:]:
            if activation.shape[1:] != separated_activations[0].shape[1:]:
                raise ValueError("ITI activation geometry changed across groups")
        first_fold, second_fold = np.array_split(np.arange(len(rows)), 2)
        if len(first_fold) == 0 or len(second_fold) == 0:
            raise ValueError("ITI grouped two-fold split is empty")
        _probes_ab, accuracy_ab = self.train_probes(
            random_state,
            first_fold,
            second_fold,
            separated_activations,
            separated_labels,
            num_layers,
            num_heads,
        )
        _probes_ba, accuracy_ba = self.train_probes(
            random_state,
            second_fold,
            first_fold,
            separated_activations,
            separated_labels,
            num_layers,
            num_heads,
        )
        accuracy_ab = np.asarray(accuracy_ab).reshape(-1)
        accuracy_ba = np.asarray(accuracy_ba).reshape(-1)
        if len(accuracy_ab) != num_layers * num_heads or len(accuracy_ba) != num_layers * num_heads:
            raise ValueError("official ITI probe trainer returned the wrong head coverage")
        candidate_rows: list[dict[str, Any]] = []
        component_rows: list[dict[str, Any]] = []
        for layer in range(num_layers):
            for head in range(num_heads):
                flat = layer * num_heads + head
                component_id = f"L{layer}.attn.h{head}"
                score_ab = float(accuracy_ab[flat])
                score_ba = float(accuracy_ba[flat])
                mean_score = (score_ab + score_ba) / 2.0
                candidate_rows.append(
                    {
                        "component_id": component_id,
                        "layer_idx": layer,
                        "head_idx": head,
                        "fold_A_to_B_accuracy": score_ab,
                        "fold_B_to_A_accuracy": score_ba,
                        "mean_grouped_twofold_heldout_accuracy": mean_score,
                        "ranking_score": mean_score,
                        "selection_score": mean_score,
                    }
                )
                component_rows.extend(
                    [
                        {"sample_id": "fold_A_to_B", "component_id": component_id, "score": score_ab},
                        {"sample_id": "fold_B_to_A", "component_id": component_id, "score": score_ba},
                    ]
                )
        candidate_rows.sort(
            key=lambda row: (-float(row["ranking_score"]), int(row["layer_idx"]), int(row["head_idx"]))
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        candidate_scores = request.output_dir / "candidate_scores.jsonl"
        component_scores = request.output_dir / "component_scores.jsonl"
        trajectory = request.output_dir / "trajectory.jsonl"
        _write_jsonl(candidate_scores, candidate_rows)
        _write_jsonl(component_scores, component_rows)
        trajectory_path: Path | None = None
        if trajectory_rows:
            trajectory_path = trajectory
            _write_jsonl(trajectory, trajectory_rows)
        execution = request.output_dir / "scan_execution.json"
        execution.write_text(
            json.dumps(
                {
                    "method": "iti",
                    "model_id": resolve_model_id(request.model_config, mode="inference"),
                    "groups": len(rows),
                    "states": len(rows) * 2,
                    "fold_A_group_indices": first_fold.tolist(),
                    "fold_B_group_indices": second_fold.tolist(),
                    "layers": num_layers,
                    "heads_per_layer": num_heads,
                    "random_state": random_state,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return PositionScanResult(
            candidate_scores=candidate_scores,
            execution_manifest=execution,
            trajectory_path=trajectory_path,
            component_scores_path=component_scores,
        )
