"""Callback-based generation backend for the shared experiment flow.

The backend owns row iteration and evidence files.  Model loading and text
generation are injected, so no checkpoint path or machine-specific registry is
embedded in the framework.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments.shared.contracts import GenerationRequest, GenerationResult
from .loaders import load_model, load_rows, resolve_model_id


def _rows(path: Path, root: Path) -> list[dict[str, Any]]:
    return load_rows(path, root=root)


def _model_id(config: Mapping[str, Any]) -> str:
    return resolve_model_id(config, mode="inference")


def _prompt(row: Mapping[str, Any]) -> Any:
    value = row.get("prompt", row.get("question"))
    if isinstance(value, str) and value:
        return value
    base_state = row.get("base_state")
    if isinstance(base_state, Mapping):
        return base_state.get("prompt")
    return value


def _load_generation_checkpoint(path: Path) -> list[dict[str, Any]]:
    """Load the durable generation prefix, discarding only a torn final write."""

    if not path.is_file():
        return []
    raw = path.read_bytes()
    complete_length = raw.rfind(b"\n") + 1
    if complete_length != len(raw):
        with path.open("r+b") as handle:
            handle.truncate(complete_length)
        raw = raw[:complete_length]
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"generation checkpoint contains an invalid completed row {index}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"generation checkpoint row {index} must be an object")
        records.append(record)
    return records


def _append_generation_checkpoint(path: Path, record: Mapping[str, Any]) -> None:
    """Durably publish one completed sample/alpha generation."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class CallbackGenerationBackend:
    """Run generation with externally supplied model and generation callbacks."""

    def __init__(
        self,
        *,
        model_provider: Any,
        generate: Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Any],
    ) -> None:
        self.model_provider = model_provider
        self._generate_callback = generate

    def generate_rows(self, request: GenerationRequest) -> GenerationResult:
        model_config = request.model_config
        if not isinstance(model_config, Mapping):
            raise ValueError("generation model_config must be an object")
        model = load_model(self.model_provider, model_config, mode="inference")
        rows = _rows(request.data_role_manifest, request.evidence_root)
        if not rows:
            raise ValueError("generation data role is empty")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        predictions = request.output_dir / "predictions.jsonl"
        trajectory = request.output_dir / "trajectory.jsonl"
        component_scores = request.output_dir / "component_scores.jsonl"
        checkpoint_path = request.output_dir / "generation_checkpoint.jsonl"
        checkpoint_records = _load_generation_checkpoint(checkpoint_path)
        checkpoint_index = 0
        trace_enabled = bool(request.trajectory_config.get("enabled", True))
        prediction_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        generation_config = dict(request.generation_config)
        if request.selected_alpha is not None and request.selected_alpha_grid is None:
            generation_config["alpha"] = request.selected_alpha
        alpha_grid = tuple(request.selected_alpha_grid or ())
        alpha_values: tuple[Any, ...] = alpha_grid if alpha_grid else (request.selected_alpha,)
        expanded_curve = bool(alpha_grid)
        for index, row in enumerate(rows):
            sample_id = row.get("sample_id", row.get("id", index))
            if not isinstance(sample_id, str):
                sample_id = str(sample_id)
            prompt = _prompt(row)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"generation row {index} needs prompt or question")
            for alpha in alpha_values:
                output_id = (
                    f"{sample_id}::alpha={alpha}"
                    if expanded_curve
                    else sample_id
                )
                call_config = dict(generation_config)
                if alpha is not None:
                    call_config["alpha"] = alpha
                # Runtime-only context for the injected callback. These are
                # already-resolved artifact paths, not scientific parameters.
                if request.controller_manifest is not None:
                    call_config["_controller_manifest_path"] = str(
                        request.controller_manifest.resolve()
                    )
                call_config["_data_role_manifest_path"] = str(request.data_role_manifest.resolve())
                call_config["_evidence_root"] = str(request.evidence_root.resolve())
                checkpoint_request = {
                    "alpha": alpha,
                    "generation_config": json.loads(
                        json.dumps(generation_config, ensure_ascii=False, sort_keys=True)
                    ),
                    "model_id": _model_id(model_config),
                    "output_id": output_id,
                    "prompt": prompt,
                    "trace_enabled": trace_enabled,
                }
                if generation_config.get("prediction_mode") == "paired_completion_scores":
                    checkpoint_request["preference_pair"] = {
                        role: row.get(role) for role in ("chosen", "rejected")
                    }
                if checkpoint_index < len(checkpoint_records):
                    checkpoint = checkpoint_records[checkpoint_index]
                    if checkpoint.get("request") != checkpoint_request:
                        raise ValueError(
                            "generation checkpoint differs from the current request at "
                            f"{output_id}"
                        )
                    prediction = checkpoint.get("prediction")
                    trajectory_row = checkpoint.get("trajectory")
                    current_score_rows = checkpoint.get("component_scores")
                    if not isinstance(prediction, dict) or not isinstance(
                        current_score_rows, list
                    ):
                        raise ValueError(
                            f"generation checkpoint payload is invalid for {output_id}"
                        )
                else:
                    generated = self._generate_callback(
                        model,
                        {"sample_id": sample_id, **row},
                        call_config,
                    )
                    if isinstance(generated, str):
                        text = generated
                        sample_scores: Any = []
                        extra: Mapping[str, Any] = {}
                        score_status: Any = None
                    elif isinstance(generated, Mapping):
                        text = generated.get("generated_text", generated.get("text"))
                        sample_scores = generated.get("component_scores", [])
                        extra = generated
                        score_status = generated.get("component_scores_status")
                    else:
                        raise ValueError("generation callback must return text or an object")
                    if not isinstance(text, str):
                        raise ValueError(
                            f"generation callback returned no text for {output_id}"
                        )
                    prediction = {
                        "sample_id": output_id,
                        "prompt": prompt,
                        "generated_text": text,
                        **(
                            {"source_sample_id": sample_id, "alpha": alpha}
                            if expanded_curve
                            else {}
                        ),
                        **{
                            key: value
                            for key, value in extra.items()
                            if key
                            not in {
                                "generated_text",
                                "text",
                                "component_scores",
                                "component_scores_status",
                            }
                        },
                    }
                    trajectory_row = None
                    current_score_rows: list[dict[str, Any]] = []
                    if trace_enabled:
                        if not isinstance(sample_scores, list):
                            raise ValueError(
                                f"component_scores must be a list for {output_id}"
                            )
                        normalized_scores: list[dict[str, Any]] = []
                        if not sample_scores:
                            if not (
                                isinstance(score_status, str)
                                and score_status.startswith("not_applicable_")
                            ):
                                raise ValueError(
                                    "trajectory is enabled but no component scores were returned "
                                    f"for {output_id}"
                                )
                        else:
                            if score_status not in (None, "available"):
                                raise ValueError(
                                    "non-empty component scores require status 'available'"
                                )
                            for score in sample_scores:
                                if not isinstance(score, Mapping):
                                    raise ValueError(
                                        "component_scores entries must be objects"
                                    )
                                component_id = score.get("component_id")
                                value = score.get("score")
                                if not isinstance(component_id, str) or not component_id:
                                    raise ValueError("component score needs component_id")
                                if isinstance(value, bool) or not isinstance(
                                    value, (int, float)
                                ):
                                    raise ValueError("component score needs numeric score")
                                normalized_score = {
                                    "component_id": component_id,
                                    "score": float(value),
                                }
                                normalized_scores.append(normalized_score)
                                current_score_rows.append(
                                    {
                                        "sample_id": output_id,
                                        "component_id": component_id,
                                        "score": float(value),
                                    }
                                )
                        trajectory_row = {
                            "sample_id": output_id,
                            "generated_text": text,
                            "component_scores": normalized_scores,
                            "component_scores_status": score_status or "available",
                        }
                    _append_generation_checkpoint(
                        checkpoint_path,
                        {
                            "component_scores": current_score_rows,
                            "prediction": prediction,
                            "request": checkpoint_request,
                            "trajectory": trajectory_row,
                        },
                    )
                prediction_rows.append(prediction)
                score_rows.extend(current_score_rows)
                if trace_enabled:
                    if not isinstance(trajectory_row, dict):
                        raise ValueError(
                            f"generation checkpoint trajectory is invalid for {output_id}"
                        )
                    trajectory_rows.append(trajectory_row)
                checkpoint_index += 1
        if len(checkpoint_records) > checkpoint_index:
            raise ValueError("generation checkpoint has surplus rows")
        with predictions.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        execution_manifest = request.output_dir / "backend_execution_manifest.json"
        execution_manifest.write_text(
            json.dumps(
                {
                    "model_id": _model_id(model_config),
                    "rows": len(prediction_rows),
                    "selected_alpha": request.selected_alpha,
                    "selected_alpha_grid": list(request.selected_alpha_grid or ()),
                    "trajectory_enabled": trace_enabled,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not trace_enabled:
            return GenerationResult(predictions=predictions, execution_manifest=execution_manifest)
        with trajectory.open("w", encoding="utf-8") as handle:
            for row in trajectory_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with component_scores.open("w", encoding="utf-8") as handle:
            for row in score_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return GenerationResult(
            predictions=predictions,
            execution_manifest=execution_manifest,
            trajectory_path=trajectory,
            component_scores_path=component_scores,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.generate_rows(request)
