"""Callback-based generation backend for the shared experiment flow.

The backend owns row iteration and evidence files.  Model loading and text
generation are injected, so no checkpoint path or machine-specific registry is
embedded in the framework.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments.shared.contracts import GenerationRequest, GenerationResult
from .loaders import load_model, load_rows, resolve_model_id


def _rows(path: Path, root: Path) -> list[dict[str, Any]]:
    return load_rows(path, root=root)


def _model_id(config: Mapping[str, Any]) -> str:
    return resolve_model_id(config, mode="inference")


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
            prompt = row.get("prompt", row.get("question"))
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"generation row {index} needs prompt or question")
            for alpha in alpha_values:
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
                output_id = (
                    f"{sample_id}::alpha={alpha}"
                    if expanded_curve
                    else sample_id
                )
                if not isinstance(text, str):
                    raise ValueError(f"generation callback returned no text for {output_id}")
                prediction = {
                    "sample_id": output_id,
                    "prompt": prompt,
                    "generated_text": text,
                    **({"source_sample_id": sample_id, "alpha": alpha} if expanded_curve else {}),
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
                prediction_rows.append(prediction)
                if trace_enabled:
                    if not isinstance(sample_scores, list):
                        raise ValueError(
                            f"component_scores must be a list for {output_id}"
                        )
                    normalized_scores: list[dict[str, Any]] = []
                    if not sample_scores:
                        if score_status != "not_applicable_no_active_controller":
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
                                raise ValueError("component_scores entries must be objects")
                            component_id = score.get("component_id")
                            value = score.get("score")
                            if not isinstance(component_id, str) or not component_id:
                                raise ValueError("component score needs component_id")
                            if isinstance(value, bool) or not isinstance(value, (int, float)):
                                raise ValueError("component score needs numeric score")
                            normalized_score = {
                                "component_id": component_id,
                                "score": float(value),
                            }
                            normalized_scores.append(normalized_score)
                            score_rows.append(
                                {
                                    "sample_id": output_id,
                                    "component_id": component_id,
                                    "score": float(value),
                                }
                            )
                    trajectory_rows.append(
                        {
                            "sample_id": output_id,
                            "generated_text": text,
                            "component_scores": normalized_scores,
                            "component_scores_status": (
                                score_status or "available"
                            ),
                        }
                    )
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
