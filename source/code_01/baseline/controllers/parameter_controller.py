"""Resolve method parameters without introducing scientific defaults.

The experiment protocol supplies every value. This controller validates the
method/runtime relationship and emits a transportable plan for the mature
implementation entry.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .control_objective_controller import (
    ControlObjectiveError,
    resolve_control_objective,
)


METHODS = frozenset(
    {"direct_policy", "cast", "loreft", "bipo", "rcm_zero", "rcm_patch", "iti", "random", "caa"}
)
METHOD_KINDS = {
    "direct_policy": "policy_baseline",
    "cast": "intervention_baseline",
    "loreft": "native_external_baseline",
    "bipo": "native_external_baseline",
    "rcm_zero": "position_baseline",
    "rcm_patch": "position_baseline",
    "iti": "position_baseline",
    "random": "position_baseline",
    "caa": "position_baseline",
}
CAST_FAMILIES = frozenset({"sv", "reft"})
TRANSFER_MODES = frozenset({"none", "controller_transfer", "checkpoint_transfer"})


class ParameterContractError(ValueError):
    """Raised when a resolved method contract is incomplete or inconsistent."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ParameterContractError(f"{field} must be an object")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterContractError(f"{field} must be a non-empty string")
    return value


def _require(mapping: Mapping[str, Any], key: str, *, field: str) -> Any:
    if key not in mapping:
        raise ParameterContractError(f"{field}.{key} is required")
    return mapping[key]


def _artifact_ref(value: Any, *, field: str) -> dict[str, str]:
    item = _mapping(value, field=field)
    if "artifact" in item:
        if set(item) != {"artifact"}:
            raise ParameterContractError(
                f"{field} symbolic reference cannot include materialized fields"
            )
        artifact = _string(item["artifact"], field=f"{field}.artifact")
        if not artifact.startswith("artifact://"):
            raise ParameterContractError(f"{field}.artifact must use artifact://")
        return {"artifact": artifact}
    path = _string(_require(item, "path", field=field), field=f"{field}.path")
    return {"path": path}


def _timings(value: Any) -> list[dict[str, Any]]:
    raw = value if isinstance(value, list) else [value]
    if not raw:
        raise ParameterContractError("baseline.parameters.controller.timings cannot be empty")
    resolved: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if isinstance(item, str):
            timing = {"name": _string(item, field=f"timings[{index}]")}
        else:
            timing = dict(_mapping(item, field=f"timings[{index}]"))
            timing["name"] = _string(
                _require(timing, "name", field=f"timings[{index}]"),
                field=f"timings[{index}].name",
            )
        name = timing["name"]
        if name in names:
            raise ParameterContractError(f"duplicate intervention timing: {name}")
        names.add(name)
        resolved.append(timing)
    return resolved


def _resolve_models(parameters: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _mapping(_require(parameters, "model", field="baseline.parameters"), field="model")
    train = dict(_mapping(_require(model, "train", field="model"), field="model.train"))
    inference = dict(
        _mapping(_require(model, "inference", field="model"), field="model.inference")
    )
    for name, item in (("train", train), ("inference", inference)):
        _string(
            _require(item, "checkpoint", field=f"model.{name}"),
            field=f"model.{name}.checkpoint",
        )
        _string(
            _require(item, "revision", field=f"model.{name}"),
            field=f"model.{name}.revision",
        )
    return train, inference


def _resolve_inference_model(parameters: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(_require(parameters, "model", field="baseline.parameters"), field="model")
    inference = dict(
        _mapping(_require(model, "inference", field="model"), field="model.inference")
    )
    _string(
        _require(inference, "checkpoint", field="model.inference"),
        field="model.inference.checkpoint",
    )
    _string(
        _require(inference, "revision", field="model.inference"),
        field="model.inference.revision",
    )
    return inference


def _resolve_transfer(
    parameters: Mapping[str, Any],
    train_model: Mapping[str, Any],
    inference_model: Mapping[str, Any],
) -> dict[str, Any]:
    transfer = dict(
        _mapping(_require(parameters, "transfer", field="baseline.parameters"), field="transfer")
    )
    mode = _string(_require(transfer, "mode", field="transfer"), field="transfer.mode")
    if mode not in TRANSFER_MODES:
        raise ParameterContractError(f"unsupported transfer.mode: {mode}")
    same_model = (
        train_model["checkpoint"] == inference_model["checkpoint"]
        and train_model["revision"] == inference_model["revision"]
    )
    if mode == "none" and not same_model:
        raise ParameterContractError(
            "transfer.mode=none requires identical train and inference model revisions"
        )
    if mode != "none":
        compatibility = dict(
            _mapping(
                _require(transfer, "compatibility", field="transfer"),
                field="transfer.compatibility",
            )
        )
        if compatibility != {
            "component_mapping": "identity_by_layer_and_head",
            "architecture_match": True,
        }:
            raise ParameterContractError(
                "controller transfer requires an identity layer/head mapping and "
                "architecture_match=true"
            )
        train_architecture = train_model.get("architecture")
        inference_architecture = inference_model.get("architecture")
        if (
            not isinstance(train_architecture, Mapping)
            or not isinstance(inference_architecture, Mapping)
            or dict(train_architecture) != dict(inference_architecture)
        ):
            raise ParameterContractError(
                "controller transfer requires equal registered model architectures"
            )
        transfer["compatibility"] = compatibility
    return transfer


def resolve_method_parameters(
    config: Mapping[str, Any],
    *,
    expected_method: str | None = None,
) -> dict[str, Any]:
    """Validate and resolve one baseline runtime plan."""

    baseline = _mapping(_require(config, "baseline", field="config"), field="baseline")
    method = _string(_require(baseline, "method", field="baseline"), field="baseline.method")
    if method not in METHODS:
        raise ParameterContractError(f"unknown baseline method: {method}")
    if expected_method is not None and method != expected_method:
        raise ParameterContractError(
            f"configured method {method!r} does not match entry {expected_method!r}"
        )
    parameters = _mapping(
        _require(baseline, "parameters", field="baseline"),
        field="baseline.parameters",
    )
    data = _mapping(_require(parameters, "data", field="baseline.parameters"), field="data")
    plan: dict[str, Any] = {
        "method": method,
        "method_kind": METHOD_KINDS[method],
        "hyperparameters": copy.deepcopy(
            _mapping(
                _require(parameters, "hyperparameters", field="baseline.parameters"),
                field="hyperparameters",
            )
        ),
    }

    if method == "direct_policy":
        inference_model = _resolve_inference_model(parameters)
        plan["model"] = {"inference": copy.deepcopy(inference_model)}
        plan["data"] = {
            "inference_manifest": _artifact_ref(
                _require(data, "inference_manifest", field="data"),
                field="data.inference_manifest",
            )
        }
        plan["inference"] = copy.deepcopy(
            _mapping(
                _require(parameters, "inference", field="baseline.parameters"),
                field="inference",
            )
        )
        return plan

    if method == "cast":
        try:
            plan["control_objective"] = resolve_control_objective(
                _require(parameters, "control_objective", field="baseline.parameters"),
                expected_family="relative_advantage",
            )
        except ControlObjectiveError as exc:
            raise ParameterContractError(str(exc)) from exc
        train_model, inference_model = _resolve_models(parameters)
        resolved_data = {
            "training_manifest": _artifact_ref(
                _require(data, "training_manifest", field="data"),
                field="data.training_manifest",
            ),
        }
        for optional_role in (
            "validation_manifest",
            "inference_manifest",
            "selector_manifest",
        ):
            if optional_role in data:
                resolved_data[optional_role] = _artifact_ref(
                    data[optional_role], field=f"data.{optional_role}"
                )
        plan["model"] = {
            "train": copy.deepcopy(train_model),
            "inference": copy.deepcopy(inference_model),
        }
        plan["transfer"] = _resolve_transfer(parameters, train_model, inference_model)
        plan["data"] = resolved_data
        controller = dict(
            _mapping(
                _require(parameters, "controller", field="baseline.parameters"),
                field="controller",
            )
        )
        family = _string(
            _require(controller, "family", field="controller"),
            field="controller.family",
        )
        if family not in CAST_FAMILIES:
            raise ParameterContractError(f"unsupported CAST family: {family}")
        controller["family"] = family
        legacy_timings = controller.get("timings")
        training_timings = controller.get("training_timings")
        inference_timings = controller.get("inference_timings")
        if legacy_timings is not None:
            if training_timings is not None or inference_timings is not None:
                raise ParameterContractError(
                    "controller uses either timings or explicit training/inference timings"
                )
            resolved_timings = _timings(legacy_timings)
            controller["timings"] = resolved_timings
            controller["training_timings"] = copy.deepcopy(resolved_timings)
            controller["inference_timings"] = copy.deepcopy(resolved_timings)
        else:
            if training_timings is None or inference_timings is None:
                raise ParameterContractError(
                    "controller requires both training_timings and inference_timings"
                )
            controller["training_timings"] = _timings(training_timings)
            controller["inference_timings"] = _timings(inference_timings)
        controller["position_manifest"] = _artifact_ref(
            _require(controller, "position_manifest", field="controller"),
            field="controller.position_manifest",
        )
        plan["controller"] = controller
        plan["training"] = copy.deepcopy(
            _mapping(
                _require(parameters, "training", field="baseline.parameters"),
                field="training",
            )
        )
        plan["inference"] = copy.deepcopy(
            _mapping(
                _require(parameters, "inference", field="baseline.parameters"),
                field="inference",
            )
        )
        if "execution_adapters" in parameters:
            plan["execution_adapters"] = copy.deepcopy(
                _mapping(
                    parameters["execution_adapters"],
                    field="baseline.parameters.execution_adapters",
                )
            )
    elif method in {"loreft", "bipo"}:
        train_model, inference_model = _resolve_models(parameters)
        if (
            train_model["checkpoint"] != inference_model["checkpoint"]
            or train_model["revision"] != inference_model["revision"]
        ):
            raise ParameterContractError(
                f"native {method} trains and evaluates on the same registered base model"
            )
        plan["model"] = {
            "train": copy.deepcopy(train_model),
            "inference": copy.deepcopy(inference_model),
        }
        resolved_data = {}
        for role in (
            "training_manifest",
            "validation_manifest",
            "selector_manifest",
            "inference_manifest",
        ):
            resolved_data[role] = _artifact_ref(
                _require(data, role, field="data"), field=f"data.{role}"
            )
        plan["data"] = resolved_data
        for block in ("operator", "training", "search"):
            plan[block] = copy.deepcopy(
                _mapping(
                    _require(parameters, block, field="baseline.parameters"),
                    field=block,
                )
            )
        if method == "loreft":
            from baseline.implementations.loreft import validate_loreft_plan

            validate_loreft_plan(plan)
        else:
            plan["official_code_path"] = _string(
                _require(parameters, "official_code_path", field="baseline.parameters"),
                field="baseline.parameters.official_code_path",
            )
            if "task_name" in parameters:
                plan["task_name"] = _string(parameters["task_name"], field="task_name")
            from baseline.implementations.bipo import validate_bipo_plan

            validate_bipo_plan(plan)
    else:
        inference_model = _resolve_inference_model(parameters)
        plan["model"] = {"inference": copy.deepcopy(inference_model)}
        plan["data"] = {
            "selector_manifest": _artifact_ref(
                _require(data, "selector_manifest", field="data"),
                field="data.selector_manifest",
            )
        }
        if method == "caa":
            plan["data"]["training_manifest"] = _artifact_ref(
                _require(data, "training_manifest", field="data"),
                field="data.training_manifest",
            )
        plan["position_search"] = copy.deepcopy(
            _mapping(
                _require(parameters, "position_search", field="baseline.parameters"),
                field="position_search",
            )
        )
        if method in {"rcm_zero", "rcm_patch"}:
            try:
                plan["control_objective"] = resolve_control_objective(
                    _require(parameters, "control_objective", field="baseline.parameters"),
                    expected_family="state_contrast",
                    expected_variant=method.removeprefix("rcm_"),
                )
            except ControlObjectiveError as exc:
                raise ParameterContractError(str(exc)) from exc

    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve one baseline method parameter plan.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    plan = resolve_method_parameters(config, expected_method=args.method)
    rendered = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
