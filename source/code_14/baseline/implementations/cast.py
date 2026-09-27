"""Shared CAST executor used by Site and Performance.

The method owns only CAST mathematics and execution. Model providers, paths,
datasets, scorers, and orchestration are injected by callers.
"""

from __future__ import annotations

import csv
import copy
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import torch
from torch import Tensor, nn

from experiments.shared.contracts import (
    BankComposeRequest,
    BankTrainResult,
    ComponentPayloadResult,
)


@dataclass(frozen=True)
class CastTrainingRequest:
    """CAST-only request; scientific execution carries only explicit values."""

    bank_id: str
    ordered_positions: tuple[Mapping[str, Any], ...]
    method_plan: Mapping[str, Any]
    execution_config: Mapping[str, Any]
    training_data_manifest: Path
    validation_data_manifest: Path | None
    evidence_root: Path
    output_dir: Path


class CastModel(Protocol):
    model: nn.Module
    tokenizer: Any
    attention_heads_per_layer: int
    head_dim: int

    def projection(self, layer: int) -> nn.Module:
        """Return one attention pre-o projection module."""


class CastModelProvider(Protocol):
    """Injected model registry; no machine path is owned by CAST."""

    def load(self, model_id: str, *, mode: str) -> CastModel:
        ...


_TIMING_ID_BY_ALIAS = {
    "generation_decision_states": "continuation_decision_states",
    "decision_tokens": "continuation_decision_states",
    "decode": "continuation_decision_states",
    "all": "all_token_states",
    "all_positions": "all_token_states",
    "prefill": "prefill_states",
    "prefill_only": "prefill_states",
    "prompt": "prefill_states",
    "prompt_last": "prompt_last_state",
}


def validate_timing_reference(name: str, timing_id: str | None = None) -> str:
    """Resolve a historical hook label to one registered timing meaning."""

    canonical = _TIMING_ID_BY_ALIAS.get(name)
    if canonical is None:
        raise ValueError(f"unsupported CAST timing: {name!r}")
    if timing_id is not None and timing_id != canonical:
        raise ValueError(
            f"timing label {name!r} must use timing_id {canonical!r}, not {timing_id!r}"
        )
    return canonical


@dataclass(frozen=True)
class CastTiming:
    name: str
    mode: str | None = None

    def indices(self, prompt_length: int, sequence_length: int) -> range:
        timing_id = validate_timing_reference(self.name)
        mode = self.mode or {
            "continuation_decision_states": "generation",
            "all_token_states": "all",
            "prefill_states": "prefill",
            "prompt_last_state": "prefill",
        }[timing_id]
        if mode == "all":
            return range(sequence_length)
        if mode in {"decision_tokens", "generation", "decode"}:
            return range(max(0, prompt_length - 1), sequence_length)
        if mode == "prefill":
            return range(max(0, prompt_length - 1), min(prompt_length, sequence_length))
        raise ValueError(f"unsupported CAST timing: {self.name!r} (mode={mode!r})")


def _read_rows(path: Path, root: Path | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        rows: Any = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, Mapping):
            artifact = rows.get("role_artifact")
            if isinstance(artifact, Mapping) and root is not None and isinstance(artifact.get("relative_path"), str):
                return _read_rows(root / str(artifact["relative_path"]), root)
            rows = rows.get("rows")
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported CAST data file: {path.suffix}")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("CAST data must contain object rows")
    return [dict(row) for row in rows]


def _model_id(plan: Mapping[str, Any], execution: Mapping[str, Any], *, mode: str = "train") -> str:
    value = execution.get(f"{mode}_model_id") or plan.get(f"{mode}_model_id")
    if value is None:
        value = execution.get("model_id") or plan.get("model_id")
    if isinstance(value, str) and value:
        return value
    for source in (execution.get("model"), plan.get("model")):
        if isinstance(source, Mapping):
            preferred = source.get(mode)
            candidates = (preferred,) if preferred is not None else ()
            candidates = (*candidates, source)
            for item in candidates:
                if isinstance(item, Mapping):
                    for field in ("id", "model_id", "checkpoint"):
                        if isinstance(item.get(field), str) and item[field]:
                            return str(item[field])
        elif mode == "train" and isinstance(source, str) and source:
            return source
    raise ValueError("CAST requires a model identifier from the external model registry")


def _timings(plan: Mapping[str, Any], phase: str) -> tuple[CastTiming, ...]:
    controller = plan.get("controller")
    raw = controller.get(f"{phase}_timings") if isinstance(controller, Mapping) else None
    if raw is None and isinstance(controller, Mapping):
        raw = controller.get("timings")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError(f"CAST controller.{phase}_timings must be non-empty")
    result: list[CastTiming] = []
    for item in raw:
        name = item if isinstance(item, str) else item.get("name") if isinstance(item, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid CAST {phase} timing")
        timing_id = item.get("timing_id") if isinstance(item, Mapping) else None
        if timing_id is not None and not isinstance(timing_id, str):
            raise ValueError(f"invalid CAST {phase} timing_id")
        validate_timing_reference(name, timing_id)
        mode = item.get("hook") if isinstance(item, Mapping) else None
        if mode == "decode":
            mode = "generation"
        result.append(CastTiming(name, str(mode) if isinstance(mode, str) else None))
    return tuple(result)


def _reft_mode(plan: Mapping[str, Any]) -> str:
    controller = plan.get("controller")
    hyperparameters = plan.get("hyperparameters")
    values = (
        plan.get("reft_mode"),
        controller.get("reft_mode") if isinstance(controller, Mapping) else None,
        hyperparameters.get("reft_mode") if isinstance(hyperparameters, Mapping) else None,
    )
    mode = next((str(value) for value in values if value is not None), "low_rank")
    if mode not in {"low_rank", "gated_vector"}:
        raise ValueError("CAST ReFT mode must be low_rank or gated_vector")
    return mode


def _parse_position(component: Mapping[str, Any], heads: int) -> tuple[str, int, int]:
    value = component.get("component_id")
    if not isinstance(value, str) or not value.startswith("L") or ".attn.h" not in value:
        raise ValueError(f"invalid CAST position {value!r}")
    layer_text, head_text = value[1:].split(".attn.h", 1)
    layer, head = int(layer_text), int(head_text)
    if layer < 0 or head < 0 or head >= heads:
        raise ValueError(f"CAST position outside model geometry: {value}")
    return value, layer, head


def _input_ids(model: CastModel, prompt: str, answer: str) -> tuple[Tensor, int, int]:
    prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = model.tokenizer(answer, add_special_tokens=False).input_ids
    if not prompt_ids or not answer_ids:
        raise ValueError("CAST prompt and answer must tokenize to non-empty sequences")
    device = next(model.model.parameters()).device
    return torch.tensor([prompt_ids + answer_ids], device=device), len(prompt_ids), len(answer_ids)


def _attach(model: CastModel, parameters: Mapping[str, Mapping[str, Tensor]], alpha: float, timing: CastTiming, prompt_length: int, sequence_length: int) -> list[Any]:
    hooks: list[Any] = []
    for component, parameter in parameters.items():
        _component, layer, head = _parse_position({"component_id": component}, model.attention_heads_per_layer)
        projection = model.projection(layer)

        def pre_hook(_module: nn.Module, args: tuple[Any, ...], parameter=parameter, head=head):
            value = args[0]
            updated = value.clone()
            current_length = int(value.shape[1])
            current_prompt = (
                prompt_length
                if current_length > 1
                else (
                    1
                    if timing.name in {"generation_decision_states", "decision_tokens", "decode"}
                    else prompt_length
                )
            )
            active = set(timing.indices(current_prompt, current_length))
            state = updated[..., head * model.head_dim:(head + 1) * model.head_dim]
            mask = torch.zeros(updated.shape[:2], dtype=torch.bool, device=updated.device)
            for index in active:
                if index < mask.shape[1]:
                    mask[:, index] = True
            if "vector" in parameter and "gate_weight" in parameter:
                gate_weight = parameter["gate_weight"].to(device=state.device, dtype=state.dtype)
                gate_bias = parameter["gate_bias"].to(device=state.device, dtype=state.dtype)
                denom = float(max(state.shape[-1], 1)) ** 0.5
                gate = torch.sigmoid(
                    (state * gate_weight.view(1, 1, -1)).sum(dim=-1, keepdim=True) / denom
                    + gate_bias
                )
                delta = gate * parameter["vector"].to(device=state.device, dtype=state.dtype).view(1, 1, -1)
            elif "vector" in parameter:
                delta = parameter["vector"].to(device=state.device, dtype=state.dtype).view(1, 1, -1).expand_as(state)
            else:
                rank = int(parameter["A"].shape[0])
                if rank <= 0:
                    raise ValueError("CAST ReFT rank must be positive")
                low = torch.matmul(state, parameter["A"].to(device=state.device, dtype=state.dtype).transpose(0, 1))
                delta = torch.matmul(low, parameter["B"].to(device=state.device, dtype=state.dtype).transpose(0, 1))
                delta = delta / float(rank) ** 0.5
            replacement = state + alpha * delta
            replacement = torch.where(mask.unsqueeze(-1), replacement, state)
            begin = head * model.head_dim
            end = (head + 1) * model.head_dim
            updated = torch.cat((updated[..., :begin], replacement, updated[..., end:]), dim=-1)
            return (updated, *args[1:])

        hooks.append(projection.register_forward_pre_hook(pre_hook))
    return hooks


def _remove(hooks: Sequence[Any]) -> None:
    for hook in hooks:
        hook.remove()


def _margin(model: CastModel, prompt: str, answer: str, parameters: Mapping[str, Mapping[str, Tensor]] | None, alpha: float, timing: CastTiming) -> Tensor:
    ids, prompt_length, answer_length = _input_ids(model, prompt, answer)
    hooks = _attach(model, parameters, alpha, timing, prompt_length, ids.shape[1]) if parameters else []
    try:
        logits = model.model(input_ids=ids).logits[0, prompt_length - 1:prompt_length - 1 + answer_length]
        labels = ids[0, prompt_length:prompt_length + answer_length]
        correct = logits.gather(-1, labels[:, None]).squeeze(-1)
        other = logits.masked_fill(torch.nn.functional.one_hot(labels, logits.shape[-1]).bool(), float("-inf")).amax(-1)
        return (correct - other).mean()
    finally:
        _remove(hooks)


def _parameter_set(model: CastModel, positions: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, dict[str, Tensor]]:
    controller = plan.get("controller")
    operator = plan.get("operator")
    if operator is None and isinstance(controller, Mapping):
        operator = controller.get("operator") or controller.get("family")
    operator = str(operator or "")
    if operator not in {"sv", "reft"}:
        raise ValueError("CAST operator must be explicitly configured as sv or reft")
    device = next(model.model.parameters()).device
    hyperparameters = plan.get("hyperparameters") if isinstance(plan.get("hyperparameters"), Mapping) else {}
    rank_value = plan.get("rank", hyperparameters.get("rank", controller.get("rank") if isinstance(controller, Mapping) else None))
    if operator == "reft" and rank_value is None:
        raise ValueError("CAST ReFT requires an explicit rank")
    rank = int(rank_value) if rank_value is not None else 0
    if operator == "reft" and rank <= 0:
        raise ValueError("CAST ReFT rank must be positive")
    reft_mode = _reft_mode(plan) if operator == "reft" else "sv"
    gate_init_value = plan.get("gate_init", hyperparameters.get("gate_init"))
    if reft_mode == "gated_vector":
        if gate_init_value is None:
            raise ValueError("CAST gated_vector requires an explicit gate_init")
        gate_init = float(gate_init_value)
        if not 0.0 < gate_init < 1.0:
            raise ValueError("CAST gate_init must lie strictly between zero and one")
        gate_bias = torch.tensor(
            [torch.log(torch.tensor(gate_init / (1.0 - gate_init))).item()],
            device=device,
            dtype=torch.float32,
        )
    result: dict[str, dict[str, Tensor]] = {}
    for item in positions:
        component, _layer, _head = _parse_position(item, model.attention_heads_per_layer)
        if operator == "sv":
            result[component] = {"vector": nn.Parameter(torch.zeros(model.head_dim, device=device, dtype=torch.float32))}
        elif reft_mode == "gated_vector":
            result[component] = {
                "vector": nn.Parameter(torch.zeros(model.head_dim, device=device, dtype=torch.float32)),
                "gate_weight": nn.Parameter(torch.zeros(model.head_dim, device=device, dtype=torch.float32)),
                "gate_bias": nn.Parameter(gate_bias.clone()),
            }
        else:
            # Preserve a zero initial intervention while keeping the low-rank
            # factor trainable on the first optimizer step.  Initialising both
            # factors to zero would make every ReFT gradient zero.
            result[component] = {
                "A": nn.Parameter(
                    torch.randn(rank, model.head_dim, device=device, dtype=torch.float32) * 0.01
                ),
                "B": nn.Parameter(
                    torch.zeros(model.head_dim, rank, device=device, dtype=torch.float32)
                ),
            }
    return result


class CastBackend:
    """Shared CAST bank executor for Site and Performance."""

    def __init__(self, model_provider: CastModelProvider | None = None) -> None:
        self.model_provider = model_provider

    def train_bank(self, request: CastTrainingRequest | Any) -> BankTrainResult:
        if self.model_provider is None:
            raise RuntimeError("CAST requires an injected model provider; machine registration is outside the method")
        plan = request.method_plan
        model = self.model_provider.load(_model_id(plan, request.execution_config, mode="train"), mode="train")
        model.model.eval()
        for parameter in model.model.parameters():
            parameter.requires_grad_(False)
        rows = _read_rows(request.training_data_manifest, request.evidence_root)
        if not rows:
            raise ValueError("CAST training role is empty")
        if not request.ordered_positions:
            raise ValueError("CAST requires at least one registered position")
        parameters = _parameter_set(model, request.ordered_positions, plan)
        training = plan.get("training") if isinstance(plan.get("training"), Mapping) else {}
        required_training = {
            "lr",
            "weight_decay",
            "epochs",
            "dpo_beta",
            "alpha_train",
            "lambda_norm",
            "epoch_shuffle_seed",
        }
        # ReFT's shared protocol names the same deterministic epoch order as
        # `seed`; consume it explicitly when the SV alias is absent.
        if "epoch_shuffle_seed" not in training and isinstance(training.get("seed"), int) and not isinstance(training.get("seed"), bool):
            training = dict(training)
            training["epoch_shuffle_seed"] = training["seed"]
        missing_training = sorted(required_training - set(training))
        if missing_training:
            raise ValueError(f"CAST training configuration is missing: {missing_training}")
        lr, decay = float(training["lr"]), float(training["weight_decay"])
        epochs, beta = int(training["epochs"]), float(training["dpo_beta"])
        alpha = float(training["alpha_train"])
        timings = _timings(plan, "training")
        learnable = [
            value
            for item in parameters.values()
            for value in item.values()
            if isinstance(value, nn.Parameter)
        ]
        optimizer = torch.optim.AdamW(learnable, lr=lr, weight_decay=decay)
        steps = 0
        for _epoch in range(epochs):
            random.Random(int(training["epoch_shuffle_seed"]) + _epoch).shuffle(rows)
            for row in rows:
                prompt = str(row.get("prompt", row.get("question", "")))
                chosen_value = row.get("chosen", row.get("target", row.get("y_plus", "")))
                rejected_value = row.get("rejected", row.get("non_target", row.get("y_minus", "")))
                if isinstance(chosen_value, Sequence) and not isinstance(chosen_value, (str, bytes)):
                    chosen_value = chosen_value[0] if chosen_value else ""
                if isinstance(rejected_value, Sequence) and not isinstance(rejected_value, (str, bytes)):
                    rejected_value = rejected_value[0] if rejected_value else ""
                chosen = str(chosen_value)
                rejected = str(rejected_value)
                if not prompt or not chosen or not rejected:
                    raise ValueError("CAST rows require prompt, chosen, and rejected")
                with torch.no_grad():
                    reference = _margin(model, prompt, chosen, None, 0.0, timings[0]) - _margin(model, prompt, rejected, None, 0.0, timings[0])
                controlled = torch.zeros((), device=next(model.model.parameters()).device)
                for timing in timings:
                    controlled = controlled + _margin(model, prompt, chosen, parameters, alpha, timing) - _margin(model, prompt, rejected, parameters, alpha, timing)
                loss = -torch.nn.functional.logsigmoid(beta * (controlled / len(timings) - reference))
                loss = loss + float(training["lambda_norm"]) * sum(value.square().mean() for value in learnable)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
        request.output_dir.mkdir(parents=True, exist_ok=True)
        payload_values = {component: {key: value.detach().cpu() for key, value in item.items()} for component, item in parameters.items()}
        payload = request.output_dir / "cast_payload.pt"
        operator = "sv" if any("vector" in item and "gate_weight" not in item for item in payload_values.values()) else "reft"
        torch.save({"format": "cast_v1", "operator": operator, "reft_mode": _reft_mode(plan) if operator == "reft" else None, "head_dim": model.head_dim, "parameters": payload_values, "positions": list(payload_values), "training_timings": [item.name for item in timings], "inference_timings": [item.name for item in _timings(plan, "inference")]}, payload)
        component_results: list[ComponentPayloadResult] = []
        component_dir = request.output_dir / "components"
        component_dir.mkdir(parents=True, exist_ok=True)
        for index, component_id in enumerate(payload_values):
            component_payload = component_dir / f"component_{index:04d}.pt"
            torch.save(
                {
                    "format": "cast_component_v1",
                    "component_id": component_id,
                    "operator": operator,
                    "reft_mode": _reft_mode(plan) if operator == "reft" else None,
                    "head_dim": model.head_dim,
                    "parameters": payload_values[component_id],
                },
                component_payload,
            )
            extraction_manifest = component_dir / f"source_{index:04d}.json"
            extraction_manifest.write_text(
                json.dumps(
                    {
                        "method": "cast",
                        "operation": "extract_component_payload",
                        "component_id": component_id,
                        "source_bank": request.bank_id,
                        "operator": operator,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            component_results.append(
                ComponentPayloadResult(
                    component_id=component_id,
                    payload_path=component_payload,
                    extraction_manifest_path=extraction_manifest,
                )
            )
        manifest = request.output_dir / "cast_training.json"
        manifest.write_text(json.dumps({"method": "cast", "bank_id": request.bank_id, "operator": operator, "reft_mode": _reft_mode(plan) if operator == "reft" else None, "positions": list(payload_values), "training_rows": len(rows), "epochs": epochs, "optimizer_steps": steps, "learning_rate": lr, "weight_decay": decay, "training_timings": [item.name for item in timings], "inference_timings": [item.name for item in _timings(plan, "inference")], "model_id": _model_id(plan, request.execution_config, mode="train")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return BankTrainResult(
            payload_path=payload,
            training_manifest_path=manifest,
            optimizer_steps=steps,
            component_payloads=tuple(component_results),
        )

    def compose_bank(self, request: BankComposeRequest | Any) -> BankTrainResult:
        """Compose audited component payloads without another optimizer step."""

        component_ids = tuple(str(item) for item in request.ordered_component_ids)
        if not component_ids or len(set(component_ids)) != len(component_ids):
            raise ValueError("CAST composition requires unique ordered component IDs")
        manifests = {
            str(item.get("component_id")): item
            for item in request.component_manifests
            if isinstance(item, Mapping) and isinstance(item.get("component_id"), str)
        }
        if set(manifests) != set(component_ids):
            raise ValueError("CAST composition component manifests do not match the ordered IDs")
        parameters: dict[str, dict[str, Tensor]] = {}
        operator: str | None = None
        head_dim: int | None = None
        for component_id in component_ids:
            manifest = manifests[component_id]
            artifact = manifest.get("component_payload")
            if isinstance(artifact, Mapping):
                relative = artifact.get("relative_path")
                if not isinstance(relative, str) or not relative:
                    raise ValueError(f"component payload path is missing for {component_id}")
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"component payload escapes evidence root for {component_id}")
                payload_path = request.evidence_root / relative_path
            else:
                payload_path = Path(str(artifact)) if artifact is not None else None
            if payload_path is None or not payload_path.is_file():
                raise FileNotFoundError(f"component payload is unavailable for {component_id}: {payload_path}")
            try:
                loaded = torch.load(payload_path, map_location="cpu", weights_only=False)
            except TypeError:
                loaded = torch.load(payload_path, map_location="cpu")
            if not isinstance(loaded, Mapping):
                raise ValueError(f"component payload must be a mapping: {payload_path}")
            raw = loaded.get("parameters", loaded.get("payload", loaded))
            if isinstance(raw, Mapping) and component_id in raw and isinstance(raw[component_id], Mapping):
                raw = raw[component_id]
            elif isinstance(raw, Mapping) and len(raw) == 1 and isinstance(next(iter(raw.values())), Mapping):
                raw = next(iter(raw.values()))
            if not isinstance(raw, Mapping):
                raise ValueError(f"component payload parameters are invalid: {payload_path}")
            tensors = {
                str(name): value.detach().clone().cpu()
                for name, value in raw.items()
                if isinstance(value, Tensor)
            }
            if not tensors:
                raise ValueError(f"component payload has no tensor parameters: {payload_path}")
            if "vector" in tensors:
                current_operator = "reft" if "gate_weight" in tensors else "sv"
                current_head_dim = int(tensors["vector"].shape[-1])
            elif "A" in tensors and "B" in tensors:
                current_operator = "reft"
                current_head_dim = int(tensors["B"].shape[0])
            else:
                raise ValueError(f"unsupported CAST component payload parameters: {sorted(tensors)}")
            if operator is not None and current_operator != operator:
                raise ValueError("CAST composition cannot mix SV and ReFT component payloads")
            if head_dim is not None and current_head_dim != head_dim:
                raise ValueError("CAST component payload head dimensions disagree")
            operator, head_dim = current_operator, current_head_dim
            parameters[component_id] = tensors
        request.output_dir.mkdir(parents=True, exist_ok=True)
        payload = request.output_dir / "cast_payload.pt"
        controller = request.method_plan.get("controller")
        controller = controller if isinstance(controller, Mapping) else {}
        training_timings = controller.get("training_timings", controller.get("timings", []))
        inference_timings = controller.get("inference_timings", controller.get("timings", []))
        torch.save(
            {
                "format": "cast_v1",
                "operator": operator,
                "reft_mode": _reft_mode(request.method_plan) if operator == "reft" else None,
                "head_dim": head_dim,
                "parameters": parameters,
                "positions": list(component_ids),
                "training_timings": list(training_timings) if isinstance(training_timings, Sequence) else [],
                "inference_timings": list(inference_timings) if isinstance(inference_timings, Sequence) else [],
            },
            payload,
        )
        manifest = request.output_dir / "cast_composition.json"
        manifest.write_text(
            json.dumps(
                {
                    "method": "cast",
                    "bank_id": request.bank_id,
                    "operation": "compose_bank",
                    "optimizer_steps": 0,
                    "operator": operator,
                    "reft_mode": _reft_mode(request.method_plan) if operator == "reft" else None,
                    "positions": list(component_ids),
                    "source_component_ids": list(component_ids),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return BankTrainResult(
            payload_path=payload,
            training_manifest_path=manifest,
            optimizer_steps=0,
        )


def resolve_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("CAST config must be an object")
    baseline = config.get("baseline")
    if not isinstance(baseline, Mapping) or baseline.get("method") != "cast":
        raise ValueError("CAST config must select baseline.method=cast")
    parameters = baseline.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("CAST baseline.parameters must be an object")
    for field in ("model", "controller", "training", "inference"):
        if not isinstance(parameters.get(field), Mapping):
            raise ValueError(f"CAST parameters.{field} must be an object")
    return {"method": "cast", "method_kind": "actuator_baseline", **copy.deepcopy(dict(parameters))}


def train(*args: Any, **kwargs: Any) -> BankTrainResult:
    backend = kwargs.pop("backend", None) or CastBackend(kwargs.pop("model_provider", None))
    request = args[0] if args else kwargs.get("request")
    if request is None or not all(hasattr(request, name) for name in ("bank_id", "ordered_positions", "method_plan", "execution_config", "training_data_manifest", "evidence_root", "output_dir")):
        raise TypeError("CAST.train expects a CAST training request")
    return backend.train_bank(request)


def train_from_files(
    *,
    method_plan: Mapping[str, Any],
    model_provider: CastModelProvider,
    bank_id: str,
    positions: Sequence[Mapping[str, Any]],
    training_data_manifest: Path,
    validation_data_manifest: Path | None,
    evidence_root: Path,
    output_dir: Path,
    execution_config: Mapping[str, Any],
) -> BankTrainResult:
    """Script-facing entrypoint with every runtime choice supplied externally."""

    request = CastTrainingRequest(
        bank_id=bank_id,
        ordered_positions=tuple(dict(item) for item in positions),
        method_plan=dict(method_plan),
        execution_config=dict(execution_config),
        training_data_manifest=Path(training_data_manifest),
        validation_data_manifest=(
            Path(validation_data_manifest) if validation_data_manifest is not None else None
        ),
        evidence_root=Path(evidence_root),
        output_dir=Path(output_dir),
    )
    return CastBackend(model_provider).train_bank(request)


def load_payload(payload_path: Path) -> dict[str, Any]:
    """Load one CAST bank payload without resolving a machine/model path."""

    path = Path(payload_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch releases do not expose weights_only
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format") != "cast_v1":
        raise ValueError("CAST payload must be a cast_v1 mapping")
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("CAST payload parameters must be non-empty")
    operator = payload.get("operator")
    if operator not in {"sv", "reft"}:
        raise ValueError("CAST payload operator must be sv or reft")
    normalized: dict[str, Any] = dict(payload)
    normalized["parameters"] = {
        str(component): {
            str(name): value.detach().clone() if isinstance(value, Tensor) else value
            for name, value in values.items()
        }
        for component, values in parameters.items()
        if isinstance(values, Mapping)
    }
    if len(normalized["parameters"]) != len(parameters):
        raise ValueError("CAST payload contains an invalid component parameter set")
    return normalized


@contextmanager
def applied_payload(
    model: CastModel,
    payload: Mapping[str, Any],
    *,
    alpha: float,
    timings: Sequence[CastTiming],
    prompt_length: int,
    sequence_length: int,
) -> Iterator[CastModel]:
    """Apply one frozen CAST bank around an externally owned generation call."""

    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("CAST payload parameters must be non-empty")
    hooks: list[Any] = []
    try:
        for timing in timings:
            hooks.extend(
                _attach(
                    model,
                    parameters,
                    float(alpha),
                    timing,
                    int(prompt_length),
                    int(sequence_length),
                )
            )
        yield model
    finally:
        _remove(hooks)


def infer_from_files(
    *,
    method_plan: Mapping[str, Any],
    model_provider: CastModelProvider,
    payload_path: Path,
    execution_config: Mapping[str, Any],
    alpha: float,
    prompt_length: int,
    sequence_length: int,
    generate: Callable[[CastModel], Any],
) -> Any:
    """Run an external generator with a CAST bank and inference model choice.

    Model loading and text generation remain injected.  CAST only loads the
    bank, installs its configured timing hooks, and removes them afterwards.
    """

    model = model_provider.load(
        _model_id(method_plan, execution_config, mode="inference"), mode="inference"
    )
    payload = load_payload(payload_path)
    timings = _timings(method_plan, "inference")
    with applied_payload(
        model,
        payload,
        alpha=alpha,
        timings=timings,
        prompt_length=prompt_length,
        sequence_length=sequence_length,
    ):
        return generate(model)
