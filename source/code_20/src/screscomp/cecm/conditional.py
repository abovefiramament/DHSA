from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONDITIONAL_ACTUATOR_KIND = "cast_conditional_actuator"


@dataclass(frozen=True, slots=True)
class ConditionalComponentAction:
    component_id: str
    layer_idx: int
    component_type: str
    vector: Any
    group: str
    vector_key: str = ""
    condition_vector: Any | None = None
    similarity_threshold_lower: float = 0.0
    similarity_threshold_upper: float = 1.0
    similarity_temperature: float = 8.0
    projection_threshold_lower: float = -100.0
    projection_threshold_upper: float = 100.0
    projection_temperature: float = 1.0
    train_apply_mode: str = "prompt_last"
    generation_apply_mode: str = "prefill"


@dataclass(frozen=True, slots=True)
class ConditionalHeadAction:
    head_id: str
    layer_idx: int
    head_idx: int
    vector: Any
    group: str
    vector_key: str = ""
    condition_vector: Any | None = None
    similarity_threshold_lower: float = 0.0
    similarity_threshold_upper: float = 1.0
    similarity_temperature: float = 8.0
    projection_threshold_lower: float = -100.0
    projection_threshold_upper: float = 100.0
    projection_temperature: float = 1.0
    train_apply_mode: str = "all"
    generation_apply_mode: str = "all"


@dataclass(frozen=True, slots=True)
class ConditionalGroupSpec:
    group: str
    alpha_max: float
    similarity_temperature: float
    projection_temperature: float = 1.0
    gate_mode: str = "soft"


def default_generation_apply_mode(train_apply_mode: str) -> str:
    mode = str(train_apply_mode or "").strip()
    if mode in {"prompt_last", "prefill", "prompt"}:
        return "prefill"
    if mode == "decision_tokens":
        return "first_decode"
    return mode or "all"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_gate_mode(value: Any, *, default: str = "soft") -> str:
    mode = str(value or default).strip().lower()
    if mode not in {"soft", "hard"}:
        raise ValueError(f"Unsupported conditional gate mode: {value!r}")
    return mode


def _lookup_payload_value(mapping: dict[str, Any], primary_key: str, fallback_key: str | None = None) -> Any:
    if primary_key in mapping:
        value = mapping[primary_key]
        if value is not None:
            return value
    if fallback_key is not None and fallback_key != primary_key:
        return mapping.get(fallback_key)
    return mapping.get(primary_key)


def _row_similarity_bounds(row: dict[str, Any]) -> tuple[float, float]:
    lower = _to_float(
        row.get("similarity_threshold_lower", row.get("similarity_threshold", row.get("direction_threshold", 0.0))),
        0.0,
    )
    upper = _to_float(
        row.get("similarity_threshold_upper", row.get("direction_threshold_upper", 1.0)),
        1.0,
    )
    return (min(lower, upper), max(lower, upper))


def _row_projection_bounds(row: dict[str, Any]) -> tuple[float, float]:
    lower = _to_float(
        row.get("projection_threshold_lower", row.get("magnitude_threshold_lower", -100.0)),
        -100.0,
    )
    upper = _to_float(
        row.get("projection_threshold_upper", row.get("magnitude_threshold_upper", 100.0)),
        100.0,
    )
    return (min(lower, upper), max(lower, upper))


def load_conditional_actuator_payload(
    path: Path,
) -> tuple[list[ConditionalComponentAction], list[ConditionalHeadAction], dict[str, ConditionalGroupSpec]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    kind = str(payload.get("kind", ""))
    if kind != CONDITIONAL_ACTUATOR_KIND:
        raise ValueError(f"Unsupported conditional actuator payload kind={kind!r} path={path}")

    raw_group_specs = dict(payload.get("group_specs", {}) or {})
    actuator_vectors = dict(payload.get("vectors", {}) or {})
    condition_vectors = dict(payload.get("condition_vectors", {}) or {})
    group_specs = {
        str(group): ConditionalGroupSpec(
            group=str(group),
            alpha_max=_to_float(spec.get("alpha_max", 0.0)),
            similarity_temperature=_to_float(
                spec.get("similarity_temperature", spec.get("direction_temperature", 8.0)),
                8.0,
            ),
            projection_temperature=_to_float(
                spec.get("projection_temperature", 1.0),
                1.0,
            ),
            gate_mode=_normalize_gate_mode(spec.get("gate_mode", payload.get("gate_mode", "soft"))),
        )
        for group, spec in raw_group_specs.items()
    }
    components: list[ConditionalComponentAction] = []
    for row in payload.get("components", []) or []:
        group = str(row["group"])
        component_id = str(row["component_id"])
        vector_key = str(row.get("vector_key") or f"{group}:{component_id}")
        similarity_threshold_lower, similarity_threshold_upper = _row_similarity_bounds(row)
        projection_threshold_lower, projection_threshold_upper = _row_projection_bounds(row)
        components.append(
            ConditionalComponentAction(
                component_id=component_id,
                layer_idx=int(row["layer_idx"]),
                component_type=str(row["component_type"]),
                vector=_lookup_payload_value(actuator_vectors, vector_key, component_id),
                group=group,
                vector_key=vector_key,
                condition_vector=_lookup_payload_value(condition_vectors, vector_key, component_id),
                similarity_threshold_lower=similarity_threshold_lower,
                similarity_threshold_upper=similarity_threshold_upper,
                similarity_temperature=_to_float(
                    row.get(
                        "similarity_temperature",
                        group_specs.get(group, ConditionalGroupSpec(group, 0.0, 8.0)).similarity_temperature,
                    ),
                    8.0,
                ),
                projection_threshold_lower=projection_threshold_lower,
                projection_threshold_upper=projection_threshold_upper,
                projection_temperature=_to_float(
                    row.get(
                        "projection_temperature",
                        group_specs.get(group, ConditionalGroupSpec(group, 0.0, 8.0)).projection_temperature,
                    ),
                    1.0,
                ),
                train_apply_mode=str(row.get("train_apply_mode") or "prompt_last"),
                generation_apply_mode=str(
                    row.get("generation_apply_mode")
                    or default_generation_apply_mode(str(row.get("train_apply_mode") or "prompt_last"))
                ),
            )
        )

    heads: list[ConditionalHeadAction] = []
    for row in payload.get("heads", []) or []:
        group = str(row["group"])
        head_id = str(row["head_id"])
        vector_key = str(row.get("vector_key") or f"{group}:{head_id}")
        similarity_threshold_lower, similarity_threshold_upper = _row_similarity_bounds(row)
        projection_threshold_lower, projection_threshold_upper = _row_projection_bounds(row)
        heads.append(
            ConditionalHeadAction(
                head_id=head_id,
                layer_idx=int(row["layer_idx"]),
                head_idx=int(row["head_idx"]),
                vector=_lookup_payload_value(actuator_vectors, vector_key, head_id),
                group=group,
                vector_key=vector_key,
                condition_vector=_lookup_payload_value(condition_vectors, vector_key, head_id),
                similarity_threshold_lower=similarity_threshold_lower,
                similarity_threshold_upper=similarity_threshold_upper,
                similarity_temperature=_to_float(
                    row.get(
                        "similarity_temperature",
                        group_specs.get(group, ConditionalGroupSpec(group, 0.0, 8.0)).similarity_temperature,
                    ),
                    8.0,
                ),
                projection_threshold_lower=projection_threshold_lower,
                projection_threshold_upper=projection_threshold_upper,
                projection_temperature=_to_float(
                    row.get(
                        "projection_temperature",
                        group_specs.get(group, ConditionalGroupSpec(group, 0.0, 8.0)).projection_temperature,
                    ),
                    1.0,
                ),
                train_apply_mode=str(row.get("train_apply_mode") or "all"),
                generation_apply_mode=str(
                    row.get("generation_apply_mode")
                    or default_generation_apply_mode(str(row.get("train_apply_mode") or "all"))
                ),
            )
        )

    missing_groups = {
        *(action.group for action in components),
        *(action.group for action in heads),
    } - set(group_specs)
    if missing_groups:
        raise ValueError(f"Missing group specs for conditional payload {path}: {sorted(missing_groups)}")
    missing_vectors = [
        action.component_id
        for action in components
        if action.vector is None
    ] + [
        action.head_id
        for action in heads
        if action.vector is None
    ]
    if missing_vectors:
        raise ValueError(f"Missing actuator vectors for conditional payload {path}: {sorted(missing_vectors)}")
    missing_conditions = [
        action.component_id
        for action in components
        if action.condition_vector is None
    ] + [
        action.head_id
        for action in heads
        if action.condition_vector is None
    ]
    if missing_conditions:
        raise ValueError(f"Missing condition vectors for conditional payload {path}: {sorted(missing_conditions)}")
    return components, heads, group_specs


def conditional_gate_values(
    torch_module: Any,
    *,
    hidden_slice: Any,
    condition_vector: Any,
    projection_vector: Any,
    similarity_threshold_lower: Any,
    similarity_threshold_upper: Any,
    similarity_temperature: Any,
    projection_threshold_lower: Any,
    projection_threshold_upper: Any,
    projection_temperature: Any,
    gate_mode: str = "soft",
) -> dict[str, Any]:
    if int(hidden_slice.shape[-1]) <= 0:
        raise ValueError("hidden_slice must have a positive feature dimension")
    pooled = hidden_slice.float().mean(dim=1)
    eps = 1e-6

    device = pooled.device
    dtype = pooled.dtype

    def to_tensor(value: Any) -> Any:
        if hasattr(value, "to"):
            return value.to(device=device, dtype=dtype)
        return torch_module.tensor(float(value), device=device, dtype=dtype)

    condition = to_tensor(condition_vector)
    projection = to_tensor(projection_vector)
    similarity_threshold_lower_value = to_tensor(similarity_threshold_lower)
    similarity_threshold_upper_value = to_tensor(similarity_threshold_upper)
    similarity_temperature_value = to_tensor(similarity_temperature)
    projection_threshold_lower_value = to_tensor(projection_threshold_lower)
    projection_threshold_upper_value = to_tensor(projection_threshold_upper)
    projection_temperature_value = to_tensor(projection_temperature)
    pooled_norm = pooled.norm(dim=-1).clamp_min(eps)
    condition_norm = condition.norm().clamp_min(eps)
    projection_norm = projection.norm().clamp_min(eps)
    similarity = (pooled * condition.unsqueeze(0)).sum(dim=-1) / (pooled_norm * condition_norm)
    projection_value = (pooled * (projection / projection_norm).unsqueeze(0)).sum(dim=-1)
    similarity_lower = torch_module.minimum(similarity_threshold_lower_value, similarity_threshold_upper_value)
    similarity_upper = torch_module.maximum(similarity_threshold_lower_value, similarity_threshold_upper_value)
    projection_lower = torch_module.minimum(projection_threshold_lower_value, projection_threshold_upper_value)
    projection_upper = torch_module.maximum(projection_threshold_lower_value, projection_threshold_upper_value)

    normalized_gate_mode = _normalize_gate_mode(gate_mode)
    similarity_lower_enabled = bool(float(similarity_lower.detach().cpu().item()) > -1.0 + 1e-5)
    similarity_upper_enabled = bool(float(similarity_upper.detach().cpu().item()) < 1.0 - 1e-5)
    projection_lower_enabled = bool(float(projection_lower.detach().cpu().item()) > -100.0 + 1e-5)
    projection_upper_enabled = bool(float(projection_upper.detach().cpu().item()) < 100.0 - 1e-5)
    if normalized_gate_mode == "hard":
        gate = torch_module.ones_like(similarity, dtype=dtype)
        if similarity_lower_enabled:
            gate = gate * (similarity >= similarity_lower).to(dtype)
        if similarity_upper_enabled:
            gate = gate * (similarity <= similarity_upper).to(dtype)
        if projection_lower_enabled:
            gate = gate * (projection_value >= projection_lower).to(dtype)
        if projection_upper_enabled:
            gate = gate * (projection_value <= projection_upper).to(dtype)
    else:
        gate = torch_module.ones_like(similarity, dtype=dtype)
        if similarity_lower_enabled:
            gate = gate * torch_module.sigmoid(
                similarity_temperature_value * (similarity - similarity_lower)
            )
        if similarity_upper_enabled:
            gate = gate * torch_module.sigmoid(
                similarity_temperature_value * (similarity_upper - similarity)
            )
        if projection_lower_enabled:
            gate = gate * torch_module.sigmoid(
                projection_temperature_value * (projection_value - projection_lower)
            )
        if projection_upper_enabled:
            gate = gate * torch_module.sigmoid(
                projection_temperature_value * (projection_upper - projection_value)
            )
    return {
        "gate": gate,
        "similarity": similarity,
        "projection": projection_value,
        "similarity_threshold_lower": similarity_lower,
        "similarity_threshold_upper": similarity_upper,
        "projection_threshold_lower": projection_lower,
        "projection_threshold_upper": projection_upper,
        "gate_mode": normalized_gate_mode,
    }
