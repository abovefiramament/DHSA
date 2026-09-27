from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CONTROL_VECTOR_KINDS = {"comp", "head_act"}
CONTROL_NO_ALPHA_KINDS = {"head_scale", "cond"}


@dataclass(frozen=True, slots=True)
class ControlPart:
    kind: str
    name: str
    value: float | None = None
    apply_mode: str = ""

    def to_text(self) -> str:
        if self.kind in CONTROL_VECTOR_KINDS:
            if self.value is None:
                raise ValueError(f"{self.kind} control requires an alpha")
            text = f"{self.kind}:{self.name}:{_format_float(self.value)}"
            return f"{text}:{self.apply_mode}" if self.apply_mode else text
        if self.kind == "cond":
            if self.value is not None:
                raise ValueError("cond control does not take an explicit alpha")
            if self.apply_mode:
                raise ValueError("cond control does not support apply_mode overrides")
            return f"cond:{self.name}"
        if self.kind == "head_scale":
            if self.value is not None:
                raise ValueError("head_scale control weights live in --head-scalings and cannot be alpha-scaled here")
            text = f"head_scale:{self.name}"
            return f"{text}:{self.apply_mode}" if self.apply_mode else text
        raise ValueError(f"Unsupported control part kind: {self.kind!r}")


def _format_float(value: float) -> str:
    rounded = 0.0 if abs(value) < 1e-15 else float(value)
    return f"{rounded:.12g}"


def split_control_parts(parts: str) -> list[str]:
    return [part.strip() for part in str(parts or "").split("+") if part.strip()]


def parse_control_part(part: str) -> ControlPart:
    fields = [field.strip() for field in part.split(":")]
    if not fields or not fields[0]:
        raise ValueError(f"Empty control part: {part!r}")
    kind = fields[0]
    if kind in CONTROL_VECTOR_KINDS:
        if len(fields) not in {3, 4}:
            raise ValueError(f"{kind} part must be {kind}:<name>:<alpha>[:apply_mode], got {part!r}")
        return ControlPart(kind=kind, name=fields[1], value=float(fields[2]), apply_mode=fields[3] if len(fields) == 4 else "")
    if kind == "cond":
        if len(fields) != 2:
            raise ValueError(f"cond part must be cond:<name>, got {part!r}")
        return ControlPart(kind=kind, name=fields[1], value=None, apply_mode="")
    if kind == "head_scale":
        if len(fields) not in {2, 3}:
            raise ValueError(f"head_scale part must be head_scale:<name>[:apply_mode], got {part!r}")
        apply_mode = ""
        if len(fields) == 3:
            apply_mode = fields[2]
        return ControlPart(kind=kind, name=fields[1], value=None, apply_mode=apply_mode)
    raise ValueError(f"Unsupported control part: {part!r}")


def parse_control_parts(parts: str) -> list[ControlPart]:
    return [parse_control_part(part) for part in split_control_parts(parts)]


def join_control_parts(*groups: str) -> str:
    parts: list[str] = []
    for group in groups:
        parts.extend(split_control_parts(group))
    return "+".join(parts)


def scale_control_parts(parts: str, scale: float) -> str:
    scaled: list[str] = []
    for part in parse_control_parts(parts):
        if part.kind in CONTROL_VECTOR_KINDS:
            scaled.append(
                ControlPart(
                    kind=part.kind,
                    name=part.name,
                    value=float(part.value or 0.0) * float(scale),
                    apply_mode=part.apply_mode,
                ).to_text()
            )
        elif part.kind == "head_scale":
            if abs(float(scale) - 1.0) > 1e-12:
                raise ValueError("head_scale controls cannot be scaled by prev_scale; use head_act vectors for multi-round state")
            scaled.append(ControlPart(kind=part.kind, name=part.name, value=None, apply_mode=part.apply_mode).to_text())
        elif part.kind == "cond":
            if abs(float(scale) - 1.0) > 1e-12:
                raise ValueError("cond controls cannot be scaled by prev_scale; train a new conditional payload instead")
            scaled.append(ControlPart(kind=part.kind, name=part.name, value=None, apply_mode="").to_text())
    return "+".join(scaled)


def remap_control_parts(
    parts: str,
    *,
    component_names: Mapping[str, str] | None = None,
    head_names: Mapping[str, str] | None = None,
    scale: float = 1.0,
) -> str:
    component_names = component_names or {}
    head_names = head_names or {}
    remapped: list[str] = []
    for part in parse_control_parts(parts):
        name = part.name
        if part.kind == "comp":
            name = component_names.get(name, name)
            value = float(part.value or 0.0) * float(scale)
        elif part.kind == "head_act":
            name = head_names.get(name, name)
            value = float(part.value or 0.0) * float(scale)
        elif part.kind == "head_scale":
            name = head_names.get(name, name)
            if abs(float(scale) - 1.0) > 1e-12:
                raise ValueError("head_scale controls cannot be scaled by prev_scale; use head_act vectors for multi-round state")
            value = None
        elif part.kind == "cond":
            if abs(float(scale) - 1.0) > 1e-12:
                raise ValueError("cond controls cannot be scaled by prev_scale; train a new conditional payload instead")
            value = None
        else:
            raise ValueError(f"Unsupported control part kind: {part.kind!r}")
        remapped.append(ControlPart(kind=part.kind, name=name, value=value, apply_mode=part.apply_mode).to_text())
    return "+".join(remapped)


def max_abs_alpha(parts: str) -> float:
    values = [
        abs(float(part.value or 0.0))
        for part in parse_control_parts(parts)
        if part.kind in CONTROL_VECTOR_KINDS
    ]
    return max(values) if values else 0.0
