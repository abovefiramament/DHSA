"""Code-owned capability registry, separate from methods and runtime paths.

Experiments select an immutable implementation ID and supply scientific
parameters. They cannot register executable paths or inject callables.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import require_operations


COMPONENT_KINDS = frozenset(
    {
        "data_adapter",
        "position_scanner",
        "position_backend",
        "bank_backend",
        "native_baseline_backend",
        "generation_backend",
        "evaluator",
        "alpha_selector",
    }
)


class ComponentRegistryError(ValueError):
    """Raised when a capability registration or binding is ambiguous."""


@dataclass(frozen=True, slots=True)
class ComponentRegistration:
    kind: str
    component_id: str
    version: int
    implementation: Any
    operations: tuple[str, ...]
    input_contract: str
    output_contract: str


class ComponentRegistry:
    """Strict code registry with no scientific values or machine paths."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, int], ComponentRegistration] = {}

    def register(
        self,
        *,
        kind: str,
        component_id: str,
        version: int,
        implementation: Any,
        operations: Sequence[str],
        input_contract: str,
        output_contract: str,
    ) -> ComponentRegistration:
        if kind not in COMPONENT_KINDS:
            raise ComponentRegistryError(f"unsupported component kind: {kind!r}")
        if not isinstance(component_id, str) or not component_id.strip():
            raise ComponentRegistryError("component_id must be non-empty")
        if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
            raise ComponentRegistryError("component version must be a positive integer")
        operation_tuple = tuple(operations)
        if not operation_tuple or len(operation_tuple) != len(set(operation_tuple)):
            raise ComponentRegistryError(
                "operations must be a non-empty unique sequence"
            )
        if not all(isinstance(item, str) and item for item in operation_tuple):
            raise ComponentRegistryError("operation names must be non-empty strings")
        if not isinstance(input_contract, str) or not input_contract:
            raise ComponentRegistryError("input_contract must be non-empty")
        if not isinstance(output_contract, str) or not output_contract:
            raise ComponentRegistryError("output_contract must be non-empty")
        require_operations(implementation, operation_tuple)
        key = (kind, component_id, version)
        if key in self._entries:
            raise ComponentRegistryError(f"duplicate component registration: {key}")
        registration = ComponentRegistration(
            kind=kind,
            component_id=component_id,
            version=version,
            implementation=implementation,
            operations=operation_tuple,
            input_contract=input_contract,
            output_contract=output_contract,
        )
        self._entries[key] = registration
        return registration

    def resolve(
        self,
        spec: Mapping[str, Any],
        *,
        expected_kind: str,
        required_operations: Sequence[str] = (),
    ) -> ComponentRegistration:
        if set(spec) != {"id", "version"}:
            raise ComponentRegistryError(
                "component binding must contain exactly id and version"
            )
        component_id = spec.get("id")
        version = spec.get("version")
        if not isinstance(component_id, str) or not component_id:
            raise ComponentRegistryError("component binding id must be non-empty")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ComponentRegistryError("component binding version must be an integer")
        key = (expected_kind, component_id, version)
        registration = self._entries.get(key)
        if registration is None:
            raise ComponentRegistryError(f"unregistered component binding: {key}")
        unsupported = sorted(set(required_operations) - set(registration.operations))
        if unsupported:
            raise ComponentRegistryError(
                f"component {component_id!r} lacks required operations: {unsupported}"
            )
        return registration

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": entry.kind,
                "id": entry.component_id,
                "version": entry.version,
                "operations": list(entry.operations),
                "input_contract": entry.input_contract,
                "output_contract": entry.output_contract,
            }
            for _key, entry in sorted(self._entries.items())
        ]

    def clone(self) -> "ComponentRegistry":
        cloned = ComponentRegistry()
        cloned._entries = copy.copy(self._entries)
        return cloned


COMPONENT_REGISTRY = ComponentRegistry()
