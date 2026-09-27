"""Validate the formal handoff from frozen legacy evidence to the new matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REGISTRY = Path("configs/evidence/evidence_registry_v1.json")


class FrozenHandoffError(ValueError):
    """Raised when a frozen registration is ambiguous or names a current invalid cell."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenHandoffError(f"cannot read frozen evidence registry: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise FrozenHandoffError("frozen evidence registry schema_version must be 1")
    return value


def _forbid_new_hash_contract(value: Any, path: str = "registry") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if (
                lowered == "hash"
                or lowered.startswith("hash_")
                or lowered.endswith("_hash")
                or "sha256" in lowered
                or lowered == "digest"
                or lowered.endswith("_digest")
            ):
                raise FrozenHandoffError(f"new frozen registry cannot contain hash field: {path}.{key}")
            _forbid_new_hash_contract(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbid_new_hash_contract(child, f"{path}[{index}]")


def validate_frozen_handoff(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    from experiments.performance.performance_protocol_binding import iter_declared_performance_cells
    from experiments.site.site_protocol_binding import iter_bound_site_jobs

    registry = _load(path)
    _forbid_new_hash_contract(registry)
    packages = registry.get("packages")
    registrations = registry.get("frozen_registrations")
    rcm_sources = registry.get("shared_rcm_sources")
    if not isinstance(packages, Mapping) or not isinstance(registrations, list):
        raise FrozenHandoffError("packages and frozen_registrations are required")
    if not isinstance(rcm_sources, list):
        raise FrozenHandoffError("shared_rcm_sources must be a list")

    site_rows = [dict(row) for row in iter_bound_site_jobs()]
    performance_rows = [dict(row) for row in iter_declared_performance_cells()]
    site_by_id = {row["job_id"]: row for row in site_rows}
    performance_by_id = {row["job_id"]: row for row in performance_rows}
    registration_ids: set[str] = set()
    resolved: dict[str, dict[str, Any]] = {}

    for registration in registrations:
        if not isinstance(registration, Mapping):
            raise FrozenHandoffError("frozen registration must be an object")
        registration_id = registration.get("registration_id")
        package_id = registration.get("package_id")
        experiment = registration.get("experiment")
        if not isinstance(registration_id, str) or registration_id in registration_ids:
            raise FrozenHandoffError("frozen registration_id is invalid or duplicated")
        registration_ids.add(registration_id)
        if package_id not in packages:
            raise FrozenHandoffError(f"unknown frozen package: {package_id}")
        if experiment == "site":
            selector = registration.get("target_selector")
            if not isinstance(selector, Mapping):
                raise FrozenHandoffError(f"Site registration {registration_id} needs target_selector")
            targets = [
                row["job_id"]
                for row in site_rows
                if all(str(row.get(field)) == str(expected) for field, expected in selector.items())
            ]
        elif experiment == "performance":
            targets = list(registration.get("target_job_ids", []))
            if not targets or any(target not in performance_by_id for target in targets):
                raise FrozenHandoffError(f"Performance registration {registration_id} has unknown target")
        else:
            raise FrozenHandoffError(f"unknown experiment in {registration_id}")
        if len(targets) != registration.get("expected_target_count"):
            raise FrozenHandoffError(f"target count drift for {registration_id}")
        for target in targets:
            if target in resolved:
                raise FrozenHandoffError(f"multiple frozen registrations target {target}")
            resolved[target] = dict(registration)

    primary_targets = {
        target for target, registration in resolved.items() if registration.get("primary") is True
    }
    if any(
        resolved[target].get("eligibility") != "exact_cell" for target in primary_targets
    ):
        raise FrozenHandoffError("only exact_cell frozen evidence can be primary")

    source_ids: set[str] = set()
    rcm_consumers: set[str] = set()
    for source in rcm_sources:
        if not isinstance(source, Mapping):
            raise FrozenHandoffError("shared RCM source must be an object")
        source_id = source.get("rcm_source_id")
        if not isinstance(source_id, str) or source_id in source_ids:
            raise FrozenHandoffError("shared RCM source id is invalid or duplicated")
        source_ids.add(source_id)
        if source.get("package_id") not in packages:
            raise FrozenHandoffError(f"shared RCM source {source_id} has unknown package")
        for job_id in source.get("consumer_job_ids", []):
            row = performance_by_id.get(job_id)
            if row is None:
                raise FrozenHandoffError(f"shared RCM source {source_id} has unknown consumer")
            if row.get("dataset") != source.get("dataset"):
                raise FrozenHandoffError(f"shared RCM source {source_id} crosses datasets")
            if row.get("baseline", {}).get("train_base") != "sft":
                raise FrozenHandoffError(f"shared RCM source {source_id} targets non-SFT localization")
            if row.get("models", {}).get("sft", {}).get("model_registry_id") != source.get(
                "model_registry_id"
            ):
                raise FrozenHandoffError(f"shared RCM source {source_id} crosses model identities")
            rcm_consumers.add(job_id)

    return {
        "package_count": len(packages),
        "registered_target_count": len(resolved),
        "site_exact_cell_primary_count": sum(
            1
            for target in primary_targets
            if target in site_by_id
        ),
        "site_exact_stage_count": sum(
            1
            for target, registration in resolved.items()
            if target in site_by_id and registration.get("eligibility") == "exact_stage"
        ),
        "performance_comparable_count": sum(
            1
            for target, registration in resolved.items()
            if target in performance_by_id
            and registration.get("eligibility") == "comparable_external"
        ),
        "shared_rcm_source_count": len(rcm_sources),
        "shared_rcm_consumer_count": len(rcm_consumers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen evidence handoff registrations")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    print(json.dumps(validate_frozen_handoff(args.registry), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
