"""Create an inspectable onboarding manifest for legacy RCM/K8 evidence.

This command never invents scientific conditions.  It copies declared values,
checks the referenced real files, and assigns either ``verified_import`` or
``attested_legacy``.  The execution scanner accepts only the former.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping


class PositionImportOnboardingError(ValueError):
    """Raised when a legacy declaration cannot form an import manifest."""


def _load(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PositionImportOnboardingError(f"cannot read {field}: {path}") from exc
    if not isinstance(value, dict):
        raise PositionImportOnboardingError(f"{field} must contain one object")
    return value


def _inside(source_dir: Path, relative: Any) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    path = (source_dir / relative).resolve()
    try:
        path.relative_to(source_dir.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def build_onboarding_manifest(
    *, source_dir: Path, declaration: Mapping[str, Any]
) -> dict[str, Any]:
    source = source_dir.expanduser().resolve()
    if not source.is_dir():
        raise PositionImportOnboardingError(f"legacy source directory is missing: {source}")
    missing: list[str] = []
    artifact_kind = declaration.get("artifact_kind")
    if artifact_kind not in {"rcm_scan", "fixed_k8"}:
        missing.append("artifact_kind")
    conditions = declaration.get("scientific_conditions")
    required_conditions = {"method", "dataset", "model_registry_id", "scan_config"}
    if not isinstance(conditions, Mapping):
        missing.extend(f"scientific_conditions.{name}" for name in sorted(required_conditions))
        conditions = {}
    else:
        for name in required_conditions:
            if name not in conditions:
                missing.append(f"scientific_conditions.{name}")
    files = declaration.get("files")
    if not isinstance(files, Mapping):
        files = {}
        missing.append("files")
    required_files = {"sample_scores"} if artifact_kind == "rcm_scan" else {"positions"}
    checked_files: dict[str, Any] = {}
    for name, relative in files.items():
        if relative is None:
            checked_files[str(name)] = None
            continue
        if _inside(source, relative) is None:
            missing.append(f"files.{name}")
        checked_files[str(name)] = relative
    for name in required_files:
        if name not in checked_files:
            missing.append(f"files.{name}")
    evidence = declaration.get("evidence")
    required_evidence = {"model", "data", "scorer", "rcm"}
    checked_evidence: dict[str, Any] = {}
    if not isinstance(evidence, Mapping):
        missing.extend(f"evidence.{name}" for name in sorted(required_evidence))
    else:
        for name in required_evidence:
            relative = evidence.get(name)
            checked_evidence[name] = relative
            if _inside(source, relative) is None:
                missing.append(f"evidence.{name}")
    if artifact_kind == "fixed_k8" and not isinstance(
        declaration.get("selector_sample_ids"), list
    ):
        missing.append("selector_sample_ids")
    if declaration.get("candidate_scope") not in {
        "exact_registered_pool",
        "all_model_heads",
        "shared_source_head_pool",
    }:
        missing.append("candidate_scope")
    missing = sorted(set(missing))
    return {
        "schema_version": 1,
        "manifest_type": "rcm_external_import",
        "eligibility": "verified_import" if not missing else "attested_legacy",
        "artifact_kind": artifact_kind,
        "candidate_scope": declaration.get("candidate_scope"),
        "scientific_conditions": dict(conditions),
        "selector_sample_ids": declaration.get("selector_sample_ids"),
        "files": checked_files,
        "evidence": checked_evidence,
        "onboarding": {
            "source_directory": str(source),
            "missing_or_unverifiable_fields": missing,
            "manual_values_are_not_self-validating": True,
        },
    }


def write_onboarding_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise PositionImportOnboardingError(
            f"onboarding manifest already exists: {output}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Onboard legacy RCM/K8 evidence")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = _load(args.declaration, field="legacy import declaration")
    manifest = build_onboarding_manifest(
        source_dir=args.source_dir, declaration=declaration
    )
    write_onboarding_manifest(args.output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest["eligibility"] == "verified_import" else 2


if __name__ == "__main__":
    raise SystemExit(main())
