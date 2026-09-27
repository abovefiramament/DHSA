"""Protocol-owned prompt registry resolution for the Site framework.

Prompt text is scientific configuration, never a machine-registry value. Dataset
controllers choose named entries and preserve their rendered bytes unchanged.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .controller_api import ControllerError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def _load_registry(relative_path: str) -> dict[str, Any]:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ControllerError("prompt registry path must be repository-relative")
    path = (PROJECT_ROOT / candidate).resolve()
    if PROJECT_ROOT not in path.parents:
        raise ControllerError("prompt registry path escapes project root")
    if not path.is_file():
        raise ControllerError(f"prompt registry does not exist: {candidate}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        raise ControllerError("prompt registry must contain an entries object")
    return payload


def resolve_prompt(
    prompt_spec: Mapping[str, Any],
    prompt_id: str,
    values: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Render one registered prompt without normalization or trimming."""

    registry_path = prompt_spec.get("registry_path")
    if not isinstance(registry_path, str) or not registry_path:
        raise ControllerError("data.prompt_registry.registry_path must be non-empty")
    registry = _load_registry(registry_path)
    entries = registry["entries"]
    entry = entries.get(prompt_id)
    if not isinstance(entry, Mapping):
        raise ControllerError(f"unknown registered prompt id: {prompt_id}")
    template = entry.get("template")
    required = entry.get("required_fields")
    if not isinstance(template, str) or not isinstance(required, list):
        raise ControllerError(f"registered prompt {prompt_id!r} is malformed")
    missing = [field for field in required if field not in values]
    if missing:
        raise ControllerError(f"registered prompt {prompt_id!r} is missing fields: {missing}")
    try:
        rendered = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise ControllerError(f"unable to render registered prompt {prompt_id!r}") from exc
    if entry.get("preserve_exact") is not True:
        raise ControllerError(f"registered prompt {prompt_id!r} must preserve exact bytes")
    return rendered, {
        "registry_id": registry.get("registry_id"),
        "registry_path": registry_path,
        "prompt_id": prompt_id,
        "rendering": entry.get("rendering"),
    }