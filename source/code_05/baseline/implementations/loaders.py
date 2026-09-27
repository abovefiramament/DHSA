"""Small, model/path-neutral loaders shared by executable method adapters.

Loaders resolve only values supplied by the caller.  They do not discover
checkpoints, datasets, devices, or machine registries on their own.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_rows(path: Path, *, root: Path | None = None) -> list[dict[str, Any]]:
    """Load JSON/JSONL/CSV rows and follow an explicit role artifact reference."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        value: Any = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            artifact = value.get("role_artifact")
            if (
                root is not None
                and isinstance(artifact, Mapping)
                and isinstance(artifact.get("relative_path"), str)
            ):
                return load_rows(Path(root) / str(artifact["relative_path"]), root=root)
            value = value.get("rows", value.get("examples", value.get("predictions", value)))
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            value = list(csv.DictReader(handle))
    else:
        raise ValueError("row artifact must be JSON, JSONL, or CSV")
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError("row artifact must contain object rows")
    return [dict(row) for row in value]


def resolve_model_id(config: Mapping[str, Any], *, mode: str | None = None) -> str:
    """Resolve a caller-declared model identifier without reading a checkpoint path."""

    if not isinstance(config, Mapping):
        raise ValueError("model config must be an object")
    direct_fields = ([f"{mode}_model_id"] if mode else []) + [
        "model_id",
        "id",
        "checkpoint",
    ]
    for field in direct_fields:
        value = config.get(field)
        if isinstance(value, str) and value:
            return value
    for field in ((mode,) if mode else ()) + ("model",):
        value = config.get(field)
        candidates: Sequence[Any]
        if isinstance(value, Mapping):
            preferred = value.get(mode) if mode else None
            candidates = (preferred, value) if preferred is not None else (value,)
        else:
            candidates = (value,)
        for item in candidates:
            if isinstance(item, Mapping):
                for key in ("model_id", "id", "checkpoint"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate:
                        return candidate
            elif isinstance(item, str) and item:
                return item
    raise ValueError("model config needs an external model identifier")


def load_model(provider: Any, config: Mapping[str, Any], *, mode: str) -> Any:
    """Load one model through the injected provider."""

    if provider is None or not callable(getattr(provider, "load", None)):
        raise ValueError("an external model provider with load() is required")
    model = provider.load(resolve_model_id(config, mode=mode), mode=mode)
    value: Any = config
    nested = config.get("model") if isinstance(config, Mapping) else None
    if isinstance(nested, Mapping):
        preferred = nested.get(mode)
        value = preferred if isinstance(preferred, Mapping) else nested
    use_chat_template = (
        value.get("use_chat_template", False)
        if isinstance(value, Mapping)
        else False
    )
    configure = getattr(model, "configure_input", None)
    if callable(configure):
        configure(use_chat_template=bool(use_chat_template))
    return model


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Load one JSON object for a manifest or configuration artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain one object: {path}")
    return value
