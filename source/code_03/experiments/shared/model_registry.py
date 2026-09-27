"""Resolve protocol-owned model identities without machine-local paths."""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_REGISTRY = Path("configs/models/model_registry_20260831_v1.json")
REVISION = re.compile(r"^[0-9a-f]{40}$")


class ModelRegistryError(ValueError):
    """Raised when a public model registration is incomplete or ambiguous."""


def _resolve_registry_path(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ModelRegistryError("model registry path must be repository-relative")
    resolved = (PROJECT_ROOT / candidate).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ModelRegistryError("model registry path escapes repository") from exc
    return resolved


@lru_cache(maxsize=8)
def load_model_registry(
    relative_path: str | Path = DEFAULT_MODEL_REGISTRY,
) -> dict[str, Any]:
    path = _resolve_registry_path(relative_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(f"cannot load model registry: {relative_path}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), dict):
        raise ModelRegistryError("model registry identity is invalid")
    for model_id, entry in payload["entries"].items():
        if not isinstance(model_id, str) or not model_id or not isinstance(entry, Mapping):
            raise ModelRegistryError("model registry entry is invalid")
        kind = entry.get("kind", "policy_model")
        status = entry.get("status")
        source = entry.get("source")
        architecture = entry.get("architecture")
        if not isinstance(source, Mapping):
            raise ModelRegistryError(f"registered model {model_id} has invalid status/source")
        if kind == "api_judge_model":
            if status != "provider_alias" or source.get("provider") != "deepseek_openai_compatible":
                raise ModelRegistryError(f"registered API judge {model_id} is invalid")
            for field in ("model", "base_url", "documentation_url", "revision_policy"):
                if not isinstance(source.get(field), str) or not source[field]:
                    raise ModelRegistryError(f"registered API judge {model_id} lacks {field}")
            if source.get("revision") is not None:
                raise ModelRegistryError(f"registered API judge {model_id} invents a revision")
            continue
        if kind not in {"policy_model", "evaluator_huggingface_model"}:
            raise ModelRegistryError(f"registered model {model_id} kind is unsupported")
        if status not in {"ready", "pending_revision"} or source.get("provider") != "huggingface_hub":
            raise ModelRegistryError(f"registered model {model_id} has invalid Hugging Face source")
        repository = source.get("repository")
        url = source.get("url")
        revision = source.get("revision")
        revision_url = source.get("revision_url")
        if not isinstance(repository, str) or not repository or not isinstance(url, str):
            raise ModelRegistryError(f"registered model {model_id} lacks public source")
        if url != f"https://huggingface.co/{repository}":
            raise ModelRegistryError(f"registered model {model_id} source URL drifts")
        if status == "ready":
            if not isinstance(revision, str) or REVISION.fullmatch(revision) is None:
                raise ModelRegistryError(f"registered model {model_id} revision is invalid")
            if revision_url != f"{url}/tree/{revision}":
                raise ModelRegistryError(f"registered model {model_id} revision URL drifts")
        elif revision is not None or revision_url is not None:
            raise ModelRegistryError(f"pending model {model_id} must not invent a revision")
        if kind == "evaluator_huggingface_model":
            if status != "ready":
                raise ModelRegistryError(f"registered evaluator {model_id} must be ready")
            continue
        if entry.get("use_chat_template") not in {True, False}:
            raise ModelRegistryError(f"registered model {model_id} chat-template flag is invalid")
        if not isinstance(architecture, Mapping) or not all(
            isinstance(architecture.get(key), int) and architecture[key] > 0
            for key in ("layers", "attention_heads_per_layer")
        ):
            raise ModelRegistryError(f"registered model {model_id} architecture is invalid")
        artifact = entry.get("artifact")
        if artifact is not None:
            if (
                not isinstance(artifact, Mapping)
                or artifact.get("kind") != "peft_adapter"
                or not isinstance(artifact.get("base_model_registry_id"), str)
                or not artifact["base_model_registry_id"]
            ):
                raise ModelRegistryError(f"registered model {model_id} artifact is invalid")
            base = payload["entries"].get(artifact["base_model_registry_id"])
            if not isinstance(base, Mapping) or base.get("status") != "ready":
                raise ModelRegistryError(f"registered model {model_id} PEFT base is not ready")
            adapter_base = artifact.get("adapter_base_model_name")
            if adapter_base is not None and (
                not isinstance(adapter_base, str) or not adapter_base
            ):
                raise ModelRegistryError(f"registered model {model_id} PEFT base alias is invalid")
    return copy.deepcopy(payload)


def resolve_registered_model(
    model_id: str,
    *,
    registry_path: str | Path = DEFAULT_MODEL_REGISTRY,
    allow_pending: bool = False,
) -> dict[str, Any]:
    registry = load_model_registry(registry_path)
    entry = registry["entries"].get(model_id)
    if not isinstance(entry, Mapping):
        raise ModelRegistryError(f"unknown registered model: {model_id}")
    if entry.get("kind", "policy_model") != "policy_model":
        raise ModelRegistryError(f"registered model is not a policy model: {model_id}")
    if entry["status"] != "ready" and not allow_pending:
        raise ModelRegistryError(f"registered model is not ready: {model_id}")
    source = entry["source"]
    resolved = {
        "model_registry_id": model_id,
        "model_id": model_id,
        "id": source["repository"],
        "checkpoint": source["repository"],
        "revision": source["revision"],
        "source_url": source["url"],
        "revision_url": source["revision_url"],
        "status": entry["status"],
        "use_chat_template": entry["use_chat_template"],
        "architecture": copy.deepcopy(entry["architecture"]),
    }
    artifact = entry.get("artifact")
    if isinstance(artifact, Mapping):
        base_id = artifact["base_model_registry_id"]
        base = registry["entries"][base_id]
        resolved["artifact"] = {
            "kind": "peft_adapter",
            "base_model_registry_id": base_id,
            "base_checkpoint": base["source"]["repository"],
            "adapter_base_model_name": artifact.get(
                "adapter_base_model_name", base["source"]["repository"]
            ),
            "base_local_path": f"registry://models/{base_id}",
        }
    return resolved


def resolve_registered_evaluator_model(
    model_id: str,
    *,
    registry_path: str | Path = DEFAULT_MODEL_REGISTRY,
) -> dict[str, Any]:
    registry = load_model_registry(registry_path)
    entry = registry["entries"].get(model_id)
    if not isinstance(entry, Mapping):
        raise ModelRegistryError(f"unknown registered evaluator model: {model_id}")
    kind = entry.get("kind")
    source = entry["source"]
    if kind == "evaluator_huggingface_model":
        return {
            "model_registry_id": model_id,
            "kind": kind,
            "model": source["repository"],
            "revision": source["revision"],
            "source_url": source["url"],
            "revision_url": source["revision_url"],
        }
    if kind == "api_judge_model":
        return {
            "model_registry_id": model_id,
            "kind": kind,
            "provider": source["provider"],
            "model": source["model"],
            "base_url": source["base_url"],
            "documentation_url": source["documentation_url"],
            "revision": None,
            "revision_policy": source["revision_policy"],
        }
    raise ModelRegistryError(f"registered model is not an evaluator model: {model_id}")
