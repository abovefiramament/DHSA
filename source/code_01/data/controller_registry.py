"""Code-owned registry for direct dataset-controller calls.

This registry is an adapter boundary, not an experiment protocol.  It exposes
the same controller surface to Site/Performance/Tulu callers while leaving
splits, roles, and mixing in the caller's explicit data configuration.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


DATASET_CONTROLLER_MODULES = {
    "confiqa": "data.confiqa.controller",
    "imdb": "data.imdb.controller",
    "tldr": "data.tldr.controller",
    "tulu": "data.tulu.controller",
}


class DatasetControllerRegistryError(ValueError):
    """Raised when a requested dataset controller is not registered."""


def get_dataset_controller(dataset: str) -> ModuleType:
    """Return one fixed controller module for direct orchestration."""

    if not isinstance(dataset, str) or not dataset:
        raise DatasetControllerRegistryError("dataset must be a non-empty string")
    module_name = DATASET_CONTROLLER_MODULES.get(dataset)
    if module_name is None:
        raise DatasetControllerRegistryError(
            f"unregistered dataset controller: {dataset!r}"
        )
    module = import_module(module_name)
    if not callable(getattr(module, "execute", None)):
        raise DatasetControllerRegistryError(
            f"dataset controller {dataset!r} must expose execute(data_spec, output_dir)"
        )
    return module


def registered_datasets() -> tuple[str, ...]:
    """Return stable registry order for UI/configuration discovery."""

    return tuple(DATASET_CONTROLLER_MODULES)
