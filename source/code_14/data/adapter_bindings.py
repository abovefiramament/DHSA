"""Typed data adapters for the formal Site flow."""

from __future__ import annotations

from experiments.shared.component_registry import ComponentRegistry, ComponentRegistryError
from experiments.shared.contracts import DataMaterializationRequest, DataMaterializationResult


_MODEL_PROVIDER = None


def configure_data_runtime(*, model_provider: object) -> None:
    """Bind the shared model provider; no path or scientific setting is stored here."""

    global _MODEL_PROVIDER
    _MODEL_PROVIDER = model_provider


class ControllerDataAdapter:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset

    def materialize(self, request: DataMaterializationRequest) -> DataMaterializationResult:
        from data.controller_registry import get_dataset_controller

        if request.dataset != self.dataset:
            raise ValueError(f"data adapter {self.dataset!r} received {request.dataset!r}")
        controller = get_dataset_controller(self.dataset)
        kwargs = {"model_provider": _MODEL_PROVIDER} if self.dataset == "imdb" else {}
        manifest = controller.execute(dict(request.data_spec), request.output_dir, **kwargs)
        path = request.output_dir / "dataset_manifest.json"
        if not path.is_file() or manifest.get("dataset") != self.dataset:
            raise ValueError(f"dataset controller {self.dataset!r} did not emit a valid freeze")
        return DataMaterializationResult(dataset_manifest=path.resolve())


def register_data_components(registry: ComponentRegistry) -> None:
    for dataset in ("confiqa", "imdb", "tldr"):
        component_id = f"site_data_{dataset}"
        try:
            registry.register(
                kind="data_adapter",
                component_id=component_id,
                version=1,
                implementation=ControllerDataAdapter(dataset),
                operations=("materialize",),
                input_contract="DataMaterializationRequest/v1",
                output_contract="DataMaterializationResult/v1",
            )
        except ComponentRegistryError as exc:
            if "duplicate component registration" not in str(exc):
                raise
