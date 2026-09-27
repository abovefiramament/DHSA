from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ModelBackend(Protocol):
    """Model backend used by prior probing and scoring."""

    @property
    def model_id(self) -> str:
        ...

    def score_ab(self, prompt: str, option_a: str = "A", option_b: str = "B") -> tuple[float, float]:
        ...


class DatasetBuilderProtocol(Protocol):
    def build(self, seed_csv: Path, out_dir: Path, num_counters: int, seed: int) -> None:
        ...


class PriorProbeProtocol(Protocol):
    def run(self, in_jsonl: Path, out_retained: Path, out_excluded: Path) -> None:
        ...


class SplitterProtocol(Protocol):
    def split(
        self,
        retained_jsonl: Path,
        out_manifest: Path,
        group_key: str,
        seed: int,
        discovery_ratio: float,
        validation_ratio: float,
    ) -> None:
        ...


class PromptRendererProtocol(Protocol):
    def render(self, retained_jsonl: Path, split_manifest: Path, out_dir: Path) -> None:
        ...

