from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from screscomp.data import dump_csv, dump_json, load_csv

from .events import AxisSpec, default_axis_specs


@dataclass(slots=True)
class ComponentNode:
    axis: str
    component_id: str
    layer_idx: int
    component_type: str
    sign: str
    weight: float
    selection_score: float
    selection_metric: str
    timing: str
    alpha: float
    source_path: str = ""
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComponentRelation:
    component_id: str
    layer_idx: int
    component_type: str
    axes: str
    signs: str
    timings: str
    incident_edges: int
    relation_kind: str
    needs_arbitration: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AxisRelation:
    axis_a: str
    axis_b: str
    shared_components: int
    same_sign_shared: int
    opposite_sign_shared: int
    union_components: int
    overlap_jaccard: float
    relation_kind: str
    shared_component_ids: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RCMGraph:
    name: str
    axes: dict[str, AxisSpec]
    components: list[ComponentNode]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "axes": {name: spec.to_dict() for name, spec in self.axes.items()},
            "components": [node.to_dict() for node in self.components],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RCMGraph":
        axes = {name: AxisSpec(**spec) for name, spec in data.get("axes", {}).items()}
        components = [ComponentNode(**node) for node in data.get("components", [])]
        return cls(
            name=str(data.get("name", "rcm_graph")),
            axes=axes,
            components=components,
            metadata=dict(data.get("metadata", {})),
            version=str(data.get("version", "1.0")),
        )

    def component_rows(self) -> list[dict[str, Any]]:
        return [node.to_dict() for node in self.components]

    def by_axis(self, axis: str) -> list[ComponentNode]:
        return [node for node in self.components if node.axis == axis]

    def by_component(self) -> dict[str, list[ComponentNode]]:
        grouped: dict[str, list[ComponentNode]] = {}
        for node in self.components:
            grouped.setdefault(node.component_id, []).append(node)
        return grouped

    def component_relation_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for component_id, edges in sorted(self.by_component().items()):
            axes = sorted({edge.axis for edge in edges})
            signs = sorted({edge.sign for edge in edges})
            timings = sorted({edge.timing for edge in edges})
            if len(axes) == 1:
                relation_kind = "single_axis"
            elif len(signs) == 1:
                relation_kind = "shared_aligned_sign"
            else:
                relation_kind = "shared_opposite_sign"
            first = edges[0]
            rows.append(
                ComponentRelation(
                    component_id=component_id,
                    layer_idx=first.layer_idx,
                    component_type=first.component_type,
                    axes=",".join(axes),
                    signs=",".join(signs),
                    timings=",".join(timings),
                    incident_edges=len(edges),
                    relation_kind=relation_kind,
                    needs_arbitration=len(axes) > 1,
                ).to_dict()
            )
        return rows

    def axis_relation_rows(self) -> list[dict[str, Any]]:
        axes = sorted(self.axes)
        by_axis_component = {
            axis: {node.component_id: node for node in self.by_axis(axis)}
            for axis in axes
        }
        rows: list[dict[str, Any]] = []
        for idx, axis_a in enumerate(axes):
            comps_a = by_axis_component[axis_a]
            for axis_b in axes[idx + 1 :]:
                comps_b = by_axis_component[axis_b]
                shared = sorted(set(comps_a) & set(comps_b))
                union_size = len(set(comps_a) | set(comps_b))
                same = sum(1 for cid in shared if comps_a[cid].sign == comps_b[cid].sign)
                opposite = len(shared) - same
                if not shared:
                    relation_kind = "disjoint"
                elif opposite and same:
                    relation_kind = "mixed_overlap"
                elif opposite:
                    relation_kind = "opposite_sign_overlap"
                else:
                    relation_kind = "same_sign_overlap"
                rows.append(
                    AxisRelation(
                        axis_a=axis_a,
                        axis_b=axis_b,
                        shared_components=len(shared),
                        same_sign_shared=same,
                        opposite_sign_shared=opposite,
                        union_components=union_size,
                        overlap_jaccard=(len(shared) / union_size) if union_size else 0.0,
                        relation_kind=relation_kind,
                        shared_component_ids=",".join(shared),
                    ).to_dict()
                )
        return rows


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return int(float(value))


def _axis_rows(rows: list[dict[str, str]], axis: str) -> list[dict[str, str]]:
    selected = [row for row in rows if str(row.get("axis", axis)) == axis]
    return selected or rows


def _select_signed_rows(
    rows: list[dict[str, str]],
    *,
    top_up_k: int,
    top_down_k: int,
) -> list[tuple[str, dict[str, str]]]:
    usable = [row for row in rows if row.get("component_id") or (row.get("layer_idx") and row.get("component_type"))]
    positive = sorted(
        [row for row in usable if _float(row, "selection_score") >= 0.0],
        key=lambda row: (-_float(row, "selection_score"), _int(row, "selection_rank", 10**9)),
    )
    negative = sorted(
        [row for row in usable if _float(row, "selection_score") < 0.0],
        key=lambda row: (_float(row, "selection_score"), _int(row, "selection_rank", 10**9)),
    )
    output: list[tuple[str, dict[str, str]]] = []
    output.extend(("up", row) for row in positive[:top_up_k])
    output.extend(("down", row) for row in negative[:top_down_k])
    return output


def build_graph_from_summaries(
    axis_summary_csvs: dict[str, Path],
    *,
    name: str = "rcm_graph",
    top_up_k: int = 4,
    top_down_k: int = 4,
    axis_timings: dict[str, str] | None = None,
    axis_alphas: dict[str, float] | None = None,
) -> RCMGraph:
    axis_specs = default_axis_specs()
    axis_timings = axis_timings or {}
    axis_alphas = axis_alphas or {}
    components: list[ComponentNode] = []
    selected_axes: dict[str, AxisSpec] = {}

    for axis, path in axis_summary_csvs.items():
        if axis not in axis_specs:
            raise ValueError(f"Unknown axis: {axis}")
        selected_axes[axis] = axis_specs[axis]
        spec = axis_specs[axis]
        rows = _axis_rows(load_csv(path), axis)
        timing = axis_timings.get(axis, spec.default_timing)
        alpha = axis_alphas.get(axis, spec.default_alpha)
        for rank, (sign, row) in enumerate(
            _select_signed_rows(rows, top_up_k=top_up_k, top_down_k=top_down_k),
            start=1,
        ):
            layer_idx = _int(row, "layer_idx")
            component_type = str(row.get("component_type", ""))
            component_id = str(row.get("component_id") or f"L{layer_idx}.{component_type}")
            selection_score = _float(row, "selection_score")
            components.append(
                ComponentNode(
                    axis=axis,
                    component_id=component_id,
                    layer_idx=layer_idx,
                    component_type=component_type,
                    sign=sign,
                    weight=abs(selection_score),
                    selection_score=selection_score,
                    selection_metric=str(row.get("selection_metric") or spec.selection_metric),
                    timing=timing,
                    alpha=alpha,
                    source_path=str(path),
                    rank=rank,
                    metadata={
                        key: row[key]
                        for key in (
                            "pair_name",
                            "selection_rule",
                            "mean_context_only",
                            "mean_prior_only",
                            "mean_both",
                            "mean_neither",
                            "mean_form_score",
                            "mean_commitment_score",
                            "mean_source_score",
                            "mean_joint_score",
                            "mean_event_target_score",
                            "mean_event_contrast_score",
                            "mean_event_margin_score",
                            "baseline_event_margin_score",
                            "direction",
                            "signed_edge_sign",
                            "signed_edge_source_direction",
                            "target_attractor",
                            "contrast_attractor",
                            "selection_pair_names",
                            "selection_sign_rank",
                            "selection_mean_min_CI",
                            "selection_mean_C_plus_I",
                            "selection_frac_both_positive",
                            "ck_mean_min_CI",
                            "ck_mean_C_plus_I",
                            "ck_frac_both_positive",
                            "edge_role",
                            "sign_mode",
                            "target_attractor",
                            "contrast_attractor",
                            "ck_high_attractor",
                            "ck_low_attractor",
                            "ck_sign_interpretation",
                            "source_component_source",
                        )
                        if key in row
                    },
                )
            )

    return RCMGraph(
        name=name,
        axes=selected_axes,
        components=components,
        metadata={
            "top_up_k": top_up_k,
            "top_down_k": top_down_k,
            "axis_summary_csvs": {axis: str(path) for axis, path in axis_summary_csvs.items()},
        },
    )


def save_graph(graph: RCMGraph, path: Path) -> None:
    dump_json(path, graph.to_dict())


def load_graph(path: Path) -> RCMGraph:
    return RCMGraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_component_rows(graph: RCMGraph, path: Path) -> None:
    dump_csv(path, graph.component_rows())


def save_component_relation_rows(graph: RCMGraph, path: Path) -> None:
    dump_csv(path, graph.component_relation_rows())


def save_axis_relation_rows(graph: RCMGraph, path: Path) -> None:
    dump_csv(path, graph.axis_relation_rows())
