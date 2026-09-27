from __future__ import annotations

from .control import build_control_plan
from .graph import ComponentNode, RCMGraph


def compose_graphs(name: str, graphs: list[RCMGraph]) -> RCMGraph:
    """Merge axis graphs into one factorized RCM graph."""

    axes = {}
    components: list[ComponentNode] = []
    source_graphs: list[str] = []
    for graph in graphs:
        source_graphs.append(graph.name)
        for axis, spec in graph.axes.items():
            if axis in axes:
                raise ValueError(f"Duplicate axis during graph composition: {axis}")
            axes[axis] = spec
        components.extend(graph.components)
    return RCMGraph(
        name=name,
        axes=axes,
        components=components,
        metadata={"source_graphs": source_graphs},
    )


def build_factorized_target_plan(graph: RCMGraph, *, name: str = "rcm_factorized_target", global_alpha: float = 1.0):
    """Default joint controller: raise every target axis in the graph."""

    return build_control_plan(
        graph,
        axis_operations={axis: ["target"] for axis in graph.axes},
        name=name,
        global_alpha=global_alpha,
    )
