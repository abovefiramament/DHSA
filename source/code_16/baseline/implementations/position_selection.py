"""Shared, model-free position selection primitives.

The selector backends consume rows produced by an external scorer/probe.  They
never load a model, resolve a machine path, or decide a dataset split.  This
keeps the selection policy reusable by Site and Performance while leaving the
model-facing scan in the injected execution layer.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SelectionResult:
    method: str
    positions: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]


def _as_float(value: Any, *, field: str, default: float | None = None) -> float:
    if value in (None, "") and default is not None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"position row has a non-numeric {field}: {value!r}") from exc


def _as_int(value: Any, *, field: str, default: int | None = None) -> int:
    if value in (None, "") and default is not None:
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"position row has a non-integer {field}: {value!r}") from exc


def _component_id(row: Mapping[str, Any]) -> str:
    value = row.get("component_id") or row.get("head_id") or row.get("position")
    if not isinstance(value, str) or not value.strip():
        layer = row.get("layer_idx", row.get("layer"))
        head = row.get("head_idx", row.get("head"))
        if layer not in (None, "") and head not in (None, ""):
            value = f"L{int(layer)}.attn.h{int(head)}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("position row requires component_id (or layer/head fields)")
    return value.strip()


def load_rows(source: Path | Sequence[Mapping[str, Any]], *, root: Path | None = None) -> list[dict[str, Any]]:
    """Load candidate rows from CSV/JSON/JSONL or an already materialized list."""

    if not isinstance(source, (str, Path)):
        return [dict(row) for row in source if isinstance(row, Mapping)]
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows: Any = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(rows, Mapping):
            artifact = rows.get("role_artifact")
            if root is not None and isinstance(artifact, Mapping):
                relative = artifact.get("relative_path")
                if isinstance(relative, str) and relative:
                    return load_rows(root / relative, root=root)
            rows = rows.get("rows", rows.get("candidates", rows))
    else:
        raise ValueError(f"unsupported position source format: {path.suffix}")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("position source must resolve to a list of object rows")
    return [dict(row) for row in rows]


def _unique_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item = dict(row)
        component = _component_id(item)
        if component in seen:
            raise ValueError(f"duplicate position candidate: {component}")
        seen.add(component)
        item["component_id"] = component
        output.append(item)
    return output


def _score(row: Mapping[str, Any], fields: Sequence[str]) -> float:
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return _as_float(row[field], field=field)
    raise ValueError(f"position row has none of the score fields: {list(fields)}")


def _layer(row: Mapping[str, Any]) -> int:
    component = _component_id(row)
    if component.startswith("L") and ".attn.h" in component:
        return int(component[1:].split(".attn.h", 1)[0])
    return _as_int(row.get("layer_idx", row.get("layer")), field="layer_idx")


def _head(row: Mapping[str, Any]) -> int:
    component = _component_id(row)
    if component.startswith("L") and ".attn.h" in component:
        return int(component.split(".attn.h", 1)[1])
    return _as_int(row.get("head_idx", row.get("head")), field="head_idx")


def _ordered(rows: Sequence[Mapping[str, Any]], *, descending: bool, score_fields: Sequence[str]) -> list[dict[str, Any]]:
    normalized = _unique_rows(rows)
    return sorted(
        normalized,
        key=lambda row: (
            -_score(row, score_fields) if descending else _score(row, score_fields),
            _layer(row),
            _head(row),
            _component_id(row),
        ),
    )


def _signed(row: Mapping[str, Any], score_fields: Sequence[str]) -> float:
    # Role is assigned from this candidate's own measured effect.  A parent
    # layer label or an upstream bank label must not leak into head roles.
    return _score(row, score_fields)


def _oriented_ci(row: Mapping[str, Any], *, positive: bool, score_fields: Sequence[str]) -> float:
    if positive:
        return _as_float(row.get("ci95_low"), field="ci95_low", default=_score(row, score_fields))
    return -_as_float(row.get("ci95_high"), field="ci95_high", default=_score(row, score_fields))


def select_rcm(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    positive_count: int,
    negative_count: int,
    parent_layers_per_role: int = 4,
    score_fields: Sequence[str] = ("selection_score", "mean_score_delta", "effect"),
    signed_shortfall: str = "error",
) -> SelectionResult:
    """Select signed RCM heads using layer-beam then head-refinement policy."""

    if method not in {"rcm_zero", "rcm_patch"}:
        raise ValueError(f"unsupported RCM method: {method}")
    if positive_count <= 0 or negative_count <= 0 or parent_layers_per_role <= 0:
        raise ValueError("RCM quotas and parent layer count must be positive")
    candidates = _unique_rows(rows)
    if signed_shortfall not in {"error", "strict_signed_partial"}:
        raise ValueError(f"unsupported RCM head shortfall policy: {signed_shortfall}")
    partial = signed_shortfall == "strict_signed_partial"
    if partial:
        score_fields = ("mean_score_delta", "effect", *score_fields)
        if any(not math.isfinite(_signed(row, score_fields)) for row in candidates):
            raise ValueError("RCM candidate effects must be finite")
    if not partial and len(candidates) < positive_count + negative_count:
        raise ValueError("RCM candidate pool is smaller than the signed quota")

    positive_pool = [row for row in candidates if _signed(row, score_fields) >= 0.0]
    if partial:
        positive_pool = [row for row in positive_pool if _signed(row, score_fields) > 0.0]
    negative_pool = [row for row in candidates if _signed(row, score_fields) < 0.0]
    if not partial and (len(positive_pool) < positive_count or len(negative_pool) < negative_count):
        raise ValueError("RCM candidate pool cannot form both signed roles")

    # The model scanner has already performed the registered layer scan and
    # emitted heads only from the union of its positive/negative layer beams.
    # The pure selector therefore ranks those head effects directly; it must
    # not perform a second, different layer search.
    positive_layers = sorted({_layer(row) for row in positive_pool})
    negative_layers = sorted({_layer(row) for row in negative_pool})
    positive_heads = positive_pool
    negative_heads = negative_pool
    positive = sorted(
        positive_heads,
        key=lambda row: (-_oriented_ci(row, positive=True, score_fields=score_fields), _layer(row), _head(row)),
    )[:positive_count]
    negative = sorted(
        negative_heads,
        key=lambda row: (-_oriented_ci(row, positive=False, score_fields=score_fields), _layer(row), _head(row)),
    )[:negative_count]
    if not partial and (len(positive) != positive_count or len(negative) != negative_count):
        raise ValueError("RCM layer refinement cannot satisfy signed head quotas")
    if not positive and not negative:
        raise ValueError("RCM candidate pool has no non-zero signed heads")

    selected: list[dict[str, Any]] = []
    for role, members in (("target_support", positive), ("competitor_support", negative)):
        for rank, row in enumerate(members, start=1):
            selected.append(
                {
                    **dict(row),
                    "component_id": _component_id(row),
                    "layer_idx": _layer(row),
                    "head_idx": _head(row),
                    "selection_role": role,
                    "rank": rank,
                    "ranking_score": _oriented_ci(
                        row, positive=role == "target_support", score_fields=score_fields
                    ),
                    "parent_layer_beam": positive_layers if role == "target_support" else negative_layers,
                    "state_contrast": "zero_missing_to_native_present" if method == "rcm_zero" else "reference_to_target_prototype",
                }
            )
    trace = tuple(
        {
            "component_id": _component_id(row),
            "layer_idx": _layer(row),
            "head_idx": _head(row),
            "selected": _component_id(row) in {_component_id(item) for item in selected},
            "effect": _score(row, score_fields),
            "role": "zero_effect" if partial and _signed(row, score_fields) == 0 else "target_support" if _signed(row, score_fields) >= 0.0 else "competitor_support",
        }
        for row in candidates
    )
    return SelectionResult(method, tuple(selected), tuple(candidates), trace)


def select_iti(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    score_fields: Sequence[str] = ("ranking_score", "mean_grouped_twofold_heldout_accuracy", "selection_score"),
) -> SelectionResult:
    """Select ITI heads by grouped two-fold held-out probe score."""

    if count <= 0:
        raise ValueError("ITI count must be positive")
    candidates = _unique_rows(rows)
    ranked = _ordered(candidates, descending=True, score_fields=score_fields)
    if len(ranked) < count:
        raise ValueError("ITI candidate pool is smaller than the requested count")
    selected = [
        {
            **row,
            "component_id": _component_id(row),
            "layer_idx": _layer(row),
            "head_idx": _head(row),
            "selection_role": "unsigned",
            "rank": rank,
            "ranking_score": _score(row, score_fields),
            "score_name": "mean_grouped_twofold_heldout_accuracy",
        }
        for rank, row in enumerate(ranked[:count], start=1)
    ]
    selected_ids = {_component_id(row) for row in selected}
    trace = tuple(
        {
            "component_id": _component_id(row),
            "layer_idx": _layer(row),
            "head_idx": _head(row),
            "selected": _component_id(row) in selected_ids,
            "effect": _score(row, score_fields),
            "role": "unsigned",
        }
        for row in ranked
    )
    return SelectionResult("iti", tuple(selected), tuple(ranked), trace)


def select_random(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> SelectionResult:
    """Sample a fixed-size head set without replacement."""

    if count <= 0:
        raise ValueError("Random count must be positive")
    candidates = _unique_rows(rows)
    if len(candidates) < count:
        raise ValueError("Random candidate pool is smaller than the requested count")
    sampled = random.Random(int(seed)).sample(candidates, count)
    selected = [
        {
            **row,
            "component_id": _component_id(row),
            "layer_idx": _layer(row),
            "head_idx": _head(row),
            "selection_role": "unsigned",
            "rank": rank,
            "seed": int(seed),
        }
        for rank, row in enumerate(sampled, start=1)
    ]
    selected_ids = {_component_id(row) for row in selected}
    trace = tuple(
        {
            "component_id": _component_id(row),
            "layer_idx": _layer(row),
            "head_idx": _head(row),
            "selected": _component_id(row) in selected_ids,
            "effect": None,
            "role": "unsigned",
        }
        for row in candidates
    )
    return SelectionResult("random", tuple(selected), tuple(candidates), trace)


def write_selection_artifacts(result: SelectionResult, *, output_dir: Path) -> tuple[Path, Path]:
    """Write selected positions plus the complete ranking trace."""

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_manifest.jsonl"
    execution_path = output_dir / "backend_execution_manifest.json"
    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in result.positions:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    execution_path.write_text(
        json.dumps(
            {
                "method": result.method,
                "candidate_count": len(result.candidates),
                "selected_count": len(result.positions),
                "positions": [dict(row) for row in result.positions],
                "trace": [dict(row) for row in result.trace],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return candidate_path, execution_path


def plan_parameters(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the method parameters without interpreting machine settings."""

    parameters = plan.get("parameters") if isinstance(plan, Mapping) else None
    if isinstance(parameters, Mapping):
        return parameters
    return plan


def run_request(request: Any, result: SelectionResult) -> Any:
    """Persist a selector result and adapt it to the shared runner result type."""

    candidate_path, execution_path = write_selection_artifacts(
        result, output_dir=Path(request.output_dir)
    )
    # Imported lazily so the model-free selectors remain usable without the
    # shared orchestration package.
    from experiments.shared.contracts import PositionSearchResult

    return PositionSearchResult(
        candidate_manifest=candidate_path,
        execution_manifest=execution_path,
    )
