from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.cli.score_component_competition import _component_id
from screscomp.cli.score_steering_utility import _load_summary_rows
from screscomp.data import dump_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export CK-style controlled-assay component rankings as an RCM axis summary. "
            "This lets source components discovered for CK open-generation control be re-tested "
            "under the RCM axis-audit readout protocol."
        )
    )
    p.add_argument("--component_summary_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--axis", default="source_identity")
    p.add_argument("--pair_name", default="ck_e1_content_context_vs_prior")
    p.add_argument("--selection_pairs", default="e1_content,e1_content_swapped")
    p.add_argument("--max_components", type=int, default=16)
    p.add_argument("--max_up_components", type=int, default=None)
    p.add_argument("--max_down_components", type=int, default=None)
    p.add_argument(
        "--sign_mode",
        default="signed_ci",
        choices=["signed_ci", "positive_up", "prior_support_down", "duplicate_positive", "paired_positive"],
        help=(
            "How to map CK C/I scores to RCM signed edges. `signed_ci` is the bidirectional "
            "default for source_identity: positive CK mean_min_CI is prior-support/down, "
            "negative CK mean_min_CI is context-support/up. `paired_positive` exports the "
            "same top positive CK components as paired context-up/prior-down operation edges."
        ),
    )
    return p.parse_args()


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _rank_signed_components(
    summary_rows: list[dict[str, str]],
    *,
    selection_pairs: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    pair_set = set(selection_pairs)
    for row in summary_rows:
        if row.get("split") != "discovery":
            continue
        if row.get("pair_name") not in pair_set:
            continue
        grouped[(int(row["layer_idx"]), row["component_type"])].append(row)

    candidates: list[dict[str, Any]] = []
    for (layer_idx, component_type), rows in grouped.items():
        mean_min_ci = mean(_float(row, "mean_min_CI") for row in rows)
        mean_c_plus_i = mean(_float(row, "mean_C_plus_I") for row in rows)
        frac_both = mean(_float(row, "frac_both_positive") for row in rows)
        candidates.append(
            {
                "layer_idx": layer_idx,
                "component_type": component_type,
                "component_id": _component_id(layer_idx, component_type),
                "ck_mean_min_CI": mean_min_ci,
                "ck_mean_C_plus_I": mean_c_plus_i,
                "ck_frac_both_positive": frac_both,
                "selection_rows": len(rows),
                "selection_pair_names": ",".join(sorted({str(row["pair_name"]) for row in rows})),
            }
        )
    if not candidates:
        raise ValueError("No discovery component summary rows found for CK axis export.")
    return candidates


def _signed_score(candidate: dict[str, Any], sign_mode: str) -> float:
    ck_score = float(candidate["ck_mean_min_CI"])
    if sign_mode == "signed_ci":
        return -ck_score
    if sign_mode == "positive_up":
        return ck_score
    if sign_mode == "prior_support_down":
        return -abs(ck_score)
    raise ValueError(f"Unsupported single-score sign mode: {sign_mode}")


def _signed_rows(
    candidates: list[dict[str, Any]],
    *,
    sign_mode: str,
    max_up_components: int,
    max_down_components: int,
) -> list[tuple[dict[str, Any], float, str, int]]:
    if sign_mode in {"duplicate_positive", "paired_positive"}:
        ranked = sorted(
            [row for row in candidates if float(row["ck_mean_min_CI"]) > 0.0],
            key=lambda row: (
                -float(row["ck_mean_min_CI"]),
                -float(row["ck_frac_both_positive"]),
                -float(row["ck_mean_C_plus_I"]),
                int(row["layer_idx"]),
                str(row["component_type"]),
            ),
        )
        up_role = "ck_duplicate_up_candidate"
        down_role = "ck_duplicate_down_candidate"
        if sign_mode == "paired_positive":
            up_role = "ck_context_up_paired_operation"
            down_role = "ck_prior_down_paired_operation"
        output: list[tuple[dict[str, Any], float, str, int]] = []
        for rank, row in enumerate(ranked[:max_up_components], start=1):
            output.append(
                (row, abs(float(row["ck_mean_min_CI"])), up_role, rank)
            )
        for rank, row in enumerate(ranked[:max_down_components], start=1):
            output.append(
                (row, -abs(float(row["ck_mean_min_CI"])), down_role, rank)
            )
        return output

    scored = [(row, _signed_score(row, sign_mode)) for row in candidates]
    up = sorted(
        [(row, score) for row, score in scored if score > 0.0],
        key=lambda item: (
            -item[1],
            -float(item[0]["ck_frac_both_positive"]),
            -abs(float(item[0]["ck_mean_C_plus_I"])),
            int(item[0]["layer_idx"]),
            str(item[0]["component_type"]),
        ),
    )
    down = sorted(
        [(row, score) for row, score in scored if score < 0.0],
        key=lambda item: (
            item[1],
            -float(item[0]["ck_frac_both_positive"]),
            -abs(float(item[0]["ck_mean_C_plus_I"])),
            int(item[0]["layer_idx"]),
            str(item[0]["component_type"]),
        ),
    )
    output: list[tuple[dict[str, Any], float, str, int]] = []
    output.extend(
        (row, score, "ck_context_up_candidate", rank)
        for rank, (row, score) in enumerate(up[:max_up_components], start=1)
    )
    output.extend(
        (row, score, "ck_prior_down_candidate", rank)
        for rank, (row, score) in enumerate(down[:max_down_components], start=1)
    )
    return output


def _source_row(
    row: dict[str, Any],
    *,
    axis: str,
    pair_name: str,
    sign_mode: str,
    edge_role: str,
    selection_score: float,
    sign_rank: int,
    rank: int,
) -> dict[str, Any]:
    if sign_mode == "paired_positive":
        sign_interpretation = (
            "Paired CK operation edge: the component is selected once by positive CK mean_min_CI. "
            "The context-up operation applies the context-minus-prior/base delta; the prior-down "
            "operation applies the reverse delta. The two rows are paired operations, not two "
            "independent component discoveries."
        )
    elif sign_mode == "signed_ci":
        sign_interpretation = (
            "For source_identity, selection_score > 0 is context/up and selection_score < 0 is "
            "prior/down. In signed_ci mode, selection_score = -ck_mean_min_CI."
        )
    else:
        sign_interpretation = (
            "For source_identity, this export follows the requested sign_mode. Positive rows become "
            "up edges and negative rows become down edges in the RCM graph builder."
        )
    return {
        "group": f"{axis}:ck_component:{row['component_id']}:{edge_role}",
        "split": "discovery",
        "pair_name": pair_name,
        "axis": axis,
        "layer_idx": row["layer_idx"],
        "component_type": row["component_type"],
        "component_id": row["component_id"],
        "n": row.get("selection_rows", ""),
        "selection_score": selection_score,
        "selection_metric": "ck_signed_mean_min_CI",
        "selection_rule": f"ck_{sign_mode}_rank_by_signed_CI_per_sign",
        "selection_rank": rank,
        "selection_sign_rank": sign_rank,
        "selection_pair_names": row.get("selection_pair_names", ""),
        "selection_mean_min_CI": row.get("ck_mean_min_CI", ""),
        "selection_mean_C_plus_I": row.get("ck_mean_C_plus_I", ""),
        "selection_frac_both_positive": row.get("ck_frac_both_positive", ""),
        "ck_mean_min_CI": row.get("ck_mean_min_CI", ""),
        "ck_mean_C_plus_I": row.get("ck_mean_C_plus_I", ""),
        "ck_frac_both_positive": row.get("ck_frac_both_positive", ""),
        "edge_role": edge_role,
        "sign_mode": sign_mode,
        "target_attractor": "context",
        "contrast_attractor": "prior",
        "ck_high_attractor": "prior",
        "ck_low_attractor": "context",
        "ck_sign_interpretation": sign_interpretation,
        "source_component_source": "ck_controlled_assay_component_competition_summary",
    }


def main() -> None:
    args = parse_args()
    candidates = _rank_signed_components(
        _load_summary_rows(args.component_summary_csv),
        selection_pairs=_parse_csv(args.selection_pairs),
    )
    max_up = args.max_up_components if args.max_up_components is not None else args.max_components
    max_down = args.max_down_components if args.max_down_components is not None else args.max_components
    signed = _signed_rows(
        candidates,
        sign_mode=args.sign_mode,
        max_up_components=max_up,
        max_down_components=max_down,
    )
    rows = [
        _source_row(
            row,
            axis=args.axis,
            pair_name=args.pair_name,
            sign_mode=args.sign_mode,
            edge_role=edge_role,
            selection_score=selection_score,
            sign_rank=sign_rank,
            rank=rank,
        )
        for rank, (row, selection_score, edge_role, sign_rank) in enumerate(signed, start=1)
    ]
    dump_csv(args.out_csv, rows)
    up_count = sum(1 for row in rows if float(row["selection_score"]) > 0.0)
    down_count = sum(1 for row in rows if float(row["selection_score"]) < 0.0)
    print(
        f"[rcm-export-ck-axis-summary] rows={len(rows)} up={up_count} down={down_count} "
        f"sign_mode={args.sign_mode} out={args.out_csv}"
    )


if __name__ == "__main__":
    main()
