from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from screscomp.cli.score_ckplug_generation import _summarize
from screscomp.data import dump_csv, dump_jsonl, load_jsonl


HEDGE_PATTERNS = [
    r"\bhowever\b",
    r"\balthough\b",
    r"\bbut\b",
    r"\berror\b",
    r"\bincorrect\b",
    r"\binsufficient\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bdoes not\b",
    r"\bdoesn't\b",
    r"\bnot\b",
    r"\blikely\b",
    r"\bshould\b",
    r"\bconflict\b",
    r"\binstead\b",
    r"\bunfortunately\b",
    r"\bbased on\b",
    r"\bpassage\b",
    r"\btext says\b",
    r"\bstates\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Post-hoc route between two already generated CK-style methods using only "
            "label-free output-form features from the source method."
        )
    )
    p.add_argument("--source_generations_jsonl", type=Path, required=True)
    p.add_argument("--fallback_generations_jsonl", type=Path, required=True)
    p.add_argument("--source_method", type=str, default="")
    p.add_argument("--fallback_method", type=str, default="")
    p.add_argument("--rules", type=str, default="length,hedge,length_or_hedge")
    p.add_argument("--length_thresholds", type=str, default="24,32,40,48,64,80")
    p.add_argument("--out_generations_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--out_manifest_csv", type=Path, required=True)
    p.add_argument("--keep_candidate_predictions", action="store_true")
    p.add_argument("--keep_prompt", action="store_true")
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_method_rows(path: Path, method: str) -> tuple[str, dict[str, dict[str, Any]]]:
    rows = load_jsonl(path)
    methods = sorted({str(row["method"]) for row in rows})
    if not methods:
        raise ValueError(f"No methods found in {path}")
    if method:
        if method not in methods:
            raise ValueError(f"Method {method!r} not found in {path}. Available: {methods}")
        selected_method = method
    elif len(methods) == 1:
        selected_method = methods[0]
    else:
        raise ValueError(f"{path} contains multiple methods; pass an explicit method. Available: {methods}")

    selected = {str(row["sample_id"]): row for row in rows if str(row["method"]) == selected_method}
    return selected_method, selected


def _has_hedge(text: str) -> bool:
    return bool(HEDGE_RE.search(text or ""))


def _use_fallback(source_prediction: str, rule: str, threshold: int) -> tuple[bool, str]:
    text = source_prediction or ""
    too_long = len(text) > threshold
    hedged = _has_hedge(text)
    if rule == "length":
        return too_long, "length" if too_long else "source_ok"
    if rule == "hedge":
        return hedged, "hedge" if hedged else "source_ok"
    if rule == "length_or_hedge":
        if too_long or hedged:
            reasons = []
            if too_long:
                reasons.append("length")
            if hedged:
                reasons.append("hedge")
            return True, "+".join(reasons)
        return False, "source_ok"
    if rule == "length_and_hedge":
        return too_long and hedged, "length+hedge" if too_long and hedged else "source_ok"
    raise ValueError(f"Unknown routing rule: {rule}")


def _route_rows(
    *,
    source_method: str,
    fallback_method: str,
    source_rows: dict[str, dict[str, Any]],
    fallback_rows: dict[str, dict[str, Any]],
    rule: str,
    threshold: int,
    keep_candidate_predictions: bool,
    keep_prompt: bool,
) -> list[dict[str, Any]]:
    sample_ids = sorted(set(source_rows) & set(fallback_rows))
    if not sample_ids:
        raise ValueError("No paired sample ids between source and fallback generations.")

    method = (
        f"routed_{source_method}_to_{fallback_method}"
        f"__{rule}_t{threshold}"
    )
    output_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        source = source_rows[sample_id]
        fallback = fallback_rows[sample_id]
        choose_fallback, reason = _use_fallback(str(source.get("prediction", "")), rule, threshold)
        chosen = fallback if choose_fallback else source
        chosen_payload = dict(chosen)
        if not keep_prompt:
            chosen_payload.pop("prompt", None)
        row = {
            **chosen_payload,
            "method": method,
            "route_rule": rule,
            "route_threshold": threshold,
            "route_chosen": "fallback" if choose_fallback else "source",
            "route_reason": reason,
            "route_source_method": source_method,
            "route_fallback_method": fallback_method,
            "route_source_output_chars": source.get("output_chars", len(str(source.get("prediction", "")))),
            "route_fallback_output_chars": fallback.get("output_chars", len(str(fallback.get("prediction", "")))),
        }
        if keep_candidate_predictions:
            row["route_source_prediction"] = source.get("prediction", "")
            row["route_fallback_prediction"] = fallback.get("prediction", "")
        output_rows.append(row)
    return output_rows


def main() -> None:
    args = parse_args()
    rules = _parse_csv(args.rules)
    thresholds = [int(item) for item in _parse_csv(args.length_thresholds)]
    if not rules:
        raise ValueError("No routing rules provided.")
    if not thresholds:
        raise ValueError("No routing thresholds provided.")

    source_method, source_rows = _load_method_rows(args.source_generations_jsonl, args.source_method)
    fallback_method, fallback_rows = _load_method_rows(args.fallback_generations_jsonl, args.fallback_method)

    output_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for rule in rules:
        for threshold in thresholds:
            routed = _route_rows(
                source_method=source_method,
                fallback_method=fallback_method,
                source_rows=source_rows,
                fallback_rows=fallback_rows,
                rule=rule,
                threshold=threshold,
                keep_candidate_predictions=args.keep_candidate_predictions,
                keep_prompt=args.keep_prompt,
            )
            output_rows.extend(routed)
            n_fallback = sum(1 for row in routed if row["route_chosen"] == "fallback")
            manifest_rows.append(
                {
                    "source_generations_jsonl": str(args.source_generations_jsonl),
                    "fallback_generations_jsonl": str(args.fallback_generations_jsonl),
                    "source_method": source_method,
                    "fallback_method": fallback_method,
                    "rule": rule,
                    "threshold": threshold,
                    "n": len(routed),
                    "fallback_rate": n_fallback / len(routed),
                    "method": routed[0]["method"],
                }
            )

    dump_jsonl(args.out_generations_jsonl, output_rows)
    dump_csv(args.out_summary_csv, _summarize(output_rows))
    dump_csv(args.out_manifest_csv, manifest_rows)
    print(
        f"[route-ckplug-generations] methods={len(manifest_rows)} "
        f"rows={len(output_rows)} source={source_method} fallback={fallback_method}"
    )


if __name__ == "__main__":
    main()
