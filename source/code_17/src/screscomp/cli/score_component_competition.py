from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.data import dump_csv, dump_jsonl, load_jsonl
from screscomp.modeling import TransformersABBackend


PAIR_SPECS = {
    "e1_content": ("c_p", "c_e", "normal"),
    "e1_content_swapped": ("swap_c_p", "swap_c_e", "swapped"),
    "e2_task": ("e2_c1_doc_actual", "e2_c1_doc_according", "normal"),
    "e2_task_swapped": ("swap_e2_c1_doc_actual", "swap_e2_c1_doc_according", "swapped"),
    "e2_source": ("e2_c2_user_actual", "e2_c2_doc_actual", "normal"),
    "e2_source_swapped": ("swap_e2_c2_user_actual", "swap_e2_c2_doc_actual", "swapped"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute component-level C_t/I_t competition profiles.")
    p.add_argument("--rendered_jsonl", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out_rows_jsonl", type=Path, required=True)
    p.add_argument("--out_summary_csv", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--layers", type=str, default="all")
    p.add_argument("--components", type=str, default="attn,mlp")
    p.add_argument("--pairs", type=str, default="e1_content,e1_content_swapped")
    p.add_argument("--splits", type=str, default="discovery,validation,test")
    p.add_argument("--templates", type=str, default="")
    p.add_argument("--max_rows", type=int, default=None)
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_set(raw: str) -> set[str]:
    return set(_parse_csv(raw))


def _parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw == "all":
        return list(range(num_layers))
    layers: set[int] = set()
    for part in _parse_csv(raw):
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(part))
    invalid = sorted(x for x in layers if x < 0 or x >= num_layers)
    if invalid:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {invalid}")
    return sorted(layers)


def _opposite_label(label: str) -> str:
    if label == "A":
        return "B"
    if label == "B":
        return "A"
    raise ValueError(f"Unsupported label: {label}")


def _prior_label(row: dict, label_mode: str) -> str:
    if label_mode == "normal":
        return row["prior_label"]
    if label_mode == "swapped":
        return _opposite_label(row["prior_label"])
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _semantic_b(logit_a: float, logit_b: float, prior_label: str) -> float:
    if prior_label == "A":
        return logit_a - logit_b
    if prior_label == "B":
        return logit_b - logit_a
    raise ValueError(f"Unsupported prior label: {prior_label}")


def _score_b(backend: TransformersABBackend, prompt: str, prior_label: str) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab(prompt)
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _score_component_patch_b(
    backend: TransformersABBackend,
    prompt: str,
    prior_label: str,
    layer_idx: int,
    component_type: str,
    replacement: Any,
) -> tuple[float, dict[str, float]]:
    logit_a, logit_b = backend.score_ab_with_component_last_token_patch(
        prompt=prompt,
        layer_idx=layer_idx,
        component_type=component_type,
        replacement_last_token=replacement,
    )
    return _semantic_b(logit_a, logit_b, prior_label), {"A": logit_a, "B": logit_b}


def _select_rows(rows: list[dict], splits: set[str], templates: set[str], max_rows: int | None) -> list[dict]:
    selected = [row for row in rows if row["split"] in splits and (not templates or row["template_family"] in templates)]
    if max_rows is not None:
        selected = selected[:max_rows]
    if not selected:
        raise ValueError("No rows selected.")
    return selected


def _validate_keys(rows: list[dict], pair_names: list[str]) -> None:
    for pair_name in pair_names:
        high_key, low_key, _label_mode = PAIR_SPECS[pair_name]
        for row in rows[:1]:
            prompts = row["prompts"]
            missing = [key for key in (high_key, low_key) if key not in prompts]
            if missing:
                raise ValueError(f"Rendered file is missing prompt keys for {pair_name}: {missing}")


def _component_id(layer_idx: int, component_type: str) -> str:
    return f"L{layer_idx}.{component_type}"


def _score_rows(
    backend: TransformersABBackend,
    rows: list[dict],
    pair_names: list[str],
    component_specs: list[tuple[int, str]],
) -> list[dict]:
    _validate_keys(rows, pair_names)
    output_rows: list[dict] = []
    for row in tqdm(rows, desc="component C/I"):
        prompts = row["prompts"]
        for pair_name in pair_names:
            high_key, low_key, label_mode = PAIR_SPECS[pair_name]
            prior_label = _prior_label(row, label_mode)
            high_prompt = prompts[high_key]
            low_prompt = prompts[low_key]

            high_b, high_logits = _score_b(backend, high_prompt, prior_label)
            low_b, low_logits = _score_b(backend, low_prompt, prior_label)
            high_acts = backend.capture_component_last_token(high_prompt, component_specs)
            low_acts = backend.capture_component_last_token(low_prompt, component_specs)

            for layer_idx, component_type in component_specs:
                high_delta = high_acts[(layer_idx, component_type)]
                low_delta = low_acts[(layer_idx, component_type)]
                low_with_high_b, low_with_high_logits = _score_component_patch_b(
                    backend=backend,
                    prompt=low_prompt,
                    prior_label=prior_label,
                    layer_idx=layer_idx,
                    component_type=component_type,
                    replacement=high_delta,
                )
                high_with_low_b, high_with_low_logits = _score_component_patch_b(
                    backend=backend,
                    prompt=high_prompt,
                    prior_label=prior_label,
                    layer_idx=layer_idx,
                    component_type=component_type,
                    replacement=low_delta,
                )
                # C/I are competition-profile instances of the core RCM rule:
                # component contribution = B(with component write) - B(without/replaced write).
                # B_high_minus_low is only a D_t-style overlay, not a selection metric.
                c_t = low_with_high_b - low_b
                i_t = high_b - high_with_low_b
                output_rows.append(
                    {
                        "render_id": row["render_id"],
                        "sample_id": row["sample_id"],
                        "fact_id": row["fact_id"],
                        "subject_id": row["subject_id"],
                        "split": row["split"],
                        "template_family": row["template_family"],
                        "pair_name": pair_name,
                        "high_key": high_key,
                        "low_key": low_key,
                        "label_mode": label_mode,
                        "layer_idx": layer_idx,
                        "component_type": component_type,
                        "component_id": _component_id(layer_idx, component_type),
                        "B_high": high_b,
                        "B_low": low_b,
                        "B_high_minus_low": high_b - low_b,
                        "B_low_with_high_delta": low_with_high_b,
                        "B_high_with_low_delta": high_with_low_b,
                        "C_t": c_t,
                        "I_t": i_t,
                        "C_plus_I": c_t + i_t,
                        "min_CI": min(c_t, i_t),
                        "C_positive": c_t > 0,
                        "I_positive": i_t > 0,
                        "both_positive": c_t > 0 and i_t > 0,
                        "logits": {
                            "high": high_logits,
                            "low": low_logits,
                            "low_with_high_delta": low_with_high_logits,
                            "high_with_low_delta": high_with_low_logits,
                        },
                    }
                )
    return output_rows


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(vals) if vals else 0.0


def _summarize_group(group: str, rows: list[dict]) -> dict[str, object]:
    first = rows[0]
    return {
        "group": group,
        "split": first.get("split", "mixed"),
        "pair_name": first["pair_name"],
        "layer_idx": first["layer_idx"],
        "component_type": first["component_type"],
        "component_id": first["component_id"],
        "n": len(rows),
        "mean_B_high_minus_low": _mean(rows, "B_high_minus_low"),
        "mean_C_t": _mean(rows, "C_t"),
        "mean_I_t": _mean(rows, "I_t"),
        "mean_C_plus_I": _mean(rows, "C_plus_I"),
        "mean_min_CI": _mean(rows, "min_CI"),
        "frac_C_positive": _rate(rows, "C_positive"),
        "frac_I_positive": _rate(rows, "I_positive"),
        "frac_both_positive": _rate(rows, "both_positive"),
    }


def _summarize(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    all_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["split"], row["pair_name"], row["layer_idx"], row["component_type"])
        groups[key].append(row)
        all_key = ("all", row["pair_name"], row["layer_idx"], row["component_type"])
        all_groups[all_key].append({**row, "split": "all"})

    summary: list[dict[str, object]] = []
    for key in sorted(groups):
        split, pair_name, layer_idx, component_type = key
        group = f"split:{split}:pair:{pair_name}:component:L{layer_idx}.{component_type}"
        summary.append(_summarize_group(group, groups[key]))
    for key in sorted(all_groups):
        split, pair_name, layer_idx, component_type = key
        group = f"split:{split}:pair:{pair_name}:component:L{layer_idx}.{component_type}"
        summary.append(_summarize_group(group, all_groups[key]))
    return summary


def main() -> None:
    args = parse_args()
    pair_names = _parse_csv(args.pairs)
    unknown_pairs = sorted(set(pair_names) - set(PAIR_SPECS))
    if unknown_pairs:
        raise ValueError(f"Unknown pair specs: {unknown_pairs}")
    component_types = _parse_csv(args.components)
    unknown_components = sorted(set(component_types) - {"attn", "mlp"})
    if unknown_components:
        raise ValueError(f"Unsupported components: {unknown_components}")

    all_rows = load_jsonl(args.rendered_jsonl)
    rows = _select_rows(
        all_rows,
        splits=_parse_csv_set(args.splits),
        templates=_parse_csv_set(args.templates),
        max_rows=args.max_rows,
    )
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    layers = _parse_layers(args.layers, backend.num_layers)
    component_specs = [(layer_idx, component_type) for layer_idx in layers for component_type in component_types]
    scored_rows = _score_rows(
        backend=backend,
        rows=rows,
        pair_names=pair_names,
        component_specs=component_specs,
    )
    dump_jsonl(args.out_rows_jsonl, scored_rows)
    dump_csv(args.out_summary_csv, _summarize(scored_rows))
    print(
        f"[score-component-competition] rows={len(rows)} pairs={len(pair_names)} "
        f"components={len(component_specs)} output_rows={len(scored_rows)}"
    )


if __name__ == "__main__":
    main()
