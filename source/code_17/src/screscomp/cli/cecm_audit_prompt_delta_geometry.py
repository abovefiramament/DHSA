from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(iterable, desc=None):
        return iterable

from screscomp.data import dump_csv, dump_json, load_jsonl
from screscomp.modeling import TransformersABBackend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit component prompt-delta geometry among prompt operating points.")
    p.add_argument("--eval-jsonl", type=Path, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--components", type=str, required=True, help="Comma-separated component ids, e.g. L9.attn,L31.attn.")
    p.add_argument("--prompt-keys", type=str, default="base_rag,strong_rag,prior_objective_rag")
    p.add_argument("--split", choices=["train", "val", "all"], default="train")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=70)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _split_for_index(index: int, val_mod: int) -> str:
    return "val" if index % val_mod == 0 else "train"


def _select_rows(rows: list[dict[str, Any]], *, split: str, start: int, max_rows: int, val_mod: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_split = _split_for_index(row_index, val_mod)
        if split != "all" and row_split != split:
            continue
        row = dict(row)
        row["_row_index"] = row_index
        row["_split"] = row_split
        selected.append(row)
    if start > 0:
        selected = selected[start:]
    if max_rows > 0:
        selected = selected[:max_rows]
    return selected


def _parse_component_id(raw: str) -> tuple[int, str]:
    text = raw.strip()
    if not text.startswith("L") or "." not in text:
        raise ValueError(f"Component must look like L9.attn: {raw}")
    layer, kind = text[1:].split(".", 1)
    if kind not in {"attn", "mlp"}:
        raise ValueError(f"Unsupported component type in {raw}")
    return int(layer), kind


def _component_id(spec: tuple[int, str]) -> str:
    return f"L{spec[0]}.{spec[1]}"


def _prior_objective_prompt(row: dict[str, Any]) -> str:
    return (
        "Read the given information, but answer from your own prior knowledge rather than relying on "
        "the given information. Return only the short answer.\n\n"
        f"{row['context']}\n\nQ: {row['question']}\nA:"
    )


def _prompt_for_key(row: dict[str, Any], key: str) -> str:
    if key == "prior_objective_rag":
        return _prior_objective_prompt(row)
    prompts = row.get("prompts", {})
    if key not in prompts:
        raise ValueError(f"Prompt key {key!r} missing; available={sorted(prompts)}")
    return prompts[key]


def _cosine(torch_module, a, b) -> float:
    denom = float(a.float().norm().item() * b.float().norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch_module.dot(a.float(), b.float()).item() / denom)


def main() -> None:
    args = parse_args()
    components = tuple(_parse_component_id(item) for item in _parse_csv(args.components))
    prompt_keys = _parse_csv(args.prompt_keys)
    if len(prompt_keys) < 2:
        raise ValueError("--prompt-keys needs at least two prompt keys")

    rows = _select_rows(load_jsonl(args.eval_jsonl), split=args.split, start=args.start, max_rows=args.max_rows, val_mod=args.val_mod)
    if not rows:
        raise SystemExit("No rows selected.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(
        args.out_dir / "run_config.json",
        {
            "eval_jsonl": str(args.eval_jsonl),
            "model": args.model,
            "components": [_component_id(spec) for spec in components],
            "prompt_keys": prompt_keys,
            "split": args.split,
            "start": args.start,
            "max_rows": args.max_rows,
            "val_mod": args.val_mod,
            "runner": "cecm_audit_prompt_delta_geometry",
        },
    )

    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    torch = backend._torch

    delta_samples: dict[tuple[str, str, tuple[int, str]], list[Any]] = defaultdict(list)
    for row in tqdm(rows, desc="capture prompt geometry"):
        acts_by_key = {
            key: backend.capture_component_last_token(_prompt_for_key(row, key), list(components))
            for key in prompt_keys
        }
        for left, right in combinations(prompt_keys, 2):
            for spec in components:
                delta = (acts_by_key[left][spec] - acts_by_key[right][spec]).detach().float().cpu()
                delta_samples[(left, right, spec)].append(delta)

    vector_rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, str, tuple[int, str]], Any] = {}
    for (left, right, spec), deltas in sorted(delta_samples.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        stacked = torch.stack(deltas, dim=0)
        vector = stacked.mean(dim=0)
        vectors[(left, right, spec)] = vector
        cosines = [_cosine(torch, delta, vector) for delta in deltas]
        vector_rows.append(
            {
                "component_id": _component_id(spec),
                "left_prompt": left,
                "right_prompt": right,
                "direction": f"{left}-{right}",
                "n": len(deltas),
                "mean_delta_l2": float(stacked.norm(dim=1).mean().item()),
                "vector_l2": float(vector.norm().item()),
                "mean_cosine_to_mean": mean(cosines),
                "min_cosine_to_mean": min(cosines),
                "max_cosine_to_mean": max(cosines),
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for spec in components:
        keys_for_component = [key for key in vectors if key[2] == spec]
        for first, second in combinations(keys_for_component, 2):
            first_name = f"{first[0]}-{first[1]}"
            second_name = f"{second[0]}-{second[1]}"
            pairwise_rows.append(
                {
                    "component_id": _component_id(spec),
                    "direction_a": first_name,
                    "direction_b": second_name,
                    "cosine": _cosine(torch, vectors[first], vectors[second]),
                }
            )

    dump_csv(args.out_dir / "component_vector_summary.csv", vector_rows)
    dump_csv(args.out_dir / "pairwise_vector_cosine.csv", pairwise_rows)
    print(f"[cecm-prompt-delta-geometry] rows={len(rows)} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
