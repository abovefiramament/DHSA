from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from screscomp.data import dump_csv, dump_json, dump_jsonl, load_jsonl


QUERY_LENGTH = 512
# OpenAI's tldr_rm_task extends the policy's 48-token response window to 128.
RESPONSE_LENGTH = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score TL;DR summaries with OpenAI's original rm4.")
    parser.add_argument("--rm4-dir", type=Path, required=True)
    parser.add_argument("--openai-code-dir", type=Path, required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], metavar="NAME=GENERATIONS_JSONL")
    parser.add_argument(
        "--component-loo-manifest",
        type=Path,
        help="Add full and drop candidates from an existing component LOO audit manifest.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--act-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args(argv)


def _parse_candidates(values: list[str]) -> list[tuple[str, Path | None]]:
    candidates: list[tuple[str, Path | None]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"Invalid --candidate {value!r}; expected NAME=GENERATIONS_JSONL or human=")
        candidate_name = name.strip()
        candidate_path = Path(path.strip()) if path.strip() else None
        if candidate_name == "human" and candidate_path is not None:
            raise ValueError("Use human= without a path to score reference summaries")
        if candidate_name != "human" and candidate_path is None:
            raise ValueError(f"Candidate {candidate_name!r} requires a generations path")
        candidates.append((candidate_name, candidate_path))
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    return candidates


def _component_loo_candidates(
    path: Path,
) -> tuple[list[tuple[str, Path]], dict[str, Any], dict[str, float]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    def resolve(raw_path: str) -> Path:
        candidate_path = Path(raw_path)
        if candidate_path.is_absolute():
            return candidate_path
        project_relative = Path.cwd() / candidate_path
        if project_relative.exists():
            return project_relative
        return base_dir / candidate_path

    candidates: list[tuple[str, Path]] = []
    candidate_alphas: dict[str, float] = {}
    for group in manifest["groups"]:
        group_name = str(group["group"])
        alpha = float(group["alpha"])
        full_name = f"{group_name}__full"
        candidates.append((full_name, resolve(str(group["full_generations"]))))
        candidate_alphas[full_name] = alpha
        for variant in group["variants"]:
            variant_name = f"{group_name}__{variant['variant']}"
            candidates.append(
                (
                    variant_name,
                    resolve(str(variant["generations"])),
                )
            )
            candidate_alphas[variant_name] = alpha
    return candidates, manifest, candidate_alphas


def _indexed(
    rows: list[dict[str, Any]], source: Path, *, alpha: float | None = None
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if alpha is not None and abs(float(row.get("alpha", "nan")) - alpha) > 1e-9:
            continue
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id or sample_id in result:
            raise ValueError(f"Missing or duplicate sample_id={sample_id!r} in {source}")
        result[sample_id] = row
    if not result:
        suffix = f" at alpha={alpha}" if alpha is not None else ""
        raise ValueError(f"No rows found in {source}{suffix}")
    return result


def _load_original_modules(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    import summarize_from_feedback  # type: ignore
    from summarize_from_feedback.models.transformer import Transformer  # type: ignore

    return summarize_from_feedback.encoder, Transformer


def _load_piece(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_rm4(args: argparse.Namespace):
    encoder, Transformer = _load_original_modules(args.openai_code_dir)
    info = json.loads((args.rm4_dir / "info.json").read_text(encoding="utf-8"))
    hparams = info["model_hparams"]
    model = Transformer(
        n_ctx=int(hparams["n_ctx"]),
        n_vocab=int(encoder.n_vocab),
        d_model=int(hparams["d_model"]),
        n_layer=int(hparams["n_layer"]),
        heads=int(hparams["heads"]),
        attn_dropout=0.0,
        resid_dropout=0.0,
        emb_dropout=0.0,
        m_attn=float(hparams["m_attn"]),
        m_mlp=float(hparams["m_mlp"]),
        include_output_unembeddings=False,
        include_final_layer_norm=True,
        key_bias=bool(hparams.get("key_bias", False)),
    )
    act_dtype = getattr(torch, args.act_dtype)
    checkpoint = args.rm4_dir / "checkpoint"
    print("[tldr-rm4] loading input and position embeddings", flush=True)
    model.embedding.load_state_dict(_load_piece(checkpoint / "input_embeddings_shard_000.pkl"))
    model.position_embedding.load_state_dict(_load_piece(checkpoint / "position_embedding_shard_000.pkl"))
    for index, block in enumerate(model.torso.resblocks):
        print(f"[tldr-rm4] loading block={index + 1}/{len(model.torso.resblocks)}", flush=True)
        block.load_state_dict(_load_piece(checkpoint / f"resblock_{index:04d}_shard_000.pkl"))
    model.ln_f.load_state_dict(_load_piece(checkpoint / "final_layer_norm_shard_000.pkl"))
    reward_head = torch.nn.Linear(int(hparams["d_model"]), 1)
    reward_head.load_state_dict(_load_piece(checkpoint / "output_head_reward_shard_000.pkl"))
    print(
        f"[tldr-rm4] moving float32 weights to device={args.device}; act_dtype={args.act_dtype}",
        flush=True,
    )
    model = model.to(args.device).eval()
    reward_head = reward_head.to(args.device).eval()
    return encoder, model, reward_head, act_dtype


def _format_query(row: dict[str, Any]) -> str:
    subreddit = str(row.get("raw_subreddit", "")).removeprefix("r/")
    return (
        f"SUBREDDIT: r/{subreddit}\n\n"
        f"TITLE: {row.get('raw_title', '')}\n\n"
        f"POST: {row.get('raw_post', '')}\n\n"
        "TL;DR:"
    )


def _encode_query(encoder, row: dict[str, Any]) -> list[int]:
    working = dict(row)
    query = _format_query(working)
    tokens = encoder.encode(query)
    while len(tokens) > QUERY_LENGTH:
        post = str(working.get("raw_post", ""))
        if not post:
            raise ValueError(f"Could not truncate query for sample_id={row.get('sample_id')}")
        split = post.rfind("\n")
        working["raw_post"] = post[: split if split >= 0 else -1]
        query = _format_query(working)
        tokens = encoder.encode(query)
    pad_token = encoder.encode(" ")
    if len(pad_token) != 1:
        raise ValueError(f"Expected one padding token, found {pad_token}")
    return pad_token * (QUERY_LENGTH - len(tokens)) + tokens


def _encode_response(encoder, text: str) -> tuple[list[int], int]:
    tokens = encoder.encode(" " + text.strip())[: RESPONSE_LENGTH - 1]
    tokens.append(int(encoder.eot_token))
    last_response_index = len(tokens) - 1
    return tokens + [0] * (RESPONSE_LENGTH - len(tokens)), last_response_index


def _batches(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidates = _parse_candidates(args.candidate)
    component_loo_manifest: dict[str, Any] | None = None
    candidate_alphas: dict[str, float] = {}
    if args.component_loo_manifest is not None:
        component_candidates, component_loo_manifest, component_alphas = _component_loo_candidates(
            args.component_loo_manifest
        )
        candidates.extend(component_candidates)
        candidate_alphas.update(component_alphas)
    if not candidates:
        raise ValueError("Provide at least one --candidate or --component-loo-manifest")
    if len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")

    prompts = load_jsonl(args.prompts_jsonl)
    if args.max_rows is not None:
        prompts = prompts[: args.max_rows]
    prompt_by_id = _indexed(prompts, args.prompts_jsonl)
    candidate_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path in candidates:
        if path is not None:
            candidate_by_name[name] = _indexed(
                load_jsonl(path), path, alpha=candidate_alphas.get(name)
            )

    encoder, model, reward_head, act_dtype = _load_rm4(args)
    rows_to_score: list[dict[str, Any]] = []
    for prompt in prompts:
        sample_id = str(prompt["sample_id"])
        query_tokens = _encode_query(encoder, prompt)
        for name, _ in candidates:
            if name == "human":
                completion = str(prompt.get("reference_summary", ""))
            else:
                if sample_id not in candidate_by_name[name]:
                    raise ValueError(f"Missing sample_id={sample_id} in candidate={name}")
                completion = str(candidate_by_name[name][sample_id].get("completion", ""))
            if not completion.strip():
                raise ValueError(f"Empty completion candidate={name} sample_id={sample_id}")
            response_tokens, last_response_index = _encode_response(encoder, completion)
            rows_to_score.append(
                {
                    "sample_id": sample_id,
                    "candidate": name,
                    "completion": completion,
                    "tokens": query_tokens + response_tokens,
                    "reward_index": QUERY_LENGTH + last_response_index,
                }
            )

    scored: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(_batches(rows_to_score, args.batch_size), start=1):
        tokens = torch.tensor([row["tokens"] for row in batch], dtype=torch.long, device=args.device)
        reward_indices = torch.tensor([row["reward_index"] for row in batch], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            acts = model(tokens, act_dtype=act_dtype)["acts"]
            all_rewards = reward_head(acts.type(reward_head.weight.dtype)).squeeze(-1)
            rewards = all_rewards.gather(1, reward_indices[:, None]).squeeze(1).float().cpu().tolist()
        for row, reward in zip(batch, rewards, strict=True):
            scored.append(
                {
                    "sample_id": row["sample_id"],
                    "candidate": row["candidate"],
                    "reward": float(reward),
                    "completion": row["completion"],
                }
            )
        if batch_index % 10 == 0:
            print(f"[tldr-rm4] scored={len(scored)}/{len(rows_to_score)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.out_dir / "scores.jsonl", scored)
    scores_by_candidate: dict[str, dict[str, float]] = {name: {} for name, _ in candidates}
    for row in scored:
        scores_by_candidate[str(row["candidate"])][str(row["sample_id"])] = float(row["reward"])
    summary_rows: list[dict[str, Any]] = []
    for name, _ in candidates:
        values = list(scores_by_candidate[name].values())
        summary_rows.append({"kind": "candidate", "name": name, "n": len(values), "mean_reward": mean(values)})
    for left, right in itertools.combinations([name for name, _ in candidates], 2):
        common = sorted(set(scores_by_candidate[left]) & set(scores_by_candidate[right]))
        deltas = [scores_by_candidate[left][sample_id] - scores_by_candidate[right][sample_id] for sample_id in common]
        summary_rows.append(
            {
                "kind": "pair",
                "name": f"{left}_vs_{right}",
                "n": len(common),
                "mean_reward_delta": mean(deltas),
                "left_win_rate": sum(delta > 0 for delta in deltas) / len(deltas),
                "tie_rate": sum(delta == 0 for delta in deltas) / len(deltas),
            }
        )
    component_rows: list[dict[str, Any]] = []
    if component_loo_manifest is not None:
        for group in component_loo_manifest["groups"]:
            group_name = str(group["group"])
            full_name = f"{group_name}__full"
            for variant in group["variants"]:
                variant_name = f"{group_name}__{variant['variant']}"
                common = sorted(
                    set(scores_by_candidate[full_name]) & set(scores_by_candidate[variant_name])
                )
                contributions = [
                    scores_by_candidate[full_name][sample_id]
                    - scores_by_candidate[variant_name][sample_id]
                    for sample_id in common
                ]
                mean_contribution = mean(contributions)
                component_rows.append(
                    {
                        "group": group_name,
                        "kind": group["kind"],
                        "alpha": group["alpha"],
                        "variant": variant["variant"],
                        "dropped_unit": variant["dropped_unit"],
                        "n": len(common),
                        "mean_full_reward": mean(
                            scores_by_candidate[full_name][sample_id] for sample_id in common
                        ),
                        "mean_drop_reward": mean(
                            scores_by_candidate[variant_name][sample_id] for sample_id in common
                        ),
                        "mean_reward_contribution": mean_contribution,
                        "reward_effect": (
                            "positive"
                            if mean_contribution > 0
                            else "negative"
                            if mean_contribution < 0
                            else "neutral"
                        ),
                        "positive_contribution_rate": sum(value > 0 for value in contributions)
                        / len(contributions),
                        "negative_contribution_rate": sum(value < 0 for value in contributions)
                        / len(contributions),
                        "tie_rate": sum(value == 0 for value in contributions) / len(contributions),
                    }
                )
        component_rows.sort(
            key=lambda row: (str(row["group"]), -float(row["mean_reward_contribution"]))
        )
        dump_csv(args.out_dir / "component_loo_reward_summary.csv", component_rows)
        dump_json(args.out_dir / "component_loo_reward_summary.json", {"rows": component_rows})
    dump_csv(args.out_dir / "summary.csv", summary_rows)
    dump_json(args.out_dir / "summary.json", {"rows": summary_rows, "component_loo": component_rows})
    if args.device.startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(
            f"[tldr-rm4] cuda_peak_allocated_gib={peak_allocated:.3f} "
            f"cuda_peak_reserved_gib={peak_reserved:.3f}",
            flush=True,
        )
    print(f"[tldr-rm4] complete out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
