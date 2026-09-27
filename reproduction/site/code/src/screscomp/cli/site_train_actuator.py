from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any

from screscomp.cecm.actuator import limit_rows, load_actuator_pairs
from screscomp.cli.cecm_train_attention_head_actuator import (
    AttentionHeadActuatorTrainer,
    HeadActuatorConfig,
    HeadSpec,
)
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend
from screscomp.site.protocol import SiteProtocol, require_files
from screscomp.site.runtime import configure_gpu, resolve_pinned_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the common locked eight-head Site actuator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    return parser.parse_args()


SIGNED_SELECTORS = frozenset({"rcm_zero_signed", "rcm_patch_signed"})


def _head_spec(component: str) -> HeadSpec:
    layer_text, head_text = component.split(".attn.h")
    return HeadSpec(
        head_id=component,
        layer_idx=int(layer_text[1:]),
        head_idx=int(head_text),
    )


def _read_head_groups(
    path: Path,
    *,
    selector: str,
    expected_count: int,
    signed_contract: dict[str, Any],
) -> tuple[list[HeadSpec], dict[str, list[HeadSpec]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    entries = [(row.get("selection_role", ""), _head_spec(row["component_id"])) for row in rows]
    heads = [head for _role, head in entries]
    if len(heads) != expected_count or len({head.head_id for head in heads}) != expected_count:
        raise RuntimeError(
            f"Site actuator requires exactly {expected_count} distinct selected heads"
        )

    if selector in SIGNED_SELECTORS:
        bank_order = tuple(str(role) for role in signed_contract["bank_order"])
        heads_per_bank = int(signed_contract["heads_per_bank"])
        if heads_per_bank * len(bank_order) != expected_count:
            raise RuntimeError("signed RCM bank contract does not sum to the selected head count")
        if {role for role, _head in entries} != set(bank_order):
            raise RuntimeError("signed RCM selected heads do not match the registered role banks")
        groups = {
            role: [head for row_role, head in entries if row_role == role]
            for role in bank_order
        }
        if any(len(group) != heads_per_bank for group in groups.values()):
            raise RuntimeError(
                f"signed RCM requires exactly {heads_per_bank} heads in each role bank"
            )
        return heads, groups

    if {role for role, _head in entries} != {"unsigned"}:
        raise RuntimeError(f"{selector} requires one unsigned selected-head bank")
    return heads, {"unsigned": heads}


def _validate_adamw_defaults(torch, common: dict) -> None:
    signature = inspect.signature(torch.optim.AdamW)
    defaults = {
        "betas": signature.parameters["betas"].default,
        "eps": signature.parameters["eps"].default,
        "weight_decay": signature.parameters["weight_decay"].default,
        "amsgrad": signature.parameters["amsgrad"].default,
    }
    expected = {
        "betas": tuple(common["betas"]),
        "eps": common["eps"],
        "weight_decay": common["weight_decay"],
        "amsgrad": common["amsgrad"],
    }
    if defaults != expected:
        raise RuntimeError(
            "existing AttentionHeadActuatorTrainer AdamW defaults no longer match the locked Site config: "
            f"expected={expected} actual={defaults}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tag_rows(rows: list[dict[str, object]], training_group: str) -> list[dict[str, object]]:
    return [{"training_group": training_group, **row} for row in rows]


def _train_bank(
    *,
    backend: TransformersABBackend,
    heads: list[HeadSpec],
    config: HeadActuatorConfig,
    train_pairs: list[Any],
    val_pairs: list[Any],
    out_dir: Path,
    training_group: str,
) -> dict[str, Any]:
    backend._torch.manual_seed(int(config.seed))
    backend._torch.cuda.manual_seed_all(int(config.seed))
    trainer = AttentionHeadActuatorTrainer(backend=backend, heads=heads, config=config)
    out_dir.mkdir(parents=True, exist_ok=True)
    history, baseline_margins = trainer.train(train_pairs, val_pairs=val_pairs, out_dir=out_dir)
    payload = out_dir / "head_actuator.pt"
    trainer.save_payload(payload)
    history_rows = _tag_rows(history, training_group)
    vector_rows = _tag_rows(trainer.vector_summary_rows(), training_group)
    audit_rows = _tag_rows(
        [
            trainer.evaluate_alpha(
                val_pairs,
                baseline_margins=baseline_margins,
                alpha=float(config.alpha_train),
                split="val_teacher_forced_audit_only",
            )
        ],
        training_group,
    )
    dump_csv(out_dir / "training_history.csv", history_rows)
    dump_csv(out_dir / "vector_summary.csv", vector_rows)
    dump_csv(out_dir / "teacher_forced_audit.csv", audit_rows)
    result = {
        "training_group": training_group,
        "heads": [head.head_id for head in heads],
        "payload": payload,
        "history_rows": history_rows,
        "vector_rows": vector_rows,
        "audit_rows": audit_rows,
    }
    del trainer
    backend._torch.cuda.empty_cache()
    return result


def _load_payload(torch: Any, path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid actuator payload: {path}")
    return payload


def _merge_role_payloads(
    torch: Any,
    bank_runs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    loaded = [
        (str(run["training_group"]), Path(run["payload"]), _load_payload(torch, Path(run["payload"])))
        for run in bank_runs
    ]
    first = loaded[0][2]
    invariant_keys = (
        "kind", "event", "endpoint_objective", "apply_mode", "score_mode",
        "option_selection_mode", "alpha_train", "preference_loss_mode", "dpo_beta",
        "state_margin_weight", "gain_weight", "target_margin", "target_gain",
    )
    for _role, _path, payload in loaded[1:]:
        if any(payload.get(key) != first.get(key) for key in invariant_keys):
            raise RuntimeError("signed RCM role payloads do not share one downstream contract")

    heads: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    role_banks: list[dict[str, Any]] = []
    for role, path, payload in loaded:
        bank_heads = list(payload["heads"])
        bank_ids = [str(row["head_id"]) for row in bank_heads]
        if set(bank_ids) & set(vectors):
            raise RuntimeError("signed RCM role banks contain overlapping heads")
        if set(bank_ids) != set(payload["vectors"]):
            raise RuntimeError(f"role payload head/vector mismatch: {role}")
        heads.extend(bank_heads)
        vectors.update(payload["vectors"])
        role_banks.append(
            {
                "selection_role": role,
                "heads": bank_ids,
                "payload": str(path),
                "payload_sha256": _sha256(path),
            }
        )
    merged = {
        **first,
        "heads": heads,
        "vectors": vectors,
        "training_partition": "two_independently_trained_four_head_role_banks",
        "role_banks": role_banks,
        "inference_composition": "sum_both_role_banks",
        "alpha_policy": "one_shared_alpha_for_both_role_banks",
        "independent_role_alpha_search": False,
        "semantics": (
            "Vectors are trained jointly within each signed RCM role bank and independently "
            "across banks, then both banks are added at inference with one shared alpha."
        ),
    }
    torch.save(merged, output_path)


def main() -> None:
    args = parse_args()
    protocol = SiteProtocol.load(args.config)
    job = protocol.job(args.job_id)
    paths = protocol.paths(args.artifact_root, args.job_id)
    require_files([paths.selected_heads, paths.pairs_csv])
    validator = protocol.canonical_validator
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--config", str(protocol.config_path),
            "--validate-selector-output", str(paths.selected_heads),
            "--expected-selector", job.selector,
        ],
        check=True,
    )
    gpu = configure_gpu(args.device, protocol.site["execution"])
    pinned_model = resolve_pinned_model(job.model, job.model_revision)
    backend = TransformersABBackend(
        model_name_or_path=pinned_model,
        tokenizer_name_or_path=pinned_model,
        device="cuda:0",
        use_chat_template=job.use_chat_template,
        torch_dtype="auto",
    )
    common = protocol.site["downstream_common"]
    downstream = job.task["downstream"]
    _validate_adamw_defaults(backend._torch, common)
    signed_contract = protocol.site["downstream_contract"]["signed_rcm_bank_contract"]
    heads, head_groups = _read_head_groups(
        paths.selected_heads,
        selector=job.selector,
        expected_count=int(protocol.site["selection_contract"]["selected_count"]),
        signed_contract=signed_contract,
    )
    pairs = load_actuator_pairs(
        paths.pairs_csv,
        event=downstream["event"],
        endpoint_objective=downstream["endpoint_objective"],
    )
    train_pairs = limit_rows(
        [pair for pair in pairs if pair.split == "train"],
        int(downstream["train_rows"]),
    )
    val_pairs = limit_rows(
        [pair for pair in pairs if pair.split == "val"],
        int(downstream["val_rows"]),
    )
    if not train_pairs or not val_pairs:
        raise RuntimeError("fixed Site pair artifact has an empty train or validation split")
    config = HeadActuatorConfig(
        event=downstream["event"],
        endpoint_objective=downstream["endpoint_objective"],
        apply_mode=downstream["apply_mode"],
        causal_train_mask=bool(downstream["causal_train_mask"]),
        score_mode=downstream["score_mode"],
        option_selection_mode=downstream["option_selection_mode"],
        alpha_train=float(common["alpha_train"]),
        preference_loss_mode=downstream["preference_loss_mode"],
        dpo_beta=float(downstream["dpo_beta"]),
        state_margin_weight=float(downstream["state_margin_weight"]),
        gain_weight=float(downstream["gain_weight"]),
        target_margin=float(downstream.get("target_margin", 0.0)),
        target_gain=float(downstream.get("target_gain", 0.0)),
        lambda_norm=float(common["lambda_norm"]),
        lr=float(common["lr"]),
        epochs=int(common["epochs"]),
        train_batch_size=int(downstream["train_batch_size"]),
        seed=int(common["seed"]),
        max_aliases_per_side=int(downstream["max_aliases_per_side"]),
        empty_cache_every=int(common["empty_cache_every"]),
    )
    paths.actuator_dir.mkdir(parents=True, exist_ok=True)
    bank_runs: list[dict[str, Any]] = []
    for training_group, group_heads in head_groups.items():
        group_dir = (
            paths.actuator_dir / training_group
            if job.selector in SIGNED_SELECTORS
            else paths.actuator_dir
        )
        bank_runs.append(
            _train_bank(
                backend=backend,
                heads=group_heads,
                config=config,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                out_dir=group_dir,
                training_group=training_group,
            )
        )

    payload = paths.actuator_dir / "head_actuator.pt"
    if job.selector in SIGNED_SELECTORS:
        _merge_role_payloads(backend._torch, bank_runs, payload)
    elif Path(bank_runs[0]["payload"]) != payload:
        raise RuntimeError("unsigned actuator payload path drift")
    history_rows = [row for run in bank_runs for row in run["history_rows"]]
    vector_rows = [row for run in bank_runs for row in run["vector_rows"]]
    audit_rows = [row for run in bank_runs for row in run["audit_rows"]]
    dump_csv(paths.actuator_dir / "training_history.csv", history_rows)
    dump_csv(paths.actuator_dir / "vector_summary.csv", vector_rows)
    dump_csv(paths.actuator_dir / "teacher_forced_audit.csv", audit_rows)
    repo_root = protocol.config_path.parent.parent
    code_commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    dirty_tree = subprocess.check_output(["git", "-C", str(repo_root), "status", "--short"], text=True).splitlines()
    resolved_dtype = str(next(backend._model.parameters()).dtype)
    dump_json(
        paths.actuator_dir / "training_manifest.json",
        {
            **protocol.snapshot_manifest(job),
            "gpu": gpu,
            "model_path": pinned_model,
            "resolved_dtype": resolved_dtype,
            "selected_heads_csv": str(paths.selected_heads),
            "selected_heads": [head.head_id for head in heads],
            "selected_head_groups": {
                role: [head.head_id for head in group]
                for role, group in head_groups.items()
            },
            "selector_specific_downstream_variable": protocol.site["downstream_contract"][
                "selector_specific_downstream_variable"
            ],
            "actuator_training_partition": protocol.site["downstream_contract"][
                "actuator_partition_by_selector"
            ][job.selector],
            "role_bank_payloads": [
                {
                    "selection_role": run["training_group"],
                    "payload": str(run["payload"]),
                    "payload_sha256": _sha256(Path(run["payload"])),
                    "heads": run["heads"],
                }
                for run in bank_runs
            ],
            "inference_composition": (
                signed_contract["inference_composition"]
                if job.selector in SIGNED_SELECTORS
                else "single_unsigned_bank"
            ),
            "alpha_policy": (
                signed_contract["alpha_policy"]
                if job.selector in SIGNED_SELECTORS
                else "one_common_alpha"
            ),
            "independent_role_alpha_search": False,
            "locator_operations_present": False,
            "pair_artifact": str(paths.pairs_csv),
            "pair_artifact_sha256": _sha256(paths.pairs_csv),
            "actual_train_rows": len(train_pairs),
            "actual_val_rows": len(val_pairs),
            "optimizer_effective": {
                "name": common["optimizer"],
                "lr": common["lr"],
                "betas": common["betas"],
                "eps": common["eps"],
                "weight_decay": common["weight_decay"],
                "amsgrad": common["amsgrad"],
            },
            "actuator_payload": str(payload),
            "code_commit": code_commit,
            "dirty_tree_manifest": dirty_tree,
            "teacher_forced_audit_role": "per-bank training diagnostic only; never held-out evidence",
        },
    )
    print(f"[site-train] complete job={job.job_id} actuator={payload}", flush=True)


if __name__ == "__main__":
    main()
