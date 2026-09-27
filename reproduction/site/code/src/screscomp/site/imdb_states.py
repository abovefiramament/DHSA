from __future__ import annotations

import json
from typing import Any

from screscomp.data import dump_json, dump_jsonl, load_jsonl
from screscomp.site.model import PreOInterface, StateAction
from screscomp.site.protocol import (
    SiteJob,
    SitePaths,
    SiteProtocol,
    require_files,
    sha256_file,
)


BATCH_SEED_STRIDE = 1009


def imdb_condition_rows(
    job: SiteJob,
    source_rows: list[dict[str, Any]],
    condition: str,
) -> list[dict[str, Any]]:
    template = job.task["states"][f"{condition}_template"]
    return [
        {
            "sample_id": str(row["sample_id"]),
            "source_row_index": row.get("source_row_index", ""),
            "prefix": str(row["prefix"]),
            "condition": condition,
            "prompt": template.format(prefix=str(row["prefix"])),
        }
        for row in source_rows
    ]


def generate_imdb_condition(
    interface: PreOInterface,
    prompts: list[dict[str, Any]],
    *,
    action: StateAction | None,
    state_spec: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    batch_size = int(state_spec["generation_batch_size"])
    rows: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[start : start + batch_size]
        batch_seed = seed + batch_index * BATCH_SEED_STRIDE
        interface.torch.manual_seed(batch_seed)
        if interface.torch.cuda.is_available():
            interface.torch.cuda.manual_seed_all(batch_seed)
        generated = interface.generate_batch(
            [str(row["prompt"]) for row in batch],
            action=action,
            max_new_tokens=int(state_spec["max_new_tokens"]),
            do_sample=bool(state_spec["do_sample"]),
            temperature=float(state_spec["temperature"]),
            top_p=float(state_spec["top_p"]),
            top_k=int(state_spec["generation_top_k"]),
        )
        for source, completion in zip(batch, generated, strict=True):
            rows.append({**source, **completion, "batch_seed": batch_seed})
    return rows


def _condition_artifacts(paths: SitePaths, condition: str):
    if condition == "good":
        return paths.selector_good_generations, paths.selector_good_generation_manifest
    if condition == "base":
        return paths.selector_base_generations, paths.selector_base_generation_manifest
    raise ValueError(f"unknown IMDb selector condition: {condition}")


def _generation_contract(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    state_spec: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    artifact, _manifest = _condition_artifacts(paths, condition)
    return {
        "protocol_id": protocol.data["protocol_id"],
        "protocol_version": protocol.data["version"],
        "config": str(protocol.config_path),
        "config_sha256": protocol.config_sha256,
        "dataset": job.dataset,
        "subset": job.subset,
        "model_key": job.model_key,
        "model": job.model,
        "model_revision": job.model_revision,
        "use_chat_template": job.use_chat_template,
        "source_artifact": str(paths.selector_prompts),
        "source_artifact_sha256": sha256_file(paths.selector_prompts),
        "state_artifact": str(artifact),
        "condition": condition,
        "prompt_template": job.task["states"][f"{condition}_template"],
        "selector_state_generation": state_spec,
        "batch_seed_stride": BATCH_SEED_STRIDE,
    }


def _validate_state_rows(
    rows: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    state_spec: dict[str, Any],
    *,
    condition: str,
) -> None:
    expected_ids = [row["sample_id"] for row in prompts]
    actual_ids = [str(row.get("sample_id", "")) for row in rows]
    if actual_ids != expected_ids:
        raise RuntimeError(f"shared IMDb {condition} states do not match selector prompt order")
    if [str(row.get("prompt", "")) for row in rows] != [
        str(row["prompt"]) for row in prompts
    ]:
        raise RuntimeError(f"shared IMDb {condition} state prompt drift")
    batch_size = int(state_spec["generation_batch_size"])
    expected_seeds = [
        int(state_spec["generation_seed"]) + (index // batch_size) * BATCH_SEED_STRIDE
        for index in range(len(rows))
    ]
    if [row.get("batch_seed") for row in rows] != expected_seeds:
        raise RuntimeError(f"shared IMDb {condition} state seed schedule drift")
    for row in rows:
        token_ids = row.get("generated_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"shared IMDb {condition} state has no generated token ids")
        if int(row.get("generated_tokens", -1)) != len(token_ids):
            raise RuntimeError(f"shared IMDb {condition} state token count drift")


def _load_or_generate_condition(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
    state_spec: dict[str, Any],
    prompts: list[dict[str, Any]],
    condition: str,
) -> list[dict[str, Any]]:
    artifact, manifest_path = _condition_artifacts(paths, condition)
    contract = _generation_contract(protocol, job, paths, state_spec, condition)
    if artifact.is_file():
        require_files([manifest_path])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: {"registered": manifest.get(key), "locked": value}
            for key, value in contract.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"shared IMDb {condition} generation contract drift: {mismatches}"
            )
        rows = load_jsonl(artifact)
        if manifest.get("actual_n") != len(rows):
            raise RuntimeError(f"shared IMDb {condition} manifest has the wrong actual_n")
    else:
        if manifest_path.is_file():
            raise RuntimeError(
                f"shared IMDb {condition} manifest exists without its JSONL artifact"
            )
        rows = generate_imdb_condition(
            interface,
            prompts,
            action=None,
            state_spec=state_spec,
            seed=int(state_spec["generation_seed"]),
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        dump_jsonl(artifact, rows)
        dump_json(
            manifest_path,
            {
                **contract,
                "actual_n": len(rows),
                "batch_seed_schedule": sorted({row["batch_seed"] for row in rows}),
            },
        )
    _validate_state_rows(rows, prompts, state_spec, condition=condition)
    return rows


def ensure_imdb_selector_states(
    protocol: SiteProtocol,
    job: SiteJob,
    paths: SitePaths,
    interface: PreOInterface,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require_files([paths.selector_prompts])
    state_spec = protocol.selector_state_generation(job)
    if state_spec is None:
        raise RuntimeError("IMDb selector state-generation contract is missing")
    if state_spec["conditions"] != ["good", "base"]:
        raise RuntimeError("IMDb selector conditions must be ordered as good, base")
    source_rows = [
        row
        for row in load_jsonl(paths.selector_prompts)
        if str(row.get("admitted", "1")) not in {"0", "false", "False"}
    ][: int(state_spec["prompt_rows"])]
    if len(source_rows) != int(state_spec["prompt_rows"]):
        raise ValueError(
            f"IMDb selector requires {state_spec['prompt_rows']} admitted prompts, "
            f"got {len(source_rows)}"
        )
    good_prompts = imdb_condition_rows(job, source_rows, "good")
    base_prompts = imdb_condition_rows(job, source_rows, "base")
    good_rows = _load_or_generate_condition(
        protocol, job, paths, interface, state_spec, good_prompts, "good"
    )
    base_rows = _load_or_generate_condition(
        protocol, job, paths, interface, state_spec, base_prompts, "base"
    )
    good_ids = [row["sample_id"] for row in good_rows]
    base_ids = [row["sample_id"] for row in base_rows]
    if good_ids != base_ids:
        raise RuntimeError("IMDb good/base state pairing drift")
    if [row["batch_seed"] for row in good_rows] != [
        row["batch_seed"] for row in base_rows
    ]:
        raise RuntimeError("IMDb good/base states must use common random numbers")
    pair_manifest = {
        "protocol_id": protocol.data["protocol_id"],
        "protocol_version": protocol.data["version"],
        "config_sha256": protocol.config_sha256,
        "dataset": job.dataset,
        "subset": job.subset,
        "model_key": job.model_key,
        "model": job.model,
        "model_revision": job.model_revision,
        "source_artifact": str(paths.selector_prompts),
        "source_artifact_sha256": sha256_file(paths.selector_prompts),
        "good_state_artifact": str(paths.selector_good_generations),
        "good_state_sha256": sha256_file(paths.selector_good_generations),
        "base_state_artifact": str(paths.selector_base_generations),
        "base_state_sha256": sha256_file(paths.selector_base_generations),
        "pair_groups": len(good_rows),
        "sample_ids": good_ids,
        "positive_state": job.task["states"]["iti_positive_state"],
        "negative_state": job.task["states"]["iti_negative_state"],
        "common_random_numbers_across_conditions": True,
        "exact_generated_token_ids_shared": True,
        "selector_state_generation": state_spec,
    }
    if paths.selector_state_pair_manifest.is_file():
        registered = json.loads(
            paths.selector_state_pair_manifest.read_text(encoding="utf-8")
        )
        if registered != pair_manifest:
            raise RuntimeError("IMDb shared good/base state-pair manifest drift")
    else:
        dump_json(paths.selector_state_pair_manifest, pair_manifest)
    return good_rows, base_rows
