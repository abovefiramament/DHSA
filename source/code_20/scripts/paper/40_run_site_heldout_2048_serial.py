#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from screscomp.data import dump_csv, dump_json, load_jsonl  # noqa: E402
from screscomp.site.runtime import resolve_pinned_model  # noqa: E402
from screscomp.tldr_site.head_audit import (  # noqa: E402
    load_frozen_alpha,
    load_frozen_configurations,
)


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixed implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREP = _import_file(
    "site_heldout_2048_prepare",
    REPO_ROOT / "scripts/paper/39_prepare_site_heldout_2048_protocol.py",
)
TLDR_V6 = _import_file(
    "tldr_site_direct_k8_v6_runner",
    REPO_ROOT / "scripts/paper/36_run_tldr_site_reft_direct_k8_disjoint_alpha_dev_serial.py",
)

Heldout2048Protocol = PREP.Heldout2048Protocol
require = PREP.require
sha256_file = PREP.sha256_file
TARGET_ROWS = PREP.TARGET_ROWS
TLDR_JUDGE_AMENDMENT = REPO_ROOT / PREP.TLDR_JUDGE_AMENDMENT_RELATIVE_PATH


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


class SerialHeldout2048Runner:
    def __init__(self, protocol: Heldout2048Protocol) -> None:
        self.protocol = protocol
        self.root = protocol.run_root
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        input_manifest = self.root / "protocol/input_manifest.json"
        fixed_manifest = self.root / "protocol/fixed_protocol_manifest.json"
        require(input_manifest.is_file() and fixed_manifest.is_file(), "run CPU preparation before GPU execution")
        require(load_json(input_manifest).get("config_sha256") == protocol.config_sha256, "input manifest config drift")
        require(load_json(fixed_manifest).get("config_sha256") == protocol.config_sha256, "fixed protocol manifest drift")

    def command(self, name: str, argv: list[str], *, env: dict[str, str] | None = None) -> None:
        log_path = self.logs / f"{safe_id(name)}.log"
        record_path = self.logs / f"{safe_id(name)}.command.json"
        started = now()
        record = {
            **self.protocol.snapshot(),
            "name": name,
            "argv": argv,
            "cwd": str(REPO_ROOT),
            "started_at": started,
            "status": "running",
            "log": str(log_path),
        }
        dump_json(record_path, record)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                argv,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record.update(
            {
                "ended_at": now(),
                "returncode": completed.returncode,
                "status": "complete" if completed.returncode == 0 else "failed",
            }
        )
        dump_json(record_path, record)
        if completed.returncode != 0:
            raise RuntimeError(f"command failed: {name}; see {log_path}")

    def gpu_environment(self, stage: str) -> dict[str, str]:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = []
        for line in query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                continue
            rows.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_used_mib": int(fields[3]),
                    "memory_total_mib": int(fields[4]),
                    "utilization_percent": int(fields[5]),
                }
            )
        candidates = [
            row
            for row in rows
            if row["index"] in self.protocol.data["execution"]["preferred_physical_gpu_indices"]
        ]
        require(len(candidates) == 2, "physical GPU 2/3 inventory is incomplete")
        selected = sorted(
            candidates,
            key=lambda row: (
                row["memory_used_mib"],
                row["utilization_percent"],
                row["index"],
            ),
        )[0]
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        snapshot = {
            **self.protocol.snapshot(),
            "stage": stage,
            "observed_at": now(),
            "gpus": rows,
            "selected": selected,
            "visible_compute_processes": [
                line.strip() for line in processes.stdout.splitlines() if line.strip()
            ],
            "co_tenancy_allowed": True,
        }
        snapshot_path = self.root / "protocol" / f"gpu_launch_{safe_id(stage)}.json"
        dump_json(snapshot_path, snapshot)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
        env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}"
        return env

    @staticmethod
    def _confiqa_generation_rows(directory: Path) -> Path:
        return directory / "generation_rows.jsonl"

    @staticmethod
    def _confiqa_generation_summary(directory: Path) -> Path:
        return directory / "generation_summary.csv"

    def _confiqa_complete(self, directory: Path, expected_ids: set[str]) -> bool:
        rows_path = self._confiqa_generation_rows(directory)
        summary_path = self._confiqa_generation_summary(directory)
        if not rows_path.is_file() or not summary_path.is_file():
            return False
        rows = load_jsonl(rows_path)
        ids = [str(row["sample_id"]) for row in rows]
        require(len(ids) <= TARGET_ROWS, f"too many ConFiQA generation rows: {rows_path}")
        if len(ids) != TARGET_ROWS:
            return False
        require(len(set(ids)) == TARGET_ROWS, f"duplicate ConFiQA generation rows: {rows_path}")
        require(set(ids) == expected_ids, f"ConFiQA generation sample mismatch: {rows_path}")
        with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
            summary = list(csv.DictReader(stream))
        require(len(summary) == 1, f"unexpected ConFiQA summary row count: {summary_path}")
        require(int(summary[0]["n"]) == TARGET_ROWS, f"ConFiQA summary n drift: {summary_path}")
        return True

    def _confiqa_command(
        self,
        *,
        model_path: str,
        use_chat_template: bool,
        input_rows: Path,
        output_dir: Path,
        controls: str,
        actuator: Path | None,
        downstream: dict[str, Any],
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "screscomp.cli.cecm_run_joint_actuator_generation",
            "--model",
            model_path,
            "--eval-open-rows",
            str(input_rows),
            "--controls",
            controls,
            "--generation-prompt-key",
            "official_rag",
            "--prior-source",
            "dataset_orig",
            "--scoring-kind",
            "source",
            "--split",
            "",
            "--start",
            "0",
            "--max-rows",
            str(TARGET_ROWS),
            "--val-mod",
            str(downstream["val_mod"]),
            "--generation-apply-mode",
            "all",
            "--max-new-tokens",
            str(downstream["generation"]["max_new_tokens"]),
            "--stop-strings",
            ",".join(downstream["generation"]["stop_strings"]),
            "--empty-cache-every",
            "25",
            "--device",
            "cuda:0",
            "--torch-dtype",
            "auto",
            "--out-dir",
            str(output_dir),
        ]
        if actuator is not None:
            command.extend(["--head-actuators", f"site={actuator}"])
        if downstream["generation"]["do_sample"]:
            command.append("--do-sample")
        if use_chat_template:
            command.append("--use-chat-template")
        return command

    def _source_confiqa_jobs(self, model_key: str) -> Iterable[dict[str, Any]]:
        source = self.protocol.confiqa_source
        extension = self.protocol.data["confiqa"]
        matrix = next(
            (
                row
                for row in source["site"]["matrix"]
                if row["dataset"] == "confiqa" and row["model_key"] == model_key
            ),
            None,
        )
        require(matrix is not None, f"unknown ConFiQA model key: {model_key}")
        selectors = source["site"]["selectors"]
        for subset in extension["subsets"]:
            for selector, spec in selectors.items():
                if selector == "random":
                    replicates = [f"seed_{seed}" for seed in spec["replicate_seeds"]]
                else:
                    replicates = ["primary"]
                for replicate in replicates:
                    job_id = "__".join(["confiqa", subset, model_key, selector, replicate])
                    yield {
                        "job_id": job_id,
                        "subset": subset,
                        "model": matrix["model"],
                        "model_revision": matrix["revision"],
                        "model_key": model_key,
                        "use_chat_template": bool(matrix["use_chat_template"]),
                        "selector": selector,
                        "replicate": replicate,
                    }

    def _validate_source_confiqa_job(self, job: dict[str, Any]) -> dict[str, Any]:
        extension = self.protocol.data["confiqa"]
        root = Path(extension["source_run_root"]) / job["job_id"]
        required = [root / rel for rel in extension["source_artifact_requirements"]]
        missing = [str(path) for path in required if not path.is_file()]
        require(not missing, f"source Site job is not frozen and complete: {missing}")
        source_hash = extension["source_config_sha256"]
        manifests = {
            "training": root / "actuator/training_manifest.json",
            "dev": root / "evaluation/dev_manifest.json",
            "test": root / "evaluation/test_manifest.json",
        }
        for name, path in manifests.items():
            payload = load_json(path)
            require(payload.get("config_sha256") == source_hash, f"source Site {name} config drift: {job['job_id']}")
            require(payload.get("job_id") == job["job_id"], f"source Site {name} job drift: {job['job_id']}")
        training = load_json(manifests["training"])
        actuator = root / "actuator/head_actuator.pt"
        require(training.get("actuator_payload") == str(actuator), f"source actuator path drift: {job['job_id']}")
        selection = root / "evaluation/alpha_selection.json"
        selected_alpha = float(load_json(selection)["selected_alpha"])
        source_grid = [
            float(value)
            for value in self.protocol.confiqa_source["site"]["tasks"]["confiqa"]["downstream"]["alpha_nonzero_grid"]
        ]
        require(selected_alpha in source_grid, f"source selected alpha outside fixed grid: {job['job_id']}")
        return {
            "root": root,
            "actuator": actuator,
            "actuator_sha256": sha256_file(actuator),
            "training_manifest": manifests["training"],
            "training_manifest_sha256": sha256_file(manifests["training"]),
            "alpha_selection": selection,
            "alpha_selection_sha256": sha256_file(selection),
            "selected_alpha": selected_alpha,
            "dev_manifest": manifests["dev"],
            "dev_manifest_sha256": sha256_file(manifests["dev"]),
            "test_manifest": manifests["test"],
            "test_manifest_sha256": sha256_file(manifests["test"]),
        }

    def run_confiqa(self, model_keys: list[str]) -> None:
        downstream = self.protocol.confiqa_source["site"]["tasks"]["confiqa"]["downstream"]
        data_root = Path(self.protocol.data["confiqa"]["sample_construction"]["data_root"])
        env = self.gpu_environment("confiqa_" + "_".join(model_keys))
        results: list[dict[str, Any]] = []
        for model_key in model_keys:
            jobs = list(self._source_confiqa_jobs(model_key))
            model_path = resolve_pinned_model(jobs[0]["model"], jobs[0]["model_revision"])
            for subset in self.protocol.data["confiqa"]["subsets"]:
                input_rows = data_root / subset / "heldout_test_2048.jsonl"
                input_manifest = data_root / subset / "input_manifest.json"
                require(input_rows.is_file() and input_manifest.is_file(), f"missing fixed ConFiQA input: {subset}")
                expected_ids = {str(row["sample_id"]) for row in load_jsonl(input_rows)}
                require(len(expected_ids) == TARGET_ROWS, f"ConFiQA fixed input n drift: {subset}")
                base_dir = self.root / "confiqa" / subset / model_key / "base"
                if not self._confiqa_complete(base_dir, expected_ids):
                    self.command(
                        f"confiqa_{subset}_{model_key}_base",
                        self._confiqa_command(
                            model_path=model_path,
                            use_chat_template=jobs[0]["use_chat_template"],
                            input_rows=input_rows,
                            output_dir=base_dir,
                            controls="base=",
                            actuator=None,
                            downstream=downstream,
                        ),
                        env=env,
                    )
                require(self._confiqa_complete(base_dir, expected_ids), f"ConFiQA base incomplete: {subset}/{model_key}")
                for job in [row for row in jobs if row["subset"] == subset]:
                    source = self._validate_source_confiqa_job(job)
                    alpha = source["selected_alpha"]
                    alpha_name = "alpha_" + f"{alpha:.1f}".replace(".", "p")
                    output_dir = self.root / "confiqa" / subset / model_key / job["job_id"]
                    if not self._confiqa_complete(output_dir, expected_ids):
                        self.command(
                            f"{job['job_id']}_heldout_2048",
                            self._confiqa_command(
                                model_path=model_path,
                                use_chat_template=job["use_chat_template"],
                                input_rows=input_rows,
                                output_dir=output_dir,
                                controls=f"{alpha_name}=head_act:site:{alpha}:all",
                                actuator=source["actuator"],
                                downstream=downstream,
                            ),
                            env=env,
                        )
                    require(self._confiqa_complete(output_dir, expected_ids), f"ConFiQA job incomplete: {job['job_id']}")
                    with self._confiqa_generation_summary(output_dir).open(
                        "r", encoding="utf-8-sig", newline=""
                    ) as stream:
                        summary = list(csv.DictReader(stream))[0]
                    result = {
                        "dataset": "confiqa",
                        "subset": subset,
                        "model_key": model_key,
                        "job_id": job["job_id"],
                        "selector": job["selector"],
                        "replicate": job["replicate"],
                        "alpha": alpha,
                        "actual_n": int(summary["n"]),
                        **{
                            key: value
                            for key, value in summary.items()
                            if key not in {"control_name", "n"}
                        },
                    }
                    manifest = {
                        **self.protocol.snapshot(),
                        "dataset": "confiqa",
                        "subset": subset,
                        "model_key": model_key,
                        "job_id": job["job_id"],
                        "selector": job["selector"],
                        "replicate": job["replicate"],
                        "input": str(input_rows),
                        "input_sha256": sha256_file(input_rows),
                        "input_manifest": str(input_manifest),
                        "input_manifest_sha256": sha256_file(input_manifest),
                        "source_v12_config": self.protocol.data["confiqa"]["source_config"],
                        "source_v12_config_sha256": self.protocol.data["confiqa"]["source_config_sha256"],
                        "source_artifacts": {
                            key: str(value) if isinstance(value, Path) else value
                            for key, value in source.items()
                        },
                        "output_dir": str(output_dir),
                        "generation_rows_sha256": sha256_file(self._confiqa_generation_rows(output_dir)),
                        "generation_summary_sha256": sha256_file(self._confiqa_generation_summary(output_dir)),
                        "actual_n": TARGET_ROWS,
                        "status": "complete",
                    }
                    dump_json(output_dir / "heldout_2048_manifest.json", manifest)
                    results.append(result)
        summary_path = self.root / "confiqa/confiqa_heldout_2048_results.csv"
        merged: dict[str, dict[str, Any]] = {}
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
                merged.update(
                    {
                        str(row["job_id"]): dict(row)
                        for row in csv.DictReader(stream)
                    }
                )
        merged.update({str(row["job_id"]): row for row in results})
        ordered = sorted(
            merged.values(),
            key=lambda row: (
                str(row["model_key"]),
                str(row["subset"]),
                str(row["selector"]),
                str(row["replicate"]),
            ),
        )
        dump_csv(summary_path, ordered)
        print(
            f"[site-heldout-2048] ConFiQA complete new_rows={len(results)} "
            f"total_rows={len(ordered)} summary={summary_path}",
            flush=True,
        )

    def _load_tldr_source(self):
        source_protocol = TLDR_V6.DirectK8Protocol.load(
            self.protocol.tldr_source_config_path,
            validate_files=True,
        )
        source_runner = TLDR_V6.DirectK8Runner(source_protocol)
        source_root = Path(self.protocol.data["tldr"]["source_run_root"])
        require(source_protocol.paths.root.resolve() == source_root.resolve(), "TLDR source run-root drift")
        for relative in self.protocol.data["tldr"]["frozen_state_requirements"]:
            require((source_root / relative).is_file(), f"missing frozen TLDR source state: {relative}")
        return source_protocol, source_runner

    def validate_sources(self, model_keys: list[str]) -> None:
        confiqa_jobs = 0
        for model_key in model_keys:
            for job in self._source_confiqa_jobs(model_key):
                self._validate_source_confiqa_job(job)
                confiqa_jobs += 1
        source_protocol, source_runner = self._load_tldr_source()
        configurations = {
            (str(row["selector"]), str(row["pool_id"])): row
            for row in load_frozen_configurations(source_protocol)
        }
        frozen = load_frozen_alpha(source_protocol)
        require(len(frozen) == 6, "TLDR frozen-alpha coverage drift")
        for row in frozen:
            key = (str(row["selector"]), str(row["pool_id"]))
            require(key in configurations, f"TLDR frozen configuration missing: {key}")
            source_runner._final_actuator_paths(configurations[key])
        print(
            f"[site-heldout-2048] validate-only complete "
            f"confiqa_jobs={confiqa_jobs} tldr_configurations={len(frozen)}",
            flush=True,
        )

    @staticmethod
    def _tldr_output_complete(
        output: Path,
        expected_ids: set[str],
        *,
        control_name: str,
        alpha: float,
        actuators: list[Path],
    ) -> bool:
        manifest_path = output.with_suffix(".manifest.json")
        if not output.is_file() or not manifest_path.is_file():
            return False
        rows = load_jsonl(output)
        require(len(rows) <= TARGET_ROWS, f"too many TLDR generation rows: {output}")
        if len(rows) != TARGET_ROWS:
            return False
        ids = [str(row["sample_id"]) for row in rows]
        require(len(set(ids)) == TARGET_ROWS and set(ids) == expected_ids, f"TLDR generation sample drift: {output}")
        expected_actuators = [str(path) for path in actuators]
        for row in rows:
            require(str(row["control_name"]) == control_name, f"TLDR control drift: {output}")
            require(float(row["alpha"]) == float(alpha), f"TLDR alpha drift: {output}")
            require(row.get("reft_actuator_paths", []) == expected_actuators, f"TLDR actuator drift: {output}")
        manifest = load_json(manifest_path)
        require(int(manifest.get("completed_rows", -1)) == TARGET_ROWS, f"TLDR generation manifest n drift: {output}")
        require(manifest.get("control_name") == control_name, f"TLDR generation manifest control drift: {output}")
        require(manifest.get("reft_actuators") == expected_actuators, f"TLDR generation manifest actuator drift: {output}")
        return True

    def _tldr_candidates(self) -> dict[str, Path]:
        generation_dir = self.root / "tldr/generations"
        candidates = {"sft": generation_dir / "sft.jsonl"}
        source_protocol, _source_runner = self._load_tldr_source()
        for row in load_frozen_alpha(source_protocol):
            name = f"F__{safe_id(str(row['selector']))}__{safe_id(str(row['pool_id']))}"
            candidates[name] = generation_dir / f"{name}.jsonl"
        require(all(path.is_file() for path in candidates.values()), "TLDR 2048 generation set is incomplete")
        return candidates

    def run_tldr_generate(self) -> None:
        source_protocol, source_runner = self._load_tldr_source()
        prompts = Path(self.protocol.data["tldr"]["output_file"])
        require(prompts.is_file() and len(load_jsonl(prompts)) == TARGET_ROWS, "fixed TLDR 2048 input drift")
        expected_ids = {str(row["sample_id"]) for row in load_jsonl(prompts)}
        require(len(expected_ids) == TARGET_ROWS, "TLDR fixed input sample IDs are not unique")
        env = self.gpu_environment("tldr_generate")
        generation_dir = self.root / "tldr/generations"
        generation_dir.mkdir(parents=True, exist_ok=True)
        base = generation_dir / "sft.jsonl"
        if not self._tldr_output_complete(
            base,
            expected_ids,
            control_name="sft",
            alpha=0.0,
            actuators=[],
        ):
            self.command(
                "tldr_heldout_2048_sft",
                source_runner._generation_command(
                    prompts=prompts,
                    output=base,
                    control_name="sft",
                    alphas=[0.0],
                    actuators=[],
                    max_rows=TARGET_ROWS,
                ),
                env=env,
            )
        require(
            self._tldr_output_complete(base, expected_ids, control_name="sft", alpha=0.0, actuators=[]),
            "TLDR SFT 2048 generation incomplete",
        )
        configurations = {
            (str(row["selector"]), str(row["pool_id"])): row
            for row in load_frozen_configurations(source_protocol)
        }
        for selection in load_frozen_alpha(source_protocol):
            key = (str(selection["selector"]), str(selection["pool_id"]))
            require(key in configurations, f"TLDR frozen-alpha configuration missing: {key}")
            name = f"F__{safe_id(key[0])}__{safe_id(key[1])}"
            output = generation_dir / f"{name}.jsonl"
            actuators = source_runner._final_actuator_paths(configurations[key])
            alpha = float(selection["alpha"])
            if not self._tldr_output_complete(
                output,
                expected_ids,
                control_name=name,
                alpha=alpha,
                actuators=actuators,
            ):
                self.command(
                    f"tldr_heldout_2048_{name}",
                    source_runner._generation_command(
                        prompts=prompts,
                        output=output,
                        control_name=name,
                        alphas=[alpha],
                        actuators=actuators,
                        max_rows=TARGET_ROWS,
                    ),
                    env=env,
                )
            require(
                self._tldr_output_complete(
                    output,
                    expected_ids,
                    control_name=name,
                    alpha=alpha,
                    actuators=actuators,
                ),
                f"TLDR 2048 generation incomplete: {name}",
            )
            dump_json(
                output.with_suffix(".heldout_2048.json"),
                {
                    **self.protocol.snapshot(),
                    "dataset": "tldr",
                    "source_v6_config": str(source_protocol.config_path),
                    "source_v6_config_sha256": source_protocol.config_sha256,
                    "source_frozen_alpha": str(source_protocol.paths.frozen_alpha),
                    "source_frozen_alpha_sha256": sha256_file(source_protocol.paths.frozen_alpha),
                    "control_name": name,
                    "selector": key[0],
                    "pool_id": key[1],
                    "alpha": alpha,
                    "actuators": [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in actuators
                    ],
                    "input": str(prompts),
                    "input_sha256": sha256_file(prompts),
                    "output": str(output),
                    "output_sha256": sha256_file(output),
                    "generation_manifest_sha256": sha256_file(output.with_suffix(".manifest.json")),
                    "actual_n": TARGET_ROWS,
                    "status": "complete",
                },
            )
        print("[site-heldout-2048] TLDR generation complete configurations=6", flush=True)

    def run_tldr_evaluate(self) -> None:
        source_protocol, source_runner = self._load_tldr_source()
        prompts = Path(self.protocol.data["tldr"]["output_file"])
        candidates = self._tldr_candidates()
        require(TLDR_JUDGE_AMENDMENT.is_file(), "missing TLDR human-only judge amendment")
        amendment = load_json(TLDR_JUDGE_AMENDMENT)
        require(
            amendment.get("protocol_id")
            == "site_heldout_2048_tldr_human_only_judge_amendment_20260727_v1",
            "unexpected TLDR judge amendment id",
        )
        require(amendment.get("version") == 1, "unexpected TLDR judge amendment version")
        require(amendment.get("status") == "locked_after_generation_before_valid_judging", "TLDR judge amendment is not locked")
        require(
            amendment["parent"]["config_sha256"] == self.protocol.config_sha256,
            "TLDR judge amendment parent-config drift",
        )
        judge = amendment["judge"]
        require(judge["comparison_scope"] == "each_fixed_candidate_vs_human_reference_only", "TLDR judge scope drift")
        require(judge["runner_mode"] == "human", "TLDR judge runner mode drift")
        require(int(judge["expected_rows"]) == TARGET_ROWS, "TLDR human-only judge n drift")
        require(int(judge["candidate_count"]) == len(candidates), "TLDR human-only candidate-count drift")
        require(list(judge["candidate_names"]) == list(candidates), "TLDR human-only candidate order drift")
        require(int(judge["first_pass_packets"]) == TARGET_ROWS * len(candidates), "TLDR first-pass count drift")
        require(float(judge["review_fraction"]) == 1.0, "TLDR full order-swap review drift")
        require(int(judge["maximum_total_judgments"]) == 2 * TARGET_ROWS * len(candidates), "TLDR total judge count drift")
        require(
            float(source_protocol.data["evaluation"]["judge"]["review_fraction"])
            == float(judge["review_fraction"]),
            "TLDR source review-fraction drift",
        )
        judge_dir = self.root / judge["output_dir"]
        health = self.root / "tldr/health.csv"
        judge_summary = judge_dir / "pairwise_summary.json"
        if not judge_summary.is_file():
            self.command(
                "tldr_heldout_2048_human_only_judge",
                source_runner._judge_command(
                    prompts=prompts,
                    candidates=candidates,
                    out_dir=judge_dir,
                    mode=judge["runner_mode"],
                    expected_rows=TARGET_ROWS,
                ),
            )
        if not health.is_file():
            self.command(
                "tldr_heldout_2048_health",
                source_runner._health_command(
                    prompts,
                    candidates,
                    health,
                    expected_rows=TARGET_ROWS,
                ),
            )
        require(judge_summary.is_file() and health.is_file(), "TLDR 2048 evaluation incomplete")
        pair_rows = load_json(judge_summary)["pairwise"]
        with health.open("r", encoding="utf-8-sig", newline="") as stream:
            health_rows = {str(row["method"]): row for row in csv.DictReader(stream)}
        frozen = load_frozen_alpha(source_protocol)
        table_rows = []
        table_candidates = [
            {
                "row_label": "SFT",
                "name": "sft",
                "selector": "sft",
                "pool_id": "",
                "alpha": 0.0,
            }
        ]
        table_candidates.extend(
            {
                "row_label": f"{row['selector']} + per-head audit",
                "name": f"F__{safe_id(str(row['selector']))}__{safe_id(str(row['pool_id']))}",
                "selector": row["selector"],
                "pool_id": row["pool_id"],
                "alpha": row["alpha"],
            }
            for row in frozen
        )
        for candidate in table_candidates:
            match = next(
                (
                    row
                    for row in pair_rows
                    if {str(row["left"]), str(row["right"])}
                    == {str(candidate["name"]), "human"}
                ),
                None,
            )
            require(match is not None, f"TLDR human comparison missing: {candidate['name']}")
            metrics = source_runner._oriented_pair(match, str(candidate["name"]))
            require(int(metrics["n"]) == TARGET_ROWS, f"TLDR judge n drift: {candidate['name']}")
            require(candidate["name"] in health_rows, f"TLDR health row missing: {candidate['name']}")
            row = {
                **candidate,
                "first_human_win_rate": metrics["first"],
                "swap_human_win_rate": metrics["swap"],
                "balanced_human_win_rate": metrics["balanced"],
                "order_swap_agreement": metrics["agreement"],
                "actual_n": metrics["n"],
            }
            for field in source_protocol.data["evaluation"]["health"]["required_metrics"]:
                row[field] = health_rows[candidate["name"]][field]
            table_rows.append(row)
        table = self.root / "tldr/site_selector_final_table_2048.csv"
        dump_csv(table, table_rows)
        manifest = {
            **self.protocol.snapshot(),
            "dataset": "tldr",
            "source_v6_config": str(source_protocol.config_path),
            "source_v6_config_sha256": source_protocol.config_sha256,
            "judge_amendment": str(TLDR_JUDGE_AMENDMENT),
            "judge_amendment_sha256": sha256_file(TLDR_JUDGE_AMENDMENT),
            "judge_comparison_scope": judge["comparison_scope"],
            "superseded_all_pairs_output": str(self.root / judge["superseded_output_dir"]),
            "input": str(prompts),
            "input_sha256": sha256_file(prompts),
            "judge": str(judge_summary),
            "judge_sha256": sha256_file(judge_summary),
            "health": str(health),
            "health_sha256": sha256_file(health),
            "table": str(table),
            "table_sha256": sha256_file(table),
            "actual_n": TARGET_ROWS,
            "status": "complete",
        }
        dump_json(self.root / "tldr/final_manifest.json", manifest)
        print(f"[site-heldout-2048] TLDR evaluation complete table={table}", flush=True)

    def curate(self) -> None:
        curated = REPO_ROOT / self.protocol.data["execution"]["curated_results"]
        curated.mkdir(parents=True, exist_ok=True)
        candidates = {
            "fixed_protocol_manifest.json": self.root / "protocol/fixed_protocol_manifest.json",
            "input_manifest.json": self.root / "protocol/input_manifest.json",
            "scientific_config.json": self.root / "protocol/scientific_config.json",
            "confiqa_heldout_2048_results.csv": self.root / "confiqa/confiqa_heldout_2048_results.csv",
            "tldr_site_selector_final_table_2048.csv": self.root / "tldr/site_selector_final_table_2048.csv",
            "tldr_final_manifest.json": self.root / "tldr/final_manifest.json",
        }
        published = {}
        for name, source in candidates.items():
            if not source.is_file():
                continue
            destination = curated / name
            shutil.copy2(source, destination)
            require(sha256_file(destination) == sha256_file(source), f"curated copy drift: {name}")
            published[name] = {
                "source": str(source),
                "sha256": sha256_file(destination),
            }
        dump_json(
            curated / "source_run.json",
            {
                **self.protocol.snapshot(),
                "source_run_root": str(self.root),
                "files": published,
                "status": "complete",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Site controllers on fixed 2048-row held-out inputs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=[
            "validate",
            "confiqa",
            "tldr-generate",
            "tldr-evaluate",
            "tldr",
            "all",
            "curate",
        ],
        default="all",
    )
    parser.add_argument(
        "--model-key",
        default="all",
        help="ConFiQA model key or all; ignored for TLDR-only stages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = Heldout2048Protocol.load(args.config)
    runner = SerialHeldout2048Runner(protocol)
    if args.model_key == "all":
        model_keys = list(protocol.data["confiqa"]["model_keys"])
    else:
        require(args.model_key in protocol.data["confiqa"]["model_keys"], "unknown ConFiQA model key")
        model_keys = [args.model_key]
    if args.task == "validate":
        runner.validate_sources(model_keys)
    if args.task in {"confiqa", "all"}:
        runner.run_confiqa(model_keys)
    if args.task in {"tldr-generate", "tldr", "all"}:
        runner.run_tldr_generate()
    if args.task in {"tldr-evaluate", "tldr", "all"}:
        runner.run_tldr_evaluate()
    if args.task in {"curate", "all", "tldr"}:
        runner.curate()


if __name__ == "__main__":
    main()
