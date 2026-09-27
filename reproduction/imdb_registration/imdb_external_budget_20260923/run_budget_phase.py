import time
started = time.monotonic()
import json, sys, os, platform, importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
from experiments.shared.flow_executor import FlowAwaitingAudit, execute_flow, DEFAULT_HANDLERS
from experiments.shared.runtime_registry import load_runtime_registry
from experiments.site.run_site_reproduction import prepare_site_runtime
method, mode = sys.argv[1:]
assert method in ("bipo", "loreft") and mode in ("controller", "test")
run = Path(__file__).resolve().parent
root = Path("LOCAL_HOME/rpec_outputs/imdb_external_budget_v7/performance/formal") / f"performance__imdb__gpt2_large__{method}_native"
registry = run / "runtime_registry.json"
load_runtime_registry(registry)
prepare_site_runtime(device="cuda")({"flow": {"runtime_cell_root": str(root)}})
job = os.environ.get("SLURM_JOB_ID")
events = run / "logs" / f"{job}_{method}_{mode}_stages.jsonl"
def stamp(handler):
    def timed(context):
        begin = time.monotonic()
        with events.open("a") as handle:
            handle.write(json.dumps({"event": "start", "stage": context.stage["stage_id"], "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
        try:
            return handler(context)
        finally:
            with events.open("a") as handle:
                handle.write(json.dumps({"event": "end", "stage": context.stage["stage_id"], "utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - begin}) + "\n")
    return timed
def pause_before_test(context):
    raise FlowAwaitingAudit("Candidate controller complete; final test is a separately timed dependent job")
overrides = {name: stamp(handler) for name, handler in DEFAULT_HANDLERS.items()}
if mode == "controller": overrides["test_gate"] = pause_before_test
packages = {}
for name in ("torch", "transformers", "trl", "datasets", "pyreft", "peft", "accelerate", "numpy"):
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name] = None
environment = {"job_id": job, "method": method, "phase": mode, "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages, "code_revision": "3c610fe1", "registry": str(registry)}
import torch
environment.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
(run / f"{job}_{method}_{mode}_environment.json").write_text(json.dumps(environment, indent=2) + "\n")
result = execute_flow(root, handler_overrides=overrides, freeze=mode == "test")
if mode == "controller":
    expected = 10 if method == "bipo" else 6
    stages = result["stages"]
    trained = [name for name, value in stages.items() if name.startswith(method + "_train_") and value["status"] == "completed"]
    assert len(trained) == expected, trained
    assert stages[method + "_candidate_freeze"]["status"] == "completed"
    reg = json.loads(registry.read_text())["entries"]
    for role in ("training", "selector", "validation", "test"):
        manifest = json.loads((root / f"data/roles/{role}.manifest.json").read_text())
        actual = [json.loads(line) for line in (root / manifest["role_artifact"]["relative_path"]).open()]
        source = [json.loads(line) for line in Path(reg["datasets/imdb/shared_frozen_" + role]["path"]).open()]
        assert actual == source, f"shared {role} role drift"
    measurement = {"job_id": job, "method": method, "controller_elapsed_seconds": time.monotonic() - started, "candidates_completed": len(trained), "target_seconds_approx": 2400, "test_included": False, "data_role_exact_reuse_verified": True, "gpu_trace": str(run / "logs" / f"{job}_{method}_{mode}_gpu.csv"), "stage_trace": str(events), "time_source_note": "Python entrypoint elapsed; final H100 occupancy uses completed Slurm ElapsedRaw."}
    (run / f"{method}_controller_measurement.json").write_text(json.dumps(measurement, indent=2) + "\n")
    print(json.dumps(measurement))
else:
    assert result["status"] == "frozen", result["status"]
    print(json.dumps({"job_id": job, "method": method, "phase": mode, "status": result["status"]}))
