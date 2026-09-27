"""Official pyReFT LoReFT baseline with preregistered four-layer tuples.

This module keeps the AxBench native geometry: four jointly trained rank-4
LoReFT interventions on ``block_output`` and shared ``f5+l5`` positions.  It
does not know dataset paths, checkpoint paths, scorers, or test partitions.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from experiments.shared.contracts import (
    NativeBaselineTrainRequest,
    NativeBaselineTrainResult,
)
from .generation_backend import CallbackGenerationBackend
from .loaders import load_rows, resolve_model_id
from .pyreft_compat import register_pyreft_model
from .sequence_scoring import paired_completion_scores
from .model_runtime import generation_cache_kwargs


PYREFT_SOURCE = {
    "repository": "https://github.com/stanfordnlp/pyreft",
    "revision": "dafd0995a366d7b47160a337dcc388eda7431821",
}
AXBENCH_SOURCE = {
    "repository": "https://github.com/stanfordnlp/axbench",
    "revision": "41c8332543e5a631f9a8c0a9df38799893ace758",
    "configuration": "axbench/sweep/wuzhengx/2b/l10/loreft.yaml",
}

NORMALIZED_TUPLES: dict[str, tuple[float, float, float, float]] = {
    "author": (0.20, 0.40, 0.60, 0.80),
    "shallow": (0.10, 0.30, 0.50, 0.70),
    "deep": (0.30, 0.50, 0.70, 0.90),
    "stratified": (0.125, 0.375, 0.625, 0.875),
}
AXBENCH_FACTORS = (
    0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
    1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
)


def _training_tokenizer(tokenizer: Any, model_config: Any) -> Any:
    """Replace an unspecified tokenizer limit with the model context capacity."""
    from transformers.tokenization_utils_base import LARGE_INTEGER

    if tokenizer.model_max_length <= LARGE_INTEGER:
        return tokenizer
    capacity = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("LoReFT needs a finite tokenizer limit or model context capacity")
    if capacity > LARGE_INTEGER:
        raise ValueError("LoReFT model context capacity is not finite")
    prepared = copy.deepcopy(tokenizer)
    prepared.model_max_length = capacity
    return prepared


def _pyreft(code_path: Path) -> Any:
    code_path = code_path.expanduser().resolve()
    if not (code_path / "pyreft" / "__init__.py").is_file():
        raise ValueError("registered pyReFT source lacks pyreft/__init__.py")
    loaded = sys.modules.get("pyreft")
    if loaded is not None:
        module_file = getattr(loaded, "__file__", None)
        if module_file and code_path in Path(module_file).resolve().parents:
            return loaded
        raise RuntimeError(
            "Another pyReFT package is loaded; start LoReFT in its registered runtime"
        )
    sys.path.insert(0, str(code_path))
    try:
        pyreft = importlib.import_module("pyreft")
    except ImportError as exc:
        raise RuntimeError(
            "LoReFT requires the official pyreft package; install the pinned "
            f"source revision {PYREFT_SOURCE['revision']}"
        ) from exc
    finally:
        if sys.path[0] == str(code_path):
            sys.path.pop(0)
    if code_path not in Path(pyreft.__file__).resolve().parents:
        raise RuntimeError("LoReFT imported a non-registered pyReFT implementation")
    return pyreft


def map_normalized_tuple(
    layer_count: int, normalized: Sequence[float]
) -> tuple[int, int, int, int]:
    """Map an architecture-neutral tuple with round-half-up on ``L-1``."""

    if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count < 4:
        raise ValueError("layer_count must be an integer >= 4")
    if len(normalized) != 4 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0
        for value in normalized
    ):
        raise ValueError("normalized LoReFT tuple must contain four values in [0, 1]")
    layers = tuple(int(math.floor(float(value) * (layer_count - 1) + 0.5)) for value in normalized)
    if len(set(layers)) != 4 or tuple(sorted(layers)) != layers:
        raise ValueError("normalized tuple does not map to four ordered unique layers")
    return layers  # type: ignore[return-value]


def registered_tuple_candidates(
    layer_count: int,
    normalized_tuples: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, dict[str, Any]]:
    tuples = NORMALIZED_TUPLES if normalized_tuples is None else normalized_tuples
    return {
        name: {
            "candidate_id": name,
            "normalized_depths": list(depths),
            "layers": list(map_normalized_tuple(layer_count, depths)),
        }
        for name, depths in tuples.items()
    }


def resolve_training_batches(plan: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve a registered single-device batch split without changing effective batch."""
    training = plan["training"]
    execution = plan.get("training_execution", {})
    if not isinstance(execution, Mapping) or set(execution) - {
        "batch_size", "gradient_accumulation_steps", "rationale",
    }:
        raise ValueError("Invalid LoReFT training_execution settings")
    batch = execution.get("batch_size", training["batch_size"])
    accumulation = execution.get(
        "gradient_accumulation_steps", training["gradient_accumulation_steps"]
    )
    if any(type(value) is not int or value <= 0 for value in (batch, accumulation)):
        raise ValueError("LoReFT execution batches must be positive integers")
    if batch * accumulation != training["batch_size"] * training["gradient_accumulation_steps"]:
        raise ValueError("LoReFT execution must preserve the registered effective batch")
    return batch, accumulation


def validate_loreft_plan(plan: Mapping[str, Any]) -> None:
    operator = plan.get("operator")
    training = plan.get("training")
    search = plan.get("search")
    if not all(isinstance(value, Mapping) for value in (operator, training, search)):
        raise ValueError("LoReFT plan requires operator, training, and search blocks")
    expected_operator = {
        "component": "block_output",
        "rank": 4,
        "positions": "f5+l5",
        "share_weights_across_position_groups": True,
        "joint_layer_training": True,
    }
    if any(operator.get(key) != value for key, value in expected_operator.items()):
        raise ValueError("LoReFT native operator geometry drift")
    expected_training = {
        "batch_size": 18,
        "gradient_accumulation_steps": 2,
        "epochs": 24,
        "learning_rate": 0.0009,
        "weight_decay": 0.0,
        "lr_scheduler": "linear",
        "seed": 42,
        "train_on_negative": False,
        "exclude_bos": True,
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("LoReFT author-side training configuration drift")
    resolve_training_batches(plan)
    candidate_order = search.get("candidate_order")
    candidates = search.get("candidate_tuples")
    compiled_candidates = candidates is not None
    if not isinstance(candidate_order, list) or not candidate_order:
        raise ValueError("LoReFT tuple order is missing")
    if candidates is None:
        candidates = search.get("normalized_tuples")
    if (
        not isinstance(candidates, Mapping)
        or len(candidates) != len(candidate_order)
        or set(candidates) != set(candidate_order)
    ):
        raise ValueError("LoReFT tuple order does not match registered candidates")
    for candidate_id in candidate_order if compiled_candidates else ():
        candidate = candidates[candidate_id]
        if not isinstance(candidate, Mapping):
            raise ValueError("LoReFT candidate must be a mapping")
        normalized = candidate.get("normalized_depths")
        layers = candidate.get("layers")
        if not isinstance(normalized, list) or len(normalized) != 4:
            raise ValueError("LoReFT candidate needs four normalized depths")
        if not isinstance(layers, list) or len(layers) != 4:
            raise ValueError("LoReFT candidate needs four layers")
        if len(set(layers)) != 4 or sorted(layers) != layers:
            raise ValueError("LoReFT candidate layer geometry drift")
    if search.get("tuple_selection_factor") != 1.0:
        raise ValueError("LoReFT tuple selection must use factor 1.0")
    if search.get("factor_grid") != list(AXBENCH_FACTORS):
        raise ValueError("LoReFT factor grid drift")
    if search.get("cartesian_tuple_factor_search") is not False:
        raise ValueError("LoReFT tuple x factor Cartesian search is forbidden")
    if search.get("componentwise_human_audit") is not False:
        raise ValueError("LoReFT componentwise human audit is forbidden")
    code = plan.get("official_code_path")
    if (
        not isinstance(code, str)
        or not code
        or not (code.startswith("registry://") or Path(code).is_absolute())
    ):
        raise ValueError(
            "LoReFT official_code_path must be registered or machine-resolved"
        )


def _without_bos(tokenizer: Any, text: str, enabled: bool) -> str:
    bos = getattr(tokenizer, "bos_token", None)
    if enabled and isinstance(bos, str) and bos and text.startswith(bos):
        return text[len(bos):]
    return text


def _payload_manifest(payload_dir: Path, root: Path, config: Mapping[str, Any]) -> Path:
    files = []
    for path in sorted(item for item in payload_dir.rglob("*") if item.is_file()):
        files.append(
            {
                "relative_path": path.resolve().relative_to(root.resolve()).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    if not files:
        raise ValueError("pyReFT save produced no payload files")
    path = payload_dir.parent / "payload_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_type": "loreft_payload",
                "status": "complete",
                "payload_directory": payload_dir.resolve().relative_to(root.resolve()).as_posix(),
                "configuration": copy.deepcopy(dict(config)),
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


class LoReFTBackend:
    """Train exactly one four-layer candidate through official pyReFT."""

    def __init__(self, model_provider: Any) -> None:
        if not callable(getattr(model_provider, "load_fresh", None)):
            raise ValueError("LoReFT requires a model provider with load_fresh()")
        self.model_provider = model_provider

    def train_candidate(
        self, request: NativeBaselineTrainRequest
    ) -> NativeBaselineTrainResult:
        validate_loreft_plan(request.method_plan)
        pyreft = _pyreft(Path(request.method_plan["official_code_path"]))
        layers = request.candidate_config.get("layers")
        if (
            not isinstance(layers, list)
            or len(layers) != 4
            or not all(isinstance(layer, int) and not isinstance(layer, bool) for layer in layers)
            or layers != sorted(set(layers))
        ):
            raise ValueError("LoReFT candidate needs four ordered unique layers")
        model_id = resolve_model_id(request.model_config, mode="train")
        adapter = self.model_provider.load_fresh(model_id, mode="train")
        adapter.configure_input(
            use_chat_template=bool(request.model_config.get("use_chat_template", False))
        )
        # Full-sequence training does not consume the generation KV cache.
        # Avoid retaining it alongside the four intervention training graphs.
        adapter.model.config.use_cache = False
        rows = load_rows(request.training_data_manifest, root=request.evidence_root)
        prompts: list[str] = []
        outputs: list[str] = []
        for index, row in enumerate(rows):
            prompt = row.get("prompt")
            chosen = row.get("chosen")
            if not isinstance(prompt, str) or not prompt or not isinstance(chosen, str) or not chosen:
                raise ValueError(f"LoReFT training row {index} requires prompt and chosen")
            prompts.append(
                _without_bos(
                    adapter.tokenizer,
                    adapter.format_prompt(prompt),
                    bool(request.method_plan["training"]["exclude_bos"]),
                )
            )
            outputs.append(chosen)
        if not rows:
            raise ValueError("LoReFT training role is empty")
        seed = int(request.method_plan["training"]["seed"])
        from transformers import TrainingArguments, set_seed

        set_seed(seed)
        model_dtype = next(adapter.model.parameters()).dtype
        representations = [
            {
                "layer": layer,
                "component": "block_output",
                "low_rank_dimension": 4,
                "intervention": pyreft.LoreftIntervention(
                    embed_dim=adapter.model.config.hidden_size,
                    low_rank_dimension=4,
                    dtype=model_dtype,
                    dropout=0.0,
                ),
            }
            for layer in layers
        ]
        register_pyreft_model(adapter.model)
        reft_model = pyreft.get_reft_model(
            adapter.model, pyreft.ReftConfig(representations=representations)
        )
        reft_model.set_device(str(next(adapter.model.parameters()).device))
        data_module = pyreft.make_multiple_position_supervised_data_module(
            _training_tokenizer(adapter.tokenizer, adapter.model.config),
            adapter.model,
            prompts,
            outputs,
            positions="f5+l5",
            num_interventions=4,
            nonstop=True,
            share_weights=True,
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        trainer_dir = request.output_dir / "trainer"
        training = request.method_plan["training"]
        batch_size, accumulation = resolve_training_batches(request.method_plan)
        arguments = TrainingArguments(
            output_dir=str(trainer_dir),
            num_train_epochs=float(training["epochs"]),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            lr_scheduler_type=str(training["lr_scheduler"]),
            warmup_ratio=0.0,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=20,
            save_strategy="no",
            report_to=[],
            seed=seed,
            remove_unused_columns=False,
            bf16=model_dtype == torch.bfloat16,
            fp16=model_dtype == torch.float16,
        )
        trainer = pyreft.ReftTrainerForCausalLM(
            model=reft_model,
            tokenizer=adapter.tokenizer,
            args=arguments,
            **data_module,
        )
        print(
            f"[loreft] candidate={request.candidate_id} microbatch={batch_size} "
            f"gradient_accumulation={accumulation} world_size={arguments.world_size} "
            f"effective_batch={batch_size * accumulation * arguments.world_size}",
            flush=True,
        )
        result = trainer.train()
        payload_dir = request.output_dir / "pyreft"
        reft_model.set_device("cpu")
        reft_model.save(str(payload_dir))
        parameters = int(reft_model.count_parameters(include_model=False))
        training_manifest = request.output_dir / "training_manifest.json"
        training_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_type": "native_baseline_training",
                    "status": "complete",
                    "method": "loreft",
                    "candidate_id": request.candidate_id,
                    "layers": layers,
                    "training_rows": len(rows),
                    "per_device_train_batch_size": batch_size,
                    "gradient_accumulation_steps": accumulation,
                    "effective_batch_size": batch_size * accumulation * arguments.world_size,
                    "world_size": arguments.world_size,
                    "author_side_training": dict(training),
                    "training_execution": dict(request.method_plan.get("training_execution", {})),
                    "trainable_parameters": parameters,
                    "global_steps": int(getattr(result, "global_step", 0)),
                    "train_loss": float(getattr(result, "training_loss", float("nan"))),
                    "model_id": model_id,
                    "model_dtype": str(model_dtype),
                    "pyreft_source": PYREFT_SOURCE,
                    "axbench_configuration_source": AXBENCH_SOURCE,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        payload_manifest = _payload_manifest(
            payload_dir,
            request.evidence_root,
            {
                "candidate_id": request.candidate_id,
                "layers": layers,
                "rank": 4,
                "positions": "f5+l5",
                "share_weights": True,
            },
        )
        del trainer, reft_model, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return NativeBaselineTrainResult(
            payload_manifest=payload_manifest,
            training_manifest=training_manifest,
            trainable_parameters=parameters,
        )


def _resolve_payload(controller_path: Path, evidence_root: Path) -> tuple[Path, dict[str, Any]]:
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    record = controller.get("payload_manifest")
    if not isinstance(record, Mapping) or not isinstance(record.get("relative_path"), str):
        raise ValueError("LoReFT controller has no payload manifest")
    manifest_path = evidence_root / record["relative_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = manifest.get("payload_directory")
    if not isinstance(relative, str):
        raise ValueError("LoReFT payload directory is missing")
    for item in manifest.get("files", []):
        path = evidence_root / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
            raise ValueError("LoReFT payload inventory is incomplete")
    return evidence_root / relative, controller


def _intervention_modules(reft_model: Any) -> Iterator[torch.nn.Module]:
    for value in reft_model.interventions.values():
        module = value if isinstance(value, torch.nn.Module) else value[0]
        if not isinstance(module, torch.nn.Module):
            raise TypeError("LoReFT intervention entry must contain a torch module")
        yield module


@contextlib.contextmanager
def _scaled_interventions(reft_model: Any, factor: float) -> Iterator[None]:
    originals = []
    for module in _intervention_modules(reft_model):
        original = module.forward

        def scaled(base: torch.Tensor, source: Any = None, subspaces: Any = None, *, _original=original):
            output = _original(base, source=source, subspaces=subspaces)
            return base + float(factor) * (output - base)

        module.forward = scaled
        originals.append((module, original))
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original


def _unit_locations(pyreft: Any, prompt_length: int) -> dict[str, Any]:
    first_n, last_n = pyreft.parse_positions("f5+l5")
    locations = pyreft.get_intervention_locations(
        last_position=prompt_length,
        first_n=first_n,
        last_n=last_n,
        pad_mode="last",
        num_interventions=4,
        share_weights=True,
    )
    return {"sources->base": (None, [[list(group)] for group in locations])}


def _policy_metrics(
    *,
    reft_model: Any,
    base_adapter: Any,
    reference_adapter: Any,
    output: torch.Tensor,
    prompt_length: int,
    locations: Mapping[str, Any],
    factor: float,
) -> dict[str, float]:
    continuation = output[:, prompt_length:]
    with torch.inference_mode(), _scaled_interventions(reft_model, factor):
        _base, controlled = reft_model.forward(
            base={"input_ids": output, "attention_mask": torch.ones_like(output)},
            unit_locations=locations,
            use_cache=False,
        )
    controlled_logits = controlled.logits[:, prompt_length - 1:-1].float()
    reference_output = output.to(next(reference_adapter.model.parameters()).device)
    with torch.inference_mode():
        reference_logits = reference_adapter.model(
            input_ids=reference_output,
            attention_mask=torch.ones_like(reference_output),
            use_cache=False,
        ).logits[:, prompt_length - 1:-1].float().to(controlled_logits.device)
    from .policy_metrics import sequence_policy_metrics

    return sequence_policy_metrics(controlled_logits, reference_logits, continuation)


def loreft_generation_backend(model_provider: Any) -> CallbackGenerationBackend:
    """Build the shared generation backend for frozen LoReFT controllers."""

    active: dict[str, Any] = {}

    def generate(_outer_adapter: Any, row: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
        controller_value = config.get("_controller_manifest_path")
        root_value = config.get("_evidence_root")
        if not isinstance(controller_value, str) or not isinstance(root_value, str):
            raise ValueError("LoReFT generation requires a frozen controller")
        code_value = config.get("official_code_path")
        if not isinstance(code_value, str) or not Path(code_value).is_absolute():
            raise ValueError("LoReFT generation requires registered official code")
        key = f"{controller_value}::{_outer_adapter.model_id}::{code_value}"
        if active.get("key") != key:
            active.clear()
            payload_dir, controller = _resolve_payload(Path(controller_value), Path(root_value))
            pyreft = _pyreft(Path(code_value))
            register_pyreft_model(_outer_adapter.model)
            reft_model = pyreft.ReftModel.load(str(payload_dir), _outer_adapter.model)
            reft_model.set_device(str(next(_outer_adapter.model.parameters()).device))
            _outer_adapter.model.eval()
            for module in _intervention_modules(reft_model):
                module.eval()
            active.update(
                key=key,
                adapter=_outer_adapter,
                reft_model=reft_model,
                controller=controller,
                pyreft=pyreft,
            )
        adapter = active["adapter"]
        reft_model = active["reft_model"]
        prompt = row.get("prompt", row.get("question"))
        if not isinstance(prompt, str) or not prompt:
            base_state = row.get("base_state")
            prompt = base_state.get("prompt") if isinstance(base_state, Mapping) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("LoReFT generation row requires a prompt")
        prompt = _without_bos(adapter.tokenizer, adapter.format_prompt(prompt), True)
        encoded = adapter.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        device = next(adapter.model.parameters()).device
        ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        locations = _unit_locations(active["pyreft"], int(ids.shape[1]))
        if config.get("prediction_mode") == "paired_completion_scores":
            factor = float(config["alpha"])
            def forward(pair_ids):
                return reft_model.forward(
                    base={"input_ids": pair_ids, "attention_mask": torch.ones_like(pair_ids)},
                    unit_locations=locations, use_cache=False,
                )[1]
            with _scaled_interventions(reft_model, factor):
                return paired_completion_scores(
                    tokenizer=adapter.tokenizer, prompt_ids=ids, row=row, forward=forward,
                )
        locked = config.get("locked_generation", config.get("generation", {}))
        locked = locked if isinstance(locked, Mapping) else {}
        kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(locked.get("max_new_tokens", 8))),
            "do_sample": bool(locked.get("do_sample", False)),
            "pad_token_id": adapter.tokenizer.pad_token_id,
        }
        kwargs.update(generation_cache_kwargs(locked))
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(locked.get("temperature", 1.0))
            if "top_p" in locked:
                kwargs["top_p"] = float(locked["top_p"])
            if "top_k" in locked:
                kwargs["top_k"] = int(locked["top_k"])
        factor = float(config.get("alpha", config.get("factor", 1.0)))
        with torch.inference_mode(), _scaled_interventions(reft_model, factor):
            _base, generated = reft_model.generate(
                {"input_ids": ids, "attention_mask": attention_mask},
                unit_locations=locations,
                intervene_on_prompt=True,
                **kwargs,
            )
        text = adapter.tokenizer.decode(
            generated[0, ids.shape[1]:], skip_special_tokens=True
        )
        result: dict[str, Any] = {
            "generated_text": text,
            "component_scores": [],
            "component_scores_status": "not_applicable_native_operator",
        }
        policy = config.get("policy_audit")
        if isinstance(policy, Mapping) and bool(policy.get("enabled", False)):
            reference = adapter
            reference_config = policy.get("reference_model")
            if isinstance(reference_config, Mapping):
                reference_id = resolve_model_id(reference_config, mode="reference")
                reference = model_provider.load(reference_id, mode="reference")
                reference.configure_input(
                    use_chat_template=bool(reference_config.get("use_chat_template", False))
                )
            result.update(
                _policy_metrics(
                    reft_model=reft_model,
                    base_adapter=adapter,
                    reference_adapter=reference,
                    output=generated,
                    prompt_length=int(ids.shape[1]),
                    locations=locations,
                    factor=factor,
                )
            )
        return result

    return CallbackGenerationBackend(model_provider=model_provider, generate=generate)
