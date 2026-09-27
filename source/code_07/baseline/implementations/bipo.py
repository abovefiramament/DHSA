"""Native BiPO training and inference through the registered official trainer."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import random
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.shared.contracts import NativeBaselineTrainRequest, NativeBaselineTrainResult
from .generation_backend import CallbackGenerationBackend
from .loaders import load_rows, resolve_model_id
from .model_runtime import _block_addition
from .policy_metrics import sequence_policy_metrics


OFFICIAL_REVISION = "cef1d00ab108d0e265578589e6b344b200ea3db8"


def validate_bipo_plan(plan: Mapping[str, Any]) -> None:
    operator = plan.get("operator")
    training = plan.get("training")
    search = plan.get("search")
    if not all(isinstance(value, Mapping) for value in (operator, training, search)):
        raise ValueError("BiPO plan requires operator, training, and search blocks")
    required_operator = {
        "component": "block_output",
        "layers_per_candidate": 1,
        "trainable_vectors_per_candidate": 1,
        "application_positions": "all_token_positions",
        "bidirectional_shared_vector": True,
        "initialization": "zeros",
    }
    if any(operator.get(key) != value for key, value in required_operator.items()):
        raise ValueError("BiPO native operator geometry drift")
    required_training = {
        "beta": 0.1,
        "batch_size": 4,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0005,
        "weight_decay": 0.05,
        "optimizer": "adamw_torch",
        "lr_scheduler": "cosine",
        "warmup_steps": 100,
        "seed": 42,
        "loss_type": "sigmoid",
    }
    if any(training.get(key) != value for key, value in required_training.items()):
        raise ValueError("BiPO author-side training configuration drift")
    order = search.get("candidate_order")
    if not isinstance(order, list) or not order or order != [
        f"caa_rank_{i}" for i in range(1, len(order) + 1)
    ]:
        raise ValueError("BiPO candidate order drift")
    if search.get("candidate_selection_multiplier") != 1.0:
        raise ValueError("BiPO candidate selection must use multiplier 1")
    if search.get("layer_multiplier_cartesian_search") is not False:
        raise ValueError("BiPO layer x multiplier Cartesian search is forbidden")
    code = plan.get("official_code_path")
    if (
        not isinstance(code, str)
        or not code
        or not (code.startswith("registry://") or Path(code).is_absolute())
    ):
        raise ValueError("BiPO official_code_path must be registered or machine-resolved")


class _BiPOBlock(torch.nn.Module):
    """Architecture-neutral form of the official BiPO BlockWrapper."""

    def __init__(self, block: torch.nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.block = block
        self.vec = torch.nn.Parameter(torch.zeros(hidden_size))
        self.multiplier = 1.0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.block(*args, **kwargs)
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden + (self.multiplier * self.vec).to(hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)




def _merge_adapter(model: torch.nn.Module) -> torch.nn.Module:
    merge = getattr(model, "merge_and_unload", None)
    return merge() if callable(merge) else model


def _official_trainer(code_path: Path) -> tuple[Any, Any]:
    trainer_file = code_path / "trl" / "trainer" / "bipo_trainer.py"
    if not trainer_file.is_file():
        raise ValueError("registered BiPO source lacks trl/trainer/bipo_trainer.py")
    for name in tuple(sys.modules):
        if name == "trl" or name.startswith("trl."):
            module_file = getattr(sys.modules[name], "__file__", None)
            if module_file and code_path not in Path(module_file).resolve().parents:
                raise RuntimeError(
                    "Another TRL package is loaded; start BiPO in its registered runtime"
                )
    sys.path.insert(0, str(code_path))
    try:
        trl = importlib.import_module("trl")
    finally:
        if sys.path[0] == str(code_path):
            sys.path.pop(0)
    if code_path not in Path(trl.__file__).resolve().parents:
        raise RuntimeError("BiPO imported a non-registered TRL implementation")
    return trl.BiPOTrainer, trl.DPOConfig


def _selected_layer(candidate: Mapping[str, Any]) -> int:
    position = candidate.get("position")
    component = str(position.get("component_id", "")) if isinstance(position, Mapping) else ""
    if not component.startswith("L") or not component.endswith(".block"):
        raise ValueError("BiPO requires one block position resolved by the position controller")
    layer = int(component[1:-6])
    if layer < 0:
        raise ValueError("BiPO layer must be non-negative")
    return layer


class BiPOBackend:
    def __init__(self, model_provider: Any) -> None:
        self.model_provider = model_provider

    def train_candidate(self, request: NativeBaselineTrainRequest) -> NativeBaselineTrainResult:
        validate_bipo_plan(request.method_plan)
        layer = _selected_layer(request.candidate_config)
        model_id = resolve_model_id(request.model_config, mode="train")
        policy = self.model_provider.load_fresh(model_id, mode="train")
        reference = self.model_provider.load_fresh(model_id, mode="reference")
        policy.model = _merge_adapter(policy.model)
        reference.model = _merge_adapter(reference.model)
        hidden_size = int(
            getattr(policy.model.config, "hidden_size", getattr(policy.model.config, "n_embd", 0))
        )
        policy.configure_input(use_chat_template=bool(request.model_config.get("use_chat_template", False)))
        if policy.tokenizer.bos_token_id is None:
            if policy.tokenizer.pad_token_id is None:
                raise ValueError("BiPO requires a BOS or padding token")
            policy.tokenizer.bos_token = policy.tokenizer.pad_token
        if reference.tokenizer.bos_token_id is None:
            reference.tokenizer.bos_token = policy.tokenizer.bos_token
        container = policy.block_container()
        wrapper = _BiPOBlock(container[layer], hidden_size).to(next(policy.model.parameters()).device)
        container[layer] = wrapper
        native_model = getattr(policy.model, "model", None)
        if not hasattr(native_model, "layers"):
            object.__setattr__(policy.model, "model", SimpleNamespace(layers=container))
        for parameter in policy.model.parameters():
            parameter.requires_grad = False
        wrapper.vec.requires_grad = True
        for parameter in reference.model.parameters():
            parameter.requires_grad = False
        policy.model.config.use_cache = False
        rows = load_rows(request.training_data_manifest, root=request.evidence_root)
        from datasets import Dataset
        training_rows = []
        for index, row in enumerate(rows):
            prompt, chosen, rejected = row.get("prompt"), row.get("chosen"), row.get("rejected")
            if not all(isinstance(value, str) and value for value in (prompt, chosen, rejected)):
                raise ValueError(f"BiPO training row {index} lacks prompt/chosen/rejected")
            training_rows.append(
                {
                    "prompt": policy.format_prompt(prompt),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        seed = int(request.method_plan["training"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        Trainer, DPOConfig = _official_trainer(Path(request.method_plan["official_code_path"]))
        training = request.method_plan["training"]
        request.output_dir.mkdir(parents=True, exist_ok=True)
        args = DPOConfig(
            output_dir=str(request.output_dir / "trainer"),
            per_device_train_batch_size=int(training["batch_size"]),
            per_device_eval_batch_size=int(training["eval_batch_size"]),
            num_train_epochs=float(training["epochs"]),
            logging_steps=1,
            save_strategy="no",
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            eval_strategy="no",
            report_to=[],
            lr_scheduler_type=str(training["lr_scheduler"]),
            warmup_steps=int(training["warmup_steps"]),
            optim=str(training["optimizer"]),
            bf16=next(policy.model.parameters()).dtype == torch.bfloat16,
            fp16=next(policy.model.parameters()).dtype == torch.float16,
            remove_unused_columns=False,
            max_prompt_length=int(training["max_prompt_length"]),
            max_length=int(training["max_length"]),
            seed=seed,
        )
        # The author trainer writes ./vector; confine it to this candidate's output.
        previous_directory = Path.cwd()
        try:
            os.chdir(request.output_dir)
            trainer = Trainer(
                policy.model,
                ref_model=reference.model,
                args=args,
                beta=float(training["beta"]),
                loss_type=str(training["loss_type"]),
                train_dataset=Dataset.from_list(training_rows),
                eval_dataset=None,
                tokenizer=policy.tokenizer,
                behavior=str(request.method_plan.get("task_name", "registered_task")),
                layer=layer,
                name="registered",
            )
            result = trainer.train()
        finally:
            os.chdir(previous_directory)
        vector_path = request.output_dir / "bipo_vector.pt"
        torch.save(wrapper.vec.detach().float().cpu(), vector_path)
        payload_manifest = request.output_dir / "payload_manifest.json"
        payload_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_type": "bipo_payload",
                    "status": "complete",
                    "method": "bipo",
                    "candidate_id": request.candidate_id,
                    "layer": layer,
                    "vector_file": vector_path.relative_to(request.evidence_root).as_posix(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        training_manifest = request.output_dir / "training_manifest.json"
        training_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_type": "native_baseline_training",
                    "status": "complete",
                    "method": "bipo",
                    "official_revision": OFFICIAL_REVISION,
                    "candidate_id": request.candidate_id,
                    "layer": layer,
                    "training_rows": len(rows),
                    "trainable_parameters": int(wrapper.vec.numel()),
                    "global_steps": int(getattr(result, "global_step", 0)),
                    "train_loss": float(getattr(result, "training_loss", float("nan"))),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return NativeBaselineTrainResult(
            payload_manifest=payload_manifest,
            training_manifest=training_manifest,
            trainable_parameters=int(wrapper.vec.numel()),
        )


def _payload(controller_path: Path, root: Path) -> tuple[int, torch.Tensor]:
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    record = controller.get("payload_manifest")
    if not isinstance(record, Mapping) or not isinstance(record.get("relative_path"), str):
        raise ValueError("BiPO controller lacks a payload manifest")
    payload = json.loads((root / record["relative_path"]).read_text(encoding="utf-8"))
    vector_file = payload.get("vector_file")
    layer = payload.get("layer")
    if not isinstance(vector_file, str) or isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("BiPO payload manifest is incomplete")
    return layer, torch.load(root / vector_file, map_location="cpu", weights_only=True)




def bipo_generation_backend(model_provider: Any) -> CallbackGenerationBackend:
    active: dict[str, Any] = {}

    def generate(adapter: Any, row: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
        controller = config.get("_controller_manifest_path")
        root = config.get("_evidence_root")
        if not isinstance(controller, str) or not isinstance(root, str):
            raise ValueError("BiPO generation requires a frozen controller")
        key = f"{controller}::{adapter.model_id}"
        if active.get("key") != key:
            layer, vector = _payload(Path(controller), Path(root))
            active.clear()
            active.update(key=key, layer=layer, vector=vector)
        prompt = row.get("prompt", row.get("question"))
        if not isinstance(prompt, str) or not prompt:
            state = row.get("base_state")
            prompt = state.get("prompt") if isinstance(state, Mapping) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("BiPO generation row requires prompt")
        encoded = adapter.tokenizer(
            adapter.format_prompt(prompt), return_tensors="pt", add_special_tokens=False
        )
        device = next(adapter.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        locked = config.get("locked_generation", config.get("generation", {}))
        locked = locked if isinstance(locked, Mapping) else {}
        kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(locked.get("max_new_tokens", 8))),
            "do_sample": bool(locked.get("do_sample", False)),
            "pad_token_id": adapter.tokenizer.pad_token_id,
            "eos_token_id": adapter.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(locked.get("temperature", 1.0))
            if "top_p" in locked:
                kwargs["top_p"] = float(locked["top_p"])
            if "top_k" in locked:
                kwargs["top_k"] = int(locked["top_k"])
        factor = float(config.get("alpha", config.get("factor", 1.0)))
        with torch.inference_mode(), _block_addition(
            adapter, layer=active["layer"], vector=active["vector"], factor=factor
        ):
            output = adapter.model.generate(**encoded, **kwargs)
        width = int(encoded["input_ids"].shape[1])
        result: dict[str, Any] = {
            "generated_text": adapter.tokenizer.decode(
                output[0, width:], skip_special_tokens=True
            ),
            "component_scores": [],
            "component_scores_status": "not_applicable_native_operator",
        }
        policy = config.get("policy_audit")
        if isinstance(policy, Mapping) and bool(policy.get("enabled", False)):
            continuation = output[:, width:]
            attention = torch.ones_like(output)
            with torch.inference_mode(), _block_addition(
                adapter, layer=active["layer"], vector=active["vector"], factor=factor
            ):
                controlled_logits = adapter.model(
                    input_ids=output, attention_mask=attention, use_cache=False
                ).logits[:, width - 1 : -1].float()
            with torch.inference_mode():
                reference_logits = adapter.model(
                    input_ids=output, attention_mask=attention, use_cache=False
                ).logits[:, width - 1 : -1].float()
            result.update(sequence_policy_metrics(controlled_logits, reference_logits, continuation))
        return result

    return CallbackGenerationBackend(model_provider=model_provider, generate=generate)
