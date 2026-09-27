"""Reusable model execution primitives for Site/Performance baselines.

This module owns model loading, tokenization, forward passes, attention-head
state hooks, RCM/ITI measurement, and CAST generation. A
machine supplies only an explicit model-id -> absolute-path mapping and an
optional device string.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import cast as cast_impl
from .sequence_scoring import completion_score


class SiteHFModel:
    """Architecture adapter with no checkpoint discovery or machine paths."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.use_chat_template = False
        cfg = model.config
        self.attention_heads_per_layer = int(
            getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))
        )
        self.layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
        hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
        self.head_dim = int(
            getattr(cfg, "head_dim", hidden // max(self.attention_heads_per_layer, 1))
        )
        if min(self.layers, self.attention_heads_per_layer, self.head_dim) <= 0:
            raise ValueError(f"unsupported model geometry for {model_id}")

    def configure_input(self, *, use_chat_template: bool) -> None:
        self.use_chat_template = bool(use_chat_template)

    def format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("use_chat_template=True but tokenizer has no chat template")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode_prompts(self, prompts: Sequence[str]) -> Mapping[str, torch.Tensor]:
        formatted = [self.format_prompt(str(prompt)) for prompt in prompts]
        if not formatted:
            raise ValueError("cannot encode an empty prompt batch")
        old_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in encoded.items()}

    def projection(self, layer: int) -> torch.nn.Module:
        pending = [self.model]
        seen: set[int] = set()
        while pending:
            current = pending.pop(0)
            if id(current) in seen:
                continue
            seen.add(id(current))
            transformer = getattr(current, "transformer", None)
            if transformer is not None and hasattr(transformer, "h"):
                attention = transformer.h[layer].attn
                if hasattr(attention, "c_proj"):
                    return attention.c_proj
                if hasattr(attention, "out_proj"):
                    return attention.out_proj
            if hasattr(current, "layers"):
                attention = current.layers[layer].self_attn
                if hasattr(attention, "o_proj"):
                    return attention.o_proj
                if hasattr(attention, "out_proj"):
                    return attention.out_proj
            for attribute in ("model", "base_model"):
                child = getattr(current, attribute, None)
                if isinstance(child, torch.nn.Module) and id(child) not in seen:
                    pending.append(child)
        raise ValueError(f"cannot resolve attention output projection for layer {layer}")

    def block_container(self) -> torch.nn.ModuleList:
        """Resolve the registered model's blocks for all block-level methods."""

        pending = [self.model]
        seen: set[int] = set()
        while pending:
            current = pending.pop(0)
            if id(current) in seen:
                continue
            seen.add(id(current))
            transformer = getattr(current, "transformer", None)
            if transformer is not None and hasattr(transformer, "h"):
                return transformer.h
            layers = getattr(current, "layers", None)
            if isinstance(layers, (torch.nn.ModuleList, list, tuple)):
                return layers
            for attribute in ("model", "base_model"):
                child = getattr(current, attribute, None)
                if isinstance(child, torch.nn.Module) and id(child) not in seen:
                    pending.append(child)
        raise ValueError("cannot resolve Transformer blocks")

    def block(self, layer: int) -> torch.nn.Module:
        return self.block_container()[layer]


class HuggingFaceModelProvider:
    """Load models from caller-supplied paths; never discovers machine paths."""

    def __init__(
        self,
        model_paths: Mapping[str, str] | None = None,
        *,
        device: str = "cuda",
        dtype_by_family: Mapping[str, torch.dtype] | None = None,
    ) -> None:
        self.paths = {str(key): str(value) for key, value in (model_paths or {}).items()}
        self.adapter_base_paths: dict[str, str] = {}
        self.device = str(device)
        self.dtype_by_family = dict(dtype_by_family or {})
        self.cache: dict[str, SiteHFModel] = {}


    def register_path(self, model_id: str, path: str) -> None:
        """Bind one portable model identifier to one resolved machine path."""

        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("model path must be an absolute path")
        existing = self.paths.get(model_id)
        if existing is not None and existing != path:
            raise ValueError(f"conflicting registered paths for model_id={model_id!r}")
        self.paths[model_id] = path

    def register_adapter_base(self, adapter_id: str, base_path: str) -> None:
        """Bind one registered PEFT adapter to its protocol-declared base path."""

        if not isinstance(adapter_id, str) or not adapter_id:
            raise ValueError("adapter_id must be a non-empty string")
        if not isinstance(base_path, str) or not Path(base_path).is_absolute():
            raise ValueError("PEFT base path must be an absolute path")
        existing = self.adapter_base_paths.get(adapter_id)
        if existing is not None and existing != base_path:
            raise ValueError(f"conflicting PEFT base paths for adapter_id={adapter_id!r}")
        self.adapter_base_paths[adapter_id] = base_path

    def load(self, model_id: str, *, mode: str) -> SiteHFModel:
        del mode
        if model_id in self.cache:
            # A fresh candidate may have offloaded this shared inference model.
            self.cache[model_id].model.to(self.device)
            return self.cache[model_id]
        path = self.paths.get(model_id)
        if path is None:
            raise KeyError(f"model id is not registered: {model_id}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter_config_path = Path(path) / "adapter_config.json"
        adapter_config = (
            json.loads(adapter_config_path.read_text(encoding="utf-8"))
            if adapter_config_path.is_file()
            else None
        )
        base_path: str | None = None
        if isinstance(adapter_config, Mapping):
            base_id = adapter_config.get("base_model_name_or_path")
            if not isinstance(base_id, str) or not base_id:
                raise ValueError(f"PEFT adapter lacks base_model_name_or_path: {path}")
            base_path = self.adapter_base_paths.get(model_id) or self.paths.get(base_id)
            if base_path is None:
                raise KeyError(f"PEFT base model is not registered: {base_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_path or path,
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        family_key = "gpt2" if "gpt2" in model_id.lower() else (
            "gptj" if "gptj" in model_id.lower() or "tldr" in model_id.lower() else "default"
        )
        dtype = self.dtype_by_family.get(
            family_key,
            torch.float16 if family_key == "gpt2" else torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_path or path,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if base_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("loading a registered PEFT adapter requires peft") from exc
            model = PeftModel.from_pretrained(model, path, local_files_only=True)
        model.to(self.device).eval()
        wrapped = SiteHFModel(model, tokenizer, model_id)
        self.cache[model_id] = wrapped
        print(
            f"[baseline] loaded {model_id} geometry={wrapped.layers}x"
            f"{wrapped.attention_heads_per_layer} head_dim={wrapped.head_dim}",
            flush=True,
        )
        return wrapped

    def load_fresh(self, model_id: str, *, mode: str) -> SiteHFModel:
        """Load an uncached model for an independently trained native candidate."""

        cached = self.cache.pop(model_id, None)
        try:
            if cached is not None:
                # Keep cached weights reusable without retaining a second GPU
                # backbone while an independent native candidate is trained.
                cached.model.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fresh = self.load(model_id, mode=mode)
            self.cache.pop(model_id, None)
            return fresh
        finally:
            if cached is not None:
                self.cache[model_id] = cached


def register_model_paths_from_runtime_config(
    provider: HuggingFaceModelProvider,
    runtime_config: Mapping[str, Any],
) -> dict[str, str]:
    """Register resolved train/inference model paths from one compiled cell.

    Scientific bundles provide portable model identifiers; compilation resolves
    the local_path through the machine path registry. This function only joins
    those two already-declared values and does not discover either one.
    """

    baseline = runtime_config.get("baseline")
    parameters = baseline.get("parameters") if isinstance(baseline, Mapping) else None
    model_block = parameters.get("model") if isinstance(parameters, Mapping) else None
    if not isinstance(model_block, Mapping):
        raise ValueError("compiled runtime config needs baseline.parameters.model")
    registered: dict[str, str] = {}
    for mode in ("train", "inference", "reference"):
        model = model_block.get(mode)
        if not isinstance(model, Mapping):
            continue
        path = model.get("local_path")
        aliases = tuple(dict.fromkeys(
            model.get(field)
            for field in ("model_registry_id", "model_id", "id", "checkpoint")
            if isinstance(model.get(field), str) and model.get(field)
        ))
        if not aliases or not isinstance(path, str):
            raise ValueError(f"compiled {mode} model needs checkpoint/id and local_path")
        for model_id in aliases:
            provider.register_path(model_id, path)
        artifact = model.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get("kind") == "peft_adapter":
            base_path = artifact.get("base_local_path")
            base_aliases = tuple(dict.fromkeys(
                artifact.get(field)
                for field in (
                    "base_model_registry_id",
                    "base_checkpoint",
                    "adapter_base_model_name",
                )
                if isinstance(artifact.get(field), str) and artifact.get(field)
            ))
            if not isinstance(base_path, str) or not base_aliases:
                raise ValueError(f"compiled {mode} PEFT adapter lacks its registered base")
            for base_id in base_aliases:
                provider.register_path(base_id, base_path)
            for adapter_id in aliases:
                provider.register_adapter_base(adapter_id, base_path)
        registered[mode] = aliases[0]
    if not registered:
        raise ValueError("compiled runtime config did not register a train or inference model")
    return registered
def _text(row: Mapping[str, Any], key: str, fallback: str = "") -> str:
    value = row.get(key, fallback)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "")


def _prompt_pair(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _text(row, "prompt", _text(row, "question")),
        _text(row, "chosen", _text(row, "target", _text(row, "cf_answer"))),
        _text(row, "rejected", _text(row, "non_target", _text(row, "orig_answer"))),
    )


def _ids(adapter: SiteHFModel, prompt: str, answer: str = "") -> tuple[torch.Tensor, int]:
    prompt_ids = adapter.tokenizer(adapter.format_prompt(prompt), add_special_tokens=False).input_ids
    answer_ids = adapter.tokenizer(answer, add_special_tokens=False).input_ids if answer else []
    if not prompt_ids:
        prompt_ids = [adapter.tokenizer.eos_token_id or 0]
    return (
        torch.tensor([prompt_ids + answer_ids], device=next(adapter.model.parameters()).device),
        len(prompt_ids),
    )


def _parse_component(component_id: str) -> tuple[int, int | None]:
    match = re.fullmatch(r"L(\d+)\.attn(?:\.h(\d+))?", component_id)
    if not match:
        raise ValueError(f"unsupported component id: {component_id}")
    return int(match.group(1)), (
        int(match.group(2)) if match.group(2) is not None else None
    )


def _timing(config: Mapping[str, Any]) -> cast_impl.CastTiming:
    value = config.get("state_timing", config.get("timing", "generation_decision_states"))
    timing_id: str | None = None
    if isinstance(value, Mapping):
        raw_id = value.get("timing_id")
        if raw_id is not None and not isinstance(raw_id, str):
            raise ValueError("RCM timing_id must be a string")
        timing_id = raw_id
        value = value.get("inference", value.get("name", "generation_decision_states"))
    name = str(value)
    cast_impl.validate_timing_reference(name, timing_id)
    # RCM scores a materialized trajectory, not an optimizer training window.
    return cast_impl.CastTiming(name, phase="replay")


@contextlib.contextmanager
def _head_state_override(
    adapter: SiteHFModel,
    layer: int,
    head: int | None,
    *,
    mode: str,
    replacement: torch.Tensor | None,
    timing: cast_impl.CastTiming,
    prompt_length: int,
    sequence_length: int,
):
    module = adapter.projection(layer)

    def hook(_module: torch.nn.Module, args: tuple[Any, ...]):
        value = args[0]
        updated = value.clone()
        current_length = int(value.shape[1])
        current_prompt = (
            prompt_length
            if current_length > 1
            else (1 if timing.name in {"generation_decision_states", "decision_tokens", "decode"} else prompt_length)
        )
        active = set(timing.indices(current_prompt, current_length))
        mask = torch.zeros(value.shape[:2], dtype=torch.bool, device=value.device)
        for index in active:
            if index < mask.shape[1]:
                mask[:, index] = True
        if head is None:
            begin, end = 0, value.shape[-1]
        else:
            begin, end = head * adapter.head_dim, (head + 1) * adapter.head_dim
        state = updated[..., begin:end]
        if mode == "zero":
            state = torch.zeros_like(state)
        elif mode == "patch":
            if replacement is None:
                raise ValueError("patch intervention requires a replacement state")
            state = replacement.to(device=state.device, dtype=state.dtype).view(1, 1, -1).expand_as(state)
        else:
            raise ValueError(f"unknown state override mode: {mode}")
        updated[..., begin:end] = torch.where(mask.unsqueeze(-1), state, updated[..., begin:end])
        return (updated, *args[1:])

    handle = module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _answer_margin(
    adapter: SiteHFModel,
    prompt: str,
    answer: str,
    *,
    layer: int | None = None,
    head: int | None = None,
    mode: str | None = None,
    replacement: torch.Tensor | None = None,
    timing: cast_impl.CastTiming | None = None,
    score_mode: str = "answer_rest_margin",
) -> float:
    ids, prompt_length = _ids(adapter, prompt, answer)
    labels = ids[0, prompt_length:]
    if labels.numel() == 0:
        return 0.0
    timing = timing or cast_impl.CastTiming("generation_decision_states")
    context = (
        contextlib.nullcontext()
        if layer is None or mode is None
        else _head_state_override(
            adapter,
            layer,
            head,
            mode=mode,
            replacement=replacement,
            timing=timing,
            prompt_length=prompt_length,
            sequence_length=int(ids.shape[1]),
        )
    )
    with torch.inference_mode(), context:
        logits = adapter.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        ).logits[0, prompt_length - 1 : prompt_length - 1 + labels.numel()]
    return float(completion_score(logits, labels, score_mode).item())


def _target_aliases(row: Mapping[str, Any]) -> list[str]:
    values = row.get("chosen_answers", row.get("target_answers"))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        aliases = [str(item) for item in values if isinstance(item, str) and item]
        if aliases:
            return aliases
    _prompt, chosen, _rejected = _prompt_pair(row)
    return [chosen] if chosen else []


def _imdb_selector_row(row: Mapping[str, Any]) -> bool:
    return row.get("selector_semantics") == "shared_model_native_good_base_complete_states"


def _state(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = row.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"IMDb selector row lacks {name}")
    prompt = value.get("prompt")
    token_ids = value.get("generated_token_ids")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"IMDb {name} lacks its registered prompt")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids)
    ):
        raise ValueError(f"IMDb {name} lacks complete generated token IDs")
    return value


def _state_input_ids(
    adapter: SiteHFModel,
    state: Mapping[str, Any],
    *,
    exclude_terminal_eos: bool = False,
) -> tuple[torch.Tensor, int, int]:
    prompt = str(state["prompt"])
    token_ids = [int(token) for token in state["generated_token_ids"]]
    eos_id = adapter.tokenizer.eos_token_id
    if (
        exclude_terminal_eos
        and eos_id is not None
        and len(token_ids) > 1
        and token_ids[-1] == eos_id
    ):
        token_ids.pop()
    prompt_ids = adapter.tokenizer(adapter.format_prompt(prompt), add_special_tokens=False).input_ids
    if not prompt_ids or not token_ids:
        raise ValueError("IMDb complete state has an empty prompt or continuation")
    device = next(adapter.model.parameters()).device
    ids = torch.tensor([prompt_ids + token_ids], device=device)
    return ids, len(prompt_ids), len(token_ids)


def _capture_imdb_layer_means(
    adapter: SiteHFModel,
    state: Mapping[str, Any],
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    ids, prompt_length, continuation_length = _state_input_ids(adapter, state)
    start = max(prompt_length - 1, 0)
    stop = min(prompt_length + continuation_length - 1, int(ids.shape[1]))
    if stop <= start:
        raise ValueError("IMDb target state has no generation-decision trajectory")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], *, layer: int = layer):
            captured[layer] = args[0][0, start:stop].detach().float().mean(dim=0).cpu()

        handles.append(adapter.projection(layer).register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise ValueError("IMDb patch prototype capture missed a registered layer")
    return captured


def _capture_imdb_final_heads(
    adapter: SiteHFModel,
    state: Mapping[str, Any],
) -> np.ndarray:
    ids, _prompt_length, _continuation_length = _state_input_ids(
        adapter,
        state,
        exclude_terminal_eos=True,
    )
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in range(adapter.layers):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], *, layer: int = layer):
            captured[layer] = args[0][0, -1].detach().float().cpu()

        handles.append(adapter.projection(layer).register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != adapter.layers:
        raise ValueError("IMDb ITI capture missed a registered layer")
    result = np.zeros(
        (adapter.layers, adapter.attention_heads_per_layer, adapter.head_dim),
        dtype=np.float32,
    )
    for layer, value in captured.items():
        result[layer] = value.numpy().reshape(
            adapter.attention_heads_per_layer,
            adapter.head_dim,
        )
    return result


_IMDB_SCORERS: dict[tuple[str, str, str], Any] = {}
_IMDB_NATIVE_REWARD_CACHE: dict[tuple[Any, ...], tuple[float, ...]] = {}


def _imdb_reward_scores(
    adapter: SiteHFModel,
    texts: Sequence[str],
    config: Mapping[str, Any],
) -> list[float]:
    scorer = config.get("sentiment_scorer")
    if not isinstance(scorer, Mapping):
        raise ValueError("IMDb RCM scan requires sentiment_scorer settings")
    path = scorer.get("local_path")
    revision = scorer.get("revision")
    batch_size = scorer.get("reward_batch_size")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError("IMDb RCM scan requires a resolved sentiment scorer path")
    if not isinstance(revision, str) or not revision:
        raise ValueError("IMDb RCM scan requires a sentiment scorer revision")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("IMDb RCM scan requires a positive reward_batch_size")
    device = str(next(adapter.model.parameters()).device)
    key = (path, revision, device)
    evaluator = _IMDB_SCORERS.get(key)
    if evaluator is None:
        from evaluators.imdb import IMDbSentimentEvaluator

        evaluator = IMDbSentimentEvaluator(device=device)
        _IMDB_SCORERS[key] = evaluator
    return evaluator.score_completions(
        list(texts),
        path=path,
        revision=revision,
        batch_size=batch_size,
    )


def _imdb_native_rewards(
    adapter: SiteHFModel,
    rows: Sequence[Mapping[str, Any]],
    state_name: str,
    config: Mapping[str, Any],
) -> list[float]:
    states = [_state(row, state_name) for row in rows]
    texts = [str(state.get("generated_text", "")) for state in states]
    if not all(texts):
        raise ValueError(f"IMDb {state_name} lacks generated text for reward scoring")
    scorer = config.get("sentiment_scorer", {})
    cache_key = (
        adapter.model_id,
        state_name,
        str(scorer.get("local_path")),
        str(scorer.get("revision")),
        tuple(str(row.get("sample_id", index)) for index, row in enumerate(rows)),
        tuple(texts),
    )
    cached = _IMDB_NATIVE_REWARD_CACHE.get(cache_key)
    if cached is None:
        cached = tuple(_imdb_reward_scores(adapter, texts, config))
        _IMDB_NATIVE_REWARD_CACHE[cache_key] = cached
    return list(cached)


def _imdb_changed_generations(
    adapter: SiteHFModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    state_name: str,
    layer: int,
    head: int | None,
    mode: str,
    replacement: torch.Tensor | None,
    config: Mapping[str, Any],
) -> list[str]:
    generation = config.get("selector_state_generation")
    if not isinstance(generation, Mapping):
        raise ValueError("IMDb RCM scan requires selector_state_generation settings")
    timing = _timing(config)
    prompts = [str(_state(row, state_name)["prompt"]) for row in rows]
    seeds = [row.get("batch_seed") for row in rows]
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError("IMDb selector rows require their registered batch_seed")

    groups: list[tuple[int, int, int]] = []
    start = 0
    while start < len(rows):
        seed = int(seeds[start])
        stop = start + 1
        while stop < len(rows) and seeds[stop] == seed:
            stop += 1
        groups.append((start, stop, seed))
        start = stop

    output_texts: list[str] = []
    device = next(adapter.model.parameters()).device
    for start, stop, seed in groups:
        batch_prompts = prompts[start:stop]
        encoded = adapter.encode_prompts(batch_prompts)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        context = _head_state_override(
            adapter,
            layer,
            head,
            mode=mode,
            replacement=replacement,
            timing=timing,
            prompt_length=int(input_ids.shape[1]),
            sequence_length=int(input_ids.shape[1]),
        )
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(generation["max_new_tokens"]),
            "do_sample": bool(generation["do_sample"]),
            "pad_token_id": adapter.tokenizer.pad_token_id,
            "eos_token_id": adapter.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs.update(
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
            )
        with torch.inference_mode(), context:
            output = adapter.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        width = int(input_ids.shape[1])
        output_texts.extend(
            adapter.tokenizer.decode(tokens[width:], skip_special_tokens=True)
            for tokens in output
        )
    if len(output_texts) != len(rows):
        raise ValueError("IMDb changed generation count drift")
    return output_texts


def rcm_prepare_patch_prototypes(
    adapter: SiteHFModel,
    rows: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Build one global target prototype for every registered component."""

    if not rows or not positions:
        raise ValueError("RCM-patch prototype preparation requires selector rows and positions")
    parsed = {
        str(position["component_id"]): _parse_component(str(position["component_id"]))
        for position in positions
    }
    layers = sorted({layer for layer, _head in parsed.values()})

    imdb_flags = [_imdb_selector_row(row) for row in rows]
    if any(imdb_flags) and not all(imdb_flags):
        raise ValueError("RCM selector rows mix incompatible state definitions")
    if all(imdb_flags):
        sample_maps = [
            _capture_imdb_layer_means(adapter, _state(row, "good_state"), layers)
            for row in rows
        ]
        layer_means = {
            layer: torch.stack([sample[layer] for sample in sample_maps], dim=0).mean(dim=0)
            for layer in layers
        }
    else:
        timing = _timing(config)
        sample_sums: dict[int, torch.Tensor] = {}
        sample_count = 0
        for row in rows:
            prompt, _chosen, _rejected = _prompt_pair(row)
            aliases = _target_aliases(row)
            if not prompt or not aliases:
                raise ValueError("RCM-patch target rows require a prompt and at least one target alias")
            alias_maps: list[dict[int, torch.Tensor]] = []
            for answer in aliases:
                ids, prompt_length = _ids(adapter, prompt, answer)
                captured: dict[int, torch.Tensor] = {}
                handles = []
                for layer in layers:
                    def hook(_module: torch.nn.Module, args: tuple[Any, ...], *, layer: int = layer):
                        value = args[0][0].detach().float()
                        active = list(timing.indices(prompt_length, int(value.shape[0])))
                        if not active:
                            raise ValueError("registered RCM timing selected no target states")
                        captured[layer] = value[active].mean(dim=0).cpu()

                    handles.append(adapter.projection(layer).register_forward_pre_hook(hook))
                try:
                    with torch.inference_mode():
                        adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                finally:
                    for handle in handles:
                        handle.remove()
                if set(captured) != set(layers):
                    raise ValueError("RCM-patch prototype capture missed a registered layer")
                alias_maps.append(captured)
            sample_map = {
                layer: torch.stack([alias_map[layer] for alias_map in alias_maps], dim=0).mean(dim=0)
                for layer in layers
            }
            for layer, value in sample_map.items():
                sample_sums[layer] = value if layer not in sample_sums else sample_sums[layer] + value
            sample_count += 1
        layer_means = {
            layer: total / float(sample_count)
            for layer, total in sample_sums.items()
        }

    result: dict[str, torch.Tensor] = {}
    for component_id, (layer, head) in parsed.items():
        value = layer_means[layer]
        if head is not None:
            begin, end = head * adapter.head_dim, (head + 1) * adapter.head_dim
            value = value[begin:end]
        result[component_id] = value.contiguous()
    return result


def rcm_measure(
    adapter: SiteHFModel,
    row: Mapping[str, Any],
    position: Mapping[str, Any],
    method: str,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Compute one non-IMDb registered RCM state contrast."""

    if _imdb_selector_row(row):
        raise ValueError("IMDb RCM must use the batchwise complete-generation measurement")
    prompt, chosen, rejected = _prompt_pair(row)
    layer, head = _parse_component(str(position["component_id"]))
    timing = _timing(config)
    if not prompt or not chosen or not rejected:
        raise ValueError("RCM pair measurement requires prompt, chosen, and rejected endpoints")
    score_mode = str(config.get("score_mode", "answer_rest_margin"))
    native = _answer_margin(adapter, prompt, chosen, score_mode=score_mode) - _answer_margin(adapter, prompt, rejected, score_mode=score_mode)
    if method == "rcm_zero":
        zero = (
            _answer_margin(adapter, prompt, chosen, layer=layer, head=head, mode="zero", timing=timing, score_mode=score_mode)
            - _answer_margin(adapter, prompt, rejected, layer=layer, head=head, mode="zero", timing=timing, score_mode=score_mode)
        )
        return {"effect": native - zero, "native_margin": native, "intervened_margin": zero}
    if method != "rcm_patch":
        raise ValueError(f"unsupported RCM method: {method}")
    prototype = config.get("_rcm_patch_prototype")
    if not isinstance(prototype, torch.Tensor):
        raise ValueError("RCM-patch measurement requires its precomputed global target prototype")
    patched = (
        _answer_margin(adapter, prompt, chosen, layer=layer, head=head, mode="patch", replacement=prototype, timing=timing, score_mode=score_mode)
        - _answer_margin(adapter, prompt, rejected, layer=layer, head=head, mode="patch", replacement=prototype, timing=timing, score_mode=score_mode)
    )
    return {"effect": patched - native, "native_margin": native, "intervened_margin": patched}


def rcm_measure_many(
    adapter: SiteHFModel,
    rows: Sequence[Mapping[str, Any]],
    position: Mapping[str, Any],
    method: str,
    config: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """Measure one component across a complete registered selector batch."""

    if not rows:
        return []
    imdb_flags = [_imdb_selector_row(row) for row in rows]
    if any(imdb_flags) and not all(imdb_flags):
        raise ValueError("RCM selector rows mix incompatible state definitions")
    if not all(imdb_flags):
        return [rcm_measure(adapter, row, position, method, config) for row in rows]

    layer, head = _parse_component(str(position["component_id"]))
    if method == "rcm_zero":
        state_name = "good_state"
        native = _imdb_native_rewards(adapter, rows, state_name, config)
        changed_text = _imdb_changed_generations(
            adapter,
            rows,
            state_name=state_name,
            layer=layer,
            head=head,
            mode="zero",
            replacement=None,
            config=config,
        )
        changed = _imdb_reward_scores(adapter, changed_text, config)
        effects = [base - intervention for base, intervention in zip(native, changed, strict=True)]
    elif method == "rcm_patch":
        state_name = "base_state"
        native = _imdb_native_rewards(adapter, rows, state_name, config)
        prototype = config.get("_rcm_patch_prototype")
        if not isinstance(prototype, torch.Tensor):
            raise ValueError("IMDb RCM-patch requires its global good-state prototype")
        changed_text = _imdb_changed_generations(
            adapter,
            rows,
            state_name=state_name,
            layer=layer,
            head=head,
            mode="patch",
            replacement=prototype,
            config=config,
        )
        changed = _imdb_reward_scores(adapter, changed_text, config)
        effects = [intervention - base for base, intervention in zip(native, changed, strict=True)]
    else:
        raise ValueError(f"unsupported RCM method: {method}")

    return [
        {
            "effect": effect,
            "native_margin": base,
            "intervened_margin": intervention,
            "generated_text": text,
        }
        for effect, base, intervention, text in zip(
            effects,
            native,
            changed,
            changed_text,
            strict=True,
        )
    ]


def _capture_block_final(
    adapter: SiteHFModel,
    ids: torch.Tensor,
) -> np.ndarray:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in range(adapter.layers):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            layer: int = layer,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[0, -1].detach().float().cpu()

        handles.append(adapter.block(layer).register_forward_hook(hook))
    try:
        with torch.inference_mode():
            adapter.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != adapter.layers:
        raise ValueError("CAA capture missed a Transformer block")
    return np.stack([captured[layer].numpy() for layer in range(adapter.layers)])


def caa_capture_pair(
    adapter: SiteHFModel,
    row: Mapping[str, Any],
    _config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Capture the official CAA positive/negative final response state."""

    if _imdb_selector_row(row):
        positive_ids, _prompt, _tokens = _state_input_ids(
            adapter, _state(row, "good_state"), exclude_terminal_eos=True
        )
        negative_ids, _prompt, _tokens = _state_input_ids(
            adapter, _state(row, "base_state"), exclude_terminal_eos=True
        )
    else:
        prompt, chosen, rejected = _prompt_pair(row)
        positive_ids, _ = _ids(adapter, prompt, chosen)
        negative_ids, _ = _ids(adapter, prompt, rejected)
    return {
        "positive_activation": _capture_block_final(adapter, positive_ids),
        "negative_activation": _capture_block_final(adapter, negative_ids),
    }


@contextlib.contextmanager
def _block_addition(
    adapter: SiteHFModel,
    *,
    layer: int,
    vector: np.ndarray,
    factor: float,
):
    module = adapter.block(layer)

    def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        delta = torch.as_tensor(vector, device=hidden.device, dtype=hidden.dtype)
        changed = hidden + float(factor) * delta.view(1, 1, -1)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _block_answer_margin(
    adapter: SiteHFModel,
    prompt: str,
    answer: str,
    *,
    layer: int,
    vector: np.ndarray,
    factor: float,
) -> float:
    with _block_addition(
        adapter, layer=layer, vector=vector, factor=factor
    ):
        return _answer_margin(adapter, prompt, answer)


def _caa_imdb_generations(
    adapter: SiteHFModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    layer: int,
    vector: np.ndarray,
    factor: float,
    config: Mapping[str, Any],
) -> list[str]:
    generation = config.get("selector_state_generation")
    if not isinstance(generation, Mapping):
        raise ValueError("IMDb CAA requires selector_state_generation settings")
    output_texts: list[str] = []
    for row in rows:
        state = _state(row, "base_state")
        encoded = adapter.encode_prompts([str(state["prompt"])])
        seed = row.get("batch_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("IMDb CAA selector rows require batch_seed")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(generation["max_new_tokens"]),
            "do_sample": bool(generation["do_sample"]),
            "pad_token_id": adapter.tokenizer.pad_token_id,
            "eos_token_id": adapter.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs.update(
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
            )
        with torch.inference_mode(), _block_addition(
            adapter, layer=layer, vector=vector, factor=factor
        ):
            output = adapter.model.generate(**encoded, **kwargs)
        width = int(encoded["input_ids"].shape[1])
        output_texts.append(
            adapter.tokenizer.decode(output[0, width:], skip_special_tokens=True)
        )
    return output_texts


def caa_measure_many(
    adapter: SiteHFModel,
    rows: Sequence[Mapping[str, Any]],
    layer: int,
    vector: np.ndarray,
    config: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    """Evaluate one CAA layer at the registered symmetric +/-1 setting."""

    if rows and all(_imdb_selector_row(row) for row in rows):
        positive_text = _caa_imdb_generations(
            adapter, rows, layer=layer, vector=vector, factor=1.0, config=config
        )
        negative_text = _caa_imdb_generations(
            adapter, rows, layer=layer, vector=vector, factor=-1.0, config=config
        )
        positive = _imdb_reward_scores(adapter, positive_text, config)
        negative = _imdb_reward_scores(adapter, negative_text, config)
        return [
            {
                "selection_score": 0.5 * (pos - neg),
                "generated_text": pos_text,
                "negative_generated_text": neg_text,
            }
            for pos, neg, pos_text, neg_text in zip(
                positive, negative, positive_text, negative_text, strict=True
            )
        ]
    output = []
    for row in rows:
        prompt, chosen, rejected = _prompt_pair(row)
        plus = _block_answer_margin(
            adapter, prompt, chosen, layer=layer, vector=vector, factor=1.0
        ) - _block_answer_margin(
            adapter, prompt, rejected, layer=layer, vector=vector, factor=1.0
        )
        minus = _block_answer_margin(
            adapter, prompt, chosen, layer=layer, vector=vector, factor=-1.0
        ) - _block_answer_margin(
            adapter, prompt, rejected, layer=layer, vector=vector, factor=-1.0
        )
        output.append({"selection_score": 0.5 * (plus - minus)})
    return output


def iti_capture_pair(
    adapter: SiteHFModel,
    row: Mapping[str, Any],
    _config: Mapping[str, Any],
) -> Mapping[str, Any]:
    if _imdb_selector_row(row):
        expected_policy = (
            "exclude_terminal_eos_if_content_tokens_remain_to_match_official_prompt_last_token"
        )
        if _config.get("iti_terminal_eos_policy") != expected_policy:
            raise ValueError("IMDb ITI terminal EOS policy differs from the registered definition")
        return {
            "positive_activation": _capture_imdb_final_heads(adapter, _state(row, "good_state")),
            "negative_activation": _capture_imdb_final_heads(adapter, _state(row, "base_state")),
            "positive_text": str(_state(row, "good_state").get("generated_text", "")),
            "negative_text": str(_state(row, "base_state").get("generated_text", "")),
        }

    prompt, chosen, rejected = _prompt_pair(row)

    def capture(answer: str) -> np.ndarray:
        ids, _ = _ids(adapter, prompt, answer)
        states: dict[int, torch.Tensor] = {}
        handles = []
        for layer in range(adapter.layers):
            def hook(_module: torch.nn.Module, args: tuple[Any, ...], *, layer: int = layer):
                states[layer] = args[0][0, -1].detach().float().cpu()

            handles.append(adapter.projection(layer).register_forward_pre_hook(hook))
        try:
            with torch.inference_mode():
                adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        result = np.zeros(
            (adapter.layers, adapter.attention_heads_per_layer, adapter.head_dim),
            dtype=np.float32,
        )
        for layer, value in states.items():
            result[layer] = value.numpy().reshape(
                adapter.attention_heads_per_layer,
                adapter.head_dim,
            )
        return result

    return {
        "positive_activation": capture(chosen),
        "negative_activation": capture(rejected),
        "positive_text": chosen,
        "negative_text": rejected,
    }


def iti_train_probes(
    random_state: int,
    train_groups: Any,
    eval_groups: Any,
    activations: Any,
    labels: Any,
    layers: int,
    heads: int,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    scores, probes = [], []
    for layer in range(layers):
        for head in range(heads):
            x_train = np.concatenate(
                [np.asarray(activations[int(group)])[:, layer, head, :] for group in train_groups]
            )
            y_train = np.concatenate([np.asarray(labels[int(group)]) for group in train_groups])
            x_eval = np.concatenate(
                [np.asarray(activations[int(group)])[:, layer, head, :] for group in eval_groups]
            )
            y_eval = np.concatenate([np.asarray(labels[int(group)]) for group in eval_groups])
            probe = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                tol=1e-4,
                fit_intercept=True,
                random_state=random_state,
            )
            probe.fit(x_train, y_train)
            probes.append(probe)
            scores.append(float(accuracy_score(y_eval, probe.predict(x_eval))))
    return probes, np.asarray(scores)


def _load_controller(
    config: Mapping[str, Any],
    cache: dict[str, tuple[dict[str, dict[str, torch.Tensor]], list[str]]],
):
    path = str(config.get("_controller_manifest_path", ""))
    root = Path(str(config.get("_evidence_root", "")))
    if not path:
        return {}, []
    if path in cache:
        return cache[path]
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters: dict[str, dict[str, torch.Tensor]] = {}
    components: list[str] = []
    for pair in manifest.get("ordered_pairs", []):
        relative = pair.get("vector_payload", {}).get("relative_path")
        if not isinstance(relative, str):
            continue
        loaded = torch.load(root / relative, map_location="cpu", weights_only=False)
        raw = loaded.get("parameters", loaded)
        for component, values in raw.items():
            if isinstance(values, Mapping):
                parameters[str(component)] = {
                    str(key): value
                    for key, value in values.items()
                    if isinstance(value, torch.Tensor)
                }
        components.extend(str(item) for item in pair.get("ordered_component_ids", []))
    cache[path] = parameters, components
    return parameters, components


def _component_scores(
    adapter: SiteHFModel,
    prompt: str,
    components: list[str],
) -> list[dict[str, Any]]:
    captured: dict[str, float] = {}
    by_layer: dict[int, list[tuple[str, int | None]]] = {}
    for component in components:
        layer, head = _parse_component(component)
        by_layer.setdefault(layer, []).append((component, head))
    ids, _ = _ids(adapter, prompt)
    handles = []
    for layer, items in by_layer.items():
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], items=items):
            value = args[0][0, -1].detach().float()
            for component, head in items:
                state = value if head is None else value[
                    head * adapter.head_dim : (head + 1) * adapter.head_dim
                ]
                captured[component] = float(state.norm().item())

        handles.append(adapter.projection(layer).register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            adapter.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [
        {"component_id": component, "score": captured.get(component, 0.0)}
        for component in components
    ]


def _policy_audit(
    adapter: SiteHFModel,
    output: torch.Tensor,
    *,
    prompt_length: int,
    parameters: Mapping[str, Mapping[str, torch.Tensor]],
    alpha: float,
    timing: cast_impl.CastTiming,
    reference_adapter: SiteHFModel | None = None,
) -> dict[str, float]:
    """Measure controlled versus frozen-base policy divergence on sampled tokens."""

    continuation = output[:, prompt_length:]
    if continuation.numel() == 0:
        raise ValueError("policy audit requires a non-empty generated continuation")
    attention_mask = torch.ones_like(output)
    hooks = cast_impl._attach(
        adapter,
        parameters,
        alpha,
        cast_impl.CastTiming(timing.name, timing.mode, phase="replay"),
        prompt_length,
        int(output.shape[1]),
    ) if parameters else []
    try:
        with torch.inference_mode():
            controlled_logits = adapter.model(
                input_ids=output,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, prompt_length - 1 : -1].float()
    finally:
        cast_impl._remove(hooks)
    reference = reference_adapter or adapter
    reference_device = next(reference.model.parameters()).device
    reference_output = output.to(reference_device)
    with torch.inference_mode():
        reference_logits = reference.model(
            input_ids=reference_output,
            attention_mask=torch.ones_like(reference_output),
            use_cache=False,
        ).logits[:, prompt_length - 1 : -1].float().to(controlled_logits.device)
    if controlled_logits.shape[1] != continuation.shape[1]:
        raise ValueError("policy audit token alignment drift")
    controlled_logp = torch.log_softmax(controlled_logits, dim=-1)
    reference_logp = torch.log_softmax(reference_logits, dim=-1)
    token_log_ratio = (
        controlled_logp.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
        - reference_logp.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
    )
    token_kl = (
        torch.softmax(controlled_logits, dim=-1)
        * (controlled_logp - reference_logp)
    ).sum(dim=-1)
    result = {
        "sampled_sequence_logprob_ratio": float(token_log_ratio.sum().item()),
        "token_kl_audit": float(token_kl.mean().item()),
        "generated_token_count": float(continuation.shape[1]),
    }
    if reference is not adapter and parameters:
        with torch.inference_mode():
            base_logits = adapter.model(
                input_ids=output,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, prompt_length - 1 : -1].float()
        base_logp = torch.log_softmax(base_logits, dim=-1)
        incremental_ratio = (
            controlled_logp.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
            - base_logp.gather(-1, continuation.unsqueeze(-1)).squeeze(-1)
        )
        incremental_kl = (
            torch.softmax(controlled_logits, dim=-1)
            * (controlled_logp - base_logp)
        ).sum(dim=-1)
        result.update(
            {
                "incremental_sampled_sequence_logprob_ratio": float(
                    incremental_ratio.sum().item()
                ),
                "incremental_token_kl_audit": float(incremental_kl.mean().item()),
            }
        )
    return result


def cast_generation_callback_factory(model_provider: Any):
    cache: dict[str, tuple[dict[str, dict[str, torch.Tensor]], list[str]]] = {}

    def generate(
        adapter: SiteHFModel,
        row: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        prompt = _text(row, "prompt", _text(row, "question"))
        parameters, components = _load_controller(config, cache)
        ids, _ = _ids(adapter, prompt)
        generation = config.get("generation", {})
        generation = generation if isinstance(generation, Mapping) else {}
        locked = config.get("locked_generation", generation)
        locked = locked if isinstance(locked, Mapping) else generation
        apply_mode = locked.get(
            "apply_mode",
            config.get("apply_mode", generation.get("apply_mode", "generation_decision_states")),
        )
        timing = cast_impl.CastTiming(str(apply_mode), phase="inference")
        alpha = float(config.get("alpha", 1.0) or 0.0)
        hooks = cast_impl._attach(
            adapter,
            parameters,
            alpha,
            timing,
            int(ids.shape[1]),
            int(ids.shape[1]),
        ) if parameters else []
        kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(locked.get("max_new_tokens", 8))),
            "do_sample": bool(locked.get("do_sample", False)),
            "pad_token_id": adapter.tokenizer.pad_token_id,
            "attention_mask": torch.ones_like(ids),
        }
        if kwargs["do_sample"]:
            if float(locked.get("temperature", 1.0)) > 0:
                kwargs["temperature"] = float(locked["temperature"])
            if "top_p" in locked:
                kwargs["top_p"] = float(locked["top_p"])
            if "top_k" in locked:
                kwargs["top_k"] = int(locked["top_k"])
        try:
            with torch.inference_mode():
                output = adapter.model.generate(input_ids=ids, **kwargs)
        finally:
            cast_impl._remove(hooks)
        text = adapter.tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=True)
        result: dict[str, Any] = {
            "generated_text": text,
            "component_scores": _component_scores(adapter, prompt, components),
            "component_scores_status": (
                "available"
                if components
                else "not_applicable_no_active_controller"
            ),
        }
        policy_audit = config.get("policy_audit")
        if isinstance(policy_audit, Mapping) and bool(policy_audit.get("enabled", False)):
            reference_config = policy_audit.get("reference_model")
            reference_adapter = None
            if reference_config is not None:
                if not isinstance(reference_config, Mapping):
                    raise ValueError("policy_audit.reference_model must be an object")
                reference_id = next(
                    (
                        reference_config.get(field)
                        for field in ("model_id", "id", "checkpoint")
                        if isinstance(reference_config.get(field), str)
                        and reference_config.get(field)
                    ),
                    None,
                )
                if not isinstance(reference_id, str):
                    raise ValueError("policy_audit reference model ID is missing")
                reference_adapter = model_provider.load(reference_id, mode="reference")
                reference_adapter.configure_input(
                    use_chat_template=bool(reference_config.get("use_chat_template", False))
                )
            result.update(
                _policy_audit(
                    adapter,
                    output,
                    prompt_length=int(ids.shape[1]),
                    parameters=parameters,
                    alpha=alpha,
                    timing=timing,
                    reference_adapter=reference_adapter,
                )
            )
        return result

    return generate
