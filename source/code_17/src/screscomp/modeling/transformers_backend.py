from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _has_model_payload(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "adapter_config.json").exists()


def _resolve_model_name_or_path(model_name_or_path: str) -> str:
    path = Path(model_name_or_path).expanduser()
    if not path.exists() and "/" in model_name_or_path and not re.match(r"^[A-Za-z]:[\\/]", model_name_or_path):
        cache_root = (
            os.environ.get("HF_HUB_CACHE", "").strip()
            or str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub")
        )
        cached_repo = Path(cache_root) / f"models--{model_name_or_path.replace('/', '--')}"
        if cached_repo.exists():
            path = cached_repo
    if not path.exists():
        return model_name_or_path
    if _has_model_payload(path):
        return str(path)

    snapshots_dir = path / "snapshots"
    if not snapshots_dir.exists():
        return str(path)

    refs_main = path / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots_dir / revision
        if _has_model_payload(snapshot):
            return str(snapshot)

    snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir() and _has_model_payload(p)]
    if not snapshots:
        return str(path)
    return str(max(snapshots, key=lambda p: p.stat().st_mtime))


def _resolve_torch_dtype(torch_module, raw: str):
    if raw == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if raw not in mapping:
        raise ValueError(f"Unsupported torch dtype: {raw}")
    return mapping[raw]


def _load_causal_lm(auto_model_cls, model_name_or_path: str, dtype):
    try:
        return auto_model_cls.from_pretrained(model_name_or_path, dtype=dtype)
    except TypeError:
        return auto_model_cls.from_pretrained(model_name_or_path, torch_dtype=dtype)


def _maybe_set_cuda_memory_limit(torch_module, device: str) -> None:
    raw = os.environ.get("CRESCOMP_CUDA_MEMORY_LIMIT_GB", "").strip()
    if not raw or not torch_module.cuda.is_available():
        return
    target_device = "cuda" if device == "auto" else device
    if not str(target_device).startswith("cuda"):
        return
    try:
        limit_gb = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid CRESCOMP_CUDA_MEMORY_LIMIT_GB={raw!r}") from exc
    if limit_gb <= 0:
        return

    device_index = torch_module.cuda.current_device()
    match = re.match(r"^cuda:(\d+)$", str(target_device))
    if match:
        device_index = int(match.group(1))
    total_bytes = float(torch_module.cuda.get_device_properties(device_index).total_memory)
    limit_bytes = limit_gb * (1024.0 ** 3)
    fraction = min(1.0, max(0.01, limit_bytes / total_bytes))
    torch_module.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    print(
        f"[transformers-backend] cuda memory cap: {limit_gb:g} GiB "
        f"(fraction={fraction:.3f}, device=cuda:{device_index})",
        flush=True,
    )


def _adapter_config_path(model_name_or_path: str) -> Path | None:
    path = Path(model_name_or_path)
    config = path / "adapter_config.json"
    return config if config.exists() else None


def _load_peft_base_path(adapter_config: Path) -> str:
    raw = json.loads(adapter_config.read_text(encoding="utf-8"))
    override = os.environ.get("SCRESCOMP_PEFT_BASE_MODEL", "").strip()
    base = override or str(raw.get("base_model_name_or_path", "")).strip()
    if not base or base == "$$":
        raise ValueError(
            f"PEFT adapter {adapter_config.parent} does not declare a usable base model. "
            "Set SCRESCOMP_PEFT_BASE_MODEL or replace base_model_name_or_path in adapter_config.json."
        )
    return _resolve_model_name_or_path(base)


@dataclass(slots=True)
class TransformersABBackend:
    model_name_or_path: str
    tokenizer_name_or_path: str | None = None
    device: str = "auto"
    use_chat_template: bool = False
    torch_dtype: str = "auto"
    resolved_model_name_or_path: str = field(init=False)
    resolved_tokenizer_name_or_path: str = field(init=False)
    _torch: Any = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers backend requires `transformers` and `torch`. "
                "Install with: pip install transformers torch"
            ) from exc

        self.resolved_model_name_or_path = _resolve_model_name_or_path(self.model_name_or_path)
        self.resolved_tokenizer_name_or_path = _resolve_model_name_or_path(
            self.tokenizer_name_or_path or self.model_name_or_path
        )
        self._torch = torch
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _maybe_set_cuda_memory_limit(torch, self.device)
        dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        adapter_config = _adapter_config_path(self.resolved_model_name_or_path)
        if adapter_config is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.resolved_tokenizer_name_or_path, use_fast=True)
            self._model = _load_causal_lm(AutoModelForCausalLM, self.resolved_model_name_or_path, dtype)
        else:
            base_model_path = _load_peft_base_path(adapter_config)
            try:
                from peft import PeftModel
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Loading a PEFT adapter requires `peft`. Install with: pip install peft"
                ) from exc
            tokenizer_path = (
                self.resolved_tokenizer_name_or_path
                if self.tokenizer_name_or_path is not None
                else base_model_path
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            base_model = _load_causal_lm(AutoModelForCausalLM, base_model_path, dtype)
            self._model = PeftModel.from_pretrained(base_model, self.resolved_model_name_or_path)

        self._model.to(self.device)
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self.resolved_model_name_or_path

    @property
    def num_layers(self) -> int:
        return len(self._decoder_layers())

    def _decoder_layers(self):
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers
        if hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            return self._model.transformer.h
        if hasattr(self._model, "gpt_neox") and hasattr(self._model.gpt_neox, "layers"):
            return self._model.gpt_neox.layers
        raise ValueError("Could not locate decoder layers for residual patching.")

    def _component_module(self, layer_idx: int, component_type: str):
        layer = self._decoder_layers()[layer_idx]
        if component_type == "attn":
            for attr in ("self_attn", "attention", "attn"):
                if hasattr(layer, attr):
                    return getattr(layer, attr)
        if component_type == "mlp":
            for attr in ("mlp", "feed_forward", "ffn"):
                if hasattr(layer, attr):
                    return getattr(layer, attr)
        raise ValueError(f"Could not locate {component_type!r} component on decoder layer {layer_idx}.")

    @staticmethod
    def _replace_last_token_output(output, replacement):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
        return hidden

    @staticmethod
    def _add_last_token_output(output, delta, alpha: float):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * delta.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, -1, :] = hidden[:, -1, :] + alpha * delta.to(device=hidden.device, dtype=hidden.dtype)
        return hidden

    def _single_token_id(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"`{text}` is not a single tokenizer token for this model.")
        return token_ids[0]

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        if not getattr(self._tokenizer, "chat_template", None):
            raise ValueError("Tokenizer has no chat template, but use_chat_template=True.")
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode_prompt(self, prompt: str) -> dict:
        prompt = self._format_prompt(prompt)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _encode_prompts(self, prompts: list[str]) -> dict:
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]
        if not formatted_prompts:
            raise ValueError("empty prompt batch")
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad_token or eos_token for batched generation.")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        old_padding_side = getattr(self._tokenizer, "padding_side", "right")
        self._tokenizer.padding_side = "left"
        try:
            inputs = self._tokenizer(formatted_prompts, return_tensors="pt", padding=True)
        finally:
            self._tokenizer.padding_side = old_padding_side
        return {k: v.to(self.device) for k, v in inputs.items()}

    def score_ab(self, prompt: str, option_a: str = "A", option_b: str = "B") -> tuple[float, float]:
        token_a = self._single_token_id(option_a)
        token_b = self._single_token_id(option_b)
        inputs = self._encode_prompt(prompt)

        with self._torch.no_grad():
            logits = self._model(**inputs, use_cache=False).logits[0, -1]

        logit_a = float(logits[token_a].item())
        logit_b = float(logits[token_b].item())
        return logit_a, logit_b

    def capture_layer_last_token(self, prompt: str, layer_indices: list[int]):
        layers = self._decoder_layers()
        wanted = set(layer_indices)
        captured: dict[int, Any] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden[0, -1, :].detach().clone()

            return hook

        for layer_idx in sorted(wanted):
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                self._model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        return captured

    def capture_component_last_token(self, prompt: str, component_specs: list[tuple[int, str]]):
        captured: dict[tuple[int, str], Any] = {}
        handles = []

        def make_hook(spec: tuple[int, str]):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[spec] = hidden[0, -1, :].detach().clone()

            return hook

        for layer_idx, component_type in component_specs:
            module = self._component_module(layer_idx=layer_idx, component_type=component_type)
            handles.append(module.register_forward_hook(make_hook((layer_idx, component_type))))

        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                self._model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        return captured

    def score_ab_with_layer_last_token_patch(
        self,
        prompt: str,
        layer_idx: int,
        replacement_last_token,
        option_a: str = "A",
        option_b: str = "B",
    ) -> tuple[float, float]:
        token_a = self._single_token_id(option_a)
        token_b = self._single_token_id(option_b)
        layer = self._decoder_layers()[layer_idx]

        def patch_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1, :] = replacement_last_token.to(device=hidden.device, dtype=hidden.dtype)
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, -1, :] = replacement_last_token.to(device=hidden.device, dtype=hidden.dtype)
            return hidden

        handle = layer.register_forward_hook(patch_hook)
        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                logits = self._model(**inputs, use_cache=False).logits[0, -1]
        finally:
            handle.remove()

        logit_a = float(logits[token_a].item())
        logit_b = float(logits[token_b].item())
        return logit_a, logit_b

    def score_ab_with_component_last_token_patch(
        self,
        prompt: str,
        layer_idx: int,
        component_type: str,
        replacement_last_token,
        option_a: str = "A",
        option_b: str = "B",
    ) -> tuple[float, float]:
        token_a = self._single_token_id(option_a)
        token_b = self._single_token_id(option_b)
        module = self._component_module(layer_idx=layer_idx, component_type=component_type)

        def patch_hook(_module, _inputs, output):
            return self._replace_last_token_output(output, replacement_last_token)

        handle = module.register_forward_hook(patch_hook)
        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                logits = self._model(**inputs, use_cache=False).logits[0, -1]
        finally:
            handle.remove()

        logit_a = float(logits[token_a].item())
        logit_b = float(logits[token_b].item())
        return logit_a, logit_b

    def score_ab_with_layer_last_token_add(
        self,
        prompt: str,
        layer_idx: int,
        direction,
        alpha: float = 1.0,
        option_a: str = "A",
        option_b: str = "B",
    ) -> tuple[float, float]:
        token_a = self._single_token_id(option_a)
        token_b = self._single_token_id(option_b)
        layer = self._decoder_layers()[layer_idx]

        def add_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                delta = direction.to(device=hidden.device, dtype=hidden.dtype)
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * delta
                return (hidden, *output[1:])
            hidden = output.clone()
            delta = direction.to(device=hidden.device, dtype=hidden.dtype)
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * delta
            return hidden

        handle = layer.register_forward_hook(add_hook)
        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                logits = self._model(**inputs, use_cache=False).logits[0, -1]
        finally:
            handle.remove()

        logit_a = float(logits[token_a].item())
        logit_b = float(logits[token_b].item())
        return logit_a, logit_b

    def score_ab_with_component_last_token_add(
        self,
        prompt: str,
        layer_idx: int,
        component_type: str,
        direction,
        alpha: float = 1.0,
        option_a: str = "A",
        option_b: str = "B",
    ) -> tuple[float, float]:
        token_a = self._single_token_id(option_a)
        token_b = self._single_token_id(option_b)
        module = self._component_module(layer_idx=layer_idx, component_type=component_type)

        def add_hook(_module, _inputs, output):
            return self._add_last_token_output(output, direction, alpha)

        handle = module.register_forward_hook(add_hook)
        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                logits = self._model(**inputs, use_cache=False).logits[0, -1]
        finally:
            handle.remove()

        logit_a = float(logits[token_a].item())
        logit_b = float(logits[token_b].item())
        return logit_a, logit_b

    def score_ab_with_component_last_token_add_many(
        self,
        prompt: str,
        additions: list[dict[str, Any]],
        option_a: str = "A",
        option_b: str = "B",
    ) -> tuple[float, float]:
        token_a = self._single_token_id(option_a)
        token_b = self._single_token_id(option_b)
        handles = []

        def make_hook(direction, alpha: float):
            def add_hook(_module, _inputs, output):
                return self._add_last_token_output(output, direction, alpha)

            return add_hook

        for addition in additions:
            module = self._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            handles.append(
                module.register_forward_hook(make_hook(addition["direction"], float(addition["alpha"])))
            )

        try:
            inputs = self._encode_prompt(prompt)
            with self._torch.no_grad():
                logits = self._model(**inputs, use_cache=False).logits[0, -1]
        finally:
            for handle in handles:
                handle.remove()

        logit_a = float(logits[token_a].item())
        logit_b = float(logits[token_b].item())
        return logit_a, logit_b

    @staticmethod
    def _cut_stop_strings(text: str, stop_strings: list[str] | None) -> str:
        if not stop_strings:
            return text
        first_idx: int | None = None
        for stop in stop_strings:
            if not stop:
                continue
            idx = text.find(stop)
            if idx >= 0 and (first_idx is None or idx < first_idx):
                first_idx = idx
        return text if first_idx is None else text[:first_idx]

    def _generate_from_inputs(
        self,
        inputs: dict,
        *,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        logits_processor=None,
    ) -> str:
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        context = getattr(self._torch, "inference_mode", self._torch.no_grad)
        with context():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        generated_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        del output_ids
        return self._cut_stop_strings(text, stop_strings).strip()

    def _generate_many_from_inputs(
        self,
        inputs: dict,
        *,
        max_new_tokens: int,
        stop_strings: list[str] | None,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[str]:
        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
        context = getattr(self._torch, "inference_mode", self._torch.no_grad)
        with context():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        texts: list[str] = []
        for row_ids in output_ids[:, prompt_len:].detach().cpu().tolist():
            text = self._tokenizer.decode(row_ids, skip_special_tokens=True)
            texts.append(self._cut_stop_strings(text, stop_strings).strip())
        del output_ids
        return texts

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> str:
        inputs = self._encode_prompt(prompt)
        return self._generate_from_inputs(
            inputs,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def generate_many(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> list[str]:
        inputs = self._encode_prompts(prompts)
        return self._generate_many_from_inputs(
            inputs,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    def generate_with_component_zero(
        self,
        prompt: str,
        *,
        layer_idx: int,
        component_type: str,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
        apply_mode: str = "all",
    ) -> str:
        def parse_apply_mode(raw: str) -> tuple[str, int | None]:
            first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
            if first_decode_match:
                first_decode_steps = int(first_decode_match.group(1) or "1")
                if first_decode_steps <= 0:
                    raise ValueError(f"Unsupported apply_mode: {raw}")
                return raw, first_decode_steps
            if raw not in {"all", "prefill", "decode"}:
                raise ValueError(f"Unsupported apply_mode: {raw}")
            return raw, None

        _, first_decode_steps = parse_apply_mode(apply_mode)
        decode_steps_seen = 0
        module = self._component_module(layer_idx=int(layer_idx), component_type=str(component_type))

        def zero_hook(_module, _inputs, output):
            nonlocal decode_steps_seen
            hidden = output[0] if isinstance(output, tuple) else output
            seq_len = int(hidden.shape[1])
            if apply_mode == "prefill" and seq_len <= 1:
                return output
            if apply_mode == "decode" and seq_len > 1:
                return output
            if first_decode_steps is not None:
                if seq_len > 1:
                    return output
                decode_steps_seen += 1
                if decode_steps_seen > first_decode_steps:
                    return output
            hidden_new = hidden.clone()
            hidden_new[:, :, :] = 0
            if isinstance(output, tuple):
                return (hidden_new, *output[1:])
            return hidden_new

        handle = module.register_forward_hook(zero_hook)
        try:
            inputs = self._encode_prompt(prompt)
            return self._generate_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            handle.remove()

    def generate_many_with_component_zero(
        self,
        prompts: list[str],
        *,
        layer_idx: int,
        component_type: str,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
        apply_mode: str = "all",
    ) -> list[str]:
        def parse_apply_mode(raw: str) -> tuple[str, int | None]:
            first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
            if first_decode_match:
                first_decode_steps = int(first_decode_match.group(1) or "1")
                if first_decode_steps <= 0:
                    raise ValueError(f"Unsupported apply_mode: {raw}")
                return raw, first_decode_steps
            if raw not in {"all", "prefill", "decode"}:
                raise ValueError(f"Unsupported apply_mode: {raw}")
            return raw, None

        _, first_decode_steps = parse_apply_mode(apply_mode)
        decode_steps_seen = 0
        module = self._component_module(layer_idx=int(layer_idx), component_type=str(component_type))

        def zero_hook(_module, _inputs, output):
            nonlocal decode_steps_seen
            hidden = output[0] if isinstance(output, tuple) else output
            seq_len = int(hidden.shape[1])
            if apply_mode == "prefill" and seq_len <= 1:
                return output
            if apply_mode == "decode" and seq_len > 1:
                return output
            if first_decode_steps is not None:
                if seq_len > 1:
                    return output
                decode_steps_seen += 1
                if decode_steps_seen > first_decode_steps:
                    return output
            hidden_new = hidden.clone()
            hidden_new[:, :, :] = 0
            if isinstance(output, tuple):
                return (hidden_new, *output[1:])
            return hidden_new

        handle = module.register_forward_hook(zero_hook)
        try:
            inputs = self._encode_prompts(prompts)
            return self._generate_many_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            handle.remove()

    def generate_with_component_last_token_add_many(
        self,
        prompt: str,
        additions: list[dict[str, Any]],
        *,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
        apply_mode: str = "all",
    ) -> str:
        handles = []
        def parse_apply_mode(raw: str) -> tuple[str, int | None]:
            first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
            if first_decode_match:
                first_decode_steps = int(first_decode_match.group(1) or "1")
                if first_decode_steps <= 0:
                    raise ValueError(f"Unsupported apply_mode: {raw}")
                return raw, first_decode_steps
            if raw not in {"all", "prefill", "decode"}:
                raise ValueError(f"Unsupported apply_mode: {raw}")
            return raw, None

        parse_apply_mode(apply_mode)

        def make_hook(direction, alpha: float, local_apply_mode: str):
            _, first_decode_steps = parse_apply_mode(local_apply_mode)
            decode_steps_seen = 0

            def add_hook(_module, _inputs, output):
                nonlocal decode_steps_seen
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = int(hidden.shape[1])
                if local_apply_mode == "prefill" and seq_len <= 1:
                    return output
                if local_apply_mode == "decode" and seq_len > 1:
                    return output
                if first_decode_steps is not None:
                    if seq_len > 1:
                        return output
                    decode_steps_seen += 1
                    if decode_steps_seen > first_decode_steps:
                        return output
                return self._add_last_token_output(output, direction, alpha)

            return add_hook

        for addition in additions:
            module = self._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            local_apply_mode = str(addition.get("apply_mode", apply_mode))
            handles.append(
                module.register_forward_hook(
                    make_hook(addition["direction"], float(addition["alpha"]), local_apply_mode)
                )
            )

        try:
            inputs = self._encode_prompt(prompt)
            return self._generate_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            for handle in handles:
                handle.remove()

    def generate_many_with_component_last_token_add_many(
        self,
        prompts: list[str],
        additions: list[dict[str, Any]],
        *,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
        apply_mode: str = "all",
    ) -> list[str]:
        handles = []

        def parse_apply_mode(raw: str) -> tuple[str, int | None]:
            first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
            if first_decode_match:
                first_decode_steps = int(first_decode_match.group(1) or "1")
                if first_decode_steps <= 0:
                    raise ValueError(f"Unsupported apply_mode: {raw}")
                return raw, first_decode_steps
            if raw not in {"all", "prefill", "decode"}:
                raise ValueError(f"Unsupported apply_mode: {raw}")
            return raw, None

        parse_apply_mode(apply_mode)

        def make_hook(direction, alpha: float, local_apply_mode: str):
            _, first_decode_steps = parse_apply_mode(local_apply_mode)
            decode_steps_seen = 0

            def add_hook(_module, _inputs, output):
                nonlocal decode_steps_seen
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = int(hidden.shape[1])
                if local_apply_mode == "prefill" and seq_len <= 1:
                    return output
                if local_apply_mode == "decode" and seq_len > 1:
                    return output
                if first_decode_steps is not None:
                    if seq_len > 1:
                        return output
                    decode_steps_seen += 1
                    if decode_steps_seen > first_decode_steps:
                        return output
                return self._add_last_token_output(output, direction, alpha)

            return add_hook

        for addition in additions:
            module = self._component_module(
                layer_idx=int(addition["layer_idx"]),
                component_type=str(addition["component_type"]),
            )
            local_apply_mode = str(addition.get("apply_mode", apply_mode))
            handles.append(
                module.register_forward_hook(
                    make_hook(addition["direction"], float(addition["alpha"]), local_apply_mode)
                )
            )

        try:
            inputs = self._encode_prompts(prompts)
            return self._generate_many_from_inputs(
                inputs,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        finally:
            for handle in handles:
                handle.remove()

    def logit_delta_from_component_additions(self, additions: list[dict[str, Any]]):
        """Project residual directions into vocabulary-logit space.

        This is an execution-layer approximation: the selected component directions
        remain the source of the intervention, but the actuation is applied directly
        to candidate token scores.
        """
        lm_head = self._model.get_output_embeddings()
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise ValueError("Model does not expose an output embedding matrix for logit projection.")
        weight = lm_head.weight.detach()
        total = None
        for addition in additions:
            direction = addition["direction"].to(device=weight.device, dtype=weight.dtype)
            scaled = float(addition.get("alpha", 1.0)) * direction
            total = scaled if total is None else total + scaled
        if total is None:
            raise ValueError("No additions were supplied for logit projection.")
        return self._torch.matmul(total, weight.t()).detach()

    def generate_with_component_logit_delta(
        self,
        prompt: str,
        additions: list[dict[str, Any]],
        *,
        logit_alpha: float = 1.0,
        relative_top: float = 0.01,
        min_tokens_to_keep: int = 10,
        max_new_tokens: int = 64,
        stop_strings: list[str] | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 100,
        apply_mode: str = "all",
    ) -> str:
        try:
            from transformers import LogitsProcessor, LogitsProcessorList
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Logit actuator requires transformers generation utilities.") from exc

        if relative_top <= 0.0 or relative_top > 1.0:
            raise ValueError("--logit_relative_top must be in (0, 1].")
        if min_tokens_to_keep <= 0:
            raise ValueError("--logit_min_tokens_to_keep must be positive.")

        inputs = self._encode_prompt(prompt)
        prompt_len = int(inputs["input_ids"].shape[-1])
        delta = self.logit_delta_from_component_additions(additions).to(self.device)

        def parse_apply_mode(raw: str) -> tuple[str, int | None]:
            first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", raw)
            if first_decode_match:
                first_decode_steps = int(first_decode_match.group(1) or "1")
                if first_decode_steps <= 0:
                    raise ValueError(f"Unsupported apply_mode: {raw}")
                return raw, first_decode_steps
            if raw not in {"all", "prefill", "decode"}:
                raise ValueError(f"Unsupported apply_mode: {raw}")
            return raw, None

        _, first_decode_steps = parse_apply_mode(apply_mode)

        class AxisLogitProcessor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                generated_len = max(0, int(input_ids.shape[-1]) - prompt_len)
                should_apply = False
                if apply_mode == "all":
                    should_apply = True
                elif apply_mode == "prefill":
                    should_apply = generated_len == 0
                elif apply_mode == "decode":
                    should_apply = generated_len > 0
                elif first_decode_steps is not None:
                    should_apply = 0 < generated_len <= first_decode_steps
                if not should_apply:
                    return scores

                log_probs = scores.log_softmax(dim=-1)
                sorted_log_probs, _ = self_outer._torch.sort(log_probs, descending=True)
                min_index = min(min_tokens_to_keep - 1, sorted_log_probs.shape[-1] - 1)
                min_threshold = sorted_log_probs[..., min_index].unsqueeze(-1)
                relative_threshold = log_probs.max(dim=-1).values.unsqueeze(-1) + self_outer._torch.log(
                    self_outer._torch.tensor(relative_top, device=scores.device, dtype=log_probs.dtype)
                )
                threshold = self_outer._torch.minimum(min_threshold, relative_threshold)
                mask = log_probs >= threshold
                adjusted = scores.clone()
                local_delta = delta.to(device=scores.device, dtype=scores.dtype).unsqueeze(0)
                adjusted = adjusted + mask.to(scores.dtype) * float(logit_alpha) * local_delta
                return adjusted

        self_outer = self
        return self._generate_from_inputs(
            inputs,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_strings,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logits_processor=LogitsProcessorList([AxisLogitProcessor()]),
        )
