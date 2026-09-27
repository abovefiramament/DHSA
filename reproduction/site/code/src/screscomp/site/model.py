from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from screscomp.cecm.actuator import ActuatorPair
from screscomp.cecm.objective import limit_options, score_text_token_slice, select_option_score


@dataclass(frozen=True, slots=True)
class SiteComponent:
    layer_idx: int
    head_idx: int | None = None

    @property
    def component_id(self) -> str:
        if self.head_idx is None:
            return f"L{self.layer_idx}.attn"
        return f"L{self.layer_idx}.attn.h{self.head_idx}"


@dataclass(frozen=True, slots=True)
class StateAction:
    component: SiteComponent
    operation: str
    prototype: Any | None = None


class PreOInterface:
    """Model-family-neutral access to concatenated attention heads before o_proj."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.model.eval()

    def attention_module(self, layer_idx: int) -> Any:
        return self.backend._component_module(layer_idx=layer_idx, component_type="attn")

    def o_proj(self, layer_idx: int) -> Any:
        attention = self.attention_module(layer_idx)
        for name in ("o_proj", "out_proj", "dense", "c_proj"):
            if hasattr(attention, name):
                return getattr(attention, name)
        raise ValueError(f"could not locate attention output projection for layer {layer_idx}")

    def geometry(self, layer_idx: int) -> tuple[int, int, int]:
        attention = self.attention_module(layer_idx)
        projection = self.o_proj(layer_idx)
        hidden_size = int(
            getattr(projection, "in_features", 0)
            or getattr(projection, "out_features", 0)
            or getattr(projection, "nf", 0)
            or getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
        )
        num_heads = int(
            getattr(attention, "num_heads", 0)
            or getattr(attention, "num_attention_heads", 0)
            or getattr(self.model.config, "num_attention_heads", 0)
            or getattr(self.model.config, "n_head", 0)
        )
        head_dim = int(getattr(attention, "head_dim", 0) or (hidden_size // num_heads if num_heads else 0))
        if hidden_size <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"could not infer pre-o head geometry for layer {layer_idx}")
        if num_heads * head_dim != hidden_size:
            num_heads = hidden_size // head_dim
        if num_heads * head_dim != hidden_size:
            raise ValueError(f"non-factorable pre-o geometry at layer {layer_idx}")
        return hidden_size, num_heads, head_dim

    def all_heads(self) -> list[SiteComponent]:
        rows: list[SiteComponent] = []
        for layer_idx in range(self.backend.num_layers):
            _hidden, num_heads, _head_dim = self.geometry(layer_idx)
            rows.extend(SiteComponent(layer_idx, head_idx) for head_idx in range(num_heads))
        return rows

    def _component_slice(self, component: SiteComponent) -> slice:
        hidden_size, num_heads, head_dim = self.geometry(component.layer_idx)
        if component.head_idx is None:
            return slice(0, hidden_size)
        if component.head_idx < 0 or component.head_idx >= num_heads:
            raise ValueError(f"invalid head {component.component_id}")
        start = component.head_idx * head_dim
        return slice(start, start + head_dim)

    @contextmanager
    def action_hook(
        self,
        action: StateAction | None,
        *,
        fixed_positions: slice | None = None,
        generation: bool = False,
    ) -> Iterator[None]:
        if action is None:
            yield
            return
        if action.operation not in {"zero", "clamp"}:
            raise ValueError(f"unsupported Site state operation: {action.operation}")
        target_slice = self._component_slice(action.component)
        module = self.o_proj(action.component.layer_idx)

        def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            hidden = inputs[0]
            hidden_new = hidden.clone()
            if generation:
                positions = slice(max(int(hidden.shape[1]) - 1, 0), int(hidden.shape[1]))
            else:
                if fixed_positions is None:
                    raise ValueError("fixed scoring action requires decision positions")
                positions = fixed_positions
            if action.operation == "zero":
                hidden_new[:, positions, target_slice] = 0
            else:
                if action.prototype is None:
                    raise ValueError("clamp action requires a prototype")
                prototype = self.torch.as_tensor(
                    action.prototype,
                    device=hidden_new.device,
                    dtype=hidden_new.dtype,
                )
                if action.component.head_idx is not None and int(prototype.numel()) != target_slice.stop - target_slice.start:
                    prototype = prototype[target_slice]
                expected = target_slice.stop - target_slice.start
                if int(prototype.numel()) != expected:
                    raise ValueError(
                        f"prototype geometry mismatch for {action.component.component_id}: "
                        f"expected {expected}, got {prototype.numel()}"
                    )
                hidden_new[:, positions, target_slice] = prototype.reshape(1, 1, expected)
            return (hidden_new, *inputs[1:])

        handle = module.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def tokenize_sequence(self, prompt: str, continuation: str) -> tuple[Any, int, int]:
        formatted = self.backend._format_prompt(prompt)
        prompt_ids = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=True)["input_ids"]
        continuation_ids = self.tokenizer(
            continuation,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if int(prompt_ids.shape[-1]) <= 0 or int(continuation_ids.shape[-1]) <= 0:
            raise ValueError("Site sequence has an empty prompt or continuation")
        input_ids = self.torch.cat([prompt_ids, continuation_ids], dim=-1).to(self.device)
        return input_ids, int(prompt_ids.shape[-1]), int(continuation_ids.shape[-1])

    @staticmethod
    def decision_positions(prompt_len: int, continuation_len: int, seq_len: int) -> slice:
        start = max(prompt_len - 1, 0)
        stop = min(prompt_len + continuation_len - 1, seq_len)
        if stop <= start:
            raise ValueError("empty generation-decision trajectory")
        return slice(start, stop)

    def collect_layer_sequence_means(self, prompt: str, continuation: str) -> tuple[dict[int, Any], int]:
        input_ids, prompt_len, continuation_len = self.tokenize_sequence(prompt, continuation)
        return self.collect_layer_sequence_means_ids(prompt, input_ids[:, prompt_len:].squeeze(0).tolist())

    def collect_layer_sequence_means_ids(self, prompt: str, output_token_ids: list[int]) -> tuple[dict[int, Any], int]:
        if not output_token_ids:
            raise ValueError("cannot collect a prototype from an empty target sequence")
        formatted = self.backend._format_prompt(prompt)
        prompt_ids = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        continuation_ids = self.torch.tensor([output_token_ids], device=self.device, dtype=prompt_ids.dtype)
        input_ids = self.torch.cat([prompt_ids, continuation_ids], dim=-1)
        positions = self.decision_positions(
            int(prompt_ids.shape[-1]),
            len(output_token_ids),
            int(input_ids.shape[-1]),
        )
        captured: dict[int, Any] = {}
        handles = []
        for layer_idx in range(self.backend.num_layers):
            def make_hook(local_layer: int):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    hidden = inputs[0]
                    captured[local_layer] = hidden[0, positions, :].float().mean(dim=0).detach().cpu()
                return hook
            handles.append(self.o_proj(layer_idx).register_forward_pre_hook(make_hook(layer_idx)))
        try:
            with self.torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=self.torch.ones_like(input_ids), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != self.backend.num_layers:
            raise RuntimeError("incomplete pre-o trajectory capture")
        return captured, len(output_token_ids)

    def capture_final_all_heads(self, prompt: str, continuation: str) -> Any:
        input_ids, prompt_len, _continuation_len = self.tokenize_sequence(prompt, continuation)
        return self.capture_final_all_heads_ids(
            prompt,
            input_ids[:, prompt_len:].squeeze(0).tolist(),
        )

    def capture_final_all_heads_ids(
        self,
        prompt: str,
        output_token_ids: list[int],
        *,
        exclude_terminal_eos: bool = False,
    ) -> Any:
        if not output_token_ids:
            raise ValueError("cannot capture an ITI state from an empty generated sequence")
        admitted_ids = list(output_token_ids)
        eos_id = self.tokenizer.eos_token_id
        if (
            exclude_terminal_eos
            and eos_id is not None
            and len(admitted_ids) > 1
            and admitted_ids[-1] == eos_id
        ):
            admitted_ids.pop()
        formatted = self.backend._format_prompt(prompt)
        prompt_ids = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"].to(self.device)
        continuation_ids = self.torch.tensor(
            [admitted_ids],
            device=self.device,
            dtype=prompt_ids.dtype,
        )
        input_ids = self.torch.cat([prompt_ids, continuation_ids], dim=-1)
        captured: dict[int, Any] = {}
        handles = []
        for layer_idx in range(self.backend.num_layers):
            def make_hook(local_layer: int):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    captured[local_layer] = inputs[0][0, -1, :].float().detach().cpu()
                return hook
            handles.append(self.o_proj(layer_idx).register_forward_pre_hook(make_hook(layer_idx)))
        try:
            with self.torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=self.torch.ones_like(input_ids), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        layer_rows = []
        for layer_idx in range(self.backend.num_layers):
            _hidden, num_heads, head_dim = self.geometry(layer_idx)
            layer_rows.append(captured[layer_idx].reshape(num_heads, head_dim))
        return self.torch.stack(layer_rows, dim=0).numpy()

    def generate_batch(
        self,
        prompts: list[str],
        *,
        action: StateAction | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []
        inputs = self.backend._encode_prompts(prompts)
        input_width = int(inputs["input_ids"].shape[-1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        with self.action_hook(action, generation=True):
            with self.torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
        rows: list[dict[str, Any]] = []
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        for output in outputs:
            token_ids = [int(value) for value in output[input_width:].detach().cpu().tolist()]
            if eos_id is not None and eos_id in token_ids:
                token_ids = token_ids[: token_ids.index(eos_id) + 1]
            elif pad_id is not None:
                while token_ids and token_ids[-1] == pad_id:
                    token_ids.pop()
            text_ids = [value for value in token_ids if value != eos_id]
            rows.append(
                {
                    "completion": self.tokenizer.decode(text_ids, skip_special_tokens=True),
                    "generated_token_ids": token_ids,
                    "generated_tokens": len(token_ids),
                }
            )
        return rows


class ClosedCompetitionRunner:
    def __init__(
        self,
        interface: PreOInterface,
        *,
        score_mode: str,
        option_selection_mode: str,
        max_aliases_per_side: int,
    ) -> None:
        if score_mode != "answer_rest_margin":
            raise ValueError("locked ConFiQA Site selector requires answer_rest_margin")
        self.interface = interface
        self.torch = interface.torch
        self.model = interface.model
        self.tokenizer = interface.tokenizer
        self.score_mode = score_mode
        self.option_selection_mode = option_selection_mode
        self.max_aliases_per_side = max_aliases_per_side

    def candidate_score(self, prompt: str, continuation: str, action: StateAction | None) -> Any:
        input_ids, prompt_len, continuation_len = self.interface.tokenize_sequence(prompt, continuation)
        positions = self.interface.decision_positions(prompt_len, continuation_len, int(input_ids.shape[-1]))
        with self.interface.action_hook(action, fixed_positions=positions):
            logits = self.model(
                input_ids=input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                use_cache=False,
            ).logits
        targets = input_ids[:, prompt_len : prompt_len + continuation_len]
        predictions = logits[:, prompt_len - 1 : prompt_len + continuation_len - 1, :].float()
        token_logits = predictions.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        top_values, top_indices = predictions.topk(k=2, dim=-1)
        rest = self.torch.where(
            top_indices[..., 0] == targets,
            top_values[..., 1],
            top_values[..., 0],
        )
        return (token_logits - rest).mean()

    def endpoint(self, prompt: str, options: tuple[str, ...], action: StateAction | None) -> Any:
        admitted = limit_options(options, max_aliases_per_side=self.max_aliases_per_side)
        if not admitted:
            raise ValueError("empty endpoint alias set")
        scores = [self.candidate_score(prompt, option, action) for option in admitted]
        return select_option_score(self.torch, scores, selection_mode=self.option_selection_mode)

    def margin(self, pair: ActuatorPair, action: StateAction | None = None) -> float:
        with self.torch.no_grad():
            plus = self.endpoint(pair.prompt, pair.y_plus_options, action)
            minus = self.endpoint(pair.prompt, pair.y_minus_options, action)
            return float((plus - minus).detach().cpu().item())
