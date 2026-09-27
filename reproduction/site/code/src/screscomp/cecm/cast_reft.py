
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from screscomp.cecm.components import parse_component_id


@dataclass(frozen=True, slots=True)
class CastReftSite:
    site_id: str
    site_kind: str
    layer_idx: int
    component_type: str = ""
    component_id: str = ""
    head_idx: int = -1
    hook_site: str = ""


def safe_param_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "__", str(value))
    key = re.sub(r"__+", "__", key).strip("_")
    return key or "site"


def parse_head_id(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"L(\d+)\.attn\.h(\d+)", raw.strip())
    if not match:
        raise ValueError(f"Unsupported head id: {raw!r}; expected L<layer>.attn.h<head>")
    return int(match.group(1)), int(match.group(2))


def parse_head_ids(raw: str) -> list[CastReftSite]:
    sites: list[CastReftSite] = []
    for item in str(raw or "").replace("\n", ",").split(","):
        head_id = item.strip()
        if not head_id:
            continue
        layer_idx, head_idx = parse_head_id(head_id)
        sites.append(
            CastReftSite(
                site_id=head_id,
                site_kind="head",
                layer_idx=layer_idx,
                component_type="attn",
                component_id=head_id,
                head_idx=head_idx,
                hook_site="pre_output_projection",
            )
        )
    return sites


def load_head_sites_file(path: Path) -> list[CastReftSite]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    first = text.splitlines()[0]
    if "," in first and ("head_id" in first or "component_id" in first):
        sites: list[CastReftSite] = []
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head_id = str(row.get("head_id") or row.get("component_id") or "").strip()
                if not head_id:
                    continue
                sites.extend(parse_head_ids(head_id))
        return sites
    return parse_head_ids(text)


def load_component_sites_csv(path: Path, *, hook_site: str = "post_module_output") -> list[CastReftSite]:
    if hook_site not in {"post_module_output", "pre_module_input"}:
        raise ValueError(f"Unsupported component hook site: {hook_site!r}")
    sites: list[CastReftSite] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw_component_id = str(row.get("component_id", "")).strip()
            if not raw_component_id:
                continue
            parsed = parse_component_id(raw_component_id)
            component_type = str(row.get("component_type") or parsed.component_type)
            layer_idx = int(row.get("layer_idx") or parsed.layer_idx)
            site_id = f"{parsed.component_id}@{hook_site}"
            sites.append(
                CastReftSite(
                    site_id=site_id,
                    site_kind="component",
                    layer_idx=layer_idx,
                    component_type=component_type,
                    component_id=parsed.component_id,
                    head_idx=-1,
                    hook_site=hook_site,
                )
            )
    return sites


def dedupe_sites(sites: list[CastReftSite]) -> list[CastReftSite]:
    seen: set[str] = set()
    out: list[CastReftSite] = []
    for site in sites:
        if site.site_id in seen:
            continue
        seen.add(site.site_id)
        out.append(site)
    return out


def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))


def parse_generation_apply_mode(raw: str) -> tuple[str, int | None]:
    mode = str(raw or "all")
    first_decode_match = re.fullmatch(r"first(?:_(\d+))?_decode", mode)
    if first_decode_match:
        steps = int(first_decode_match.group(1) or "1")
        if steps <= 0:
            raise ValueError(f"Unsupported generation apply mode: {raw}")
        return mode, steps
    if mode not in {"all", "all_positions", "prefill", "decode", "prompt", "prompt_last"}:
        raise ValueError(f"Unsupported generation apply mode: {raw}")
    return mode, None


class CastReftController:
    def __init__(
        self,
        *,
        backend: Any,
        sites: list[CastReftSite],
        reft_mode: str = "gated_vector",
        rank: int = 4,
        gate_init: float = 0.98,
        init_std: float = 0.01,
    ) -> None:
        if reft_mode not in {"gated_vector", "low_rank"}:
            raise ValueError(f"Unsupported ReFT mode: {reft_mode!r}")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.backend = backend
        self.torch = backend._torch
        self.model = backend._model
        self.device = backend.device
        self.sites = dedupe_sites(sites)
        self.reft_mode = reft_mode
        self.rank = int(rank)
        self.gate_init = float(gate_init)
        self.init_std = float(init_std)
        self.params = self.torch.nn.ParameterDict()
        self.site_dims: dict[str, int] = {}
        self.head_dims: dict[int, int] = {}
        self.num_heads_by_layer: dict[int, int] = {}
        for site in self.sites:
            dim = self._site_dim(site)
            self.site_dims[site.site_id] = dim
            prefix = self._prefix(site)
            if self.reft_mode == "gated_vector":
                self.params[f"{prefix}__vector"] = self.torch.nn.Parameter(
                    self.torch.zeros(dim, device=self.device, dtype=self.torch.float32)
                )
                self.params[f"{prefix}__gate_weight"] = self.torch.nn.Parameter(
                    self.torch.zeros(dim, device=self.device, dtype=self.torch.float32)
                )
                self.params[f"{prefix}__gate_bias"] = self.torch.nn.Parameter(
                    self.torch.tensor([_logit(self.gate_init)], device=self.device, dtype=self.torch.float32)
                )
            else:
                down = self.torch.empty(dim, self.rank, device=self.device, dtype=self.torch.float32)
                self.torch.nn.init.normal_(down, mean=0.0, std=self.init_std)
                self.params[f"{prefix}__down"] = self.torch.nn.Parameter(down)
                self.params[f"{prefix}__up"] = self.torch.nn.Parameter(
                    self.torch.zeros(self.rank, dim, device=self.device, dtype=self.torch.float32)
                )

    def _prefix(self, site: CastReftSite) -> str:
        return safe_param_key(site.site_id)

    def _attention_module(self, layer_idx: int):
        return self.backend._component_module(layer_idx=layer_idx, component_type="attn")

    def _o_proj_module(self, layer_idx: int):
        attn = self._attention_module(layer_idx)
        for attr in ("o_proj", "out_proj", "dense", "c_proj"):
            if hasattr(attn, attr):
                return getattr(attn, attr)
        raise ValueError(f"Could not locate attention output projection for L{layer_idx}.attn")

    def _head_geometry(self, layer_idx: int) -> tuple[int, int, int]:
        attn = self._attention_module(layer_idx)
        o_proj = self._o_proj_module(layer_idx)
        hidden_size = int(
            getattr(o_proj, "in_features", 0)
            or getattr(o_proj, "out_features", 0)
            or getattr(o_proj, "nf", 0)
            or getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
        )
        num_heads = int(
            getattr(attn, "num_heads", 0)
            or getattr(attn, "num_attention_heads", 0)
            or getattr(self.model.config, "num_attention_heads", 0)
            or getattr(self.model.config, "n_head", 0)
        )
        head_dim = int(getattr(attn, "head_dim", 0) or (hidden_size // num_heads if num_heads else 0))
        if hidden_size <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError(f"Could not infer attention head geometry for layer {layer_idx}")
        if num_heads * head_dim != hidden_size:
            num_heads = hidden_size // head_dim
        return hidden_size, num_heads, head_dim

    def _hidden_size(self) -> int:
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or getattr(self.model.config, "n_embd", 0))
        if hidden_size <= 0:
            raise ValueError("Could not infer model hidden size")
        return hidden_size

    def _site_dim(self, site: CastReftSite) -> int:
        if site.site_kind == "head":
            _hidden, num_heads, head_dim = self._head_geometry(site.layer_idx)
            if site.head_idx < 0 or site.head_idx >= num_heads:
                raise ValueError(f"Invalid {site.site_id}; L{site.layer_idx}.attn has {num_heads} heads")
            self.num_heads_by_layer[site.layer_idx] = num_heads
            self.head_dims[site.layer_idx] = head_dim
            return head_dim
        if site.site_kind == "component":
            return self._hidden_size()
        raise ValueError(f"Unsupported site kind: {site.site_kind!r}")

    def _delta(self, site: CastReftSite, hidden_slice: Any) -> Any:
        prefix = self._prefix(site)
        hidden_f = hidden_slice.float()
        if self.reft_mode == "gated_vector":
            vector = self.params[f"{prefix}__vector"]
            gate_weight = self.params[f"{prefix}__gate_weight"]
            gate_bias = self.params[f"{prefix}__gate_bias"]
            denom = math.sqrt(float(max(hidden_f.shape[-1], 1)))
            gate = self.torch.sigmoid((hidden_f * gate_weight.view(1, 1, -1)).sum(dim=-1, keepdim=True) / denom + gate_bias)
            return (gate * vector.view(1, 1, -1)).to(dtype=hidden_slice.dtype, device=hidden_slice.device)
        down = self.params[f"{prefix}__down"]
        up = self.params[f"{prefix}__up"]
        denom = math.sqrt(float(max(self.rank, 1)))
        return ((hidden_f @ down) @ up / denom).to(dtype=hidden_slice.dtype, device=hidden_slice.device)

    def _apply_site_delta(self, site: CastReftSite, hidden: Any, *, alpha: float, mask: Any) -> Any:
        hidden_new = hidden.clone()
        local_mask = mask.to(device=hidden_new.device, dtype=hidden_new.dtype).unsqueeze(-1)
        if site.site_kind == "head":
            head_dim = self.head_dims[site.layer_idx]
            start = int(site.head_idx) * head_dim
            stop = start + head_dim
            local_hidden = hidden_new[:, :, start:stop]
            delta = self._delta(site, local_hidden)
            hidden_new[:, :, start:stop] = hidden_new[:, :, start:stop] + float(alpha) * local_mask * delta
            return hidden_new
        delta = self._delta(site, hidden_new)
        return hidden_new + float(alpha) * local_mask * delta

    def register_batch_hooks(self, *, alpha: float, position_mask: Any) -> list[Any]:
        handles: list[Any] = []
        mask = position_mask
        heads_by_layer: dict[int, list[CastReftSite]] = {}
        components_by_module: dict[tuple[int, str, str], list[CastReftSite]] = {}
        for site in self.sites:
            if site.site_kind == "head":
                heads_by_layer.setdefault(site.layer_idx, []).append(site)
            elif site.site_kind == "component":
                components_by_module.setdefault((site.layer_idx, site.component_type, site.hook_site), []).append(site)
            else:
                raise ValueError(f"Unsupported site kind: {site.site_kind!r}")

        for layer_idx, sites in heads_by_layer.items():
            module = self._o_proj_module(layer_idx)

            def make_head_hook(local_sites: list[CastReftSite]):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    hidden_new = hidden
                    for site in local_sites:
                        hidden_new = self._apply_site_delta(site, hidden_new, alpha=alpha, mask=mask)
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_head_hook(sites)))

        for (layer_idx, component_type, hook_site), sites in components_by_module.items():
            module = self.backend._component_module(layer_idx=layer_idx, component_type=component_type)
            if hook_site == "pre_module_input":

                def make_pre_hook(local_sites: list[CastReftSite]):
                    def hook(_module, inputs):
                        hidden = inputs[0]
                        hidden_new = hidden
                        for site in local_sites:
                            hidden_new = self._apply_site_delta(site, hidden_new, alpha=alpha, mask=mask)
                        return (hidden_new, *inputs[1:])

                    return hook

                handles.append(module.register_forward_pre_hook(make_pre_hook(sites)))
            elif hook_site == "post_module_output":

                def make_post_hook(local_sites: list[CastReftSite]):
                    def hook(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        hidden_new = hidden
                        for site in local_sites:
                            hidden_new = self._apply_site_delta(site, hidden_new, alpha=alpha, mask=mask)
                        if isinstance(output, tuple):
                            return (hidden_new, *output[1:])
                        return hidden_new

                    return hook

                handles.append(module.register_forward_hook(make_post_hook(sites)))
            else:
                raise ValueError(f"Unsupported component hook site: {hook_site!r}")
        return handles

    @staticmethod
    def _generation_mask(torch_module: Any, hidden: Any, *, mode: str, first_decode_steps: int | None, state: dict[str, int]) -> Any | None:
        seq_len = int(hidden.shape[1])
        active = False
        if mode in {"all", "all_positions"}:
            active = True
        elif mode in {"prefill", "prompt", "prompt_last"}:
            active = seq_len > 1
        elif mode == "decode":
            active = seq_len <= 1
        elif first_decode_steps is not None:
            if seq_len <= 1:
                state["decode_steps_seen"] = state.get("decode_steps_seen", 0) + 1
                active = state["decode_steps_seen"] <= first_decode_steps
        if not active:
            return None
        mask = torch_module.zeros((int(hidden.shape[0]), seq_len), dtype=torch_module.bool, device=hidden.device)
        if mode == "all_positions":
            mask[:, :] = True
        else:
            mask[:, max(seq_len - 1, 0) : seq_len] = True
        return mask

    def register_generation_hooks(self, *, alpha: float, apply_mode: str) -> list[Any]:
        mode, first_decode_steps = parse_generation_apply_mode(apply_mode)
        handles: list[Any] = []
        heads_by_layer: dict[int, list[CastReftSite]] = {}
        components_by_module: dict[tuple[int, str, str], list[CastReftSite]] = {}
        for site in self.sites:
            if site.site_kind == "head":
                heads_by_layer.setdefault(site.layer_idx, []).append(site)
            elif site.site_kind == "component":
                components_by_module.setdefault((site.layer_idx, site.component_type, site.hook_site), []).append(site)
        for layer_idx, sites in heads_by_layer.items():
            module = self._o_proj_module(layer_idx)
            state: dict[str, int] = {}

            def make_head_hook(local_sites: list[CastReftSite], local_state: dict[str, int]):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    mask = self._generation_mask(
                        self.torch,
                        hidden,
                        mode=mode,
                        first_decode_steps=first_decode_steps,
                        state=local_state,
                    )
                    if mask is None:
                        return inputs
                    hidden_new = hidden
                    for site in local_sites:
                        hidden_new = self._apply_site_delta(site, hidden_new, alpha=alpha, mask=mask)
                    return (hidden_new, *inputs[1:])

                return hook

            handles.append(module.register_forward_pre_hook(make_head_hook(sites, state)))
        for (layer_idx, component_type, hook_site), sites in components_by_module.items():
            module = self.backend._component_module(layer_idx=layer_idx, component_type=component_type)
            state = {}
            if hook_site == "pre_module_input":

                def make_pre_hook(local_sites: list[CastReftSite], local_state: dict[str, int]):
                    def hook(_module, inputs):
                        hidden = inputs[0]
                        mask = self._generation_mask(
                            self.torch,
                            hidden,
                            mode=mode,
                            first_decode_steps=first_decode_steps,
                            state=local_state,
                        )
                        if mask is None:
                            return inputs
                        hidden_new = hidden
                        for site in local_sites:
                            hidden_new = self._apply_site_delta(site, hidden_new, alpha=alpha, mask=mask)
                        return (hidden_new, *inputs[1:])

                    return hook

                handles.append(module.register_forward_pre_hook(make_pre_hook(sites, state)))
            else:

                def make_post_hook(local_sites: list[CastReftSite], local_state: dict[str, int]):
                    def hook(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        mask = self._generation_mask(
                            self.torch,
                            hidden,
                            mode=mode,
                            first_decode_steps=first_decode_steps,
                            state=local_state,
                        )
                        if mask is None:
                            return output
                        hidden_new = hidden
                        for site in local_sites:
                            hidden_new = self._apply_site_delta(site, hidden_new, alpha=alpha, mask=mask)
                        if isinstance(output, tuple):
                            return (hidden_new, *output[1:])
                        return hidden_new

                    return hook

                handles.append(module.register_forward_hook(make_post_hook(sites, state)))
        return handles

    def load_initial_head_actuator(self, path: Path, *, scale: float = 1.0) -> int:
        if self.reft_mode != "gated_vector":
            return 0
        payload = self.torch.load(path, map_location="cpu")
        vectors = payload.get("vectors") or {}
        loaded = 0
        with self.torch.no_grad():
            for site in self.sites:
                if site.site_kind != "head":
                    continue
                source = vectors.get(site.component_id)
                if source is None:
                    source = vectors.get(site.site_id)
                if source is None:
                    continue
                target = self.params[f"{self._prefix(site)}__vector"]
                source_tensor = self.torch.as_tensor(source, dtype=target.dtype, device=self.device)
                if tuple(source_tensor.shape) != tuple(target.shape):
                    raise ValueError(f"Warm-start shape mismatch for {site.site_id}: expected {tuple(target.shape)} got {tuple(source_tensor.shape)}")
                target.copy_(float(scale) * source_tensor)
                loaded += 1
        return loaded

    def load_initial_fixed_actuator(self, path: Path, *, scale: float = 1.0) -> int:
        if self.reft_mode != "gated_vector":
            return 0
        payload = self.torch.load(path, map_location="cpu")
        vectors = payload.get("vectors") or {}
        loaded = 0
        with self.torch.no_grad():
            for site in self.sites:
                if site.site_kind != "component":
                    continue
                source = vectors.get(site.component_id)
                if source is None:
                    source = vectors.get(site.site_id)
                if source is None:
                    continue
                target = self.params[f"{self._prefix(site)}__vector"]
                source_tensor = self.torch.as_tensor(source, dtype=target.dtype, device=self.device)
                if tuple(source_tensor.shape) != tuple(target.shape):
                    raise ValueError(f"Warm-start shape mismatch for {site.site_id}: expected {tuple(target.shape)} got {tuple(source_tensor.shape)}")
                target.copy_(float(scale) * source_tensor)
                loaded += 1
        return loaded

    def norm_penalty(self):
        total = None
        for parameter in self.params.values():
            value = parameter.pow(2).mean()
            total = value if total is None else total + value
        if total is None:
            return self.torch.tensor(0.0, device=self.device)
        return total

    def summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        with self.torch.no_grad():
            for site in self.sites:
                prefix = self._prefix(site)
                row: dict[str, object] = {
                    "site_id": site.site_id,
                    "site_kind": site.site_kind,
                    "layer_idx": site.layer_idx,
                    "component_type": site.component_type,
                    "component_id": site.component_id,
                    "head_idx": site.head_idx if site.site_kind == "head" else "",
                    "hook_site": site.hook_site,
                    "dim": self.site_dims[site.site_id],
                    "reft_mode": self.reft_mode,
                }
                if self.reft_mode == "gated_vector":
                    vector = self.params[f"{prefix}__vector"].float()
                    gate_weight = self.params[f"{prefix}__gate_weight"].float()
                    gate_bias = self.params[f"{prefix}__gate_bias"].float()
                    row.update(
                        {
                            "vector_l2": float(vector.norm().detach().cpu().item()),
                            "vector_mean_abs": float(vector.abs().mean().detach().cpu().item()),
                            "vector_max_abs": float(vector.abs().max().detach().cpu().item()),
                            "gate_weight_l2": float(gate_weight.norm().detach().cpu().item()),
                            "gate_bias": float(gate_bias.detach().cpu().view(-1)[0].item()),
                            "gate_prior": float(self.torch.sigmoid(gate_bias).detach().cpu().view(-1)[0].item()),
                        }
                    )
                else:
                    down = self.params[f"{prefix}__down"].float()
                    up = self.params[f"{prefix}__up"].float()
                    row.update(
                        {
                            "down_l2": float(down.norm().detach().cpu().item()),
                            "up_l2": float(up.norm().detach().cpu().item()),
                            "down_mean_abs": float(down.abs().mean().detach().cpu().item()),
                            "up_mean_abs": float(up.abs().mean().detach().cpu().item()),
                        }
                    )
                rows.append(row)
        return rows

    def site_rows(self) -> list[dict[str, object]]:
        return [
            {
                "site_id": site.site_id,
                "site_kind": site.site_kind,
                "layer_idx": site.layer_idx,
                "component_type": site.component_type,
                "component_id": site.component_id,
                "head_idx": site.head_idx,
                "hook_site": site.hook_site,
                "dim": self.site_dims.get(site.site_id, 0),
            }
            for site in self.sites
        ]

    def save_payload(self, path: Path, *, metadata: dict[str, Any] | None = None) -> None:
        payload = {
            "kind": "cast_reft_actuator",
            "reft_mode": self.reft_mode,
            "rank": self.rank,
            "gate_init": self.gate_init,
            "init_std": self.init_std,
            "sites": self.site_rows(),
            "parameters": {name: parameter.detach().cpu() for name, parameter in self.params.items()},
            "metadata": metadata or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(payload, path)

    @classmethod
    def load_payload(cls, *, backend: Any, path: Path):
        torch_module = backend._torch
        payload = torch_module.load(path, map_location="cpu")
        if payload.get("kind") != "cast_reft_actuator":
            raise ValueError(f"Not a CAST ReFT payload: {path}")
        sites = [
            CastReftSite(
                site_id=str(row["site_id"]),
                site_kind=str(row["site_kind"]),
                layer_idx=int(row["layer_idx"]),
                component_type=str(row.get("component_type", "")),
                component_id=str(row.get("component_id", row.get("site_id", ""))),
                head_idx=int(row.get("head_idx", -1) if str(row.get("head_idx", "")) != "" else -1),
                hook_site=str(row.get("hook_site", "")),
            )
            for row in payload.get("sites", [])
        ]
        controller = cls(
            backend=backend,
            sites=sites,
            reft_mode=str(payload.get("reft_mode", "gated_vector")),
            rank=int(payload.get("rank", 4)),
            gate_init=float(payload.get("gate_init", 0.98)),
            init_std=float(payload.get("init_std", 0.01)),
        )
        params = payload.get("parameters") or {}
        with torch_module.no_grad():
            for name, source in params.items():
                if name not in controller.params:
                    continue
                target = controller.params[name]
                source_tensor = torch_module.as_tensor(source, dtype=target.dtype, device=controller.device)
                if tuple(source_tensor.shape) != tuple(target.shape):
                    raise ValueError(f"Payload parameter shape mismatch for {name}: expected {tuple(target.shape)} got {tuple(source_tensor.shape)}")
                target.copy_(source_tensor)
        return controller, payload
