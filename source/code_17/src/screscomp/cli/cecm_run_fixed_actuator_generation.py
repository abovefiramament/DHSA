from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, desc=None):
        return iterable

from screscomp.cecm.actuator import load_fixed_actuator_additions
from screscomp.cecm.scaling import classify_source_generation, load_open_eval_rows, summarize_generation_rows
from screscomp.data import dump_csv, dump_json, dump_jsonl
from screscomp.modeling import TransformersABBackend


@dataclass(frozen=True, slots=True)
class ActuatorSpec:
    name: str
    path: Path


@dataclass(frozen=True, slots=True)
class FixedControl:
    name: str
    actuator_name: str
    actuator_path: Path
    sign: float
    target_direction: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cross-sample fixed-actuator generation smoke.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--eval-open-rows", type=Path, required=True)
    p.add_argument(
        "--actuators",
        type=str,
        required=True,
        help="Semicolon-separated actuator specs, e.g. ctx=path/fixed_actuator.pt;mlp=path",
    )
    p.add_argument("--generation-prompt-key", type=str, default="base_rag")
    p.add_argument("--prior-source", choices=["auto", "model_prior", "dataset_orig"], default="dataset_orig")
    p.add_argument("--controls", type=str, default="force_target,force_start")
    p.add_argument("--alpha-sweep", type=str, default="0,0.5,1.0,1.5")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=30)
    p.add_argument("--val-mod", type=int, default=5)
    p.add_argument("--generation-apply-mode", type=str, default="all")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--stop-strings", type=str, default="Q:")
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--verbose-char-threshold", type=int, default=48)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_alphas(raw: str) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for item in _parse_csv(raw):
        value = float(item)
        if value not in seen:
            seen.add(value)
            out.append(value)
    if not out:
        raise ValueError("alpha sweep is empty")
    return out


def _parse_stop_strings(raw: str) -> list[str]:
    return [bytes(item.strip(), "utf-8").decode("unicode_escape") for item in raw.split(",") if item.strip()]


def _resolve_actuator_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "fixed_actuator.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed actuator payload: {path}")
    return path


def _parse_actuators(raw: str) -> list[ActuatorSpec]:
    specs: list[ActuatorSpec] = []
    for idx, part in enumerate(raw.split(";"), start=1):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, value = part.split("=", 1)
            name = name.strip()
            path = Path(value.strip())
        else:
            path = Path(part)
            name = path.parent.name if path.name == "fixed_actuator.pt" else path.name
            if not name:
                name = f"actuator{idx}"
        specs.append(ActuatorSpec(name=name, path=_resolve_actuator_path(path)))
    if not specs:
        raise ValueError("No actuators provided")
    return specs


def _build_controls(actuators: list[ActuatorSpec], controls: list[str]) -> list[FixedControl]:
    out: list[FixedControl] = []
    for actuator in actuators:
        for control in controls:
            if control == "force_target":
                out.append(
                    FixedControl(
                        name=f"{actuator.name}_force_target",
                        actuator_name=actuator.name,
                        actuator_path=actuator.path,
                        sign=1.0,
                        target_direction="target",
                    )
                )
            elif control == "force_start":
                out.append(
                    FixedControl(
                        name=f"{actuator.name}_force_start",
                        actuator_name=actuator.name,
                        actuator_path=actuator.path,
                        sign=-1.0,
                        target_direction="start",
                    )
                )
            else:
                raise ValueError(f"Unsupported control: {control}")
    return out


def _components_from_payload(path: Path) -> list[dict[str, object]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    return [dict(row) for row in payload.get("components", [])]


def _component_ids(components: list[dict[str, object]]) -> str:
    return ",".join(str(row.get("component_id", "")) for row in components if row.get("component_id"))


def _additions_for_control(control: FixedControl, *, alpha: float, apply_mode: str) -> list[dict[str, Any]]:
    if alpha == 0:
        return []
    return load_fixed_actuator_additions(
        control.actuator_path,
        alpha=float(control.sign) * float(alpha),
        apply_mode=apply_mode,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alphas = _parse_alphas(args.alpha_sweep)
    stop_strings = _parse_stop_strings(args.stop_strings)
    actuators = _parse_actuators(args.actuators)
    controls = _build_controls(actuators, _parse_csv(args.controls))

    print(f"[cecm-fixed-actuator-gen] loading model={args.model}", flush=True)
    backend = TransformersABBackend(
        model_name_or_path=args.model,
        device=args.device,
        use_chat_template=args.use_chat_template,
        torch_dtype=args.torch_dtype,
    )
    rows = load_open_eval_rows(
        args.eval_open_rows,
        prompt_key=args.generation_prompt_key,
        prior_source=args.prior_source,
        split=None if args.split == "all" else args.split,
        start=args.start,
        max_rows=args.max_rows,
        val_mod=args.val_mod,
    )
    if not rows:
        raise SystemExit("No eval rows selected.")

    components_by_actuator = {
        actuator.name: _components_from_payload(actuator.path)
        for actuator in actuators
    }
    dump_json(
        args.out_dir / "run_config.json",
        {
            "model": args.model,
            "eval_open_rows": str(args.eval_open_rows),
            "mode": "fixed_actuator_generation",
            "generation_prompt_key": args.generation_prompt_key,
            "prior_source": args.prior_source,
            "eval_rows": len(rows),
            "alpha_sweep": alphas,
            "generation_apply_mode": args.generation_apply_mode,
            "actuators": {actuator.name: str(actuator.path) for actuator in actuators},
            "semantics": (
                "Generation uses only a trained cross-sample fixed actuator. "
                "It does not run a target/high prompt or compute same-sample deltas on eval rows."
            ),
        },
    )
    dump_csv(
        args.out_dir / "control_plan.csv",
        [
            {
                "control_name": control.name,
                "baseline_kind": "fixed_actuator",
                "actuator_name": control.actuator_name,
                "actuator_path": str(control.actuator_path),
                "sign": control.sign,
                "target_direction": control.target_direction,
                "components": _component_ids(components_by_actuator[control.actuator_name]),
            }
            for control in controls
        ],
    )

    generation_rows: list[dict[str, object]] = []
    for control in controls:
        component_ids = _component_ids(components_by_actuator[control.actuator_name])
        for alpha in alphas:
            additions = _additions_for_control(control, alpha=alpha, apply_mode=args.generation_apply_mode)
            for row in tqdm(rows, desc=f"generate {control.name} a={alpha:g}"):
                if additions:
                    prediction = backend.generate_with_component_last_token_add_many(
                        row.prompt,
                        additions,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        apply_mode=args.generation_apply_mode,
                    )
                else:
                    prediction = backend.generate(
                        row.prompt,
                        max_new_tokens=args.max_new_tokens,
                        stop_strings=stop_strings,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                metrics = classify_source_generation(
                    prediction,
                    context_answers=row.context_answers,
                    prior_answers=row.prior_answers,
                    verbose_char_threshold=args.verbose_char_threshold,
                )
                generation_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "control_name": control.name,
                        "baseline_kind": "fixed_actuator",
                        "env_kind": "context",
                        "target_direction": control.target_direction,
                        "component_group": control.actuator_name,
                        "alpha": alpha,
                        "prediction": prediction,
                        "generation_prompt_key": args.generation_prompt_key,
                        "prior_source": args.prior_source,
                        "context_answers_json": list(row.context_answers),
                        "prior_answers_json": list(row.prior_answers),
                        "components": component_ids,
                        **metrics,
                    }
                )

    dump_jsonl(args.out_dir / "generation_rows.jsonl", generation_rows)
    dump_csv(args.out_dir / "generation_summary.csv", summarize_generation_rows(generation_rows))
    dump_csv(
        args.out_dir / "run_summary.csv",
        [
            {"metric": "eval_rows", "value": len(rows)},
            {"metric": "control_specs", "value": len(controls)},
            {"metric": "generation_rows", "value": len(generation_rows)},
        ],
    )
    print(f"[cecm-fixed-actuator-gen] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
