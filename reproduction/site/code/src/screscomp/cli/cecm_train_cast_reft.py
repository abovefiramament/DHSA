
from __future__ import annotations

import argparse
import gc
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from screscomp.cecm.actuator import (
    DPO_LOSS,
    ENDPOINT_OBJECTIVES,
    PREFERENCE_LOSS_MODES,
    actuator_loss_description,
    actuator_objective_description,
    actuator_score_description,
    competitive_margin_description,
    dpo_preference_logit,
    filter_pairs_by_source_row_index,
    is_constant_zero_y_plus,
    is_dynamic_y_minus,
    limit_rows,
    load_actuator_pairs,
    pair_question_state_weight,
    parse_alpha_list,
    parse_weight_spec,
)
from screscomp.cecm.cast_reft import CastReftController, load_component_sites_csv, load_head_sites_file, parse_head_ids
from screscomp.cecm.objective import MODEL_MAX_OPTION_SELECTION, OPTION_SELECTION_MODES, option_selection_description, score_text_token_slice
from screscomp.data import dump_csv, dump_json
from screscomp.modeling import TransformersABBackend

DEFAULT_ALPHA_SWEEP = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CAST-constrained ReFT controllers on selected sites.")
    p.add_argument("--model", required=True)
    p.add_argument("--pairs-csv", type=Path, required=True)
    p.add_argument("--event", default="tldr_summary_preference")
    p.add_argument("--endpoint-objective", default="pair_margin", choices=ENDPOINT_OBJECTIVES)
    p.add_argument("--heads", default="")
    p.add_argument("--heads-file", type=Path)
    p.add_argument("--components-csv", type=Path)
    p.add_argument("--component-hook-site", default="post_module_output", choices=["post_module_output", "pre_module_input"])
    p.add_argument("--reft-mode", default="gated_vector", choices=["gated_vector", "low_rank"])
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--gate-init", type=float, default=0.98)
    p.add_argument("--init-std", type=float, default=0.01)
    p.add_argument("--init-head-actuator", type=Path)
    p.add_argument("--init-fixed-actuator", type=Path)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--max-train-rows", type=int, default=8192)
    p.add_argument("--max-val-rows", type=int, default=512)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--min-source-row-index", type=int)
    p.add_argument("--max-source-row-index", type=int)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lambda-norm", type=float, default=1e-4)
    p.add_argument("--alpha-train", type=float, default=1.0)
    p.add_argument("--preference-loss-mode", default="dpo", choices=PREFERENCE_LOSS_MODES)
    p.add_argument("--dpo-beta", type=float, default=0.5)
    p.add_argument("--state-margin-weight", type=float, default=0.0)
    p.add_argument("--gain-weight", type=float, default=1.0)
    p.add_argument("--target-margin", type=float, default=0.0)
    p.add_argument("--target-gain", type=float, default=0.0)
    p.add_argument("--apply-mode", default="all", choices=["prefill", "prompt", "prompt_last", "decision_tokens", "decode", "all"])
    p.add_argument("--score-mode", default="avglogp", choices=["avglogp", "top_logit_gap", "answer_rest_margin"])
    p.add_argument("--option-selection-mode", default=MODEL_MAX_OPTION_SELECTION, choices=OPTION_SELECTION_MODES)
    p.add_argument("--max-aliases-per-side", type=int, default=1)
    p.add_argument("--question-state-weights", default="")
    p.add_argument("--alpha-sweep", default=DEFAULT_ALPHA_SWEEP)
    p.add_argument("--skip-alpha-summary", action="store_true")
    p.add_argument("--empty-cache-every", type=int, default=25)
    p.add_argument("--device", default="auto")
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def _resolve_payload(path: Path | None, filename: str) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        path = path / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _collect_sites(args: argparse.Namespace):
    sites = []
    sites.extend(parse_head_ids(args.heads))
    if args.heads_file is not None:
        sites.extend(load_head_sites_file(args.heads_file))
    if args.components_csv is not None:
        sites.extend(load_component_sites_csv(args.components_csv, hook_site=args.component_hook_site))
    out = []
    seen = set()
    for site in sites:
        if site.site_id in seen:
            continue
        seen.add(site.site_id)
        out.append(site)
    return out


class Trainer:
    def __init__(self, backend: TransformersABBackend, controller: CastReftController, args: argparse.Namespace):
        self.backend = backend
        self.controller = controller
        self.args = args
        self.torch = backend._torch
        self.tokenizer = backend._tokenizer
        self.model = backend._model
        self.device = backend.device
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        if args.preference_loss_mode == DPO_LOSS and args.score_mode != "avglogp":
            raise ValueError("DPO ReFT requires score_mode=avglogp")

    def _tok(self, prompt: str, continuation: str):
        prompt_ids = self.tokenizer(self.backend._format_prompt(prompt), return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        cont_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        return prompt_ids, cont_ids

    def _pad_id(self) -> int:
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id")
        return int(pad)

    def _position_slice(self, prompt_len: int, cont_len: int, seq_len: int) -> slice:
        mode = self.args.apply_mode
        if mode in {"prefill", "prompt"}:
            return slice(0, min(prompt_len, seq_len))
        if mode == "prompt_last":
            start = max(min(prompt_len, seq_len) - 1, 0)
            return slice(start, start + 1)
        if mode == "decision_tokens":
            start = max(prompt_len - 1, 0)
            stop = min(prompt_len + cont_len - 1, seq_len)
            return slice(start, max(stop, start + 1))
        if mode == "decode":
            start = min(prompt_len, seq_len)
            stop = min(prompt_len + cont_len, seq_len)
            return slice(start, max(stop, start))
        if mode == "all":
            return slice(0, seq_len)
        raise ValueError(f"Unsupported apply_mode={mode}")

    def _encode_batch(self, examples: list[tuple[str, str, str]]):
        prompt_lens = []
        cont_lens = []
        seq_lens = []
        score_slices = []
        prompt_rows = []
        cont_rows = []
        for prompt, continuation, score_text in examples:
            pids, cids = self._tok(prompt, continuation)
            prompt_lens.append(int(pids.shape[-1]))
            cont_lens.append(int(cids.shape[-1]))
            seq_lens.append(int(pids.shape[-1] + cids.shape[-1]))
            score_slices.append(score_text_token_slice(self.tokenizer, continuation, score_text))
            prompt_rows.append(pids.squeeze(0))
            cont_rows.append(cids.squeeze(0))
        input_ids = self.torch.full((len(examples), max(seq_lens)), self._pad_id(), dtype=prompt_rows[0].dtype, device=self.device)
        attention_mask = self.torch.zeros_like(input_ids)
        for i, (pids, cids) in enumerate(zip(prompt_rows, cont_rows, strict=False)):
            full = self.torch.cat([pids, cids], dim=0)
            input_ids[i, : int(full.shape[0])] = full
            attention_mask[i, : int(full.shape[0])] = 1
        return input_ids, attention_mask, prompt_lens, cont_lens, seq_lens, score_slices

    def _mask(self, prompt_lens: list[int], cont_lens: list[int], seq_lens: list[int]):
        mask = self.torch.zeros((len(prompt_lens), max(seq_lens)), dtype=self.torch.bool, device=self.device)
        for i, (p, c, s) in enumerate(zip(prompt_lens, cont_lens, seq_lens, strict=False)):
            mask[i, self._position_slice(p, c, s)] = True
        return mask

    def candidate_scores(self, examples: list[tuple[str, str, str]], alpha: float | None):
        input_ids, attention_mask, prompt_lens, cont_lens, seq_lens, score_slices = self._encode_batch(examples)
        handles = []
        if alpha is not None and alpha != 0:
            handles = self.controller.register_batch_hooks(alpha=alpha, position_mask=self._mask(prompt_lens, cont_lens, seq_lens))
        try:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        finally:
            for handle in handles:
                handle.remove()
        scores = []
        for i, score_slice in enumerate(score_slices):
            p = prompt_lens[i]
            c = cont_lens[i]
            target_ids = input_ids[i, p : p + c]
            pred_logits = logits[i, p - 1 : p + c - 1, :].float()
            if score_slice is not None:
                target_ids = target_ids[score_slice]
                pred_logits = pred_logits[score_slice, :]
            token_logits = pred_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            if self.args.score_mode == "avglogp":
                lp = self.torch.nn.functional.log_softmax(pred_logits, dim=-1)
                scores.append(lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).mean())
            elif self.args.score_mode == "top_logit_gap":
                scores.append((token_logits - pred_logits.max(dim=-1).values).mean())
            else:
                top_values, top_indices = pred_logits.topk(k=2, dim=-1)
                rest = self.torch.where(top_indices[..., 0] == target_ids, top_values[..., 1], top_values[..., 0])
                scores.append((token_logits - rest).mean())
        return self.torch.stack(scores)

    def _check_pairs(self, pairs):
        bad = [p.sample_id for p in pairs if is_constant_zero_y_plus(p) or is_dynamic_y_minus(p) or len(p.y_plus_options) != 1 or len(p.y_minus_options) != 1]
        if bad:
            raise ValueError(f"CAST-ReFT fast path requires static single-option pairs; first bad sample_id={bad[0]}")

    def margins(self, pairs, alpha: float | None):
        self._check_pairs(pairs)
        plus = [(p.prompt, p.y_plus_options[0], p.y_plus_score_text) for p in pairs]
        minus = [(p.prompt, p.y_minus_options[0], p.y_minus_score_text) for p in pairs]
        return self.candidate_scores(plus, alpha=alpha) - self.candidate_scores(minus, alpha=alpha)

    def cache_baselines(self, pairs, batch_size: int):
        out = {}
        with self.torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                values = self.margins(batch, alpha=None).detach().cpu().tolist()
                for pair, value in zip(batch, values, strict=False):
                    out[pair.sample_id] = float(value)
                if start and start % max(batch_size * 16, 1) == 0:
                    self.cleanup()
        return out

    def cleanup(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def train(self, train_pairs, val_pairs):
        self._check_pairs(train_pairs)
        self._check_pairs(val_pairs)
        train_bs = max(1, int(self.args.train_batch_size))
        eval_bs = max(1, int(self.args.eval_batch_size or train_bs))
        print(f"[cast-reft] caching train baselines n={len(train_pairs)}", flush=True)
        base_train = self.cache_baselines(train_pairs, train_bs)
        opt = self.torch.optim.AdamW(
            self.controller.params.parameters(),
            lr=float(self.args.lr),
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            eps=float(self.args.adam_eps),
            weight_decay=float(self.args.weight_decay),
            amsgrad=bool(self.args.amsgrad),
        )
        rng = random.Random(self.args.seed)
        history = []
        for epoch in range(1, int(self.args.epochs) + 1):
            ordered = list(train_pairs)
            rng.shuffle(ordered)
            losses = []
            gains = []
            pref_logits = []
            margins = []
            for start in range(0, len(ordered), train_bs):
                batch = ordered[start : start + train_bs]
                opt.zero_grad(set_to_none=True)
                margin = self.margins(batch, alpha=float(self.args.alpha_train))
                base = self.torch.tensor([base_train[p.sample_id] for p in batch], device=self.device)
                gain = margin - base
                if self.args.preference_loss_mode == DPO_LOSS:
                    pref_logit = float(self.args.dpo_beta) * gain
                    task = self.torch.nn.functional.softplus(-pref_logit)
                else:
                    pref_logit = None
                    completion = self.torch.nn.functional.softplus(self.torch.tensor(float(self.args.target_margin), device=self.device) - margin)
                    gain_loss = self.torch.nn.functional.softplus(self.torch.tensor(float(self.args.target_gain), device=self.device) - gain)
                    task = float(self.args.state_margin_weight) * completion + float(self.args.gain_weight) * gain_loss
                weights = self.torch.tensor([pair_question_state_weight(p, self.args.question_state_weights) for p in batch], device=self.device)
                loss = (weights * task).mean() + float(self.args.lambda_norm) * self.controller.norm_penalty()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
                gains.extend(gain.detach().cpu().tolist())
                margins.extend(margin.detach().cpu().tolist())
                if pref_logit is not None:
                    pref_logits.extend(pref_logit.detach().cpu().tolist())
                step = start + len(batch)
                if step == len(batch) or step % 25 < len(batch) or step == len(ordered):
                    extra = f" dpo_logit={pref_logits[-1]:.6f}" if pref_logits else ""
                    print(f"[cast-reft] epoch={epoch}/{self.args.epochs} step={step}/{len(ordered)} loss={losses[-1]:.6f} gain={gains[-1]:.6f}{extra}", flush=True)
                if self.args.empty_cache_every > 0 and step % self.args.empty_cache_every < len(batch):
                    self.cleanup()
            history.append({
                "epoch": epoch,
                "train_pairs": len(ordered),
                "mean_loss": mean(losses) if losses else math.nan,
                "mean_margin": mean(margins) if margins else math.nan,
                "mean_margin_gain": mean(gains) if gains else math.nan,
                "mean_preference_logit": mean(pref_logits) if pref_logits else math.nan,
                "dpo_win_rate": mean(float(x > 0) for x in pref_logits) if pref_logits else math.nan,
                "gain_positive_rate": mean(float(x > 0) for x in gains) if gains else math.nan,
                "norm_penalty": float(self.controller.norm_penalty().detach().cpu().item()),
            })
            self.controller.save_payload(self.args.out_dir / "cast_reft.pt", metadata=self.metadata())
        print(f"[cast-reft] caching val baselines n={len(val_pairs)}", flush=True)
        base_val = self.cache_baselines(val_pairs, eval_bs)
        return history, {**base_train, **base_val}

    def eval_alpha(self, pairs, baselines, alpha: float, split: str):
        eval_bs = max(1, int(self.args.eval_batch_size or self.args.train_batch_size))
        vals = []
        bases = []
        gains = []
        logits = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), eval_bs):
                batch = pairs[start : start + eval_bs]
                margins = [baselines[p.sample_id] for p in batch] if alpha == 0 else self.margins(batch, alpha=alpha).detach().cpu().tolist()
                for pair, margin in zip(batch, margins, strict=False):
                    base = float(baselines[pair.sample_id])
                    bases.append(base)
                    vals.append(float(margin))
                    gains.append(float(margin) - base)
                    if self.args.preference_loss_mode == DPO_LOSS:
                        logits.append(float(dpo_preference_logit(float(margin), base, beta=self.args.dpo_beta)))
        return {
            "split": split,
            "alpha": alpha,
            "n": len(vals),
            "base_mean_margin": mean(bases) if bases else math.nan,
            "mean_margin": mean(vals) if vals else math.nan,
            "mean_margin_gain": mean(gains) if gains else math.nan,
            "mean_preference_logit": mean(logits) if logits else math.nan,
            "gain_positive_rate": mean(float(x > 0) for x in gains) if gains else math.nan,
            "dpo_win_rate": mean(float(x > 0) for x in logits) if logits else math.nan,
        }

    def metadata(self):
        return {
            "event": self.args.event,
            "endpoint_objective": self.args.endpoint_objective,
            "apply_mode": self.args.apply_mode,
            "score_mode": self.args.score_mode,
            "alpha_train": self.args.alpha_train,
            "preference_loss_mode": self.args.preference_loss_mode,
            "dpo_beta": self.args.dpo_beta,
            "component_hook_site": self.args.component_hook_site,
            "objective": actuator_objective_description(state_margin_weight=self.args.state_margin_weight, gain_weight=self.args.gain_weight, preference_loss_mode=self.args.preference_loss_mode),
            "loss": actuator_loss_description(state_margin_weight=self.args.state_margin_weight, gain_weight=self.args.gain_weight, norm_label="theta_reft", preference_loss_mode=self.args.preference_loss_mode, dpo_beta=self.args.dpo_beta),
            "score": actuator_score_description(self.args.score_mode),
            "causal_operator": "CAST-constrained ReFT on selected sites with frozen base model",
        }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.question_state_weights = parse_weight_spec(args.question_state_weights)
    pairs = load_actuator_pairs(args.pairs_csv, event=args.event, endpoint_objective=args.endpoint_objective)
    pairs = filter_pairs_by_source_row_index(pairs, min_row_index=args.min_source_row_index, max_row_index=args.max_source_row_index)
    train_pool = [p for p in pairs if p.split == args.train_split]
    if args.shuffle:
        random.Random(args.seed).shuffle(train_pool)
    train_pairs = limit_rows(train_pool, args.max_train_rows)
    val_pairs = limit_rows([p for p in pairs if p.split == args.val_split], args.max_val_rows)
    if not train_pairs:
        raise SystemExit("No train pairs selected")
    if not val_pairs:
        val_pairs = train_pairs
    sites = _collect_sites(args)
    if not sites:
        raise SystemExit("No ReFT sites selected")
    alphas = parse_alpha_list(args.alpha_sweep)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_csv(args.out_dir / "reft_plan.csv", [{"site_id": s.site_id, "site_kind": s.site_kind, "layer_idx": s.layer_idx, "component_type": s.component_type, "component_id": s.component_id, "head_idx": s.head_idx if s.site_kind == "head" else "", "hook_site": s.hook_site, "reft_mode": args.reft_mode, "rank": args.rank, "train_pairs": len(train_pairs), "val_pairs": len(val_pairs), "competitive_margin": competitive_margin_description(args.endpoint_objective)} for s in sites])
    dump_json(args.out_dir / "run_config.json", {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "question_state_weights"} | {"site_count": len(sites), "alpha_sweep": alphas, "option_selection": option_selection_description(args.option_selection_mode)})
    print(f"[cast-reft] loading model={args.model} sites={len(sites)} train={len(train_pairs)} val={len(val_pairs)}", flush=True)
    backend = TransformersABBackend(model_name_or_path=args.model, device=args.device, use_chat_template=args.use_chat_template, torch_dtype=args.torch_dtype)
    random.seed(args.seed)
    backend._torch.manual_seed(args.seed)
    if backend._torch.cuda.is_available():
        backend._torch.cuda.manual_seed_all(args.seed)
    controller = CastReftController(backend=backend, sites=sites, reft_mode=args.reft_mode, rank=args.rank, gate_init=args.gate_init, init_std=args.init_std)
    head_init = _resolve_payload(args.init_head_actuator, "head_actuator.pt")
    fixed_init = _resolve_payload(args.init_fixed_actuator, "fixed_actuator.pt")
    loaded_head = controller.load_initial_head_actuator(head_init, scale=args.init_scale) if head_init else 0
    loaded_fixed = controller.load_initial_fixed_actuator(fixed_init, scale=args.init_scale) if fixed_init else 0
    dump_json(args.out_dir / "warm_start.json", {"init_head_actuator": str(head_init) if head_init else "", "init_fixed_actuator": str(fixed_init) if fixed_init else "", "loaded_head_sites": loaded_head, "loaded_component_sites": loaded_fixed, "init_scale": args.init_scale})
    trainer = Trainer(backend, controller, args)
    history, baselines = trainer.train(train_pairs, val_pairs)
    dump_csv(args.out_dir / "train_history.csv", history)
    dump_csv(args.out_dir / "reft_site_summary.csv", controller.summary_rows())
    controller.save_payload(args.out_dir / "cast_reft.pt", metadata=trainer.metadata())
    if not args.skip_alpha_summary:
        rows = []
        for split, split_pairs in ((args.train_split, train_pairs), (args.val_split, val_pairs)):
            for alpha in alphas:
                rows.append(trainer.eval_alpha(split_pairs, baselines, alpha, split))
        dump_csv(args.out_dir / "alpha_summary.csv", rows)
    print(f"[cast-reft] done out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
