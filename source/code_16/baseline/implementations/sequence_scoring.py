"""Differentiable completion scores shared by RCM measurement and CAST training."""

import torch
from torch import Tensor
from typing import Any, Callable, Mapping


def completion_score(logits: Tensor, labels: Tensor, mode: str) -> Tensor:
    """Score aligned completion tokens only, averaging within each response."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.numel() or not labels.numel():
        raise ValueError("completion scoring requires nonempty aligned logits and labels")
    logits = logits.float()
    if mode == "avglogp":
        return logits.log_softmax(-1).gather(-1, labels[:, None]).mean()
    correct = logits.gather(-1, labels[:, None]).squeeze(-1)
    if mode == "top_logit_gap":
        return (correct - logits.amax(-1)).mean()
    if mode == "answer_rest_margin":
        other = logits.masked_fill(torch.nn.functional.one_hot(labels, logits.shape[-1]).bool(), float("-inf")).amax(-1)
        return (correct - other).mean()
    raise ValueError(f"unsupported completion score mode: {mode!r}")


def paired_completion_scores(
    *, tokenizer: Any, prompt_ids: Tensor, row: Mapping[str, Any],
    forward: Callable[[Tensor], Any],
) -> dict[str, Any]:
    """Score full chosen/rejected continuations with the caller's native operator.

    Prompt and continuation are tokenized separately, as in the RCM scorer.
    This is teacher-forced inference, not generated-text or human-reference judging.
    """
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or not prompt_ids.shape[1]:
        raise ValueError("paired scoring requires one nonempty tokenized prompt")
    result: dict[str, Any] = {
        "generated_text": "", "prediction_kind": "paired_completion_scores",
        "component_scores": [], "component_scores_status": "not_applicable_native_operator",
    }
    for role in ("chosen", "rejected"):
        text = row.get(role)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"paired scoring requires nonempty {role}")
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        if not tokens:
            raise ValueError(f"paired scoring requires nonempty {role} tokens")
        labels = torch.tensor(tokens, dtype=torch.long, device=prompt_ids.device)
        ids = torch.cat((prompt_ids, labels[None, :]), dim=1)
        with torch.inference_mode():
            output = forward(ids)
            logits = output.logits[0, prompt_ids.shape[1] - 1:-1]
            result[f"{role}_avglogp"] = float(completion_score(logits, labels, "avglogp").item())
        result[f"scored_{role}"] = text
        result[f"{role}_token_count"] = len(tokens)
    result["pair_avglogp_margin"] = result["chosen_avglogp"] - result["rejected_avglogp"]
    return result
