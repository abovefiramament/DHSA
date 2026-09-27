"""Differentiable completion scores shared by RCM measurement and CAST training."""

import torch
from torch import Tensor


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
