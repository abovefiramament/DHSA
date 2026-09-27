"""Policy log-probability and KL summaries shared by native actuators."""

from __future__ import annotations

import torch


def sequence_policy_metrics(
    controlled_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    continuation: torch.Tensor,
) -> dict[str, float]:
    if controlled_logits.shape != reference_logits.shape:
        raise ValueError("policy/reference logit shapes differ")
    if continuation.shape != controlled_logits.shape[:2] or continuation.numel() == 0:
        raise ValueError("policy metrics require non-empty aligned continuation tokens")
    controlled_logp = torch.log_softmax(controlled_logits.float(), dim=-1)
    reference_logp = torch.log_softmax(reference_logits.float().to(controlled_logits.device), dim=-1)
    targets = continuation.to(controlled_logits.device).unsqueeze(-1)
    ratio = (
        controlled_logp.gather(-1, targets).squeeze(-1)
        - reference_logp.gather(-1, targets).squeeze(-1)
    )
    kl = (controlled_logp.exp() * (controlled_logp - reference_logp)).sum(-1)
    return {
        "sampled_sequence_logprob_ratio": float(ratio.sum().item()),
        "token_kl_audit": float(kl.mean().item()),
        "generated_token_count": float(continuation.shape[1]),
    }
