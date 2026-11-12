"""The exact soft-target cross-entropy plus expected-regret loss."""

import torch


def routing_loss(logits, target):
    logits = logits.float()
    y = torch.tensor(target["target_probs"], device=logits.device, dtype=torch.float32)
    u = torch.tensor(target["rewards"], device=logits.device, dtype=torch.float32)
    logp = logits.log_softmax(-1)
    p = logp.exp()
    return -(y * logp).sum() + 0.1 * (u.max() - (p * u).sum())
