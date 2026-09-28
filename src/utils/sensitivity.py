"""Semantic sensitivity assessment (SemDP-FL component 1).

s_l = sqrt(p_l * m_l)   (geometric fusion, Eq. (3) of the method doc)
  p_l : semantic prior produced by an LLM analysing layer names/functions
        (configs/semantic_prior_qwen25_05b.yaml, with per-layer rationale);
  m_l : measured sensitivity = per-group gradient norm on the PUBLIC proxy set,
        min-max normalised. Public data costs no privacy budget.
"""
from __future__ import annotations

import torch
import yaml


def measure_group_grad_norms(bundle, public_loader, device, max_batches: int = 8) -> dict[str, float]:
    model = bundle.model
    model.train()
    sq = {g: 0.0 for g in bundle.groups}
    seen = 0
    for i, batch in enumerate(public_loader):
        if i >= max_batches:
            break
        ids, mask, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        loss = torch.nn.functional.cross_entropy(model(ids, mask), y)
        loss.backward()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    g = bundle.group_of[n]
                    sq[g] += float(p.grad.float().pow(2).sum())
        model.zero_grad(set_to_none=True)
        seen += 1
    return {g: (sq[g] / max(seen, 1)) ** 0.5 for g in bundle.groups}


def combined_scores(bundle, grad_norms: dict[str, float], prior_path: str,
                    prior_only: bool = False, measured_only: bool = False) -> dict[str, float]:
    """s_l = sqrt(p_l * m_l). Ablation switches (default off, behaviour unchanged):
    prior_only -> m_l := 1; measured_only -> p_l := 1."""
    with open(prior_path) as f:
        prior = yaml.safe_load(f)["prior"]
    gmax = max(grad_norms.values()) or 1.0
    scores = {}
    for g in bundle.groups:
        p = 1.0 if measured_only else float(prior.get(g, 0.5))
        m = 1.0 if prior_only else grad_norms.get(g, 0.0) / gmax
        scores[g] = (p * max(m, 1e-3)) ** 0.5
    return scores


def assign_tiers(scores: dict[str, float], fracs: tuple[float, float, float]) -> dict[str, str]:
    """Rank groups by score: top fracs[0] -> high, next fracs[1] -> mid, rest -> low."""
    ordered = sorted(scores, key=scores.get, reverse=True)
    n = len(ordered)
    n_hi = max(1, round(n * fracs[0]))
    n_mid = max(1, round(n * fracs[1]))
    tiers = {}
    for rank, g in enumerate(ordered):
        tiers[g] = "high" if rank < n_hi else ("mid" if rank < n_hi + n_mid else "low")
    return tiers
