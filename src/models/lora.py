"""Manual LoRA (Hu et al., arXiv:2106.09685) -- no external PEFT dependency.

LoRALinear wraps a frozen nn.Linear:  y = W0 x + (alpha/r) * B A x.
The base weight stays frozen in fp16 (set by the caller); A/B are trained in
fp32, which is the standard mixed-precision PEFT recipe and keeps the small
trainable updates numerically exact for DP clipping/noise bookkeeping.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        lora = (x.float() @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return y + lora.to(y.dtype)


def inject_lora(model: nn.Module, targets: tuple[str, ...], r: int, alpha: float,
                freeze_a: bool = False) -> list[str]:
    """Replace every child Linear whose name ends with a target suffix.

    freeze_a=True implements FFA-LoRA (Sun et al., ICLR 2024): A is frozen at
    its random init and only B is trained.
    Returns the list of replaced module names.
    """
    replaced = []
    for name, module in list(model.named_modules()):
        if not any(name.endswith(sfx) for sfx in targets):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        wrapped = LoRALinear(module, r=r, alpha=alpha)
        if freeze_a:
            wrapped.lora_A.requires_grad_(False)
        setattr(parent, parts[-1], wrapped)
        replaced.append(name)
    return replaced
