"""FFA-LoRA under DP (Sun et al., ICLR 2024, arXiv:2403.12313): the A matrices
are frozen at random init (see models/lora.inject_lora(freeze_a=True)), so only
B (+ head) updates are clipped/noised -- this removes the semi-quadratic noise
amplification of jointly-trained LoRA. Global clip + uniform noise on the
remaining trainables."""
from __future__ import annotations

from .dp_fedavg import DPFedAvgServer


class DPFFALoRAServer(DPFedAvgServer):
    METHOD = "dp_ffalora"
