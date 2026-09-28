"""DP-FL with adaptive clipping (Andrew et al., NeurIPS 2021, arXiv:1905.03871).

Geometric quantile-tracking update of the clip norm:
  C <- C * exp(-eta_C * (b_t - target_q)),
where b_t is the (noised) fraction of sampled clients with ||delta|| <= C.
The noised count is accounted as an additional sampled Gaussian mechanism with
noise multiplier bit_sigma (sensitivity of the count is 1 before division by m).
"""
from __future__ import annotations

import numpy as np

from ..utils.rdp_accountant import Mechanism
from .dp_fedavg import DPFedAvgServer


class DPAdaClipServer(DPFedAvgServer):
    METHOD = "dp_adaclip"

    def extra_mechanisms(self) -> list[Mechanism]:
        return [Mechanism(self.q, self.cfg["dp"].get("bit_sigma", 1.0))]

    def post_aggregate(self, bit_sum: int) -> None:
        dp = self.cfg["dp"]
        noisy_frac = (bit_sum + float(np.random.normal(0.0, dp.get("bit_sigma", 1.0)))) / self.m
        eta = dp.get("adaclip_eta", 0.2)
        target = dp.get("adaclip_target_q", 0.5)
        self.global_clip *= float(np.exp(-eta * (noisy_frac - target)))
        self.global_clip = float(np.clip(self.global_clip, 1e-3, 1e3))
