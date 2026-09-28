"""DP-FedSAM (Shi et al., CVPR 2023, arXiv:2303.11242): SAM client optimizer
under the same client-level DP pipeline (global clip + uniform noise)."""
from __future__ import annotations

from .dp_fedavg import DPFedAvgServer


class DPFedSAMServer(DPFedAvgServer):
    METHOD = "dp_fedsam"
    USE_SAM = True
