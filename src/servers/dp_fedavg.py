"""DP-FedAvg (McMahan et al., ICLR 2018): global clip + uniform Gaussian noise,
single sampled-Gaussian mechanism on the whole LoRA update vector."""
from __future__ import annotations

from .base import BaseDPServer


class DPFedAvgServer(BaseDPServer):
    METHOD = "dp_fedavg"
