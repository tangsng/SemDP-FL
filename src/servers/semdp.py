"""SemDP-FL (ours): semantic-tier floors + utility-optimal per-group noise
allocation + per-group clipping, accounted as G composed sampled-Gaussian
mechanisms (conservative, Eq. (4) of the method doc)."""
from __future__ import annotations

import os

from ..utils.allocation import expected_distortion, optimal_weights
from ..utils.common import write_json
from ..utils.sensitivity import assign_tiers, combined_scores
from .base import BaseDPServer


class SemDPServer(BaseDPServer):
    METHOD = "semdp"

    def per_group(self) -> bool:
        return True

    def compute_weights(self, grad_norms: dict[str, float]) -> dict[str, float]:
        mc = self.cfg["method"]
        ab = mc.get("ablation", {}) or {}
        scores = combined_scores(self.bundle, grad_norms, self.cfg["semantic_prior"],
                                 prior_only=ab.get("prior_only", False),
                                 measured_only=ab.get("measured_only", False))
        tiers = assign_tiers(scores, tuple(mc.get("tier_fracs", [0.2, 0.3, 0.5])))
        if ab.get("no_floor", False):
            floor_of = {"high": 0.0, "mid": 0.0, "low": 0.0}
        else:
            floor_of = {"high": mc.get("floor_hi", 1.6), "mid": mc.get("floor_mid", 1.0),
                        "low": mc.get("floor_lo", 0.5)}
        floors = {g: floor_of[tiers[g]] for g in self.bundle.groups}
        dims = self.bundle.group_dims()
        a = {g: (self.group_clips[g] ** 2) * dims[g] / (self.m ** 2) for g in self.bundle.groups}
        w = optimal_weights(a, floors)
        # audit artifacts for the paper
        uni = expected_distortion(a, {g: 1.0 for g in self.bundle.groups})
        opt = expected_distortion(a, w)
        write_json(os.path.join(self.out_dir, "allocation_audit.json"), {
            "scores": scores, "tiers": tiers, "floors": floors,
            "a": a, "weights": w,
            "rho_theoretical": uni / max(opt, 1e-12),
            "note": "rho = uniform distortion / allocated distortion at equal budget",
        })
        return w
