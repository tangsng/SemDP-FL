"""RDP accounting for (per-group) subsampled Gaussian mechanisms.

Implements the exact integer-order RDP of the sampled Gaussian mechanism
(Mironov, Talwar & Zhang, "Renyi Differential Privacy of the Sampled Gaussian
Mechanism", arXiv:1908.10530) and the standard RDP -> (eps, delta) conversion
(Mironov, "Renyi Differential Privacy", CSF 2017, Prop. 3).

A "mechanism" is a dict: {"q": sampling prob in (0,1], "sigma": noise multiplier}.
Per-round RDP is the sum of the mechanisms' RDPs (adaptive composition, which
is a valid conservative bound when all mechanisms act on the same sampled set).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

DEFAULT_ORDERS = tuple(range(2, 65)) + (72, 80, 96, 112, 128, 160, 192, 256)


def _log_comb(n: int, k: int) -> float:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of order `alpha` (integer) for the sampled Gaussian mechanism."""
    if sigma <= 0:
        return math.inf
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * sigma * sigma)
    terms = []
    for i in range(alpha + 1):
        log_coef = _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log1p(-q)
        terms.append(log_coef + (i * i - i) / (2.0 * sigma * sigma))
    return float(logsumexp(terms) / (alpha - 1))


@dataclass
class Mechanism:
    q: float
    sigma: float


def round_rdp(mechanisms: list[Mechanism], alpha: int) -> float:
    return sum(rdp_sampled_gaussian(m.q, m.sigma, alpha) for m in mechanisms)


def get_epsilon(mechanisms: list[Mechanism], rounds: int, delta: float,
                orders=DEFAULT_ORDERS) -> tuple[float, int]:
    """Return (epsilon, best_order) after `rounds` of adaptive composition."""
    best_eps, best_order = math.inf, -1
    for a in orders:
        rdp = rounds * round_rdp(mechanisms, a)
        eps = rdp + math.log(1.0 / delta) / (a - 1)
        if eps < best_eps:
            best_eps, best_order = eps, a
    return best_eps, best_order


def calibrate_scale(weights: list[float], q: float, rounds: int, target_eps: float,
                    delta: float, extra: list[Mechanism] | None = None,
                    orders=DEFAULT_ORDERS, tol: float = 1e-3) -> float:
    """Find scale s.t. mechanisms {q, sigma=w_i*scale} (+ extras) reach target_eps.

    weights: per-mechanism relative multipliers (w_i >= 0; w_i = 0 means the
    mechanism consumes no budget, i.e. infinite noise -- excluded).
    """
    extra = extra or []
    active = [w for w in weights if w > 0]

    def eps_at(scale: float) -> float:
        mechs = [Mechanism(q, w * scale) for w in active] + list(extra)
        return get_epsilon(mechs, rounds, delta, orders)[0]

    lo, hi = 1e-3, 10.0
    while eps_at(hi) > target_eps and hi < 1e6:
        hi *= 2.0
    if hi >= 1e6:
        raise RuntimeError("calibration failed: target epsilon unreachable")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if eps_at(mid) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
