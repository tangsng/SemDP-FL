"""Utility-optimal per-group noise allocation with semantic floors (SemDP-FL core).

Solves   min sum_l a_l sigma_l^2   s.t.   sum_l c / sigma_l^2 <= R,  sigma_l >= floor_l
via the closed form of Theorem 1 plus iterative clamping (water-filling):

  unconstrained optimum:  sigma_l^2  =  sqrt(c / a_l) * (sum_j sqrt(a_j)) / R
  floor violators are clamped and the remaining budget is re-distributed.

Returns relative weights w_l = sigma_l / sigma_u (sigma_u = uniform multiplier
sqrt(c*G/R)), i.e. allocation is scale-free; the absolute scale is later set by
the exact RDP accountant (calibrate_scale), so the reported epsilon is always
the exact one.
"""
from __future__ import annotations

import math


def uniform_sigma(c: float, g: int, R: float) -> float:
    return math.sqrt(c * g / R)


def optimal_weights(a: dict[str, float], floors: dict[str, float],
                    c: float = 1.0, R: float = 1.0) -> dict[str, float]:
    """Relative noise multipliers w_l = sigma_l / sigma_u under floors.

    a: distortion weights a_l = C_l^2 * d_l / m^2 (>0).
    floors: relative floors w_min_l = floor_l (already normalised by sigma_u).
    With c = R = 1 the uniform multiplier is sqrt(G); weights are rescaled at
    the end so that sum_l 1/(w_l^2) = G  (same total budget as uniform).
    """
    groups = list(a.keys())
    G = len(groups)
    su = math.sqrt(G)  # uniform multiplier with c=R=1
    free = set(groups)
    w: dict[str, float] = {}
    budget_left = 1.0
    while True:
        sum_sqrt = sum(math.sqrt(a[g]) for g in free)
        if sum_sqrt <= 0:
            for g in free:
                w[g] = su
            break
        viol = []
        for g in free:
            # optimal sigma_g^2 on the remaining budget: sqrt(a_g) share of budget
            # x_g = 1/sigma_g^2 = budget_left * sqrt(a_g)/sum_sqrt  ->  sigma_g = 1/sqrt(x_g)
            x_g = budget_left * math.sqrt(a[g]) / sum_sqrt
            sig = 1.0 / math.sqrt(x_g)
            if sig < floors[g] * su:
                viol.append(g)
            else:
                w[g] = sig
        if not viol:
            break
        for g in viol:
            w[g] = floors[g] * su
            budget_left -= 1.0 / (w[g] ** 2)
            free.discard(g)
        if budget_left <= 0 or not free:
            for g in free:
                w[g] = su
            break
    # Rescale into sigma_u units (uniform multiplier = 1, floors read directly as
    # floor_hi/floor_mid/floor_lo): require  sum_l 1/w_l^2 = G  (uniform w=1 gives G).
    # Loop output has sum 1/w^2 = tot, so w' = w*s gives tot/s^2 = G  =>  s = sqrt(tot/G).
    # (Scale-free w.r.t. training: calibrate_scale absorbs any global factor; this
    # normalisation only fixes the reported rel_weights / audit numbers.)
    tot = sum(1.0 / (w[g] ** 2) for g in groups)
    scale = math.sqrt(tot / G)
    return {g: w[g] * scale for g in groups}


def expected_distortion(a: dict[str, float], w: dict[str, float]) -> float:
    """Relative expected distortion sum_l a_l w_l^2 (per-round, up to constants)."""
    return sum(a[g] * w[g] ** 2 for g in a)
