"""Base federated DP server: sampling, client training, aggregation, accounting,
evaluation and logging. Concrete methods subclass this and customise
clip/noise/allocation through the hooks below (unified interface)."""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from ..clients.client import Client
from ..utils.common import CsvLogger, Timer, set_seed, write_json
from ..utils.rdp_accountant import Mechanism, calibrate_scale, get_epsilon


class BaseDPServer:
    METHOD = "base"
    USE_SAM = False

    def __init__(self, cfg: dict, bundle, fed_data, device: torch.device, out_dir: str):
        self.cfg, self.bundle, self.data, self.device = cfg, bundle, fed_data, device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        set_seed(cfg["experiment"]["seed"])
        fc = cfg["fl"]
        self.N, self.m, self.T = fc["num_clients"], fc["clients_per_round"], fc["rounds"]
        self.q = self.m / self.N
        self.rng = np.random.default_rng(cfg["experiment"]["seed"] + 777)
        self.clients = [Client(k, self.data.client_loader(k), cfg, device)
                        for k in range(self.N)]
        self.sam_rho = cfg["method"].get("sam_rho", 0.05) if self.USE_SAM else 0.0
        self.global_clip: float = 1.0
        self.group_clips: dict[str, float] = {g: 1.0 for g in bundle.groups}
        self.rel_weights: dict[str, float] = {g: 1.0 for g in bundle.groups}
        self.sigma_scale: float = 0.0
        self.eps_per_round: list[Mechanism] = []
        self.logger = CsvLogger(os.path.join(out_dir, "round_log.csv"),
                                ["round", "train_loss", "test_acc", "eps", "wall_sec"])
        self.timer = Timer()

    # ---- hooks -------------------------------------------------------------
    def per_group(self) -> bool:
        return False

    def extra_mechanisms(self) -> list[Mechanism]:
        return []

    def compute_weights(self, grad_norms: dict[str, float]) -> dict[str, float]:
        return {g: 1.0 for g in self.bundle.groups}

    def sam(self) -> float:
        return self.sam_rho

    # ---- setup -------------------------------------------------------------
    def _probe(self):
        """One local training pass on the PUBLIC proxy set: per-group and total
        update norms for clip initialisation + per-group gradient norms at init
        for sensitivity measurement."""
        from ..utils.sensitivity import measure_group_grad_norms
        grad_norms = measure_group_grad_norms(
            self.bundle, self.data.public_loader(), self.device)
        probe = Client(-1, self.data.public_loader(shuffle=True), self.cfg, self.device)
        theta0 = self.bundle.trainable_state()
        delta, _, _ = probe.local_train(self.bundle, theta0, sam_rho=0.0)
        self.bundle.load_trainable(theta0)  # restore
        tot = math.sqrt(sum(float(d.pow(2).sum()) for d in delta.values()))
        per_g = {}
        for g in self.bundle.groups:
            per_g[g] = math.sqrt(sum(float(delta[n].pow(2).sum())
                                     for n in self.bundle.group_param_names(g))) or 1e-4
        cf = self.cfg["dp"].get("clip_factor", 1.0)
        self.global_clip = max(tot * cf, 1e-4)
        self.group_clips = {g: max(per_g[g] * cf, 1e-4) for g in self.bundle.groups}
        return grad_norms

    def setup(self):
        # de-cone mean (public, budget-free); ablation no_deconing skips it so the
        # head consumes raw (coned) embeddings -- default False keeps de-coning.
        if not self.cfg["train"].get("no_deconing", False):
            self.bundle.calibrate_head(self.data.public_loader())
        grad_norms = self._probe()
        self.rel_weights = self.compute_weights(grad_norms)
        dp = self.cfg["dp"]
        if dp.get("disabled", False):  # no-DP reference point (ablation only)
            self.sigma_scale = 0.0
            self.eps_per_round = []
            self.eps_final = 0.0
            return
        weights = list(self.rel_weights.values()) if self.per_group() else [1.0]
        self.sigma_scale = calibrate_scale(
            weights, self.q, self.T, dp["epsilon"], dp["delta"],
            extra=self.extra_mechanisms())
        mechs = ([Mechanism(self.q, w * self.sigma_scale) for w in weights]
                 + self.extra_mechanisms())
        self.eps_per_round = mechs
        self.eps_final, _ = get_epsilon(mechs, self.T, dp["delta"])

    # ---- aggregation -------------------------------------------------------
    def clip_delta(self, delta: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
        delta = {n: torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                 for n, d in delta.items()}  # defensive: never propagate NaN updates
        if self.per_group():
            out, bits = {}, []
            for g, names in ((g, self.bundle.group_param_names(g)) for g in self.bundle.groups):
                C = self.group_clips[g]
                nrm = math.sqrt(sum(float(delta[n].pow(2).sum()) for n in names))
                fac = min(1.0, C / max(nrm, 1e-12))
                for n in names:
                    out[n] = delta[n] * fac
                bits.append(int(nrm <= C))
            return out, int(np.mean(bits) > 0.5)
        nrm = math.sqrt(sum(float(d.pow(2).sum()) for d in delta.values()))
        fac = min(1.0, self.global_clip / max(nrm, 1e-12))
        return {n: d * fac for n, d in delta.items()}, int(nrm <= self.global_clip)

    def add_noise(self, avg: dict[str, torch.Tensor], t: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.cfg["experiment"]["seed"] * 100003 + t)
        out = {}
        for n, d in avg.items():
            g = self.bundle.group_of[n]
            if self.per_group():
                std = self.rel_weights[g] * self.sigma_scale * self.group_clips[g] / self.m
            else:
                std = self.sigma_scale * self.global_clip / self.m
            noise = torch.randn(d.shape, generator=gen) * std
            out[n] = d + noise
        return out

    def post_aggregate(self, bit_sum: int) -> None:
        pass

    # ---- main loop ---------------------------------------------------------
    def run(self) -> dict:
        self.setup()
        theta = self.bundle.trainable_state()
        server_lr = self.cfg["train"].get("server_lr", 1.0)
        server_mom = self.cfg["train"].get("server_momentum", 0.0)
        # server momentum (Andrew et al. 2021): post-processing of the noised
        # aggregate, so it does not change the DP accounting; amplifies the
        # temporally-aligned signal ~1/(1-b) while isotropic noise grows only
        # ~sqrt(1/(1-b^2)) — a major SNR win at fixed epsilon.
        velocity = {n: torch.zeros_like(theta[n]) for n in theta}
        for t in range(1, self.T + 1):
            sel = self.rng.choice(self.N, self.m, replace=False)
            deltas, losses, bit_sum = [], [], 0
            for cid in sel:
                delta, loss, _ = self.clients[cid].local_train(
                    self.bundle, theta, sam_rho=self.sam())
                clipped, bit = self.clip_delta(delta)
                deltas.append(clipped)
                losses.append(loss)
                bit_sum += bit
            avg = {n: torch.stack([d[n] for d in deltas]).mean(0) for n in theta}
            noised = self.add_noise(avg, t)
            self.post_aggregate(bit_sum)
            velocity = {n: server_mom * velocity[n] + noised[n] for n in theta}
            theta = {n: theta[n] + server_lr * velocity[n] for n in theta}
            self.bundle.load_trainable(theta)
            acc = self.evaluate()
            eps_now = 0.0 if not self.eps_per_round else \
                get_epsilon(self.eps_per_round, t, self.cfg["dp"]["delta"])[0]
            row = {"round": t, "train_loss": round(float(np.mean(losses)), 4),
                   "test_acc": round(acc, 4), "eps": round(eps_now, 3),
                   "wall_sec": int(self.timer.elapsed())}
            self.logger.log(row)
            print(f"[{self.METHOD}|r{t:02d}] loss={row['train_loss']:.4f} "
                  f"acc={acc:.4f} eps={eps_now:.3f}", flush=True)
        final = {
            "method": self.METHOD, "dataset": self.cfg["experiment"]["dataset"],
            "seed": self.cfg["experiment"]["seed"], "final_acc": acc,
            "eps": self.eps_final, "delta": self.cfg["dp"]["delta"],
            "sigma_scale": self.sigma_scale, "global_clip": self.global_clip,
            "rel_weights": self.rel_weights, "group_clips": self.group_clips,
            "wall_sec": self.timer.elapsed(),
        }
        write_json(os.path.join(self.out_dir, "final.json"), final)
        if self.cfg["experiment"].get("save_checkpoint", False):
            torch.save({k: v.cpu() for k, v in theta.items()},
                       os.path.join(self.out_dir, "final_lora.pt"))
        return final

    @torch.no_grad()
    def evaluate(self) -> float:
        self.bundle.model.eval()
        correct, total = 0, 0
        for batch in self.data.test_loader():
            ids, mask, y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            pred = self.bundle.model(ids, mask).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.numel()
        return correct / max(total, 1)
