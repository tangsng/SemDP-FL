"""Client-side local training.

Plain SGD (with momentum) by default; SAM (Foret et al., ICLR 2021) two-step
perturbation when `sam_rho > 0` (implements the client optimizer of DP-FedSAM,
Shi et al., CVPR 2023). Returns the parameter *delta* (after - before) for all
trainable parameters, on CPU in fp32.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


class Client:
    def __init__(self, cid: int, loader, cfg: dict, device: torch.device):
        self.cid = cid
        self.loader = loader
        self.cfg = cfg
        self.device = device

    def _loss(self, model, batch):
        ids, mask, y = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
        logits = model(ids, mask)
        return F.cross_entropy(logits, y)

    def local_train(self, bundle, global_state: dict[str, torch.Tensor],
                    sam_rho: float = 0.0) -> tuple[dict[str, torch.Tensor], float, int]:
        bundle.load_trainable(global_state)
        model = bundle.model
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        tc = self.cfg["train"]
        opt = torch.optim.SGD(params, lr=tc["lr"], momentum=tc["momentum"])
        max_gn = tc.get("grad_clip_local", 5.0)  # local optimisation hygiene (not DP clipping)
        total_loss, total_n = 0.0, 0

        for _ in range(tc["local_epochs"]):
            for batch in self.loader:
                bs = batch[2].numel()
                if sam_rho > 0:  # SAM two-step
                    loss = self._loss(model, batch)
                    if not torch.isfinite(loss):
                        continue
                    loss.backward()
                    with torch.no_grad():
                        gnorm = torch.norm(torch.stack([
                            p.grad.norm() for p in params if p.grad is not None]))
                        eps = [sam_rho * p.grad / (gnorm + 1e-12)
                               for p in params if p.grad is not None]
                        trainable = [p for p in params if p.grad is not None]
                        for p, e in zip(trainable, eps):
                            p.add_(e)                       # ascend to w + e
                    opt.zero_grad(set_to_none=True)
                    loss2 = self._loss(model, batch)
                    if not torch.isfinite(loss2):
                        bundle.load_trainable(global_state)
                        continue
                    loss2.backward()
                    with torch.no_grad():
                        for p, e in zip(trainable, eps):
                            p.sub_(e)                       # restore w
                    torch.nn.utils.clip_grad_norm_(params, max_gn)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    total_loss += float(loss2.detach()) * bs
                else:
                    loss = self._loss(model, batch)
                    if not torch.isfinite(loss):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_gn)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    total_loss += float(loss.detach()) * bs
                total_n += bs

        new_state = bundle.trainable_state()
        delta = {n: new_state[n] - global_state[n] for n in global_state}
        return delta, total_loss / max(total_n, 1), total_n
