"""Model bundle: Qwen2.5 backbone (frozen, fp16) + manual LoRA + classification head.

Grouping convention (drives per-group DP accounting):
  - group "layer_{i}"  : all LoRA params (A and B of every target proj) in block i
  - group "head"       : the classification head
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

from .lora import inject_lora


class ClsModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(hidden_size, num_labels)
        # small init: pretrained hidden states have large norms, a default-init
        # head would produce extreme logits and blow up local training
        nn.init.normal_(self.score.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.score.bias)
        # mean pooled embedding over the PUBLIC proxy set, set once by
        # ModelBundle.calibrate_head before training (not trainable, not in any
        # DP parameter group, estimated on public data only)
        self.register_buffer("h_mean", torch.zeros(hidden_size))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state                      # [B, L, H]
        lengths = attention_mask.sum(dim=1)            # last non-pad token
        h_last = h[torch.arange(h.size(0), device=h.device), lengths - 1].float()
        # (i) de-cone: transformer sentence embeddings live in a narrow cone
        # (anisotropy); the shared cone-direction carries no class signal yet
        # dominates both the noise projection and the majority-bias residual of
        # federated averaging, collapsing the model to a constant predictor.
        # Subtracting the public-set mean removes the attractor (identical for
        # all methods, disclosed in the paper).
        # (ii) l2-normalise: pretrained hidden-state norms are very large
        # (~1e2), which otherwise destabilises local SGD.
        h_last = h_last - self.h_mean
        h_last = h_last / h_last.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.score(h_last)


class ModelBundle:
    def __init__(self, model_name: str, num_labels: int, lora_r: int, lora_alpha: float,
                 lora_targets: tuple[str, ...], device: torch.device,
                 freeze_a: bool = False, fp16: bool = True):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        backbone = AutoModel.from_pretrained(model_name)
        hidden = backbone.config.hidden_size
        for p in backbone.parameters():
            p.requires_grad_(False)
        if fp16:
            backbone.half()
        self.replaced = inject_lora(backbone, lora_targets, r=lora_r, alpha=lora_alpha,
                                    freeze_a=freeze_a)
        self.model = ClsModel(backbone, hidden, num_labels).to(device)
        self.trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.group_of = {n: self._group(n) for n in self.trainable_names}
        self.groups = sorted(set(self.group_of.values()),
                             key=lambda g: (g != "head", int(g.split("_")[1]) if g.startswith("layer_") else 10**9))

    @staticmethod
    def _group(param_name: str) -> str:
        if param_name.startswith("score"):
            return "head"
        m = re.search(r"layers\.(\d+)\.", param_name)
        return f"layer_{m.group(1)}" if m else "other"

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().float().cpu().clone()
                for n, p in self.model.named_parameters() if p.requires_grad}

    def load_trainable(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(state[n].to(device=p.device, dtype=p.dtype))

    def group_param_names(self, group: str) -> list[str]:
        return [n for n, g in self.group_of.items() if g == group]

    @torch.no_grad()
    def calibrate_head(self, loader) -> None:
        """Estimate the cone direction h_mean = E[h_last] on the PUBLIC proxy
        set (no client data, no privacy budget). Called once before training;
        must run before any gradient measurement / clipping-norm probe so that
        every downstream step uses the centred forward."""
        self.model.eval()
        tot, cnt = None, 0
        for batch in loader:
            ids, mask = batch[0].to(self.device), batch[1].to(self.device)
            out = self.model.backbone(input_ids=ids, attention_mask=mask)
            lengths = mask.sum(dim=1)
            h_last = out.last_hidden_state[
                torch.arange(ids.size(0), device=ids.device), lengths - 1].float()
            tot = h_last.sum(dim=0) if tot is None else tot + h_last.sum(dim=0)
            cnt += ids.size(0)
        self.model.h_mean.copy_(tot / max(cnt, 1))

    def group_dims(self) -> dict[str, int]:
        dims = {g: 0 for g in self.groups}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                dims[self.group_of[n]] += p.numel()
        return dims


def build_bundle(cfg: dict, device: torch.device, freeze_a: bool = False) -> ModelBundle:
    mc = cfg["model"]
    num_labels = {"sst2": 2, "agnews": 4}[cfg["experiment"]["dataset"]]
    return ModelBundle(mc["name"], num_labels, mc["lora_r"], mc["lora_alpha"],
                       tuple(mc["lora_targets"]), device, freeze_a=freeze_a)
