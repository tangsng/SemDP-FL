"""Leakage evaluation: per-sample classification-head gradient inversion.

Protocol (disclosed in the paper as an embedding-level proxy for DLG/LAMP-style
attacks, cf. Zhu et al. 2019; Balunovic et al. 2022): for a single client sample,
the honest-but-curious server observes the NOISED per-sample head gradient
(same clipping and noise multiplier as the method's head group at equal total
epsilon), then jointly optimises the input's last-hidden embedding h and the
softmax vector p to reproduce the observed gradient. Leakage metric = cosine
similarity between recovered and true h_last (higher = more leakage) plus
label recovery accuracy. Ours should leak strictly less on the head group.

Usage:
  python3 scripts/attack_eval.py --config configs/generated/sst2_semdp_e4_s0.yaml \
      --samples 8 --iters 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.datasets.federated_data import FederatedData            # noqa: E402
from src.models.modeling import build_bundle                     # noqa: E402
from src.utils.common import mean_std, set_seed                  # noqa: E402


def head_gradient(bundle, ids, mask, y):
    """Per-sample gradient of CE loss w.r.t. classification head params."""
    bundle.model.zero_grad(set_to_none=True)
    logits = bundle.model(ids, mask)
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, list(bundle.model.score.parameters()))
    return torch.cat([g.detach().reshape(-1) for g in grads]).float().cpu(), logits.detach()


def attack_single(bundle, ids, mask, y_true, g_obs, iters=400, lr=0.05):
    """Recover (h_last, p) by gradient matching on the head."""
    hidden = bundle.model.score.in_features
    num_labels = bundle.model.score.out_features
    device = next(bundle.model.parameters()).device
    h = torch.randn(hidden, device=device, requires_grad=True)
    logit_p = torch.zeros(num_labels, device=device, requires_grad=True)
    opt = torch.optim.Adam([h, logit_p], lr=lr)
    W = bundle.model.score.weight  # [C, H]
    b = bundle.model.score.bias    # [C]
    g_target = g_obs.to(device)
    for _ in range(iters):
        p = torch.softmax(logit_p, dim=0)
        # per-sample head grad given (h, p, y): dL/dW = (p - onehot) x h ; dL/db = p - onehot
        # attacker does not know y: optimise over soft label q as well via p only;
        # use expected one-hot = p (standard relaxed iDLG matching).
        # NOTE: the deployed head consumes the centred + l2-normalised embedding
        # (see modeling.ClsModel), so the attacker's surrogate forward must match it.
        h_c = h - bundle.model.h_mean
        h_n = h_c / h_c.norm().clamp_min(1e-8)
        logits = h_n @ W.t() + b
        loss_ce = -(torch.log_softmax(logits, dim=0) * p).sum()
        gw, gb = torch.autograd.grad(loss_ce, [W, b], create_graph=True)
        g_hat = torch.cat([gw.reshape(-1), gb.reshape(-1)])
        match = F.mse_loss(g_hat, g_target)
        opt.zero_grad()
        match.backward()
        opt.step()
    with torch.no_grad():
        # true h_last (same centre + l2 normalisation as the deployed head)
        out = bundle.model.backbone(input_ids=ids, attention_mask=mask)
        lengths = mask.sum(dim=1)
        h_true = out.last_hidden_state[0, lengths[0] - 1].float()
        h_true = h_true - bundle.model.h_mean
        h_true = h_true / h_true.norm().clamp_min(1e-8)
        h_rec = h.detach().float() - bundle.model.h_mean
        h_rec = h_rec / h_rec.norm().clamp_min(1e-8)
        cos = F.cosine_similarity(h_rec, h_true, dim=0).item()
        # label guess: sign structure of bias part (iDLG-style)
        gb_obs = g_obs[-num_labels:]
        label_guess = int(torch.argmin(gb_obs).item())
    return cos, int(label_guess == int(y_true.item()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--checkpoint", default=None,
                    help="optional final_lora.pt to attack the TRAINED model "
                         "(default: attack the initial model, as before)")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # pull the method's calibrated noise from the finished run
    final_path = os.path.join(cfg["experiment"]["out_dir"], "final.json")
    with open(final_path) as f:
        final = json.load(f)
    set_seed(cfg["experiment"]["seed"] + 4242)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_bundle(cfg, device, freeze_a=cfg["experiment"]["method"] == "dp_ffalora")
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        bundle.load_trainable({k: v.to(device) for k, v in state.items()})
        print(f"[attack] loaded trained checkpoint {args.checkpoint}")
    fc = cfg["fl"]
    data = FederatedData(fc_task := cfg["experiment"]["dataset"], fc["num_clients"],
                         fc["dirichlet_alpha"], fc["max_samples_per_client"],
                         fc["public_samples"], bundle.tokenizer, cfg["model"]["max_len"],
                         cfg["train"]["batch_size"], cfg["experiment"]["seed"],
                         fc.get("test_samples", 2000))
    m = fc["clients_per_round"]
    bundle.calibrate_head(data.public_loader())  # match the deployed centred head
    per_group = cfg["experiment"]["method"] == "semdp"
    if per_group:
        std_head = final["rel_weights"]["head"] * final["sigma_scale"] \
            * final["group_clips"]["head"] / m
    else:
        std_head = final["sigma_scale"] * final["global_clip"] / m
    head_dim = sum(p.numel() for p in bundle.model.score.parameters())
    gen = torch.Generator().manual_seed(cfg["experiment"]["seed"] + 999)

    cos_sims, label_hits = [], []
    loader = data.client_loader(0, shuffle=True)
    seen = 0
    for ids, mask, y in loader:
        for j in range(ids.size(0)):
            if seen >= args.samples:
                break
            xi, mi, yi = ids[j:j + 1].to(device), mask[j:j + 1].to(device), y[j:j + 1].to(device)
            g_true, _ = head_gradient(bundle, xi, mi, yi)
            g_clip = g_true * min(1.0, (final["group_clips"]["head"] if per_group
                                        else final["global_clip"]) / max(float(g_true.norm()), 1e-12))
            noise = torch.randn(g_clip.shape, generator=gen) * std_head
            g_obs = g_clip + noise
            cos, hit = attack_single(bundle, xi, mi, yi, g_obs, iters=args.iters)
            cos_sims.append(cos)
            label_hits.append(hit)
            seen += 1
        if seen >= args.samples:
            break
    mc, sc = mean_std(cos_sims)
    ml, sl = mean_std([float(x) for x in label_hits])
    res = {"method": final["method"], "eps": final["eps"], "n": seen,
           "trained_model": bool(args.checkpoint),
           "emb_cos_mean": mc, "emb_cos_std": sc,
           "label_acc_mean": ml, "label_acc_std": sl, "std_head": std_head,
           "head_dim": head_dim}
    out = os.path.join(cfg["experiment"]["out_dir"],
                       "attack_eval_trained.json" if args.checkpoint else "attack_eval.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
