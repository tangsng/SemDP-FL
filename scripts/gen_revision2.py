"""Revision grid wave 2 (review round 1): no-DP reference + ablations + checkpoints.

Batch C (no-DP reference, 3 runs):
  SST-2 dp_fedavg with dp.disabled=true x seeds {0,1,2}

Batch D (SemDP-FL ablations @ SST-2 eps=4, 3 seeds each, 21 runs):
  prior_only / measured_only / no_floor / floor_hi=1.2 / floor_hi=1.8 /
  no_deconing / no_server_momentum

Batch B-prep (checkpoint re-runs for trained-model attack, 2 runs):
  dp_fedavg + semdp @ SST-2 eps=4 seed 0 with save_checkpoint=true
  (out_dir suffixed _ckpt so the original final.json is preserved)

Merges ALL configs (existing 75 + new 26) into gpu{0,1}.list.
Run ONLY after wave-1 grid finished (queue via queue_revision2.sh).
"""
from __future__ import annotations

import glob
import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "configs", "generated")


def load_base(dataset: str) -> dict:
    with open(os.path.join(ROOT, "configs", f"base_{dataset}.yaml")) as f:
        return yaml.safe_load(f)


def emit(cfg: dict, dataset: str, method: str, tag: str, seed: int) -> str:
    name = f"{dataset}_{tag}_s{seed}"
    cfg["experiment"].update(name=name, dataset=dataset, method=method, seed=seed,
                             out_dir=f"results/{dataset}/{tag}_s{seed}")
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    new_paths = []
    for s in [0, 1, 2]:                                # Batch C: no-DP
        cfg = load_base("sst2")
        cfg["dp"]["disabled"] = True
        new_paths.append(emit(cfg, "sst2", "dp_fedavg", "fedavg_nodp", s))
    ablations = [                                      # Batch D
        ("prioronly", {"ablation": {"prior_only": True}}),
        ("measuredonly", {"ablation": {"measured_only": True}}),
        ("nofloor", {"ablation": {"no_floor": True}}),
        ("floor12", {"floor_hi": 1.2}),
        ("floor18", {"floor_hi": 1.8}),
        ("nodeconing", {}),
        ("nomomentum", {}),
    ]
    for tag, override in ablations:
        for s in [0, 1, 2]:
            cfg = load_base("sst2")
            cfg["method"].update(override)
            if tag == "nodeconing":
                cfg["train"]["no_deconing"] = True
            if tag == "nomomentum":
                cfg["train"]["server_momentum"] = 0.0
            new_paths.append(emit(cfg, "sst2", "semdp", f"semdp_{tag}_e4", s))
    for m in ["dp_fedavg", "semdp"]:                   # Batch B-prep
        cfg = load_base("sst2")
        cfg["experiment"]["save_checkpoint"] = True
        new_paths.append(emit(cfg, "sst2", m, f"{m}_e4_ckpt", 0))

    all_cfgs = sorted(glob.glob(os.path.join(OUT, "*.yaml")))
    rel = [os.path.relpath(p, ROOT).replace("\\", "/") for p in all_cfgs]
    for gpu in [0, 1]:
        with open(os.path.join(OUT, f"gpu{gpu}.list"), "w") as f:
            f.write("\n".join(rel[gpu::2]) + "\n")
    print(f"new configs: {len(new_paths)}; total in lists: {len(rel)}")


if __name__ == "__main__":
    main()
