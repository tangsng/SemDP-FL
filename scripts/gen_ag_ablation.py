"""Cross-dataset ablation: AG News x 7 SemDP-FL variants x 3 seeds + no-DP ref.

Adds an AG News column to the ablation table (does the floor pricing /
prior-vs-measured finding replicate cross-dataset?) and an AG News no-DP
ceiling reference. 24 new runs, merged into the 146-config lists.

Usage: python3 scripts/gen_ag_ablation.py
"""
from __future__ import annotations

import glob
import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "configs", "generated")

ABLATIONS = [
    ("prioronly", {"ablation": {"prior_only": True}}),
    ("measuredonly", {"ablation": {"measured_only": True}}),
    ("nofloor", {"ablation": {"no_floor": True}}),
    ("floor12", {"floor_hi": 1.2}),
    ("floor18", {"floor_hi": 1.8}),
    ("nodeconing", {}),
    ("nomomentum", {}),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    new = []
    for tag, override in ABLATIONS:
        for s in [0, 1, 2]:
            with open(os.path.join(ROOT, "configs", "base_agnews.yaml")) as f:
                cfg = yaml.safe_load(f)
            cfg["method"].update(override)
            if tag == "nodeconing":
                cfg["train"]["no_deconing"] = True
            if tag == "nomomentum":
                cfg["train"]["server_momentum"] = 0.0
            name = f"agnews_semdp_{tag}_e4_s{s}"
            cfg["experiment"].update(name=name, dataset="agnews", method="semdp", seed=s,
                                     out_dir=f"results/agnews/semdp_{tag}_e4_s{s}")
            cfg["dp"]["epsilon"] = 4.0
            p = os.path.join(OUT, f"{name}.yaml")
            with open(p, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            new.append(p)
    for s in [0, 1, 2]:  # AG News no-DP ceiling
        with open(os.path.join(ROOT, "configs", "base_agnews.yaml")) as f:
            cfg = yaml.safe_load(f)
        cfg["dp"]["disabled"] = True
        name = f"agnews_fedavg_nodp_s{s}"
        cfg["experiment"].update(name=name, dataset="agnews", method="dp_fedavg", seed=s,
                                 out_dir=f"results/agnews/fedavg_nodp_s{s}")
        p = os.path.join(OUT, f"{name}.yaml")
        with open(p, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        new.append(p)
    all_cfgs = sorted(glob.glob(os.path.join(OUT, "*.yaml")))
    rel = [os.path.relpath(p, ROOT).replace("\\", "/") for p in all_cfgs]
    for gpu in [0, 1]:
        with open(os.path.join(OUT, f"gpu{gpu}.list"), "w") as f:
            f.write("\n".join(rel[gpu::2]) + "\n")
    print(f"new: {len(new)}; total in lists: {len(rel)}")


if __name__ == "__main__":
    main()
