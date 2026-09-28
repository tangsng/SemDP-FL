"""Extend ALL remaining 3-seed cells to 5 seeds (seeds {3,4} added).

Cells covered (36 new runs):
  SST-2  (16): 7 ablation variants x s{3,4} (14) + no-DP ref x s{3,4} (2)
  AG News(20): dp_adaclip/dp_fedsam e4 x s{3,4} (4) + 7 ablation variants
               x s{3,4} (14) + no-DP ref x s{3,4} (2)

After this run every cell in Tables II / III / V and both no-DP reference
rows is 5 seeds. Existing done-marked configs are skipped by run_grid.sh.

Usage: python3 scripts/gen_seed5_completion.py
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

SEEDS = [3, 4]


def load(base):
    with open(os.path.join(ROOT, "configs", base)) as f:
        return yaml.safe_load(f)


def write(cfg, name, p):
    cfg["experiment"]["name"] = name
    p = os.path.join(OUT, f"{name}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    new = []

    # ---- SST-2 ablations x s{3,4} ----
    for tag, override in ABLATIONS:
        for s in SEEDS:
            cfg = load("base_sst2.yaml")
            cfg["method"].update(override)
            if tag == "nodeconing":
                cfg["train"]["no_deconing"] = True
            if tag == "nomomentum":
                cfg["train"]["server_momentum"] = 0.0
            name = f"sst2_semdp_{tag}_e4_s{s}"
            cfg["experiment"].update(dataset="sst2", method="semdp", seed=s,
                                     out_dir=f"results/sst2/semdp_{tag}_e4_s{s}")
            cfg["dp"]["epsilon"] = 4.0
            new.append(write(cfg, name, None))

    # ---- SST-2 no-DP ref x s{3,4} ----
    for s in SEEDS:
        cfg = load("base_sst2.yaml")
        cfg["dp"]["disabled"] = True
        name = f"sst2_fedavg_nodp_s{s}"
        cfg["experiment"].update(dataset="sst2", method="dp_fedavg", seed=s,
                                 out_dir=f"results/sst2/fedavg_nodp_s{s}")
        new.append(write(cfg, name, None))

    # ---- AG News baselines (adaclip/fedsam) e4 x s{3,4} ----
    for m in ["dp_adaclip", "dp_fedsam"]:
        for s in SEEDS:
            cfg = load("base_agnews.yaml")
            name = f"agnews_{m}_e4_s{s}"
            cfg["experiment"].update(dataset="agnews", method=m, seed=s,
                                     out_dir=f"results/agnews/{m}_e4_s{s}")
            cfg["dp"]["epsilon"] = 4.0
            new.append(write(cfg, name, None))

    # ---- AG News ablations x s{3,4} ----
    for tag, override in ABLATIONS:
        for s in SEEDS:
            cfg = load("base_agnews.yaml")
            cfg["method"].update(override)
            if tag == "nodeconing":
                cfg["train"]["no_deconing"] = True
            if tag == "nomomentum":
                cfg["train"]["server_momentum"] = 0.0
            name = f"agnews_semdp_{tag}_e4_s{s}"
            cfg["experiment"].update(dataset="agnews", method="semdp", seed=s,
                                     out_dir=f"results/agnews/semdp_{tag}_e4_s{s}")
            cfg["dp"]["epsilon"] = 4.0
            new.append(write(cfg, name, None))

    # ---- AG News no-DP ref x s{3,4} ----
    for s in SEEDS:
        cfg = load("base_agnews.yaml")
        cfg["dp"]["disabled"] = True
        name = f"agnews_fedavg_nodp_s{s}"
        cfg["experiment"].update(dataset="agnews", method="dp_fedavg", seed=s,
                                 out_dir=f"results/agnews/fedavg_nodp_s{s}")
        new.append(write(cfg, name, None))

    all_cfgs = sorted(glob.glob(os.path.join(OUT, "*.yaml")))
    rel = [os.path.relpath(p, ROOT).replace("\\", "/") for p in all_cfgs]
    for gpu in [0, 1]:
        with open(os.path.join(OUT, f"gpu{gpu}.list"), "w") as f:
            f.write("\n".join(rel[gpu::2]) + "\n")
    print(f"new: {len(new)}; total in lists: {len(rel)}")


if __name__ == "__main__":
    main()
