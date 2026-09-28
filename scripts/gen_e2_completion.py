"""Complete the eps=2 matrix: {dp_adaclip, dp_fedsam, dp_ffalora} x 3 seeds.

Makes Table III a full 5-method x 3-eps matrix (fedavg/semdp already have
5 seeds at eps=2; the three baselines get 3 seeds, matching the eps=8
completion protocol). Merges all configs (existing done-marked ones are
skipped by run_grid.sh) into gpu{0,1}.list.

Usage: python3 scripts/gen_e2_completion.py
"""
from __future__ import annotations

import glob
import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "configs", "generated")


def main():
    os.makedirs(OUT, exist_ok=True)
    new = []
    for m in ["dp_adaclip", "dp_fedsam", "dp_ffalora"]:
        for s in [0, 1, 2]:
            with open(os.path.join(ROOT, "configs", "base_sst2.yaml")) as f:
                cfg = yaml.safe_load(f)
            name = f"sst2_{m}_e2_s{s}"
            cfg["experiment"].update(name=name, dataset="sst2", method=m, seed=s,
                                     out_dir=f"results/sst2/{m}_e2_s{s}")
            cfg["dp"]["epsilon"] = 2.0
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
