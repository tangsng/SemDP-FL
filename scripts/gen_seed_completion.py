"""Unify Table III at 5 seeds: {adaclip, fedsam, ffalora} x eps{2,8} x seeds{3,4}.

12 new runs; merged into the 122-config lists (done-marked ones are skipped).
After this, every cell of the SST-2 eps-matrix is 5 seeds.

Usage: python3 scripts/gen_seed_completion.py
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
        for eps in [2.0, 8.0]:
            for s in [3, 4]:
                with open(os.path.join(ROOT, "configs", "base_sst2.yaml")) as f:
                    cfg = yaml.safe_load(f)
                name = f"sst2_{m}_e{int(eps)}_s{s}"
                cfg["experiment"].update(name=name, dataset="sst2", method=m, seed=s,
                                         out_dir=f"results/sst2/{m}_e{int(eps)}_s{s}")
                cfg["dp"]["epsilon"] = eps
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
