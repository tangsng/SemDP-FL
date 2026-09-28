"""Revision grid (review round 1): seeds extension + baseline-matrix completion.

Batch A (statistical power, seeds {3,4}):
  SST-2  : all 5 methods x seeds {3,4} @ eps=4            (main table ext)
  SST-2  : {dp_fedavg, semdp} x eps {2,8} x seeds {3,4}   (curve ext)
  AG News: {dp_fedavg, dp_ffalora, semdp} x seeds {3,4}   (generalisation ext)

Batch E (matrix completion, seeds {0,1,2}):
  AG News: {dp_adaclip, dp_fedsam} @ eps=4
  SST-2  : {dp_fedsam, dp_adaclip, dp_ffalora} @ eps=8

Rewrites gpu{0,1}.list to include BOTH the completed 36 configs (skipped by
final_marks) and the new ones, then run_grid.sh resumes everything.

Usage: python3 scripts/gen_revision.py
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "configs", "generated")

MAIN_METHODS = ["dp_fedavg", "dp_adaclip", "dp_fedsam", "dp_ffalora", "semdp"]


def load_base(dataset: str) -> dict:
    with open(os.path.join(ROOT, "configs", f"base_{dataset}.yaml")) as f:
        return yaml.safe_load(f)


def emit(cfg: dict, dataset: str, method: str, eps: float, seed: int) -> str:
    eps_tag = f"e{int(eps)}"
    name = f"{dataset}_{method}_{eps_tag}_s{seed}"
    cfg["experiment"].update(name=name, dataset=dataset, method=method, seed=seed,
                             out_dir=f"results/{dataset}/{method}_{eps_tag}_s{seed}")
    cfg["dp"]["epsilon"] = eps
    path = os.path.join(OUT, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    new_paths = []
    for m in MAIN_METHODS:                            # Batch A: main ext
        for s in [3, 4]:
            new_paths.append(emit(load_base("sst2"), "sst2", m, 4.0, s))
    for m in ["dp_fedavg", "semdp"]:                  # Batch A: curve ext
        for eps in [2.0, 8.0]:
            for s in [3, 4]:
                new_paths.append(emit(load_base("sst2"), "sst2", m, eps, s))
    for m in ["dp_fedavg", "dp_ffalora", "semdp"]:    # Batch A: agnews ext
        for s in [3, 4]:
            new_paths.append(emit(load_base("agnews"), "agnews", m, 4.0, s))
    for m in ["dp_adaclip", "dp_fedsam"]:             # Batch E: agnews x2
        for s in [0, 1, 2]:
            new_paths.append(emit(load_base("agnews"), "agnews", m, 4.0, s))
    for m in ["dp_fedsam", "dp_adaclip", "dp_ffalora"]:  # Batch E: eps=8 x3
        for s in [0, 1, 2]:
            new_paths.append(emit(load_base("sst2"), "sst2", m, 8.0, s))

    # merge with existing 36 configs (done-marked ones will be skipped by run_grid)
    import glob as _g
    all_cfgs = sorted(_g.glob(os.path.join(OUT, "*.yaml")))
    rel = [os.path.relpath(p, ROOT).replace("\\", "/") for p in all_cfgs]
    for gpu in [0, 1]:
        with open(os.path.join(OUT, f"gpu{gpu}.list"), "w") as f:
            f.write("\n".join(rel[gpu::2]) + "\n")
    print(f"new configs: {len(new_paths)}; total in lists: {len(rel)} "
          f"(gpu0={len(rel[0::2])}, gpu1={len(rel[1::2])})")


if __name__ == "__main__":
    main()
