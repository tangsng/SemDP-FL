"""Generate one config file per experiment + round-robin GPU lists.

Grid (Phase 4 design):
  SST-2  : all 5 methods x seeds {0,1,2} @ eps=4            (main table)
  SST-2  : {dp_fedavg, semdp} x eps {2,8} x seeds {0,1,2}   (privacy-utility curve)
  AG News: {dp_fedavg, dp_ffalora, semdp} x seeds {0,1,2}   (generalization)

Usage: python3 scripts/gen_configs.py
"""
from __future__ import annotations

import os

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "configs", "generated")

SEEDS = [0, 1, 2]
ALL_METHODS = ["dp_fedavg", "dp_adaclip", "dp_fedsam", "dp_ffalora", "semdp"]


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
    paths = []
    for m in ALL_METHODS:                       # SST-2 main @ eps=4
        for s in SEEDS:
            paths.append(emit(load_base("sst2"), "sst2", m, 4.0, s))
    for m in ["dp_fedavg", "semdp"]:            # privacy-utility curve
        for eps in [2.0, 8.0]:
            for s in SEEDS:
                paths.append(emit(load_base("sst2"), "sst2", m, eps, s))
    for m in ["dp_fedavg", "dp_ffalora", "semdp"]:  # AG News
        for s in SEEDS:
            paths.append(emit(load_base("agnews"), "agnews", m, 4.0, s))

    rel = [os.path.relpath(p, ROOT).replace("\\", "/") for p in paths]
    for gpu in [0, 1]:
        with open(os.path.join(OUT, f"gpu{gpu}.list"), "w") as f:
            f.write("\n".join(rel[gpu::2]) + "\n")
    print(f"generated {len(rel)} configs; gpu0={len(rel[0::2])}, gpu1={len(rel[1::2])}")


if __name__ == "__main__":
    main()
