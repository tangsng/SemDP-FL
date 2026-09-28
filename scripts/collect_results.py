"""Aggregate per-run final.json files -> summary CSV + markdown table
(mean +/- std over seeds, ddof=1). Usage: python3 scripts/collect_results.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", "*", "*", "final.json")):
        with open(p) as f:
            j = json.load(f)
        rows.append({"dataset": j["dataset"], "method": j["method"],
                     "seed": j["seed"], "eps": int(round(j["eps"])),
                     "eps_exact": round(j["eps"], 4),
                     "final_acc": j["final_acc"], "wall_sec": round(j["wall_sec"], 0)})
    if not rows:
        print("no results yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "all_runs.csv"), index=False)

    def agg(g):
        m, s = g["final_acc"].mean(), g["final_acc"].std(ddof=1)
        n = len(g)
        return pd.Series({"n_seeds": n, "acc_mean": m,
                          "acc_std": 0.0 if n < 2 else s,
                          "acc_fmt": f"{m*100:.2f}±{(0.0 if n < 2 else s)*100:.2f}"})

    summ = (df.groupby(["dataset", "eps", "method"]).apply(agg, include_groups=False)
              .reset_index().sort_values(["dataset", "eps", "acc_mean"],
                                         ascending=[True, True, False]))
    summ.to_csv(os.path.join(ROOT, "results", "summary.csv"), index=False)
    pd.set_option("display.width", 160)
    print(summ.to_string(index=False))
    # per-run dump for audit
    for _, r in df.sort_values(["dataset", "method", "seed"]).iterrows():
        print(f"  run {r['dataset']}/{r['method']}/s{r['seed']}: "
              f"acc={r['final_acc']*100:.2f} eps={r['eps']}", file=sys.stderr)


if __name__ == "__main__":
    main()
