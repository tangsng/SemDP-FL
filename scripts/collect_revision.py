"""Revision collector: strict per-tag grouping (main table 5 seeds, ablations,
no-DP reference, checkpoint runs) + Welch t / Cohen's d vs SemDP-FL.

The plain collect_results.py groups by the `method` field, which lumps the 21
ablation runs and checkpoint re-runs into 'semdp'. This script groups by the
run DIRECTORY name instead, so every tag is its own row.

Usage: python3 scripts/collect_revision.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# run dirs look like results/<dataset>/<tag>_s<seed>/final.json (dataset is the
# PARENT directory, not part of the run basename)
MAIN_RE = re.compile(r"^(dp_fedavg|dp_adaclip|dp_fedsam|dp_ffalora|semdp)_e(\d+)_s(\d+)$")
ABL_RE = re.compile(r"^semdp_(prioronly|measuredonly|nofloor|floor12|floor18|nodeconing|nomomentum)_e4_s(\d+)$")
NODP_RE = re.compile(r"^fedavg_nodp_s(\d+)$")
CKPT_RE = re.compile(r"^(dp_fedavg|semdp)_e4_ckpt_s0$")


def load_runs():
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", "*", "*", "final.json")):
        run = os.path.basename(os.path.dirname(p))
        ds = os.path.basename(os.path.dirname(os.path.dirname(p)))
        with open(p) as f:
            j = json.load(f)
        rows.append({"run": run, "ds": ds, "acc": j["final_acc"], "eps": j["eps"],
                     "wall_sec": j.get("wall_sec", 0)})
    return rows


def agg(group_name, rows):
    accs = np.array([r["acc"] for r in rows]) * 100
    n = len(accs)
    return {"group": group_name, "n": n,
            "mean": accs.mean(), "std": accs.std(ddof=1) if n > 1 else 0.0,
            "accs": accs}


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return t, p, d


def main():
    rows = load_runs()
    main_groups, abl_groups, nodp, ckpt = {}, {}, [], []
    for r in rows:
        m = MAIN_RE.match(r["run"])
        if m:
            meth, eps = m.group(1), int(m.group(2))
            main_groups.setdefault(f"{r['ds']} {meth} e{eps}", []).append(r)
            continue
        m = ABL_RE.match(r["run"])
        if m:
            abl_groups.setdefault(f"semdp_{m.group(1)}", []).append(r)
            continue
        if NODP_RE.match(r["run"]):
            nodp.append(r)
            continue
        if CKPT_RE.match(r["run"]):
            ckpt.append(r)

    print("=" * 78)
    print("MAIN GRID (5 seeds)")
    print("=" * 78)
    out_rows = []
    for name in sorted(main_groups):
        g = agg(name, main_groups[name])
        out_rows.append(g)
        print(f"{g['group']:28s} n={g['n']}  {g['mean']:6.2f} ± {g['std']:5.2f}")
    pd.DataFrame([{k: v for k, v in g.items() if k != "accs"} for g in out_rows]
                 ).to_csv(os.path.join(ROOT, "results", "revision_main.csv"), index=False)

    print("\n" + "=" * 78)
    print("WELCH t / Cohen's d  (SemDP-FL vs baselines, SST-2 @ eps=4, 5 seeds)")
    print("=" * 78)
    ours = agg("semdp", main_groups["sst2 semdp e4"])["accs"]
    test_rows = []
    for meth in ["dp_fedavg", "dp_adaclip", "dp_fedsam", "dp_ffalora"]:
        other = agg(meth, main_groups[f"sst2 {meth} e4"])["accs"]
        t, p, d = welch(ours, other)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        test_rows.append({"vs": meth, "t": round(t, 3), "p": round(p, 4),
                          "cohens_d": round(d, 2), "sig": sig})
        print(f"semdp vs {meth:12s}  t={t:6.3f}  p={p:.4f}  d={d:5.2f}  {sig}")
    # eps=8 vs strongest baseline there (ffalora)
    ours8 = agg("semdp8", main_groups["sst2 semdp e8"])["accs"]
    ffa8 = agg("ffa8", main_groups["sst2 dp_ffalora e8"])["accs"]
    t, p, d = welch(ours8, ffa8)
    print(f"eps=8: semdp vs dp_ffalora   t={t:6.3f}  p={p:.4f}  d={d:5.2f}")
    test_rows.append({"vs": "dp_ffalora@eps8", "t": round(t, 3), "p": round(p, 4),
                      "cohens_d": round(d, 2), "sig": ""})
    # AG News vs fedavg
    oursA = agg("semdpA", main_groups["agnews semdp e4"])["accs"]
    faA = agg("faA", main_groups["agnews dp_fedavg e4"])["accs"]
    t, p, d = welch(oursA, faA)
    print(f"agnews: semdp vs dp_fedavg   t={t:6.3f}  p={p:.4f}  d={d:5.2f}")
    test_rows.append({"vs": "dp_fedavg@agnews", "t": round(t, 3), "p": round(p, 4),
                      "cohens_d": round(d, 2), "sig": ""})
    pd.DataFrame(test_rows).to_csv(os.path.join(ROOT, "results", "revision_tests.csv"),
                                   index=False)

    print("\n" + "=" * 78)
    print("ABLATIONS (SemDP-FL variants, SST-2 @ eps=4, 3 seeds)")
    print("=" * 78)
    abl_rows = []
    full = agg("semdp_full(5seeds)", main_groups["sst2 semdp e4"])
    print(f"{'semdp_full (main, 5 seeds)':32s} n={full['n']}  {full['mean']:6.2f} ± {full['std']:5.2f}")
    for name in sorted(abl_groups):
        g = agg(name, abl_groups[name])
        abl_rows.append({k: v for k, v in g.items() if k != "accs"})
        print(f"{g['group']:32s} n={g['n']}  {g['mean']:6.2f} ± {g['std']:5.2f}  "
              f"(Δ vs full {g['mean'] - full['mean']:+5.2f})")
    pd.DataFrame(abl_rows).to_csv(os.path.join(ROOT, "results", "revision_ablations.csv"),
                                  index=False)

    print("\n" + "=" * 78)
    print("NO-DP REFERENCE (SST-2, 3 seeds)")
    print("=" * 78)
    g = agg("fedavg_nodp", nodp)
    print(f"{g['group']:32s} n={g['n']}  {g['mean']:6.2f} ± {g['std']:5.2f}")
    print("\nCHECKPOINT re-runs (for trained-model attack):")
    for r in ckpt:
        print(f"  {r['run']}: acc={r['acc']*100:.2f}")


if __name__ == "__main__":
    main()
