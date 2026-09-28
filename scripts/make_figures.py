"""Publication-grade figures (matplotlib/seaborn, >=300 DPI, PDF+PNG, English labels).

Reads results/**/round_log.csv and results/summary.csv. Run AFTER collect_results.py.
Usage: python3 scripts/make_figures.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIGDIR = os.path.join(ROOT, "results", "figures")
METHOD_STYLE = {
    "dp_fedavg": ("DP-FedAvg", "#4C72B0", "-"),
    "dp_adaclip": ("DP-AdaptClip", "#55A868", "-"),
    "dp_fedsam": ("DP-FedSAM", "#C44E52", "-"),
    "dp_ffalora": ("FFA-LoRA+DP", "#8172B3", "-"),
    "semdp": ("SemDP-FL (Ours)", "#CCB974", "-"),
}


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


import re

RUN_RE = re.compile(r"^(dp_fedavg|dp_adaclip|dp_fedsam|dp_ffalora|semdp)_e(\d+)_s(\d+)$")


def load_round_logs(dataset: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", dataset, "*", "round_log.csv")):
        run = os.path.basename(os.path.dirname(p))
        m = RUN_RE.match(run)
        if not m:  # skip ckpt re-runs / no-DP / ablation dirs (no valid e-tag)
            continue
        method, eps, seed = m.group(1), f"e{m.group(2)}", m.group(3)
        df = pd.read_csv(p)
        df["method"] = method
        df["eps_tag"] = eps
        df["run"] = run
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig_acc_vs_round(dataset: str):
    df = load_round_logs(dataset)
    df = df[df["eps_tag"] == "e4"]
    if df.empty:
        print("skip acc_vs_round", dataset)
        return
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m, (label, color, ls) in METHOD_STYLE.items():
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        g = sub.groupby("round")["test_acc"]
        mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
        ax.plot(mean.index, mean.values * 100, label=label, color=color, ls=ls, lw=2)
        ax.fill_between(mean.index, (mean - std) * 100, (mean + std) * 100,
                        color=color, alpha=0.18)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, f"fig1_acc_vs_round_{dataset}")


def fig_privacy_utility():
    df = load_round_logs("sst2")
    if df.empty:
        return
    finals = (df.sort_values("round").groupby(["run", "method", "eps_tag"])
              .tail(1))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for m in ["dp_fedavg", "dp_adaclip", "dp_fedsam", "dp_ffalora", "semdp"]:
        label, color, ls = METHOD_STYLE[m]
        sub = finals[finals["method"] == m].copy()
        sub["eps_val"] = sub["eps_tag"].str[1:].astype(float)
        g = sub.groupby("eps_val")["test_acc"]
        mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
        ax.errorbar(mean.index, mean.values * 100, yerr=std.values * 100,
                    label=label, color=color, ls=ls, marker="o", lw=2, capsize=4)
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "fig2_privacy_utility_sst2")


def fig_allocation():
    p = os.path.join(ROOT, "results", "sst2", "semdp_e4_s0", "allocation_audit.json")
    if not os.path.exists(p):
        print("skip allocation fig")
        return
    with open(p) as f:
        audit = json.load(f)
    w, tiers = audit["weights"], audit["tiers"]
    groups = sorted(w, key=lambda g: (g != "head", -w[g]))
    vals = [w[g] for g in groups]
    colors = [{"high": "#C44E52", "mid": "#CCB974", "low": "#4C72B0"}[tiers[g]]
              for g in groups]
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(range(len(groups)), vals, color=colors)
    ax.axhline(1.0, color="k", lw=1, ls="--", label="Uniform")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g.replace("layer_", "L") for g in groups], rotation=90, fontsize=7)
    ax.set_ylabel(r"Noise Multiplier $\sigma_l$ (rel. to uniform)")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color="#C44E52", label="High-sensitivity"),
               mpatches.Patch(color="#CCB974", label="Mid-sensitivity"),
               mpatches.Patch(color="#4C72B0", label="Low-sensitivity"),
               plt.Line2D([0], [0], color="k", ls="--", label="Uniform")]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    save(fig, "fig3_group_noise_allocation")


def fig_agnews_bar():
    csvp = os.path.join(ROOT, "results", "summary.csv")
    if not os.path.exists(csvp):
        return
    df = pd.read_csv(csvp)
    sub = df[(df["dataset"] == "agnews") & (df["eps"] == 4.0)].copy()
    sub = sub[sub["method"].isin(METHOD_STYLE)].drop_duplicates(subset="method")
    if sub.empty:
        return
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sub["label"] = sub["method"].map(lambda m: METHOD_STYLE[m][0])
    sub["color"] = sub["method"].map(lambda m: METHOD_STYLE[m][1])
    order = {m: i for i, m in enumerate(
        ["dp_fedavg", "dp_adaclip", "dp_fedsam", "dp_ffalora", "semdp"])}
    sub = sub.sort_values("method", key=lambda s: s.map(order))
    sub = sub.reset_index(drop=True)
    ax.bar(sub["label"], sub["acc_mean"] * 100, yerr=sub["acc_std"] * 100,
           color=list(sub["color"]), capsize=5)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xlabel("Method (AG News)")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["label"], rotation=18, ha="right", fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_agnews_bar")


if __name__ == "__main__":
    fig_acc_vs_round("sst2")
    fig_privacy_utility()
    fig_allocation()
    fig_agnews_bar()
    fig_acc_vs_round("agnews")
