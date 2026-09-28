"""Generate the Elsevier graphical abstract for SemDP-FL (JISA submission) v6.

2x2 layout answering the four things readers want first:
  TL ① Problem      - uniform DP noise ignores layer sensitivity
  TR ② Method       - SemDP-FL fused score + tier floors + water-filling
  BL ③ Results      - accuracy comparison + inversion probe
  BR ④ Application  - regulated-domain deployment scenarios

Spec: minimum 531 x 1328 px (h x w) at 300 dpi. Rendered at 2646 x 1322 px
(22.4 cm x 11.2 cm, 300 dpi). All text >= 8 pt.

Usage: python3 scripts/make_graphical_abstract.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "figures", "graphical_abstract.png")

C_HIGH, C_MID, C_LOW = "#C44E52", "#DD8452", "#55A868"
C_OURS, C_BASE = "#4C72B0", "#9A9A9A"
C_PROBLEM, C_METHOD, C_RESULT, C_APP = "#B03A3A", "#2E5A8C", "#3A7D44", "#7A5A9E"

# 每个面板: 色带标题占 0.83-0.98, 内容区 0.02-0.81
BAND_BOT = 0.83
BAR_Y0 = 0.52


def panel_box(ax, title, color):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                boxstyle="round,pad=0.006,rounding_size=0.015",
                                fc="white", ec=color, lw=1.8,
                                transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0.02, BAND_BOT), 0.96, 0.15,
                                boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=color, ec="none", transform=ax.transAxes))
    ax.text(0.5, (BAND_BOT + 0.98) / 2, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white",
            transform=ax.transAxes)


def panel_problem(ax):
    panel_box(ax, "① PROBLEM", C_PROBLEM)
    n = 21
    for i in range(n):
        is_head = (i == 0)
        ax.add_patch(FancyBboxPatch((0.06 + i * 0.043, BAR_Y0), 0.030, 0.24,
                                    boxstyle="round,pad=0.001,rounding_size=0.004",
                                    fc="#DDD" if not is_head else "#E8B4B4",
                                    ec=C_HIGH if is_head else "none",
                                    lw=1.6, transform=ax.transAxes))
    ax.annotate("classification head = most leak-prone",
                xy=(0.075, BAR_Y0 + 0.24), xytext=(0.24, 0.795), fontsize=8.5,
                color=C_HIGH, va="top",
                arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.2),
                transform=ax.transAxes)
    ax.text(0.5, 0.475, "same noise σ on every group",
            ha="center", fontsize=9.5, fontweight="bold", color="#333",
            transform=ax.transAxes)
    ax.text(0.5, 0.355, "but layers carry very different\nsemantic sensitivity",
            ha="center", fontsize=8.5, color="#333", transform=ax.transAxes)
    ax.text(0.5, 0.205, "head under-protected · robust layers\nover-perturbed · accuracy lost",
            ha="center", fontsize=8.5, color=C_HIGH, transform=ax.transAxes)
    ax.text(0.5, 0.09, "→ all four baselines stuck at chance (51% / 25%)",
            ha="center", fontsize=8, color="#666", transform=ax.transAxes)


def panel_method(ax):
    panel_box(ax, "② OUR METHOD: SemDP-FL", C_METHOD)
    tiers = [C_HIGH] * 5 + [C_MID] * 6 + [C_LOW] * 10
    rel = [1.6] * 5 + [1.0] * 6 + [2.5, 0.5, 1.8, 0.5, 0.5, 1.2, 0.5, 0.5, 0.9, 0.5]
    for i, (c, h) in enumerate(zip(tiers, rel)):
        ax.add_patch(FancyBboxPatch((0.05 + i * 0.044, BAR_Y0), 0.030,
                                    0.24 * h / 2.5,
                                    boxstyle="round,pad=0.001,rounding_size=0.004",
                                    fc=c, ec="none", transform=ax.transAxes))
    ax.annotate("head tier floor 1.6×", xy=(0.065, BAR_Y0 + 0.24 * 1.6 / 2.5),
                xytext=(0.05, 0.795), fontsize=8, color=C_HIGH, va="top",
                arrowprops=dict(arrowstyle="->", color=C_HIGH, lw=1.1),
                transform=ax.transAxes)
    ax.annotate("water-filling raises cheap groups",
                xy=(0.549, BAR_Y0 + 0.24 * 2.5 / 2.5), xytext=(0.38, 0.795),
                fontsize=8, color=C_LOW, va="top",
                arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.1),
                transform=ax.transAxes)
    ax.text(0.5, 0.455, "score = LLM prior × measured gradients",
            ha="center", fontsize=9, fontweight="bold", color="#333",
            transform=ax.transAxes)
    ax.text(0.5, 0.385, "(one public proxy pass, no client data)",
            ha="center", fontsize=8, color="#666", transform=ax.transAxes)
    ax.text(0.5, 0.27, "floors cap leakage · water-filling\nallocates the rest optimally",
            ha="center", fontsize=8.5, color=C_METHOD, transform=ax.transAxes)
    ax.text(0.5, 0.10, "identical (ε, δ) via exact RDP accounting",
            ha="center", fontsize=8, color="#666", transform=ax.transAxes)


def panel_results(ax):
    panel_box(ax, "③ RESULTS (equal ε = 4)", C_RESULT)
    scale = 240.0
    groups = [(0.13, 51, 58.6, "SST-2"), (0.56, 25, 50.6, "AG News")]
    for gx, bv, ov, label in groups:
        ax.add_patch(FancyBboxPatch((gx, BAR_Y0), 0.12, bv / scale,
                                    boxstyle="round,pad=0.002,rounding_size=0.006",
                                    fc=C_BASE, ec="none", transform=ax.transAxes))
        ax.add_patch(FancyBboxPatch((gx + 0.13, BAR_Y0), 0.12, ov / scale,
                                    boxstyle="round,pad=0.002,rounding_size=0.006",
                                    fc=C_OURS, ec="none", transform=ax.transAxes))
        ax.text(gx + 0.06, BAR_Y0 + bv / scale + 0.018, f"{bv:g}", ha="center",
                fontsize=8.5, color="#666", transform=ax.transAxes)
        ax.text(gx + 0.19, BAR_Y0 + ov / scale + 0.018, f"{ov:g}", ha="center",
                fontsize=9, fontweight="bold", color=C_OURS,
                transform=ax.transAxes)
        ax.text(gx + 0.125, BAR_Y0 - 0.055, label, ha="center", fontsize=9,
                color="#333", transform=ax.transAxes)
    ax.text(0.5, 0.355, "+5.8 pp (SST-2) · +25.5 pp (AG News, p = 0.012)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_RESULT,
            transform=ax.transAxes)
    ax.text(0.5, 0.265, "up to 86% of non-private accuracy recovered",
            ha="center", fontsize=8.5, color="#333", transform=ax.transAxes)
    ax.text(0.5, 0.185, "grey = best baseline      blue = SemDP-FL",
            ha="center", fontsize=8, color="#666", transform=ax.transAxes)
    ax.text(0.5, 0.075, "inversion probe: recovery cos 0.16 → chance",
            ha="center", fontsize=8.5, color=C_HIGH, transform=ax.transAxes)


def panel_application(ax):
    panel_box(ax, "④ APPLICATION", C_APP)
    ax.text(0.5, 0.775, "drop-in upgrade to any DP-FL pipeline",
            ha="center", fontsize=9.5, fontweight="bold", color=C_APP,
            transform=ax.transAxes)
    apps = [("Healthcare", "clinical assistants on patient notes"),
            ("Finance", "transaction-text & customer-service models"),
            ("On-device", "text personalisation with a DP guarantee")]
    yy = 0.575
    for title, desc in apps:
        ax.add_patch(FancyBboxPatch((0.08, yy), 0.84, 0.155,
                                    boxstyle="round,pad=0.006,rounding_size=0.02",
                                    fc="#F4F1FA", ec=C_APP, lw=1.2,
                                    transform=ax.transAxes))
        ax.text(0.13, yy + 0.105, title, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=C_APP, transform=ax.transAxes)
        ax.text(0.13, yy + 0.045, desc, ha="left", va="center", fontsize=8.5,
                color="#444", transform=ax.transAxes)
        yy -= 0.21
    ax.text(0.5, 0.035, "→ privacy-compliant federated LLM fine-tuning in regulated domains",
            ha="center", fontsize=8, color="#888", transform=ax.transAxes)


def main():
    fig = plt.figure(figsize=(22.4 / 2.54, 11.2 / 2.54), dpi=300)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.975,
             "SemDP-FL: Semantic-Sensitivity-Driven Adaptive Differential Privacy for Federated LoRA Fine-Tuning of LLMs",
             ha="center", va="center", fontsize=9.5, fontweight="bold")
    gs = fig.add_gridspec(2, 2, left=0.02, right=0.98, top=0.94, bottom=0.02,
                          wspace=0.06, hspace=0.10)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    fig.add_artist(FancyArrowPatch((0.505, 0.73), (0.545, 0.73),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=15, lw=2.2, color="#888"))
    fig.add_artist(FancyArrowPatch((0.505, 0.27), (0.545, 0.27),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=15, lw=2.2, color="#888"))
    panel_problem(ax1)
    panel_method(ax2)
    panel_results(ax3)
    panel_application(ax4)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
