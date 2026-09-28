# SemDP-FL: Semantic-Sensitivity-Driven Adaptive Differential Privacy for Federated LoRA Fine-Tuning of LLMs

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Runs](https://img.shields.io/badge/Runs-182-blueviolet.svg)](results/raw_runs/)

Official implementation of **SemDP-FL**, a semantic-sensitivity-driven adaptive differential privacy (DP) mechanism for federated LoRA fine-tuning of large language models.

> **Paper**: *SemDP-FL: Semantic-Sensitivity-Driven Adaptive Differential Privacy for Federated LoRA Fine-Tuning of Large Language Models* — under review.

---

## Table of Contents

- [Overview](#overview)
- [Key Idea](#key-idea)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing All Experiments](#reproducing-all-experiments)
- [Reproducing Paper Tables and Figures](#reproducing-paper-tables-and-figures)
- [Provided Results](#provided-results)
- [Main Results](#main-results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Federated LoRA fine-tuning lets clients adapt a shared LLM without exposing raw text, but the exchanged low-rank updates remain vulnerable to gradient-inversion attacks. Client-level DP is the standard defence; however, existing DP federated LoRA methods inject noise **uniformly** across all trainable parameters, overlooking the pronounced heterogeneity of semantic sensitivity across transformer layers.

**SemDP-FL** allocates a fixed client-level DP budget *across LoRA parameter groups* according to a fused semantic sensitivity score. It consists of three components:

1. **Semantic sensitivity assessment** — fuses an LLM-elicited semantic prior with gradient evidence measured on a small public proxy set (Eq. (3)); groups are stratified into high/mid/low tiers with per-tier noise floors.
2. **Utility-optimal adaptive noise allocation** — a closed-form water-filling solution (Theorem 1, $\sigma_l^\star \propto a_l^{-1/4}$) with an explicit uniform-suboptimality bound (Theorem 2, measured $\rho \approx 2.08$).
3. **LoRA-level privacy protection analysis** — a Gaussian-channel mutual-information bound that turns the tier floors into hard leakage caps (Theorem 4), corroborated by a head-gradient inversion probe.

All methods are reimplemented on a **common codebase, backbone, data split, accountant, and hyperparameters**, with noise bisection-calibrated so that every compared method spends an **identical $(\varepsilon, \delta)$**.

## Key Idea

| | Uniform DP (baselines) | SemDP-FL |
|---|---|---|
| Noise allocation | one multiplier for all parameters | per-group multipliers from a fused sensitivity score |
| Leakage-prone head | same noise as everything else | enforced floor (1.6× uniform) |
| Theory | — | closed-form optimum + suboptimality ratio + MI bound |

## Repository Structure

```
.
├── README.md                    # this file
├── LICENSE                      # MIT
├── requirements.txt             # dependencies
├── .gitignore
├── src/                         # core library
│   ├── models/                  # backbone + LoRA injection (lora.py, modeling.py)
│   ├── datasets/                # federated partitioning (Dirichlet) (federated_data.py)
│   ├── clients/                 # local client training (client.py)
│   ├── servers/                 # aggregation strategies
│   │   ├── base.py              # shared FL server
│   │   ├── dp_fedavg.py         # DP-FedAvg
│   │   ├── dp_adaclip.py        # DP-AdaptClip
│   │   ├── dp_fedsam.py         # DP-FedSAM
│   │   ├── dp_ffalora.py        # FFA-LoRA+DP
│   │   └── semdp.py             # SemDP-FL (ours)
│   └── utils/
│       ├── rdp_accountant.py    # exact integer-order sampled-Gaussian RDP
│       ├── allocation.py        # closed-form floored water-filling (Thm 1-2)
│       ├── sensitivity.py       # fused semantic score + tiering
│       └── common.py
├── scripts/                     # entry points
│   ├── train_fl.py              # single-run trainer (--config)
│   ├── attack_eval.py           # head-gradient inversion probe
│   ├── collect_results.py       # aggregate runs -> summary.csv
│   ├── collect_revision.py      # grouped stats + Welch tests
│   ├── gen_configs.py           # generate the full config grid
│   ├── gen_revision*.py         # (revision) grid completion configs
│   ├── gen_seed*_completion.py  # (revision) seed/ablation completion configs
│   ├── make_figures.py          # paper figures 1-5
│   └── make_graphical_abstract.py  # graphical abstract (2x2)
├── configs/
│   ├── base_sst2.yaml           # SST-2 base configuration
│   ├── base_agnews.yaml         # AG News base configuration
│   ├── semantic_prior_qwen25_05b.yaml  # LLM-elicited semantic prior (per-group)
│   └── smoke.yaml               # tiny smoke-test configuration
├── data/
│   ├── README.md                # dataset provenance and links
│   └── prepare_data.py          # download + cache SST-2 / AG News
├── results/
│   ├── raw_runs/                # ALL 182 runs (per-run final.json / round_log.csv / config.yaml / allocation_audit.json)
│   ├── summary.csv              # aggregated per-(dataset, eps, method)
│   ├── all_runs.csv             # flat table of every run
│   ├── revision_main.csv        # main-table grouped statistics
│   ├── revision_tests.csv       # per-pair Welch tests
│   ├── revision_ablations.csv   # ablation grouped statistics
│   └── allocation_audit_semdp_e4_s0.json  # Fig. 5 source (per-group multipliers)
└── figures/                     # paper figures 1-5 + graphical abstract
```

## Data

**No private data is used.** Both benchmarks are public; the repository ships a downloader rather than the corpora themselves.

| Dataset | Task | Source | Link |
|---|---|---|---|
| **SST-2** | binary sentiment (GLUE) | HuggingFace `datasets` | https://huggingface.co/datasets/nyu-mll/glue (subset `sst2`) |
| **AG News** | 4-class news topic | HuggingFace `datasets` | https://huggingface.co/datasets/fancyzhx/ag_news |

See [`data/README.md`](data/README.md) for provenance, licensing, and partitioning details.

```bash
python data/prepare_data.py --datasets sst2 agnews
```

The public proxy split (512 samples) used for sensitivity measurement and clip initialisation is drawn from the datasets' **public training portions** and is strictly disjoint from all client subsets (see `src/datasets/federated_data.py`).

## Installation

```bash
git clone https://github.com/<your-account>/semdp-fl.git
cd semdp-fl
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x (CUDA 12.x), and `transformers` 4.57 on 2× NVIDIA V100S 32GB.

## Quick Start

Smoke test (a few rounds, tiny client count):

```bash
python scripts/train_fl.py --config configs/smoke.yaml
```

A single full run:

```bash
# SemDP-FL on SST-2 at ε = 4, seed 0
python scripts/gen_configs.py            # materialises configs/generated/*.yaml
python scripts/train_fl.py --config configs/generated/sst2_semdp_e4_s0.yaml

# baseline for comparison (identical budget)
python scripts/train_fl.py --config configs/generated/sst2_dp_fedavg_e4_s0.yaml
```

Outputs are written to `results/<dataset>/<run_name>/` containing:

- `final.json` — final test accuracy, realised ε, wall-clock
- `round_log.csv` — per-round train loss / test accuracy
- `config.yaml` — the exact configuration used
- `allocation_audit.json` — (SemDP-FL) per-group multipliers, tiers, measured $a_l$, $\rho$

## Reproducing All Experiments

The full evidence base is **182 runs** (5 seeds). The grid runner supports dual-GPU parallelism and resumes from existing runs:

```bash
python scripts/gen_configs.py                 # base grid
python scripts/gen_e2_completion.py           # ε = 2 matrix completion
python scripts/gen_seed_completion.py         # seed extension for ε ∈ {2, 8}
python scripts/gen_ag_ablation.py             # AG News ablations + no-DP reference
python scripts/gen_seed5_completion.py        # all cells -> 5 seeds
bash scripts/run_grid.sh                      # dual-GPU, skip-if-done
```

> `run_grid.sh` is provided in `scripts/`; it writes one log per run under `results/logs/` and a completion mark under `results/final_marks/`. Expected wall-clock: ≈ 4–5 h on 2× V100 for the complete 182-run grid.

Run the leakage probe (head-gradient inversion, n = 100, initial + trained model):

```bash
python scripts/attack_eval.py --config configs/generated/sst2_dp_fedavg_e4_s0.yaml --samples 100
python scripts/attack_eval.py --config configs/generated/sst2_semdp_e4_s0.yaml   --samples 100
```

## Reproducing Paper Tables and Figures

```bash
python scripts/collect_results.py    # -> results/summary.csv, all_runs.csv
python scripts/collect_revision.py   # -> revision_main.csv, revision_tests.csv, revision_ablations.csv
python scripts/make_figures.py       # -> figures/fig1..fig5 (png + pdf)
python scripts/make_graphical_abstract.py   # -> figures/graphical_abstract.png
```

| Paper object | Source |
|---|---|
| Table II (main comparison) | `results/summary.csv` + `revision_main.csv` |
| Table III (ε matrix) | `results/summary.csv` (ε ∈ {2, 4, 8}) |
| Table IV (leakage probe) | `raw_runs/sst2/*_e4_s0/attack_eval*.json` |
| Table V (ablation) | `revision_ablations.csv` |
| Fig. 1 / 2 / 4 (curves) | `raw_runs/*/*/round_log.csv` |
| Fig. 3 (AG News bars) | `results/summary.csv` |
| Fig. 5 (allocation) | `results/allocation_audit_semdp_e4_s0.json` |
| Graphical abstract | `scripts/make_graphical_abstract.py` |

## Provided Results

All **182** raw runs are included under `results/raw_runs/`, so every number in the paper can be recomputed without re-training:

```bash
python - <<'EOF'
import json, glob, numpy as np
accs = [json.load(open(p))["final_acc"]*100
        for p in sorted(glob.glob("results/raw_runs/sst2/semdp_e4_s*/final.json"))]
a = np.array(accs)
print("SemDP-FL SST-2 @ ε=4: n=%d  %.2f ± %.2f" % (len(a), a.mean(), a.std(ddof=1)))
EOF
```

## Main Results

Test accuracy (%), mean ± std over **5 seeds**, client-level $(4, 10^{-4})$-DP, realised ε = 4.00 ± 0.01 for every run.

| Method | SST-2 | AG News |
|---|---|---|
| DP-FedAvg | 50.94 ± 0.21 | 25.11 ± 0.72 |
| DP-AdaptClip | 50.32 ± 0.99 | 25.24 ± 0.96 |
| DP-FedSAM | 50.85 ± 0.83 | 25.07 ± 0.66 |
| FFA-LoRA+DP | 52.78 ± 3.69 | 25.35 ± 1.15 |
| **SemDP-FL (ours)** | **58.62 ± 6.91** | **50.57 ± 13.03** |
| *Δ vs best baseline* | *+5.8 pp* | *+25.2 pp (p = 0.012)* |
| *No-DP ceiling (reference)* | *76.90 ± 14.58* | *87.97 ± 0.79* |

Leakage probe at equal ε = 4 (n = 100, 400-step relaxed iDLG): on the **trained** model the uniform baseline's embedding recovery rises to cosine **0.161**, while SemDP-FL stays at chance (**−0.033**).

Negative and tied outcomes are reported as measured (see the paper's §VI.H): SST-2 gaps are large-effect but not individually significant at five seeds, SemDP-FL ties FFA-LoRA+DP at ε = 8 and DP-FedAvg at ε = 2.

## Citation

```bibtex
@article{tang2026semdpfl,
  title   = {SemDP-FL: Semantic-Sensitivity-Driven Adaptive Differential Privacy for Federated LoRA Fine-Tuning of Large Language Models},
  author  = {Tang, Song and Jin, Zhigang and Wang, Zhiqiang},
  journal = {Journal of Information Security and Applications},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE). The datasets are governed by their own licences (see [`data/README.md`](data/README.md)).
