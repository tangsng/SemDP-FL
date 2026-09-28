# Data

> **No private or human-subject data is used in this project.** Both benchmarks are
> public text classification corpora. We ship a downloader instead of the corpora
> themselves to keep the repository small and to respect upstream licences.

## Datasets

| Dataset | Task | Classes | HF identifier | Official link | Licence |
|---|---|---|---|---|---|
| **SST-2** | binary sentiment | 2 | `nyu-mll/glue` (config `sst2`) | https://huggingface.co/datasets/nyu-mll/glue | see GLUE / Stanford |
| **AG News** | news topic | 4 | `fancyzhx/ag_news` | https://huggingface.co/datasets/fancyzhx/ag_news | see upstream card |

### Download

```bash
python data/prepare_data.py --datasets sst2 agnews
```

This caches the raw splits under `data/cache/` via `datasets.load_dataset`.
Approximate sizes: SST-2 ≈ 67k train / 872 validation; AG News ≈ 120k train / 7.6k test.

## How the data is used

Built on the fly by [`src/datasets/federated_data.py`](../src/datasets/federated_data.py):

- **Clients**: 100 clients, label-skewed partitions via Dirichlet($\alpha = 0.5$) [Hsu et al.],
  ≤ 384 samples per client.
- **Public proxy split**: 512 samples drawn from the datasets' **public training portions**,
  strictly disjoint from all client subsets. Used once for sensitivity measurement
  (`src/utils/sensitivity.py`) and clip-norm initialisation; it consumes no privacy budget.
- **Test split**: 2000 samples (fixed, shared by every method).
- **Backbone**: Qwen2.5-0.5B (frozen fp16), LoRA $r = 8$, $\alpha = 16$ on the $q$/$v$
  projections of all 24 transformer blocks; max sequence length 96.

## Provenance of derived artefacts

Everything under [`results/raw_runs/`](../results/raw_runs/) is *generated* by this
codebase (182 training runs, 5 seeds). Each run directory contains:

| File | Content |
|---|---|
| `final.json` | final test accuracy, realised ε, wall-clock seconds |
| `round_log.csv` | per-round train loss / test accuracy / ε |
| `config.yaml` | the exact configuration used for that run |
| `allocation_audit.json` | (SemDP-FL) per-group multipliers, tiers, measured $a_l$, $\rho$ |
| `attack_eval*.json` | (selected runs) head-gradient inversion probe results |

No client text, embedding, or model checkpoint is stored — only scalar metrics and
per-group allocation statistics, so the released artefacts carry no privacy risk.
