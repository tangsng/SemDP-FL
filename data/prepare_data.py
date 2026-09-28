"""Download and cache the public benchmarks used by SemDP-FL.

Datasets (no private data is involved):
  - SST-2  : GLUE benchmark, binary sentiment classification
             https://huggingface.co/datasets/nyu-mll/glue  (subset: sst2)
  - AG News: 4-class news topic classification
             https://huggingface.co/datasets/fancyzhx/ag_news

This script only *downloads/caches* the datasets (via `datasets.load_dataset`)
and prints a summary. The federated Dirichlet partitioning and the public proxy
split are built on the fly by `src/datasets/federated_data.py`.

Usage:
    python data/prepare_data.py --datasets sst2 agnews
    python data/prepare_data.py --datasets sst2 --cache_dir ./data/cache
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset

DATASETS = {
    "sst2": {
        "hf_name": "nyu-mll/glue",
        "config": "sst2",
        "split_map": {"train": "train", "test": "validation"},
        "text_col": "sentence",
        "label_col": "label",
        "num_labels": 2,
        "url": "https://huggingface.co/datasets/nyu-mll/glue",
    },
    "agnews": {
        "hf_name": "fancyzhx/ag_news",
        "config": None,
        "split_map": {"train": "train", "test": "test"},
        "text_col": "text",
        "label_col": "label",
        "num_labels": 4,
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["sst2", "agnews"],
                    choices=list(DATASETS))
    ap.add_argument("--cache_dir", default=os.path.join("data", "cache"))
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    for key in args.datasets:
        spec = DATASETS[key]
        print(f"\n=== {key} ({spec['url']}) ===")
        for local_split, hf_split in spec["split_map"].items():
            ds = load_dataset(spec["hf_name"], spec["config"],
                              split=hf_split, cache_dir=args.cache_dir)
            print(f"  {local_split:<5s} n={len(ds):>7d} "
                  f"classes={spec['num_labels']} cols={ds.column_names}")
    print("\nDone. Partitions are built by src/datasets/federated_data.py "
          "(Dirichlet alpha=0.5, 100 clients, <=384 samples/client; "
          "512-sample public proxy split drawn from the public training portion).")


if __name__ == "__main__":
    main()
