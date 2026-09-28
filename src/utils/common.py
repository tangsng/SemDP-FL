"""Seeding, CSV logging and statistics helpers (mean +/- std, ddof=1)."""
from __future__ import annotations

import csv
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CsvLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(fieldnames)

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fieldnames])


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Sample mean and sample std (ddof=1)."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
