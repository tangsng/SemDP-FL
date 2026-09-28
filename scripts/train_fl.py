"""Main entry: one FL-DP experiment = one config file.

Usage:  python3 scripts/train_fl.py --config configs/generated/<name>.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.datasets.federated_data import FederatedData          # noqa: E402
from src.models.modeling import build_bundle                    # noqa: E402
from src.servers.dp_adaclip import DPAdaClipServer              # noqa: E402
from src.servers.dp_fedavg import DPFedAvgServer                # noqa: E402
from src.servers.dp_fedsam import DPFedSAMServer                # noqa: E402
from src.servers.dp_ffalora import DPFFALoRAServer              # noqa: E402
from src.servers.semdp import SemDPServer                       # noqa: E402

SERVERS = {
    "dp_fedavg": DPFedAvgServer,
    "dp_adaclip": DPAdaClipServer,
    "dp_fedsam": DPFedSAMServer,
    "dp_ffalora": DPFFALoRAServer,
    "semdp": SemDPServer,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp = cfg["experiment"]
    out_dir = exp["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_a = exp["method"] == "dp_ffalora"
    bundle = build_bundle(cfg, device, freeze_a=freeze_a)

    fc = cfg["fl"]
    data = FederatedData(
        task=exp["dataset"], num_clients=fc["num_clients"],
        dirichlet_alpha=fc["dirichlet_alpha"],
        max_samples_per_client=fc["max_samples_per_client"],
        public_samples=fc["public_samples"], tokenizer=bundle.tokenizer,
        max_len=cfg["model"]["max_len"], batch_size=cfg["train"]["batch_size"],
        seed=exp["seed"], test_samples=fc.get("test_samples", 2000))

    server = SERVERS[exp["method"]](cfg, bundle, data, device, out_dir)
    final = server.run()
    print("FINAL_JSON " + json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
