"""Federated text-classification data: SST-2 / AG News with Dirichlet non-IID split.

Layout of the data (per the DP-FL public-data assumption, cf. Andrew et al. 2021):
  - a small PUBLIC proxy slice (default 512 samples) held by the server, used for
    sensitivity measurement and clip-norm initialization; it is excluded from all
    client shards (no privacy cost, disclosed in the paper);
  - the remaining training samples are partitioned across N clients with a
    Dirichlet(alpha) label-skew partition (Hsu et al., arXiv:1909.06335);
  - each client shard is capped at `max_samples_per_client` samples.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

TASKS = {
    "sst2": {"hf_name": ("glue", "sst2"), "text_col": "sentence", "label_col": "label",
             "train_split": "train", "test_split": "validation", "num_labels": 2},
    "agnews": {"hf_name": ("ag_news",), "text_col": "text", "label_col": "label",
               "train_split": "train", "test_split": "test", "num_labels": 4},
}


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def _collate(batch, tokenizer, max_len):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)


class FederatedData:
    def __init__(self, task: str, num_clients: int, dirichlet_alpha: float,
                 max_samples_per_client: int, public_samples: int,
                 tokenizer, max_len: int, batch_size: int, seed: int,
                 test_samples: int = 2000):
        spec = TASKS[task]
        self.task, self.num_labels = task, spec["num_labels"]
        self.tokenizer, self.max_len, self.batch_size = tokenizer, max_len, batch_size
        rng = np.random.default_rng(seed)

        from datasets import load_dataset
        ds = load_dataset(*spec["hf_name"])
        tr = ds[spec["train_split"]]
        texts = list(tr[spec["text_col"]])
        labels = np.asarray(tr[spec["label_col"]], dtype=int)

        # public proxy slice (excluded from clients)
        perm = rng.permutation(len(labels))
        pub_idx = perm[:public_samples]
        priv_idx = perm[public_samples:]
        self.public = TextDataset([texts[i] for i in pub_idx],
                                  [int(labels[i]) for i in pub_idx])

        # Dirichlet label-skew partition over the private remainder
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]
        for c in range(self.num_labels):
            c_idx = priv_idx[labels[priv_idx] == c]
            rng.shuffle(c_idx)
            props = rng.dirichlet([dirichlet_alpha] * num_clients)
            cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
            for k, shard in enumerate(np.split(c_idx, cuts)):
                client_idx[k].extend(shard.tolist())
        # guarantee non-empty shards: at low alpha (e.g. 0.5) some seeds (3/4)
        # yield empty client shards -> DataLoader(shuffle) crashes. Move one
        # sample from the largest shard to each empty one (no-op for seeds 0-2).
        sizes = [len(x) for x in client_idx]
        for k in range(num_clients):
            if not client_idx[k]:
                donor = int(np.argmax(sizes))
                client_idx[k].append(client_idx[donor].pop())
                sizes[donor] -= 1
                sizes[k] = 1
        self.client_datasets: list[TextDataset] = []
        for k in range(num_clients):
            idx = client_idx[k]
            rng.shuffle(idx)
            idx = idx[:max_samples_per_client]
            self.client_datasets.append(
                TextDataset([texts[i] for i in idx], [int(labels[i]) for i in idx]))

        te = ds[spec["test_split"]]
        te_texts, te_labels = list(te[spec["text_col"]]), list(te[spec["label_col"]])
        if len(te_labels) > test_samples:
            sub = rng.permutation(len(te_labels))[:test_samples]
            te_texts = [te_texts[i] for i in sub]
            te_labels = [te_labels[i] for i in sub]
        self.test = TextDataset(te_texts, [int(x) for x in te_labels])

    def client_loader(self, cid: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(self.client_datasets[cid], batch_size=self.batch_size,
                          shuffle=shuffle,
                          collate_fn=lambda b: _collate(b, self.tokenizer, self.max_len))

    def public_loader(self, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.public, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: _collate(b, self.tokenizer, self.max_len))

    def test_loader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=64, shuffle=False,
                          collate_fn=lambda b: _collate(b, self.tokenizer, self.max_len))
