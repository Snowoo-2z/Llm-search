"""Chargement des données tokenisées (memmap uint16)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, data_dir: str | Path, device: str = "cpu"):
        self.dir = Path(data_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} absent : lancer d'abord `python -m blackcode.corpus prepare`"
            )
        self.meta = json.loads(meta_path.read_text())
        self.train = np.memmap(self.dir / "train.bin", dtype=np.uint16, mode="r")
        val_path = self.dir / "val.bin"
        self.val = (np.memmap(val_path, dtype=np.uint16, mode="r")
                    if val_path.exists() and val_path.stat().st_size > 0 else self.train)
        self.device = device

    def get_batch(self, split: str, batch_size: int, seq_len: int):
        data = self.train if split == "train" else self.val
        if len(data) < seq_len + 2:
            raise ValueError(f"{split} ne contient que {len(data)} tokens")
        ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
        x = np.stack([data[i:i + seq_len] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + 1 + seq_len] for i in ix]).astype(np.int64)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    @property
    def train_tokens(self) -> int:
        return int(self.meta["train_tokens"])

    @property
    def val_tokens(self) -> int:
        return int(self.meta["val_tokens"])
