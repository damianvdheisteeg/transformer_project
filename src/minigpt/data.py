"""Character-level dataset for tiny shakespeare."""
import torch
from pathlib import Path
from typing import Literal
from collections.abc import Sequence
from torch import Tensor


class CharDataset:
    def __init__(self, path: str | Path = "data/input.txt", train_frac: float = 0.9) -> None:
        assert 0.0 < train_frac < 1.0, f"train_frac must be in (0,1), got {train_frac}"
        text = Path(path).read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(train_frac * len(data))
        self.train, self.val = data[:n], data[n:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split: Literal["train", "val"], batch_size: int, block_size: int) -> tuple[Tensor, Tensor]:
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
        return x, y