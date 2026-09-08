"""Character-level dataset for tiny shakespeare."""
import torch


class CharDataset:
    def __init__(self, path="input.txt", train_frac=0.9):
        text = open(path).read()
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(train_frac * len(data))
        self.train, self.val = data[:n], data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split, batch_size, block_size):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
        return x, y