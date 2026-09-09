"""Transformer model: attention through GPT"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from minigpt.config import ModelConfig

def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """q, k, v: (B, T, C). Returns (B, T, C)."""
    # B, T, C = q.shape
    C = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(C) # (B, T, C) * (B, C, T) -> (B, T, T)
    weights = F.softmax(scores, dim=-1) # (B, T, T) -> (B, T, T)
    output = weights @ v # (B, T, T) * (B, C, T) -> (B, T, C)
    return output

def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Like attention(), but position t only sees positions <= t.
    q, k, v: (B, T, C). Returns (B, T, C)."""
    T, C = q.size(-2), q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(C) # (B, T, C) * (B, C, T) -> (B, T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1) # (B, T, T) -> (B, T, T)
    output = weights @ v # (B, T, T) * (B, C, T) -> (B, T, C)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd={n_embd} not divisible by n_head={n_head}"
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd) # split in q,k,v 
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        # dimensions, hd is size of head
        B, T, C = x.shape
        hd = C // self.n_head

        # decompose (B, T, 3C) into q,k,v of (B, T, C)
        q, k, v = self.c_attn(x).split(C, dim=2)

        # decompose (B, T, C) into (B, T, nh, hd) into (B, nh, T, hd)
        q = q.view(B, T, self.n_head, hd).transpose(1,2)
        k = k.view(B, T, self.n_head, hd).transpose(1,2)
        v = v.view(B, T, self.n_head, hd).transpose(1,2)

        # compute causal attention using broadcasting
        y = causal_attention(q, k, v) # (B, nh, T, hd)

        # merge back into (B, T, C)
        y = y.transpose(1,2).reshape(B, T, C)

        output = self.c_proj(y)

        return output

class MLP(nn.Module):
    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4*n_embd)
        self.c_proj = nn.Linear(4*n_embd, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.vocab_size > 0, f"vocab size must be set from the dataset, got {cfg.vocab_size}"
        self.cfg = cfg
        self.block_size = cfg.block_size
        # define layers
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False) 
        # tie weights at beginning and end
        self.lm_head.weight = self.tok_emb.weight
        # init all weights
        self.apply(self._init)

    def _init(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)



    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # read off dimensions from token sequence
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        # pass through layers
        x = self.tok_emb(idx) + self.pos_emb(pos) # (B, T, C) + (B, )
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # output depending on targets
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int = 300, temperature: float = 1.0, top_k: int | None = None):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])
            logits = logits[:, -1] / temperature
            if top_k is not None:
                thresh = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        if was_training:
            self.train()
        return idx