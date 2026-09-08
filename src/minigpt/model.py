"""Transformer model: attention through GPT"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v):
    """q, k, v: (B, T, C). Returns (B, T, C)."""
    # B, T, C = q.shape
    C = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(C) # (B, T, C) * (B, C, T) -> (B, T, T)
    weights = F.softmax(scores, dim=-1) # (B, T, T) -> (B, T, T)
    output = weights @ v # (B, T, T) * (B, C, T) -> (B, T, C)
    return output

def causal_attention(q, k, v):
    """Like attention(), but position t only sees positions <= t."""
    """q, k, v: (B, T, C). Returns (B, T, C)."""
    T, C = q.size(-2), q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(C) # (B, T, C) * (B, C, T) -> (B, T, T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1) # (B, T, T) -> (B, T, T)
    output = weights @ v # (B, T, T) * (B, C, T) -> (B, T, C)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd) # split in q,k,v 
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
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
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4*n_embd)
        self.c_proj = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__()
        self.block_size = block_size
        # define layers
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False) 
        # tie weights at beginning and end
        self.lm_head.weight = self.tok_emb.weight
        # init all weights
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)



    def forward(self, idx, targets=None):
        # read off dimensions from token sequence
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long)
        # pass through layers
        x = self.tok_emb(idx) + self.pos_emb(pos) # (B, T, C) + (B, )
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # output depending on targets
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return (logits, loss)



def sample(model, prompt="\n", max_new_tokens=300, temperature=1.0):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.block_size:])
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    model.train()
    return decode(idx[0].tolist())