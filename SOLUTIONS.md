# Solutions

If you're here before the corresponding check has made you suffer at least a
little, go back. Otherwise: every solution below is the code that passes the
checks, followed by the part that matters — why it's this and not the
adjacent wrong thing.

---

## 1. Scaled dot-product attention

```python
def attention(q, k, v):
    d = q.size(-1)
    att = q @ k.transpose(-2, -1) / math.sqrt(d)   # (B, T, T)
    att = F.softmax(att, dim=-1)
    return att @ v                                  # (B, T, C)
```

Why `dim=-1`: row $t$ of the score matrix holds position $t$'s affinities to
every position; the softmax must normalize *across those*, making each row a
probability distribution over where to look. Softmax over `dim=-2` normalizes
over queries instead — outputs are still the right shape, which is exactly
why the check tests values and not just shapes.

Why $\sqrt{d}$: for entries with unit variance,
$\mathrm{Var}(q \cdot k) = d$, so raw logits have spread $\sqrt{d}$. The
softmax is a Boltzmann distribution over positions; logit scale is inverse
temperature. Unscaled, temperature drops like $1/\sqrt{d}$ with width and
attention freezes into a hard argmax whose gradient is ~0. Dividing by
$\sqrt{d}$ pins the temperature to $O(1)$ *independent of width* — a
width-invariance imposed by hand at initialization. muP is the same move
applied to training dynamics; you saw its fixed-point character in
`nanogpt-sweeps`, and here is its baby sibling inside a single forward pass.

---

## 2. Causal attention

```python
def causal_attention(q, k, v):
    d, T = q.size(-1), q.size(-2)
    att = q @ k.transpose(-2, -1) / math.sqrt(d)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    att = att.masked_fill(~mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v
```

Mask *scores* with $-\infty$, then softmax: $e^{-\infty} = 0$, and the
remaining weights renormalize over the allowed positions automatically.
Zeroing weights *after* softmax leaves rows summing to less than 1 — the
output becomes a shrunken average, magnitude depending on position, and
nothing crashes. It just trains slightly worse forever. That family of bug —
silent, behaviour-adjacent, review-passing — is the same species as the
bisect regression on the git sheet.

The light-cone test is the right test because it checks the *property*, not
the implementation: any correct masking scheme passes, any leak fails, no
matter how it's written.

Extra 2 answers: `-1e9` differs from `-inf` once logits themselves get
comparable in magnitude (fp16 overflows at 65504, so `-1e9` in half precision
is already broken in a different way — real codebases use dtype-aware
`torch.finfo(dtype).min`). And a fully masked row under `-inf` gives
softmax(-inf, ..., -inf) = NaN, which then poisons everything downstream —
the standard fix in padding-mask code is masking those rows' *outputs* after
the fact.

---

## 3. Multi-head attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = causal_attention(q, k, v)                        # broadcasts over (B, nh)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

Why the transpose is not optional: attention contracts over the *last two*
dims. You need `(…, T, hd)` so that the $T \times T$ score matrix forms
per-head. Without the transpose you'd hand it `(B, T, nh, hd)` and it would
build `nh × nh` score matrices *per position* — attending heads to heads at
fixed time, which is meaningless, produces the right output shape, and trains
to a worse loss without a single error message. The check catches it by
comparing values against a reference built from your module's own weights.

The `.contiguous()` before the final `.view`: `transpose` returns a
non-contiguous stride trick, and `view` requires contiguous memory. (Or use
`.reshape`, which copies when it must.)

Parameter count for `C=24`: `c_attn` = 24·72 + 72 = 1800; `c_proj` = 24·24 +
24 = 600. Total **2400**. The general rule — attention is $4C^2$ plus change,
the MLP is $8C^2$ — makes "how big is this model" a napkin calculation.

---

## 4. Block

```python
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

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
```

The identity test works because pre-LN gives the block the form
$x + f(x) + g(x + f(x))$: silence $f$ and $g$ and what remains is exactly
$x$. Post-LN ($\mathrm{ln}(x + f(x))$) fails it — even with silenced
branches, the trunk still gets normalized, so the map is not the identity.
And that's not a technicality: it means gradients to early layers must pass
*through* every LayerNorm, whereas pre-LN gives an unobstructed identity path
from loss to layer 1. The perturbation-series picture is literal — at depth
$L$, the stream is $x + \sum_{\ell} \delta_\ell$, each $\delta_\ell$ small,
each sublayer reading a normalized snapshot. Post-LN at depth 8 without
warmup will show you the alternative: the extra-credit experiment typically
diverges or crawls, which is the historical reason warmup existed before
pre-LN made it merely helpful.

---

## 5. GPT

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # (B,T,C) + (T,C) broadcasts
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

The init-loss check: with std-0.02 weights, logits are near zero, softmax is
near uniform, and cross-entropy is $-\ln(1/65) = 4.174$. Default PyTorch
embedding init ($\mathcal N(0,1)$) puts you far above; a leak puts you below.
The asymmetry in how you should *feel* about those two failures is the
lesson: too-high is a boring bug, too-low is a lying metric. You'll meet
too-low again as a 0.99 AUC on a mammography CV fold.

Weight tying, beyond parameter savings: the embedding maps token → vector,
the head maps vector → token scores. Tying asserts these are the same
geometry — a token's score is its embedding's dot product with the residual
stream, so "predicting token $i$" and "meaning token $i$" share
representation. Empirically it regularizes small models and is standard up
through GPT-2.

Bigram baseline (extra): count character-pair frequencies, loss =
$-\mathbb{E}\log p(c_{t+1}|c_t)$. On tiny shakespeare it lands around 2.45.
Your transformer at 2000 steps should beat it — if it doesn't, it has learned
unigram/bigram statistics and is ignoring its context window.

---

## 6. Training loop

```python
if WANDB:
    wandb.init(project="transformer-lab", name="baseline-flat-lr", config=config)

opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])
for step in range(config["max_steps"]):
    xb, yb = get_batch("train", config["batch_size"], config["block_size"])
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if WANDB:
        wandb.log({"loss/train_batch": loss.item()}, step=step)
    if step % config["eval_interval"] == 0 or step == config["max_steps"] - 1:
        L = estimate_loss(model)
        history["step"].append(step)
        history["train"].append(L["train"])
        history["val"].append(L["val"])
        if WANDB:
            wandb.log({"loss/train": L["train"], "loss/val": L["val"]}, step=step)

if WANDB:
    wandb.finish()
```

Two logging choices worth making deliberate: the noisy per-batch loss and the
smoothed eval loss go to *different keys* (`loss/train_batch` vs
`loss/train`) so W&B plots them as separate curves rather than a mess on one
axis; and `step=step` pins everything to one x-axis so runs of different
lengths overlay correctly.

Reference points from a verified run of exactly this config (0.8M params,
batch 32, block 64, lr 3e-4): loss 4.17 → ~2.64 by step 100, ~2.45 by 200,
~2.37 by 300, and with 2000 steps you should land in the low 1.x with clearly
Shakespeare-shaped output. Train and val track each other closely — at 0.8M
parameters against 1.1M characters you are nowhere near memorizing, which is
itself worth registering: overfitting is a *ratio*, not a property of models.

Gradient-norm extra: `gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
float("inf"))` measures without clipping; log it, then clip at 1.0 and
compare. The norm spikes early (the model is reorganizing) and settles; when
a future large run of yours NaNs, this chart is where the autopsy starts.

---

## 7. Learning rate

**7a.**

```python
lrs = torch.logspace(-4.5, -1, 8)
final_losses = []
for lr in lrs:
    torch.manual_seed(1337)
    m = GPT(vocab_size, 64, 4, 4, 128)
    if WANDB:
        wandb.init(project="transformer-lab", group="lr-range-test",
                   name=f"lr-{lr:.1e}", config=dict(lr=lr.item(), steps=100))
    opt = torch.optim.AdamW(m.parameters(), lr=lr.item())
    for step in range(100):
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if WANDB:
            wandb.log({"loss/train_batch": loss.item()}, step=step)
    final_losses.append(loss.item())
    if WANDB:
        wandb.finish()
```

In the W&B UI: filter to the `lr-range-test` group, color by `config.lr`, and
the phase diagram appears as a fan of curves — frozen ones flat at the top,
the basin descending fastest, the too-hot ones jittering sideways. The final-
loss scatter you plot locally is a 1-d projection of that picture.

Expected shape (verified, 30-step version): monotone improvement up to a
basin around `3e-4`–`1e-3`, then degradation from `3e-3` up. The basin is
roughly one decade wide, and the optimum sits within a factor of ~3 of the
cliff. That proximity is structural, not bad luck: the loss surface rewards
the largest step size that doesn't overshoot the local curvature, so tuned
LRs always live near the edge of stability. Adam makes the cliff a slope
rather than a wall (per-parameter normalization catches divergence), but the
phase diagram keeps its shape.

**7b.**

```python
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    min_lr = max_lr / 10
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# in the loop, before opt.step():
for g in opt.param_groups:
    g["lr"] = get_lr(step)
# and log it with the loss:
#   wandb.log({"loss/train_batch": loss.item(), "lr": get_lr(step)}, step=step)
```

Endpoints to eyeball on your plot: `get_lr(0) ≈ max_lr/100`, `get_lr(99) =
max_lr`, `get_lr(1999) = max_lr/10`. Why each phase exists: warmup because at
init the Adam second-moment estimates are garbage and the loss surface is at
its most anisotropic — big steps in random directions early are how runs die
in the first 100 steps; decay because late in training you want to settle
into a minimum rather than bounce around it at fixed step size (annealing, in
the most literal sense). The payoff vs flat LR at this scale is modest but
consistently visible in the tail of the curve; at real scale it's not
optional.

Batch-size extra: doubling batch halves gradient noise, which shifts the
usable basin toward larger LRs (linear-ish scaling at small batch). And now
the muP bridge is concrete: SP's basin *moves* when you change width; muP's
doesn't. You have a range-test harness and a sweeps repo — checking that
claim at three widths is an evening, and it is exactly the isoFLOP muscle
the Chinchilla replication needs.

---

## 8. Notebook → repository

`pyproject.toml`:

```toml
[project]
name = "minigpt"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch"]

[project.optional-dependencies]
dev = ["pytest", "matplotlib", "jupyter"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

`config.py`:

```python
from dataclasses import dataclass

@dataclass
class TrainConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    batch_size: int = 32
    max_steps: int = 2000
    max_lr: float = 1e-3
    warmup_steps: int = 100
    eval_interval: int = 100
    seed: int = 1337
```

`tests/test_model.py`, the property tests reborn — note they're *yours* now,
written against the library import path:

```python
import math, torch, pytest
from minigpt.model import GPT, Block, MultiHeadAttention

def test_causality():
    torch.manual_seed(0)
    m = GPT(65, 32, n_layer=2, n_head=2, n_embd=32).eval()
    idx = torch.randint(0, 65, (2, 32))
    idx2 = idx.clone(); idx2[:, 16:] = torch.randint(0, 65, (2, 16))
    a, b = m(idx), m(idx2)
    assert torch.allclose(a[:, :16], b[:, :16], atol=1e-4)

def test_block_is_identity_when_branches_silenced():
    torch.manual_seed(0)
    b = Block(32, 2).eval()
    with torch.no_grad():
        b.attn.c_proj.weight.zero_(); b.attn.c_proj.bias.zero_()
        b.mlp.c_proj.weight.zero_();  b.mlp.c_proj.bias.zero_()
    x = torch.randn(2, 8, 32)
    assert torch.allclose(b(x), x, atol=1e-6)

def test_init_loss_is_ln_vocab():
    torch.manual_seed(0)
    m = GPT(65, 32, 2, 2, 32).eval()
    idx = torch.randint(0, 65, (8, 32))
    _, loss = m(idx, torch.randint(0, 65, (8, 32)))
    assert abs(loss.item() - math.log(65)) < 0.25

def test_can_overfit_one_batch():
    torch.manual_seed(0)
    m = GPT(65, 32, 2, 2, 64)
    idx = torch.randint(0, 65, (4, 32)); tgt = torch.randint(0, 65, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(200):
        _, loss = m(idx, tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert loss.item() < 2.0
```

`scripts/train.py` skeleton:

```python
import argparse
from dataclasses import fields
from minigpt.config import TrainConfig
from minigpt.train import train

p = argparse.ArgumentParser()
for f in fields(TrainConfig):
    p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
p.add_argument("--wandb", action="store_true")
args = p.parse_args()
use_wandb = args.wandb; del args.wandb
cfg = TrainConfig(**vars(args))
train(cfg, use_wandb=use_wandb)
```

and inside `train()`:

```python
import dataclasses

def train(cfg, use_wandb=False):
    if use_wandb:
        wandb.init(project="transformer-lab", config=dataclasses.asdict(cfg))
    ...
```

`dataclasses.asdict(cfg)` is the point of the whole config exercise landing:
the object that configures the run and the record of the run are the same
object. When someone at Kaiko asks "what exactly did run 47 use," the answer
is a click, not an archaeology project.

A commit history that reads like a story (each one small, each one green):

```
init: empty package skeleton, pyproject, gitignore
extract model.py from notebook (verbatim, no changes)
extract data.py from notebook
add TrainConfig dataclass; remove magic numbers from train loop
port property tests to pytest (causality, identity, init-loss, overfit)
add scripts/train.py CLI
gut notebook: import from minigpt, keep plots and sampling
tag v0.1.0
```

Step 2's "verbatim, no changes" rule is the load-bearing one. A move-commit
and a change-commit are individually trivial to review; fused, they hide
anything. This is the same principle as cherry-picking the fix without the
scratch work.

The three-command test — fresh venv, `pip install -e ".[dev]"`, `pytest -q`,
`python scripts/train.py --max-steps 200` — is the definition of done because
it's the *reproducibility* criterion: anyone (including you-in-three-months,
and including a vast.ai instance in the bootstrap script of your runbook) can
go from clone to training run with no folklore. When the GRPO capstone
starts, this repo structure is the template it clones.
