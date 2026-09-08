# Transformer lab

You are going to build a GPT from an empty function to a trained model, then
promote the whole thing from a notebook into a real repository. Nothing is
pre-built: attention, the causal mask, multi-head, the block, the model, the
training loop, the learning-rate schedule — all yours. What *is* provided is a
check for every step that fails loudly, with a hint about what kind of wrong
you are, until your implementation is right.

Files:

```
transformer_lab.ipynb   the workbench — exercises 1–7 happen here
checks.py               the referee — import it, don't read it
get_data.py             downloads tiny shakespeare (1.1 MB, run once)
EXERCISES.md            this file
SOLUTIONS.md            full code + the why, per exercise
```

Everything runs on a laptop CPU. The model is ~0.8M parameters; the long runs
are ~5 minutes. No GPU, no cloud, no excuses about compute — this sheet is
about *understanding*, and the vast.ai runbook is for the projects after it.

**Ground rule:** do not open `SOLUTIONS.md` until you have tried, and do not
*read* `checks.py` (running it is the whole point; its bottom half contains
reference implementations behind a marked spoiler line). Getting stuck against
a failing check and then having the insight is how this sticks. Reading first
just produces the feeling of understanding.

Pacing: exercises 0–5 in one sitting (~2h), 6–7 in another (~2h, mostly
waiting on training runs — good time to predict the outcomes before you see
them), 8 in a third (~2h, terminal only).

---

## 0. Setup — do not skip the environment step

You know how this goes. Make the environment explicit before anything else:

```bash
cd transformer-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch matplotlib jupyter wandb
python get_data.py
wandb login        # or `wandb offline` to log locally and sync later
jupyter notebook transformer_lab.ipynb
```

`python -m pip`, always. If the first notebook cell can't import torch, fix
the kernel *now* (Kernel → Change Kernel → the `.venv` one), not after an hour
of confusion. You've met this failure mode before; greet it by name and move on.

Run the first cell. Understand the two printed shapes before continuing:
`x` and `y` are the *same tokens shifted by one*. That single line is the
entire supervision signal of language modelling. Everything else is
architecture.

---

## 1. Scaled dot-product attention

$$\mathrm{Att}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Three lines. Implement it in the notebook, pass `check_attention`.

The check does something worth noticing: it feeds `v = I` so that your output
rows *are* your attention weights, then verifies they're non-negative and sum
to 1. When a check fails, read the message — each one diagnoses a specific
bug, including the classic "forgot the $\sqrt{d_k}$", which it detects by
matching your output against the unscaled version.

**Extra:** run the variance cell below the check. The dot product of two
random $d$-dim vectors has variance $d$, so unscaled logits grow like
$\sqrt{d}$ and the softmax saturates into an argmax with vanishing gradients.
The $1/\sqrt{d_k}$ is a temperature chosen so the Boltzmann weights stay
$O(1)$ at any width. Sound familiar? It's the same class of problem muP
solves for the learning rate — width-invariance by construction. Same
disease, different organ.

---

## 2. The causal mask

Position $t$ may only see positions $\le t$. Copy your attention, add the
mask, pass `check_causal_attention`.

The check is a *light-cone test*: it perturbs tokens at positions $\ge 4$ and
asserts outputs at positions $< 4$ are bit-identical. Causality is a testable
property, not a vibe. It also catches the two classic bugs by name: mask
orientation flipped (`triu` where you meant `tril`), and masking *after*
softmax instead of before (which silently breaks normalization — the check
explains why zeroing weights post-softmax leaves rows that don't sum to 1).

**Extra 1:** run the `influence_matrix` cell — it computes
$M_{ts} = \|\partial\,\mathrm{out}_t / \partial x_s\|$ by autograd and plots
it. That matrix is the network's Green's function, and for a causal map it
must be lower triangular. Keep this tool; pointing it at the full model later
is a one-liner.

**Extra 2:** people often use `-1e9` instead of `-inf` in the mask. Find a
case where that difference is observable. Then find out what happens with
`-inf` when an *entire row* is masked (this exact bug produces NaNs in
padding-mask code in the wild).

---

## 3. Multi-head attention

One attention pattern per layer is one hypothesis about which positions
matter. $n_h$ heads run $n_h$ patterns in parallel over $d/n_h$-dim slices,
then a learned $W_O$ mixes them.

Use the prescribed layout — it's nanoGPT's, so you're learning a real
codebase's conventions for free: one combined `c_attn: Linear(C, 3C)`, one
`c_proj: Linear(C, C)`. The entire exercise is reshape gymnastics:

```
(B, T, 3C) → split → (B, T, nh, hd) → transpose(1, 2) → (B, nh, T, hd)
```

Your exercise-2 function already works on 4-d tensors unchanged, because
matmul and softmax broadcast over leading batch dims. Before moving on,
answer: why is the `transpose(1, 2)` not optional? What would attention
mix together without it? (This is *the* multi-head bug. The check catches it
and says so.)

**Extra:** predict the parameter count for `C=24` with biases by hand, then
verify with `sum(p.numel() ...)`. Being able to count parameters on a napkin
is how you sanity-check model configs forever after.

---

## 4. The transformer block

```
x = x + attn(ln_1(x))
x = x + mlp(ln_2(x))
```

Plus an MLP: `Linear(C, 4C)` → GELU → `Linear(4C, C)`.

The check contains the best test on this sheet: it zeroes both branches'
output projections and asserts the block is then *exactly* the identity map.
If you put LayerNorm on the trunk instead of the branch — the single most
common way to mis-build this — the test fails and tells you what you did.

The picture worth internalizing: the residual stream is an identity plus a
sum of small corrections; each sublayer reads a normalized copy of the stream
and writes a perturbation back. Depth is a perturbation series, not a
composition of transformations. That's *why* deep transformers train at all.

**Extra:** build a post-LN variant (`ln(x + attn(x))`) and train both at
`n_layer=8` with the exercise-6 loop. Watch which one needs warmup to
survive. In 2017 everyone used post-LN and everyone needed warmup; this
experiment is the reason the field moved.

---

## 5. Assemble the GPT

Token embedding + learned position embedding, a stack of blocks, final
LayerNorm, and a bias-free `lm_head` whose weight is *tied* to the token
embedding. Init everything `N(0, 0.02²)`.

Before running `check_gpt`, predict the untrained loss for a 65-token
vocabulary. The check will measure you against $\ln 65 = 4.17$ and its
failure message distinguishes the two ways to miss: too high means your init
is too hot; *below* it means information is leaking — a broken mask or an
unshifted target — because a model that knows nothing cannot legitimately
beat uniform guessing. A suspiciously good number is a bug until proven
otherwise. That instinct is the same one the mammography project will drill
on low-prevalence metrics.

Then `check_overfit`: 150 steps on a single batch must collapse the loss
toward zero. This is the most informative smoke test in deep learning — a
model that cannot memorize eight sequences cannot learn anything, and every
plumbing bug (optimizer over the wrong parameters, missing `zero_grad`,
unshifted targets) fails it instantly.

**Extra:** why does weight tying make sense, beyond saving parameters? And:
what loss would a bigram model get on this data? Estimate it from character
statistics with three lines of Python, and keep the number — it's your
"is my transformer actually using context?" baseline for exercise 6.

---

## 6. The first real training run

Write the loop yourself: AdamW at `3e-4`, `zero_grad(set_to_none=True)` →
`backward()` → `step()`, periodic train/val evaluation into `history`. Then
2000 steps, ~5 minutes, and your first *real* loss curve — you already know
the difference between this and decorative synthetic noise.

While it runs, predict: where will val loss be at step 2000? (Your bigram
number from exercise 5 is the mark to beat.)

Then sample from the model — at several checkpoints if you rerun — and watch
the phase transitions: noise → word-shaped noise → words → dialogue-shaped
structure with `CHARACTER:` headers. The loss curve is smooth; the
capabilities are not.

**Extra 1:** temperature-sweep the sampler over `[0.3, 0.8, 1.0, 1.5]`.
Sampling *is* a Boltzmann distribution and temperature is exactly $1/\beta$:
watch text freeze into repetition and melt into noise.

**Extra 2:** log the pre-clip gradient norm every step and plot it. Then add
`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` and compare. Grad
norm is the first chart you'll stare at when a real run diverges at 3am.

---

## 7. The learning rate

**7a — range test.** Log-spaced LRs from `1e-4.5` to `1e-1`, a fresh
identically-seeded model per LR, 100 steps each — each LR its own W&B run
under `group="lr-range-test"`, so the phase diagram lives in the UI as
overlaid loss curves as well as in your local final-loss plot on log-x. You are mapping a phase diagram, and it has the shape you'd guess:
frozen on the left, a basin, and a cliff on the right past a critical LR.
Note two things: how *wide* the basin is (in decades), and how close to the
cliff the optimum sits. That second observation is why LR tuning matters and
why divergence is always nearby.

**7b — warmup + cosine.** Implement `get_lr(step)`: linear warmup over 100
steps to `max_lr`, cosine decay to `max_lr/10` at step 2000. Plot the
schedule *before* training with it — eyeballing the schedule first is a habit
that has saved more runs than any other. Then train with `max_lr` chosen from
your own phase diagram, against the flat-`3e-4` curve from exercise 6. Log
the LR alongside the loss so the schedule is visible in the run, and put
`schedule="cosine"` vs `"constant"` in the configs — the two runs should
differ by exactly one config field. Same seed, both curves, one plot.

**Extra:** double the batch size, redo 7a, and see how the basin moves.
Then look at your `nanogpt-sweeps` repo with new eyes: muP's claim is
precisely that the basin's *location* stops moving with width. You now have
the tooling to check that claim yourself — that's the bridge to the
Chinchilla-replication project.

---

## 8. Notebook → repository

The notebook was the right tool for exercises 1–7: you were looking at
things. It is the wrong tool for what the code has become: five classes, a
training loop, and a config — things that need to be *right*, versioned, and
importable. Time to promote. Terminal only; the notebook survives, demoted to
an interface.

Target:

```
minigpt/
├── pyproject.toml
├── README.md
├── .gitignore                (checkpoints, .venv, __pycache__, wandb/, *.ipynb_checkpoints)
├── src/minigpt/
│   ├── __init__.py
│   ├── model.py              attention → GPT, nothing else
│   ├── data.py               CharDataset: encode/decode/get_batch
│   ├── config.py             @dataclass TrainConfig — every magic number lives here
│   └── train.py              the loop + get_lr, callable as a function
├── scripts/train.py          argparse CLI: python scripts/train.py --lr 1e-3 --max-steps 2000
├── tests/test_model.py       the checks, reborn as pytest
└── notebooks/transformer_lab.ipynb
```

The steps, in the order that keeps you honest:

1. `git init` first, and commit after *every* step below. The history should
   read like a story: `extract model.py from notebook`, `add config
   dataclass`, `port checks to pytest`. You built this discipline on the
   disaster repo; this is where it becomes routine.
2. Extract `model.py` and `data.py` from the notebook cells. Resist improving
   anything while moving it — moves and changes in one commit is how diffs
   become unreviewable.
3. Write `pyproject.toml` (src layout), then `python -m pip install -e ".[dev]"`.
   Test the promotion: open Python *anywhere* and `from minigpt.model import GPT`.
4. Port the checks into `tests/test_model.py` as real pytest tests —
   causality, block-identity, init-loss, overfit-one-batch. You are allowed
   to read `checks.py` now; you've earned it, and porting its property tests
   teaches you to write your own. `pytest -q` green is the definition of done.
5. `config.py`: one dataclass, every hyperparameter, no magic numbers left in
   `train.py`. Then `scripts/train.py` with argparse overriding config fields.
6. Gut the notebook: delete the class definitions, `from minigpt.model import
   GPT`, keep the plots and sampling. The notebook is now what notebooks are
   for — an experiment surface over a tested library.
7. Tag it: `git tag v0.1.0`.

The test for whether you did it right: `rm -rf` your venv, recreate it,
`pip install -e .`, `pytest -q`, `python scripts/train.py --max-steps 200`.
Three commands, from nothing to a training run. That's what "repository"
means.

**Extra 1:** the W&B logging from exercises 6–7 comes along in the
promotion — but put it behind a `--wandb` flag in the CLI and pass the whole
`TrainConfig` as the run config via `dataclasses.asdict(cfg)`. One source of
truth for hyperparameters, in the code and in the tracker. Add grad norm to
what you log.
**Extra 2:** `nbstripout` as a git filter so notebook outputs never pollute
diffs — your future reviewers thank you.
**Extra 3:** run your 7a LR range test through the CLI with a shell loop
instead of a notebook cell. Feel the difference in repeatability.

---

## When things break

| Symptom | First suspect |
|---|---|
| Loss is NaN | LR too high; or an all-masked softmax row (`-inf` everywhere → NaN) |
| Loss stuck at ~4.17 | Gradients not flowing: optimizer over wrong params, missing `backward()`, or LR ~0 |
| Loss *below* 4.17 at init | Leak: mask broken or targets not shifted — celebrate nothing, investigate |
| Overfit test won't collapse | Missing `zero_grad`; or `model.eval()` left on |
| Multi-head check: causal but wrong values | Head split without the `transpose(1, 2)` |
| Block check: not identity | LayerNorm on the trunk instead of the branch |
| `ModuleNotFoundError` in notebook | Wrong kernel — you know this one; fix the interpreter, not the code |
| Everything is slow | You're fine. 0.8M params on CPU is ~7 steps/s. Patience is part of the sheet |

## Deliberately not on this sheet

Dropout, `F.scaled_dot_product_attention` / flash attention, KV-cache
generation, mixed precision, `torch.compile`, GPUs. Each is one sentence to
*use* and a project to *understand*, and they'd blur the target here, which
is the transformer itself. The GPU material is exactly what the mammography
project is for; and when you get to the GRPO capstone, the model you'll be
post-training is — layer for layer — the thing you just built by hand.
