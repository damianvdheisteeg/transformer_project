"""Checks for the transformer lab.

Import from the notebook and call the check for the exercise you are on:

    from checks import *
    check_attention(attention)

Every check either raises AssertionError with a hint, or prints PASS.

Do not *read* this file until you have passed its checks -- the bottom
contains reference implementations (clearly marked). Running it is fine;
reading it is SOLUTIONS.md territory.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def _pass(name):
    print(f"PASS: {name}")


# ----------------------------------------------------------------------
# Exercise 1: scaled dot-product attention (single head, no mask)
# fn(q, k, v) -> out, all tensors (B, T, C)
# ----------------------------------------------------------------------
def check_attention(fn):
    # T != C on purpose: a square test case cannot distinguish scaling by
    # sqrt(d_k) from scaling by sqrt(T), and only one of those is correct.
    B, T, C = 2, 9, 5
    q, k, v = torch.randn(3, B, T, C).unbind(0)

    out = fn(q, k, v)
    assert out.shape == (B, T, C), (
        f"expected output shape {(B, T, C)}, got {tuple(out.shape)}"
    )

    # weights must be a proper distribution over positions:
    # with v = identity, each output row IS the row of attention weights
    v_eye = torch.eye(T).expand(B, T, T)
    w = fn(q, k, v_eye)
    assert (w >= -1e-6).all(), (
        "attention weights went negative -- softmax missing?"
    )
    row_sums = w.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
        "attention weights do not sum to 1 across positions -- "
        "softmax over the wrong dimension?"
    )

    ref = _ref_attention(q, k, v)
    if not torch.allclose(out, ref, atol=1e-5):
        ref_unscaled = _ref_attention(q, k, v, scale=False)
        if torch.allclose(out, ref_unscaled, atol=1e-5):
            raise AssertionError(
                "matches UNSCALED attention -- you forgot the 1/sqrt(d_k)"
            )
        att_T = (q @ k.transpose(-2, -1)) / math.sqrt(T)
        ref_T = F.softmax(att_T, dim=-1) @ v
        if torch.allclose(out, ref_T, atol=1e-5):
            raise AssertionError(
                "you are dividing by sqrt(T), the sequence length, instead of "
                "sqrt(C), the channel dimension. The scale must depend on the "
                "CONTRACTED dimension -- the number of terms summed in each "
                "dot product -- not on how many dot products you compute. "
                "(This passes on square test cases. It should not have "
                "reached you as one.)"
            )
        raise AssertionError(
            "output does not match reference attention "
            "(check: scores = q @ k^T / sqrt(d), softmax over last dim, then @ v)"
        )
    _pass("attention -- shapes, normalization, and values all correct")


# ----------------------------------------------------------------------
# Exercise 2: causal attention.  fn(q, k, v) -> out, causal.
# ----------------------------------------------------------------------
def check_causal_attention(fn):
    B, T, C = 2, 8, 6
    q, k, v = torch.randn(3, B, T, C).unbind(0)
    out = fn(q, k, v)
    assert out.shape == (B, T, C), f"wrong shape {tuple(out.shape)}"

    # light-cone test: changing the future must not change the past
    t_split = 4
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, t_split:] += torch.randn_like(q2[:, t_split:])
    k2[:, t_split:] += torch.randn_like(k2[:, t_split:])
    v2[:, t_split:] += torch.randn_like(v2[:, t_split:])
    out2 = fn(q2, k2, v2)
    assert torch.allclose(out[:, :t_split], out2[:, :t_split], atol=1e-5), (
        "perturbing tokens at positions >= 4 changed the output at "
        "positions < 4 -- information is flowing backwards in time. "
        "Check the mask orientation (tril, not triu)."
    )

    # ...but the past must still reach the present (mask not too aggressive)
    assert not torch.allclose(out[:, -1], out2[:, -1], atol=1e-5), (
        "changing the inputs did not change the last position's output -- "
        "is your mask removing everything?"
    )

    # masked weights: strictly zero above the diagonal, rows still sum to 1
    v_eye = torch.eye(T).expand(B, T, T)[..., :C] if C == T else None
    w = fn(q[:1], k[:1], torch.eye(T).expand(1, T, T))
    upper = w.triu(1)
    assert upper.abs().max() < 1e-6, (
        "nonzero attention weight on a future position -- mask applied "
        "AFTER softmax instead of before?"
    )
    rows = w.sum(-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-4), (
        "rows no longer sum to 1 -- mask with -inf before softmax, "
        "don't zero out weights after"
    )

    ref = _ref_attention(q, k, v, causal=True)
    assert torch.allclose(out, ref, atol=1e-5), (
        "causality holds but values are off -- compare against "
        "softmax(mask(q k^T / sqrt(d))) v"
    )
    _pass("causal attention -- the light cone is respected")


# ----------------------------------------------------------------------
# Exercise 3: multi-head attention module.
# Expects: module.n_head, module.c_attn (Linear C->3C), module.c_proj (Linear C->C)
# forward(x) -> (B, T, C), causal.
# ----------------------------------------------------------------------
def check_multihead(module):
    for attr in ("n_head", "c_attn", "c_proj"):
        assert hasattr(module, attr), (
            f"module needs attribute `{attr}` (use the prescribed layout: "
            "one combined qkv Linear called c_attn, output Linear c_proj)"
        )
    C = module.c_proj.out_features
    assert module.c_attn.out_features == 3 * C, (
        "c_attn must map C -> 3C (q, k, v in one matmul)"
    )
    assert C % module.n_head == 0, "n_head must divide n_embd"

    B, T = 2, 10
    x = torch.randn(B, T, C)
    module.eval()
    out = module(x)
    assert out.shape == (B, T, C), f"wrong shape {tuple(out.shape)}"

    # causality on the full module
    x2 = x.clone()
    x2[:, 5:] += torch.randn_like(x2[:, 5:])
    out2 = module(x2)
    assert torch.allclose(out[:, :5], out2[:, :5], atol=1e-5), (
        "module is not causal -- did the mask survive the head reshape?"
    )

    # exact values against a reference built from the module's own weights
    ref = _ref_multihead(x, module)
    assert torch.allclose(out, ref, atol=1e-5), (
        "causal and right shape, but values differ from the reference. "
        "Most common bug: splitting heads with .view on the wrong dim order. "
        "The sequence: (B,T,3C) -> split -> (B,T,nh,hd) -> transpose(1,2)."
    )
    _pass("multi-head attention -- heads split, attended, and merged correctly")


# ----------------------------------------------------------------------
# Exercise 4: transformer block (pre-LN).
# Expects: block.ln_1, block.attn, block.ln_2, block.mlp; forward(x)->(B,T,C)
# ----------------------------------------------------------------------
def check_block(block):
    for attr in ("ln_1", "attn", "ln_2", "mlp"):
        assert hasattr(block, attr), f"block needs submodule `{attr}`"
    C = block.attn.c_proj.out_features
    B, T = 2, 9
    x = torch.randn(B, T, C)
    block.eval()
    out = block(x)
    assert out.shape == (B, T, C), f"wrong shape {tuple(out.shape)}"

    # residual wiring: silence both branches -> block must be the identity
    import copy
    b = copy.deepcopy(block)
    with torch.no_grad():
        b.attn.c_proj.weight.zero_()
        if b.attn.c_proj.bias is not None:
            b.attn.c_proj.bias.zero_()
        for m in b.mlp.modules():
            if isinstance(m, nn.Linear):
                last = m
        last.weight.zero_()
        if last.bias is not None:
            last.bias.zero_()
    y = b(x)
    assert torch.allclose(y, x, atol=1e-6), (
        "with both branch outputs zeroed the block should be exactly the "
        "identity map. It isn't -- so x is being transformed on the trunk. "
        "Pre-LN means: x + attn(ln_1(x)), then x + mlp(ln_2(x)). "
        "LayerNorm goes on the BRANCH, never on the residual stream."
    )

    out_c = block(_perturb_future(x, 4))
    assert torch.allclose(out[:, :4], out_c[:, :4], atol=1e-5), (
        "block is not causal"
    )
    _pass("block -- residual stream intact, branches normalized, causal")


# ----------------------------------------------------------------------
# Exercise 5: the full model.
# forward(idx) -> logits (B,T,vocab);  forward(idx, targets) -> (logits, loss)
# ----------------------------------------------------------------------
def check_gpt(model, vocab_size):
    B, T = 4, 16
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    model.eval()

    logits = model(idx)
    if isinstance(logits, tuple):
        logits = logits[0]
    assert logits.shape == (B, T, vocab_size), (
        f"expected logits {(B, T, vocab_size)}, got {tuple(logits.shape)}"
    )

    out = model(idx, targets)
    assert isinstance(out, tuple) and len(out) == 2, (
        "forward(idx, targets) should return (logits, loss)"
    )
    loss = out[1]

    expected = math.log(vocab_size)
    assert abs(loss.item() - expected) < 0.25, (
        f"initial loss is {loss.item():.3f}, but an untrained model should "
        f"sit at -ln(1/vocab) = ln({vocab_size}) = {expected:.3f}. "
        "If it's far above: init std too large (use 0.02). "
        "If it's mysteriously below: information is leaking -- check the "
        "causal mask and that targets are shifted by exactly one."
    )

    # causality at the model level, in token space
    idx2 = idx.clone()
    idx2[:, 8:] = torch.randint(0, vocab_size, (B, T - 8))
    l2 = model(idx2)
    if isinstance(l2, tuple):
        l2 = l2[0]
    assert torch.allclose(logits[:, :8], l2[:, :8], atol=1e-4), (
        "changing future TOKENS changed past LOGITS -- the model can see "
        "the answer it is supposed to predict"
    )
    _pass(f"gpt -- shapes, loss plumbing, init at ln(V)={expected:.2f}, causal")


def check_overfit(model, idx, targets, steps=150, lr=1e-3):
    """The single most informative smoke test in deep learning:
    a model that cannot memorize one batch cannot learn anything."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    first = None
    for i in range(steps):
        _, loss = model(idx, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    final = loss.item()
    assert final < 0.5 * first and final < 2.0, (
        f"loss went {first:.3f} -> {final:.3f} in {steps} steps on ONE batch. "
        "It should collapse toward zero. Check: optimizer over "
        "model.parameters(), zero_grad each step, targets shifted by one."
    )
    _pass(f"overfit-one-batch -- loss {first:.3f} -> {final:.3f}. It can learn.")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _perturb_future(x, t):
    x2 = x.clone()
    x2[:, t:] += torch.randn_like(x2[:, t:])
    return x2


def influence_matrix(fn, T=12, C=8):
    """M[t, s] = ||d out_t / d x_s||: the model's Green's function.
    Plot with plt.imshow -- for a causal map it is lower triangular."""
    x = torch.randn(1, T, C, requires_grad=True)
    out = fn(x, x, x) if _takes_qkv(fn) else fn(x)
    M = torch.zeros(T, T)
    for t in range(T):
        g = torch.autograd.grad(out[0, t].sum(), x, retain_graph=True)[0]
        M[t] = g[0].norm(dim=-1)
    return M.detach()


def _takes_qkv(fn):
    import inspect
    try:
        return len(inspect.signature(fn).parameters) >= 3
    except (TypeError, ValueError):
        return False


# ======================================================================
# ============  SPOILERS BELOW: reference implementations  =============
# ======================================================================
def _ref_attention(q, k, v, causal=False, scale=True):
    d = q.size(-1)
    att = q @ k.transpose(-2, -1)
    if scale:
        att = att / math.sqrt(d)
    if causal:
        T = q.size(-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
    return F.softmax(att, dim=-1) @ v


def _ref_multihead(x, module):
    B, T, C = x.shape
    nh = module.n_head
    hd = C // nh
    qkv = module.c_attn(x)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, nh, hd).transpose(1, 2)
    k = k.view(B, T, nh, hd).transpose(1, 2)
    v = v.view(B, T, nh, hd).transpose(1, 2)
    y = _ref_attention(q, k, v, causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return module.c_proj(y)
