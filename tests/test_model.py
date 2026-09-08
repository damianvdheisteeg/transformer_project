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
