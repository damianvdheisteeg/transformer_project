import math, torch, pytest
from minigpt.model import GPT, Block, MultiHeadAttention
from minigpt.config import ModelConfig

def tiny_cfg(**kw: int) -> ModelConfig:
    base = dict(n_layer=2, n_head=2, n_embd=32, block_size=32, vocab_size=65)
    return ModelConfig(**{**base, **kw})



def test_shape():
    cfg = tiny_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = m(idx, idx)
    assert logits.shape == torch.Size([2,8, cfg.vocab_size]) and loss.shape == ()

def test_causality_via_gradients():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    x = torch.randn(1, 8, cfg.n_embd, requires_grad=True)
    attn = m.blocks[0].attn
    attn(x)[0, 3].sum().backward() # only run backward pass on element [0,3]
    influence = x.grad[0].abs().sum(-1)
    assert (influence[:4] > 0).all()
    assert (influence[4:] == 0).all()




# old tests from sheet 2

def test_causality():
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    idx = torch.randint(0, tiny_cfg().vocab_size, (2, 32))
    idx2 = idx.clone(); idx2[:, 16:] = torch.randint(0, tiny_cfg().vocab_size, (2, 16))
    a, b = m(idx)[0], m(idx2)[0]
    assert torch.allclose(a[:, :16], b[:, :16], atol=1e-4)
    assert not torch.allclose(a[:, 16:], b[:, 16:], atol=1e-4)

def test_block_is_identity_when_branches_silenced():
    torch.manual_seed(0)
    b = Block(tiny_cfg().n_embd, tiny_cfg().n_head).eval()
    with torch.no_grad():
        b.attn.c_proj.weight.zero_(); b.attn.c_proj.bias.zero_()
        b.mlp.c_proj.weight.zero_();  b.mlp.c_proj.bias.zero_()
    x = torch.randn(2, 8, 32)
    assert torch.allclose(b(x), x, atol=1e-6)

def test_init_loss_is_ln_vocab():
    torch.manual_seed(0)
    m = GPT(tiny_cfg()).eval()
    idx = torch.randint(0, 65, (8, 32))
    _, loss = m(idx, torch.randint(0, 65, (8, 32)))
    assert abs(loss.item() - math.log(65)) < 0.25

def test_can_overfit_one_batch():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 32)); tgt = torch.randint(0, cfg.vocab_size, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(200):
        _, loss = m(idx, tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    assert loss.item() < 1.0
