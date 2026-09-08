"""Training loop for minigpt."""
from dataclasses import dataclass
import math

import torch

from .config import TrainConfig
from .data import CharDataset
from .model import GPT

def get_lr(step, cfg):
    """Linear warmup to max_lr, then cosine decay to max_lr / 10. """
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    min_lr = cfg.max_lr / 10
    return min_lr + 0.5 * (cfg.max_lr - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss(model, iters=20):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for i in range(iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

def train(cfg, use_wandb=False):

    if use_wandb:
        import wandb
        wandb.init(project="transformer-lab", config=dataclasses.asdict(cfg))

    torch.manual_seed(cfg.seed)
    ds = CharDataset(cfg.data_path)
    model = GPT(ds.vocab_size, cfg.block_size, cfg.n_layer, cfg.n_head, cfg.n_embd)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr)
    history = {"step": [], "train": [], "val": []}

    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        xb, yb = get_batch("train", cfg.batch_size, cfg.block_size)
        _, loss = m(xb, yb)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if use_wandb:
            wandb.log({"loss/train_batch": loss.item(),
                       "lr": lr,
                       "grad_norm": grad_norm.item()}, step=step)

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            L = estimate_loss(model, ds, cfg)
            history["step"].append(step)
            history["train"].append(L["train"])
            history["val"].append(L["val"])
            print(f"step {step:5d}  train {L['train']:.4f}  val {L['val']:.4f}")
            if use_wandb:
                wandb.log({"loss/train": L["train"], "loss/val": L["val"]}, step=step)

    if use_wandb:
        wandb.finish()
    return model, ds, history