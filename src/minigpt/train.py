"""Training loop for minigpt."""
from dataclasses import dataclass, asdict
import math
import wandb
import torch

from .config import TrainConfig, Config
from .data import CharDataset
from .model import GPT

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to max_lr, then cosine decay to max_lr / 10. """
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    min_lr = cfg.max_lr / 10
    return min_lr + 0.5 * (cfg.max_lr - min_lr) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss(model: GPT, ds: CharDataset, cfg: Config) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.train.eval_iters)
        for i in range(cfg.train.eval_iters):
            xb, yb = ds.get_batch(split, cfg.train.batch_size, cfg.model.block_size)
            losses[i] = model(xb, yb)[1]
        out[split] = losses.mean().item()
    model.train()
    return out

def train(cfg: Config, use_wandb: bool = False) -> tuple[GPT, CharDataset, dict[str, list[int] | list[float]]]:

    if use_wandb:
        wandb.init(project="transformer-lab", config=asdict(cfg))

    torch.manual_seed(cfg.train.seed)
    ds = CharDataset(cfg.data.path)
    model = GPT(cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.max_lr)
    history = {"step": [], "train": [], "val": []}

    for step in range(cfg.train.max_steps):
        lr = get_lr(step, cfg.train)
        for g in opt.param_groups:
            g["lr"] = lr

        xb, yb = ds.get_batch("train", cfg.train.batch_size, cfg.model.block_size)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if use_wandb:
            wandb.log({"loss/train_batch": loss.item(),
                       "lr": lr,
                       "grad_norm": grad_norm.item()}, step=step)

        if step % cfg.train.eval_interval == 0 or step == cfg.train.max_steps - 1:
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