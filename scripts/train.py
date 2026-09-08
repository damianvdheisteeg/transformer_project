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
