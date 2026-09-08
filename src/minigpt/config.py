"""Training configuration."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # model
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    # data
    data_path: str = "input.txt"
    batch_size: int = 32
    # optimization
    max_lr: float = 1e-3
    warmup_steps: int = 100
    max_steps: int = 2000
    # bookkeeping
    eval_interval: int = 100
    eval_iters: int = 20
    seed: int = 1337