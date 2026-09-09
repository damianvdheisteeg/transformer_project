"""Training configuration."""
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class ModelConfig:
    # model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 64
    vocab_size: int = 65        # set from dataset at runtime

@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 2000
    lr: float = 3e-4
    dtype: str = "float32"      # "float32" | "bfloat16" | "float16"
    compile: bool = False
    eval_interval: int = 100
    eval_iters: int = 20
    seed: int = 1337

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run_name: str = "debug"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        raw = yaml.safe_load(open(path))
        return cls(model=ModelConfig(**raw.get("model", {})),
                   train=TrainConfig(**raw.get("train", {})),
                   run_name=raw.get("run_name", "debug"))
    
    @classmethod
    def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
        for item in overrides:
            key, _, value = item.partition("=")
            obj = cfg
            *parents, leaf = key.split(".")
            for p in parents:
                obj = getattr(obj, p)
            current = getattr(obj, leaf)
            setattr(obj, leaf, type(current)(value))
        return cfg

@dataclass
class DataConfig:
    path: str = "data/input.txt"
    encoding: str = "char"              # "char" | "gpt2"
    memmap: bool = False                # True for uint16 .bin in Part 2