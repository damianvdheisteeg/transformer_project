"""Entry point: parse config, run training."""
import sys

from minigpt.config import Config
from minigpt.train import train


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: train.py <config.yaml> [key=value ...]")
    cfg = Config.from_cli(sys.argv[1:])
    train(cfg)


if __name__ == "__main__":
    main()
