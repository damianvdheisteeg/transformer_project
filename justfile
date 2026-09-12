# List available recipes when you type `just`
default:
    @just --list

test:
    uv run pytest -q -m "not slow"

test-all:
    uv run pytest -q

check:
    uv run pyright src/

train-laptop *ARGS:
    uv run python scripts/train.py configs/laptop.yaml {{ARGS}}

train-gpu *ARGS:
    uv run python scripts/train.py configs/gpu_small.yaml {{ARGS}}

sync-up:
    rsync -avP --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
        data/ vast:/workspace/minigpt/data/

sync-down:
    rsync -avP vast:/workspace/minigpt/out/ ./out/