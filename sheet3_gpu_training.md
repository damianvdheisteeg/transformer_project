# Exercise sheet 3 — Your minGPT on a rented GPU

**Prerequisites:** sheet 1 (git) and sheet 2 (minGPT). You have a working `train.py` that trains a small GPT on your laptop.
**Budget:** ~$5–10 and one afternoon. A single RTX 3090 / 4090 on vast.ai costs roughly $0.20–0.45 per hour, so even a sloppy afternoon stays under $5. The expensive mistake is *forgetting to destroy the instance*, not renting it.
**Goal:** the same model you already have, but trained faster, bigger, and — more importantly — *measured*. By the end you should be able to say "my run hit X% of the GPU's peak, it was memory-bound because Y, and here is the W&B link".

Two kinds of callouts run through the sheet:

> 🧠 **DL habit** — something people who train models for a living do reflexively.
> 🛠 **Programming habit** — something people who ship code do reflexively.

Do the parts in order. Part 0 is done *on your laptop, before you spend a cent*; everything that can go wrong without a GPU should go wrong for free.

---

## Part 0 — Make the code rentable (laptop, ~1.5 h, $0)

The rule on a metered machine: **never debug for the first time on the clock.** So first turn your minGPT from "a script that works on my laptop" into "a small project that runs anywhere".

### 0.1 A real project with `uv`

1. Install `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Create the layout below, with `uv init --lib` or by hand:

   ```
   mingpt/
   ├── pyproject.toml
   ├── README.md
   ├── src/mingpt/
   │   ├── __init__.py
   │   ├── model.py        # the GPT you wrote
   │   ├── data.py         # dataset + loader
   │   └── config.py       # see 0.2
   ├── scripts/
   │   └── train.py        # thin entry point: parse config → call library
   ├── configs/
   │   ├── laptop.yaml
   │   └── gpu_small.yaml
   └── tests/
       └── test_model.py
   ```

2. Put `torch`, `pyyaml`, `wandb`, `pytest` in `pyproject.toml`; run `uv sync`. Commit `uv.lock`. Verify `uv run python -c "import mingpt"` works from a *fresh clone in a different directory*.

> 🛠 **Programming habit — the script/library split.** `scripts/train.py` should be under ~60 lines and contain no model code. If you find yourself copying a function from one script to another, that function has just become library code: move it into `src/mingpt/` and import it. This is your coworker's "throwaway script vs library" question, answered by a rule of thumb: *a thing is library code the second time you need it.*

### 0.2 Config as data, not as globals

3. Replace whatever `argparse`/global-constant soup you have with a typed config. Minimal version, no extra dependencies:

   ```python
   # src/mingpt/config.py
   from dataclasses import dataclass, field, asdict
   import yaml

   @dataclass
   class ModelConfig:
       n_layer: int = 4
       n_head: int = 4
       n_embd: int = 128
       block_size: int = 256
       vocab_size: int = 65          # tiny shakespeare, char level

   @dataclass
   class TrainConfig:
       batch_size: int = 32
       max_steps: int = 2000
       lr: float = 3e-4
       dtype: str = "float32"        # "float32" | "bfloat16" | "float16"
       compile: bool = False
       eval_every: int = 200
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
   ```

   `scripts/train.py` takes exactly one argument: the YAML path, plus optional `key=value` overrides (`train.lr=1e-3`). Write the override parser yourself (it's ~15 lines) — you will appreciate what hydra does for you later.

4. Write `configs/laptop.yaml` (tiny model, 200 steps, CPU/MPS) and `configs/gpu_small.yaml` (see Part 3 for sizes). The *only* thing that should differ between a laptop run and a GPU run is which YAML you pass.

> 🛠 **Programming habit.** Every hyperparameter lives in exactly one place, and that place is serialisable. Later you will log `asdict(cfg)` to W&B, so a run is reproducible from its dashboard page alone. "Which learning rate was that run?" should never be a question you answer from memory.

5. Add type hints to `model.py` and `config.py` and run `uv run pyright src/` (or `mypy`). Fix what it finds — typically a `Tensor | None` that you assumed was never `None`.

### 0.3 Tests that would have caught your sheet-2 bugs

6. Write three tests in `tests/test_model.py` and make `uv run pytest -q` pass:
   - **shape test:** `GPT(cfg)(idx)` on `idx` of shape `(2, 8)` returns logits of shape `(2, 8, vocab_size)`.
   - **causality test:** perturb token 5 of the input; assert logits at positions 0–4 do *not* change. (This is the test that catches a missing causal mask — a bug that trains fine and only shows up as suspiciously low loss.)
   - **overfit-one-batch smoke test:** 50 steps on a single fixed batch; assert loss drops below 1.0. Mark it `@pytest.mark.slow`.

> 🧠 **DL habit — overfit a single batch first.** Before any real run, on any new machine, confirm the model can memorise one batch. If it can't, nothing downstream matters. You just turned that habit into an automated test.

### 0.4 Remote hygiene on your laptop

7. Create a dedicated SSH key: `ssh-keygen -t ed25519 -f ~/.ssh/vast -C "vast"`. Never reuse your GitHub key for a rented box.
8. Make a W&B account (free tier is fine), `uv run wandb login`, and get the API key into your shell as `WANDB_API_KEY`. Look at *where* `wandb login` stored it (`~/.netrc`) — you'll need to know that when the remote box asks.
9. Write a `Makefile` or `justfile` with the four verbs you'll type all afternoon: `test`, `train-laptop`, `sync-up`, `sync-down` (the last two get filled in in Part 2).

**Checkpoint before renting:** fresh clone → `uv sync` → `uv run pytest` green → `uv run python scripts/train.py configs/laptop.yaml` runs 200 steps and logs to W&B. Commit. Then, and only then, open vast.ai.

---

## Part 1 — Rent, connect, don't panic (~30 min, ~$0.50)

### 1.1 Choosing a box

10. On vast.ai, filter for: 1× RTX 4090 (or 3090), **≥ 50 GB disk**, **≥ 500 Mbps down**, "verified" hosts, and a PyTorch CUDA image (e.g. `pytorch/pytorch:2.x-cuda12.x-cudnn9-devel`). Sort by $/h. Prefer **on-demand** over "interruptible" for this sheet — interruptible is cheaper but can be killed mid-run, and you don't have checkpoint-resume yet.
    The CLI equivalent is worth learning once (`pip install vastai`; `vastai search offers 'gpu_name=RTX_4090 num_gpus=1 disk_space>=50 inet_down>=500 verified=true rentable=true' -o dph`).
11. Before clicking *Rent*, write down in your lab notes: **$/h, GPU, VRAM, disk, the host's reported bandwidth**. You will compare the last one against reality in 2.1.

> 🧠 **DL habit.** Know the *peak numbers* of the card you're on before you start; you'll compute utilisation against them in Part 3. For reference (dense, no sparsity):
>
> | GPU | VRAM | memory bandwidth | bf16 tensor-core peak | fp32 (non-tensor) |
> |---|---|---|---|---|
> | RTX 3090 | 24 GB GDDR6X | ~936 GB/s | ~71 TFLOPS | ~36 TFLOPS |
> | RTX 4090 | 24 GB GDDR6X | ~1 008 GB/s | ~165 TFLOPS | ~83 TFLOPS |
> | A100 80 GB | 80 GB HBM2e | ~2 000 GB/s | ~312 TFLOPS | ~19.5 TFLOPS |
> | H100 SXM | 80 GB HBM3 | ~3 350 GB/s | ~990 TFLOPS | ~67 TFLOPS |
>
> Look up the exact figures for your card on the NVIDIA spec sheet and note the ratio *FLOPS / bandwidth* (the "ridge point" of the roofline, in FLOP per byte). A 4090 is ~160 FLOP/byte in bf16; your tiny model will not get near that, which is the point of Part 3.

### 1.2 Connect like you'll do it fifty times

12. vast.ai gives you `ssh -p <PORT> root@<IP>`. Do **not** type that. Add to `~/.ssh/config`:

    ```
    Host vast
        HostName <IP>
        Port <PORT>
        User root
        IdentityFile ~/.ssh/vast
        ServerAliveInterval 30
    ```

    Now `ssh vast` works, and so does `rsync`/`scp` with the same alias. When you rent a new box tomorrow, you change two lines.
13. First commands on the box, in this order, and *read every line of output*:

    ```bash
    nvidia-smi                       # what card, what driver, what's already using it (should be nothing)
    nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
    df -h /  /workspace              # how much disk did you actually get
    nproc; free -h                   # CPU cores + host RAM — dataloader budget
    python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    cat /proc/cpuinfo | grep 'model name' | head -1
    ```

    Questions to answer in your notes: does `torch.cuda.get_device_name(0)` match what you paid for? How many CPU cores per GPU? (Rule of thumb: you want ≥ 4 per GPU for dataloading; some vast hosts give you 2.)
14. Start a tmux session immediately: `tmux new -s train`. Everything that takes more than a minute runs inside it. Practice once, deliberately: start a `sleep 300` in tmux, detach (`Ctrl-b d`), close your laptop lid, reopen, `ssh vast`, `tmux attach -t train`. The sleep is still running. *That* is why tmux exists — a dropped Wi-Fi connection must not kill a 2-hour training run.

> 🛠 **Programming habit.** `~/.ssh/config` and a tmux session are not "advanced"; they are the difference between someone who can use a remote machine and someone who can't. Both go in your dotfiles repo.

---

## Part 2 — Ship code and data (~20 min)

### 2.1 Code goes up via git, data via rsync

15. On the box: install `uv`, `git clone` your repo (use HTTPS + a token, or forward your agent with `ssh -A vast` — understand the difference before choosing), `uv sync`, `uv run pytest -q -m "not slow"`. **Green tests on the remote before anything else.** If a test fails here and not on your laptop, that's a real finding: write down why (usually CUDA vs CPU numerics, or a path).
16. Data: use **TinyShakespeare** for the char-level baseline (1 MB — just `curl` it on the box) and, for the bigger runs in Part 3, a ~100 MB slice of something tokenised, e.g. the first shard of *TinyStories* or *OpenWebText* pre-tokenised with the GPT-2 tokenizer (`tiktoken`) into a `uint16` `.bin` file. Do the tokenisation **on your laptop** (it's CPU work, the laptop is free), then:

    ```bash
    rsync -avP --exclude '.git' --exclude '.venv' data/ vast:/workspace/mingpt/data/
    ```

    Time it. Compute the achieved MB/s and compare with the bandwidth the host advertised in exercise 11. This is the "dataloading/storage as a real bottleneck" point in miniature: a 4090 does ~10⁴ steps in the time a slow host takes to receive 100 MB.
17. Fill in `make sync-up` / `make sync-down` (down = checkpoints + samples only, never the data). Add `--dry-run` to both until you trust them; `rsync --delete` on the wrong side has ended careers.

> 🧠 **DL habit.** `np.memmap` the token file rather than loading it into RAM; random-offset batching from a memmap is how nanoGPT and most real pretraining loaders work at small scale. Check `free -h` before and after: your dataset should not show up in "used".

---

## Part 3 — Measure before you scale (~1.5 h, ~$1–2)

This is the heart of the sheet. Everything here is one model, one script, and a series of controlled changes with a *number* attached to each.

### 3.1 Instrument the training loop

18. Add to the training loop, logged every N steps to both stdout and W&B:
    - `tokens/s` — `batch_size * block_size * N / elapsed`, using `torch.cuda.synchronize()` before reading the clock (why is that necessary? write one sentence).
    - `step_ms`.
    - `torch.cuda.max_memory_allocated() / 2**30` in GB, reset per log interval.
    - **MFU** (model FLOPs utilisation): `flops_per_token ≈ 6 * n_params` (+ the attention term `12 * n_layer * n_embd * block_size`, which matters once `block_size` is large); `MFU = flops_per_token * tokens_per_s / peak_bf16_flops`. Put the peak in the YAML, since it depends on the card.
    - `lr`, `loss`, and `grad_norm` (you clip anyway; log the pre-clip norm).
19. `wandb.init(project="mingpt-gpu", name=cfg.run_name, config=asdict(cfg))`. Log the git commit hash as well (`git rev-parse --short HEAD`, or refuse to start with a dirty working tree — try that stricter version and see how it changes your behaviour).

> 🧠 **DL habit — utilisation is a first-class metric.** `nvidia-smi`'s "GPU-Util" only says a kernel was running, not how well. MFU is the honest number. Big labs report it in papers; you should report it in your notes. Anything under ~10% on a small GPU means the GPU is mostly waiting for you.

### 3.2 The baseline, and reading nvidia-smi in anger

20. `configs/gpu_small.yaml`: `n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, dtype=float32, compile=false`, 1 000 steps on the tokenised data (~10 M params — nanoGPT's "Shakespeare-char" size). Run it in tmux. In a *second* tmux window run `watch -n 1 nvidia-smi` and, in a third, `htop`.
    Record: tokens/s, MFU, peak memory, GPU-util %, power draw (W) vs the card's TDP, and how busy the CPU cores are. Which resource is the bottleneck? Justify with the numbers, not the vibes.
21. **Kill it on purpose.** From the htop window: `kill -SIGINT <pid>` (equivalent to Ctrl-C). Then run again and `kill -SIGTERM`, then `kill -9`. What happens to the W&B run in each case (does it end up "crashed", "killed", "finished")? What happens to the GPU memory — check `nvidia-smi` after each — and why does `kill -9` never leak GPU memory even though your Python never got to clean up? (Hint: who owns the CUDA context — the process or the driver?)

### 3.3 One knob at a time

Each of 22–25 is a separate W&B run, changed from the baseline by *exactly one* setting, 500 steps each. Name them so the W&B table reads like an experiment log (`base-fp32`, `bf16`, `bf16-compile`, `bf16-bs256`, ...).

22. **`dtype=bfloat16`** via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Expect ~2× tokens/s or more on a 4090. Does the loss curve after 500 steps overlay the fp32 one? (Compare in W&B with both runs selected.)
23. **`dtype=float16`, *without* a `GradScaler`.** Watch `grad_norm`. Does it go to `nan`/`inf`, or does it silently train fine at this model size? Then add `torch.amp.GradScaler("cuda")` and confirm identical behaviour to bf16. Write two sentences: why does fp16 need loss scaling and bf16 doesn't? (Numbers: fp16 has 10 mantissa bits and a minimum normal of ~6e-5; bf16 has 7 mantissa bits and the same exponent range as fp32. Which one is "less precise", which one has "less range", and which failure did you observe?)
24. **`compile=true`** (`torch.compile(model)`). Note the first-step time (compilation) versus steady-state `step_ms`. How many steps does it take to amortise? On which real-world runs would you *not* want to pay it?
25. **Batch-size sweep:** `batch_size ∈ {16, 64, 256}` with bf16. Plot tokens/s and MFU vs batch size (W&B can do this from the run table; or pull the runs via the `wandb` API into a pandas DataFrame and use matplotlib — do the latter at least once, it's how you'll make figures for a report). Explain the shape of the curve using the roofline: at small batch, each kernel launch moves weights from VRAM but does little arithmetic per byte (memory-bound + launch overhead); at large batch, arithmetic intensity rises toward the ridge point. Where does memory run out?

> 🧠 **DL habit.** The batch-size sweep is the cheapest, most informative experiment on any new hardware. Do it before the "real" run every time you change GPU type. A number people memorise: for a matmul in bf16 on a 4090 you need an arithmetic intensity above ~160 FLOP/byte to be compute-bound; a tiny model with a tiny batch is nowhere near that.

### 3.4 Where is the memory going? (stretch, but worth it)

26. Estimate the training memory from first principles for your baseline: parameters (4 B each in fp32) + gradients (same) + AdamW state (2× same) + activations (scales with `batch × block × n_embd × n_layer × constant`). Compare with `max_memory_allocated`. Then compare with what `nvidia-smi` reports as "used" — why is that larger? (PyTorch's caching allocator holds freed blocks; `torch.cuda.memory_reserved()` is the honest number for "what the allocator took from the driver".)

---

## Part 4 — The real run (~1 h wall clock, ~$0.50)

27. Using the *fastest* configuration from Part 3, scale up: `n_layer=8, n_embd=512, block_size=512`, ~25 M params, bf16, compiled, the largest batch that fits with ~20 % VRAM headroom (why headroom? Because memory fragmentation and the occasional longer eval batch will find the missing 20 %). Set `max_steps` so the run takes ~30 minutes at the tokens/s you measured. **Predict the final loss and the wall-clock time before you start; write both down.**
28. Add checkpointing every `eval_every` steps: save `model.state_dict()`, `optimizer.state_dict()`, `step`, and the config. Test that `--resume path.pt` really resumes (kill the run at step 300 with SIGINT, resume, confirm the W&B curve continues rather than restarting — use `wandb.init(id=..., resume="must")`).
29. While it runs: `nvidia-smi dmon -s pucm` for 30 seconds. Read the columns (power, utilisation, clocks, memory). Is the card power-throttling (clocks dropping while power sits at the cap)? Note it — on shared hosts this is common and it explains "why is my run 15% slower than yesterday".
30. Generate 5 samples from the final checkpoint, `rsync` the checkpoint and samples back (`make sync-down`), and generate again *on your laptop* from the same checkpoint on CPU. Same text? If not, why not (temperature/sampling RNG, CPU vs CUDA kernels, bf16 vs fp32 weights)?

> 🛠 **Programming habit.** Predict, then measure. Being off by 3× on a prediction is normal and educational; not having a prediction means you learned nothing from the measurement.

---

## Part 5 — Leave no trace (~10 min)

31. Copy anything you want off the box (checkpoints, `wandb/` offline dir if you used it, your notes). Then **destroy the instance** in the vast.ai UI — not "stop". Check the billing page ten minutes later and confirm the meter stopped. Write the total cost in your notes next to your Part 1 estimate.
32. Remove the `vast` host entry (or keep it as a template with the IP blanked out) and revoke the token you used for git clone on the box, if you used one.

> 🛠 **Programming habit.** A rented machine is untrusted the moment you stop using it. Short-lived tokens, dedicated keys, and a destroy step in your checklist.

---

## Part 6 — Write it up like a PR (~30 min, $0)

33. Your repo now has: a uv project, a typed config, tests, instrumentation, checkpoint/resume, and a YAML per hardware target. Look at `git log`. Is it a reviewable history, or is it "wip", "fix", "more fixes"? Interactive-rebase it (sheet 1!) into ~6 commits, each one concern: *"Add dataclass config + YAML loader"*, *"Add causality and shape tests"*, *"Log throughput and MFU to W&B"*, ... Open a PR against `main` — even if nobody reviews it — with a description that contains the W&B project link and your batch-size plot.
34. Write a ½-page `docs/gpu-notes.md` in the repo: the card's peak numbers, the best MFU you reached and with what settings, the one thing that surprised you, and the fp16-vs-bf16 sentence from exercise 23. This is the document you'd want from a colleague who just spent an afternoon on a new GPU.

---

## What you should be able to answer afterwards (self-check)

- Why is `torch.cuda.synchronize()` needed before timing, and what would you measure without it?
- Your baseline hit X% MFU. Name two reasons it wasn't 50%, ranked by how much they cost you.
- bf16 vs fp16: which one has more range, which one has more precision, which one needs loss scaling and why?
- What does `nvidia-smi` "GPU-Util 100%" *not* tell you?
- What's the difference between `max_memory_allocated` and `memory_reserved`, and which one explains an OOM?
- Why does `kill -9` free GPU memory but leave a corrupt checkpoint?
- Why does tmux + `ServerAliveInterval` matter for a training run, and what's the failure without them?
- Give the one-line reason `configs/laptop.yaml` and `configs/gpu_small.yaml` exist instead of two scripts.

## Where this leads (candidates for sheet 4)

- **Two GPUs, same script:** rent a 2× box, wrap in DDP, measure scaling efficiency; then the same with gradient accumulation on one GPU. Collectives, NCCL, `torchrun`.
- **Profiling:** `torch.profiler` + the Chrome trace viewer on the baseline, find the top-3 kernels, and explain the batch-size curve from the trace instead of from the roofline.
- **Docker:** build your own image for the vast.ai template so `uv sync` and data download happen at image-build time, and understand why the image is still not reproducible.
- **Dataloading at scale:** replace the memmap with a sharded, streaming loader (webdataset / parquet) and measure throughput with and without `num_workers`, pinned memory, and `spawn` vs `fork`.
