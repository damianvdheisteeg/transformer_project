"""Download tiny shakespeare (1.1 MB of text). Run once: python3 get_data.py"""
import os
import urllib.request

from minigpt.data import resolve_path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

out = resolve_path("data/input.txt")
if out.exists():
    print(f"{out} already present, nothing to do")
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, out)
    n = len(out.read_text(encoding="utf-8"))
    print(f"downloaded {out} ({n:,} characters)")
