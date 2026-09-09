"""Download tiny shakespeare (1.1 MB of text). Run once: python3 get_data.py"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if os.path.exists("input.txt"):
    print("input.txt already present, nothing to do")
else:
    urllib.request.urlretrieve(URL, "input.txt")
    n = len(open("input.txt").read())
    print(f"downloaded input.txt ({n:,} characters)")
