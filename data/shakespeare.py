"""
Tiny Shakespeare, character-level.

Also not a TODO file — this is boilerplate data loading, not architecture.
Downloads Karpathy's classic tinyshakespeare.txt (~1MB) the first time you
run it, then caches locally.
"""
import os
import requests
import torch
from torch.utils.data import Dataset

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "tinyshakespeare.txt")


def download_if_needed():
    if not os.path.exists(CACHE_PATH):
        print("Downloading Tiny Shakespeare...")
        r = requests.get(URL, timeout=30)
        r.raise_for_status()
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return f.read()


class CharVocab:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class ShakespeareCharDataset(Dataset):
    """
    Returns fixed-length chunks (x, y) where y is x shifted by one character
    (standard next-token-prediction setup).
    """

    def __init__(self, seq_len=128, split="train", train_frac=0.9):
        text = download_if_needed()
        self.vocab = CharVocab(text)
        data = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        n = int(len(data) * train_frac)
        self.data = data[:n] if split == "train" else data[n:]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


if __name__ == "__main__":
    ds = ShakespeareCharDataset(seq_len=64)
    print("vocab size:", ds.vocab.vocab_size)
    x, y = ds[0]
    print("x:", ds.vocab.decode(x.tolist()))
    print("y:", ds.vocab.decode(y.tolist()))
