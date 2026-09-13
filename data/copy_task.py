"""
Synthetic long-range dependency task: the "Copy Task".

This is deliberately NOT a TODO file. The point of this lab is architecture,
not data engineering — but understand what this generates, because your
sequence-length sweep depends on it.

Task definition
----------------
Input:  [d1, d2, ..., dk, 0, 0, ..., 0, 9, 0, ..., 0]
                          |<--- delay_len --->| ^cue
Output: [0, 0, ..., 0, d1, d2, ..., dk]
                       (right-aligned at the end)

The model sees k random "digits" (tokens 1..8), then a long run of blanks
(token 0), then a special "recall cue" token (9), then must reproduce the k
digits in order.

Why this task: the ONLY way to succeed is to carry the k digits across
`delay_len` steps of pure noise. Vanilla RNNs fail once delay_len is large
because gradient signal from the output can't survive backprop through that
many multiplicative steps. LSTM/GRU survive to a much larger delay_len via
gating. Transformers don't care about delay_len at all (self-attention
connects any two positions directly) but *do* care about total sequence
length for compute/memory (O(n^2)).

Sweep `delay_len` and hold `k` fixed to isolate the long-range-dependency
effect this task is designed to probe.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

BLANK = 0
CUE = 9
VOCAB_SIZE = 10  # tokens 0..9 (0=blank, 1..8=digits, 9=cue)


def generate_copy_batch(batch_size, k=8, delay_len=50, seed=None):
    """
    Returns:
        inputs:  LongTensor [batch_size, seq_len]
        targets: LongTensor [batch_size, seq_len]  (same length, -100 = ignore)
        seq_len = k + delay_len + 1 + k
    """
    rng = np.random.default_rng(seed)
    seq_len = k + delay_len + 1 + k

    inputs = np.full((batch_size, seq_len), BLANK, dtype=np.int64)
    targets = np.full((batch_size, seq_len), -100, dtype=np.int64)  # -100 = ignored by CE loss

    digits = rng.integers(low=1, high=9, size=(batch_size, k))  # tokens 1..8
    inputs[:, :k] = digits
    inputs[:, k + delay_len] = CUE
    targets[:, k + delay_len + 1:] = digits  # model must output digits here

    return torch.from_numpy(inputs), torch.from_numpy(targets)


class CopyTaskDataset(Dataset):
    """Infinite-style dataset: generates fresh random sequences on the fly."""

    def __init__(self, k=8, delay_len=50, num_samples=10_000, seed=None):
        self.k = k
        self.delay_len = delay_len
        self.num_samples = num_samples
        self.rng_seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seed = None if self.rng_seed is None else self.rng_seed + idx
        x, y = generate_copy_batch(1, k=self.k, delay_len=self.delay_len, seed=seed)
        return x[0], y[0]


if __name__ == "__main__":
    x, y = generate_copy_batch(batch_size=2, k=4, delay_len=10, seed=0)
    print("input :", x[0].tolist())
    print("target:", y[0].tolist())
