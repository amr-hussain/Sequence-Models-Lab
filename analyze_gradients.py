"""
Compare gradient propagation through time for:

1. Vanilla RNN (manual NumPy BPTT)
2. Manual LSTM (PyTorch autograd)

Run:

    python analyze_gradients.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.numpy_rnn import VanillaRNN
from models.lstm_cell import LSTMCellManual


# ============================================================
# VANILLA RNN
# ============================================================

def measure_rnn_grad_norms_vs_timestep(
    seq_len,
    input_size=10,
    hidden_size=32,
    output_size=10,
    seed=0,
):
    rng = np.random.default_rng(seed)

    model = VanillaRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        seed=seed,
    )

    xs = []

    for _ in range(seq_len):
        x = np.zeros(input_size)
        x[rng.integers(input_size)] = 1.0
        xs.append(x)

    ys_target = [None] * (seq_len - 1)
    ys_target.append(rng.integers(output_size))

    ys, ps, hs = model.forward(xs)

    _, grad_norms = model.backward(
        xs,
        ys_target,
        ps,
        hs,
    )

    return np.array(grad_norms)


# ============================================================
# LSTM
# ============================================================

def measure_lstm_grad_norms_vs_timestep(
    seq_len,
    vocab_size=10,
    embed_size=16,
    hidden_size=32,
    seed=0,
):
    torch.manual_seed(seed)

    embed = nn.Embedding(vocab_size, embed_size)
    cell = LSTMCellManual(embed_size, hidden_size)
    out = nn.Linear(hidden_size, vocab_size)

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, seq_len),
    )

    target = torch.randint(
        low=0,
        high=vocab_size,
        size=(1,),
    )

    embedded = embed(x)

    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)

    hs = []
    cs = []

    for t in range(seq_len):

        h, c, _ = cell(
            embedded[:, t, :],
            (h, c),
        )

        h.retain_grad()
        c.retain_grad()

        hs.append(h)
        cs.append(c)

    logits = out(h)

    loss = F.cross_entropy(logits, target)

    loss.backward()

    h_norms = []
    c_norms = []

    for h_t, c_t in zip(hs, cs):

        if h_t.grad is None:
            h_norms.append(0.0)
        else:
            h_norms.append(
                h_t.grad.norm().item()
            )

        if c_t.grad is None:
            c_norms.append(0.0)
        else:
            c_norms.append(
                c_t.grad.norm().item()
            )

    return (
        np.array(h_norms),
        np.array(c_norms),
    )


# ============================================================
# PLOT
# ============================================================

def plot_gradient_comparison(
    seq_len=40,
    save_path="outputs/rnn_vs_lstm_gradients.png",
):
    rnn_norms = measure_rnn_grad_norms_vs_timestep(
        seq_len=seq_len
    )

    lstm_h_norms, lstm_c_norms = (
        measure_lstm_grad_norms_vs_timestep(
            seq_len=seq_len
        )
    )

    timesteps = np.arange(seq_len)

    steps_before_end = (
        seq_len - 1 - timesteps
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps_before_end,
        rnn_norms,
        linewidth=2,
        label="Vanilla RNN: ||dL/dh_t||",
    )

    plt.plot(
        steps_before_end,
        lstm_h_norms,
        linewidth=2,
        label="LSTM: ||dL/dh_t||",
    )

    plt.plot(
        steps_before_end,
        lstm_c_norms,
        linewidth=2,
        label="LSTM: ||dL/dc_t||",
    )

    plt.xlabel(
        "Timesteps before loss position"
    )

    plt.ylabel(
        "Gradient Norm"
    )

    plt.yscale("log")

    plt.title(
        f"Gradient Propagation Through Time (seq_len={seq_len})"
    )

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
    )

    print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# MULTIPLE LENGTHS
# ============================================================

def plot_multiple_lengths(
    seq_lengths=(10, 20, 40, 80),
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    axes = axes.flatten()

    for ax, seq_len in zip(
        axes,
        seq_lengths,
    ):

        rnn = measure_rnn_grad_norms_vs_timestep(
            seq_len=seq_len
        )

        lstm_h, lstm_c = (
            measure_lstm_grad_norms_vs_timestep(
                seq_len=seq_len
            )
        )

        steps = (
            seq_len
            - 1
            - np.arange(seq_len)
        )

        ax.plot(
            steps,
            rnn,
            label="RNN",
        )

        ax.plot(
            steps,
            lstm_h,
            label="LSTM h",
        )

        ax.plot(
            steps,
            lstm_c,
            label="LSTM c",
        )

        ax.set_yscale("log")
        ax.set_title(
            f"seq_len={seq_len}"
        )
        ax.grid(True, alpha=0.3)

    axes[0].legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/all_lengths.png",
        dpi=150,
    )

    plt.show()


if __name__ == "__main__":

    os.makedirs(
        "outputs",
        exist_ok=True,
    )

    plot_gradient_comparison(
        seq_len=40
    )

    plot_multiple_lengths(
        seq_lengths=(10, 20, 40, 80)
    )