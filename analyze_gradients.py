"""
The core diagnostic experiment: plot gradient magnitude vs. timestep, for
increasing sequence length, for each architecture. This is what makes
vanishing gradients a *visible, measured* phenomenon instead of a claim you
read in a textbook.

Run AFTER you've implemented models/numpy_rnn.py (both forward and backward).

Usage:
    python analyze_gradients.py
"""
import numpy as np
import matplotlib.pyplot as plt
from models.numpy_rnn import VanillaRNN


def measure_grad_norms_vs_timestep(seq_len, input_size=10, hidden_size=32, output_size=10, seed=0):
    """
    Builds a random sequence with a single loss target at the LAST timestep
    only (classic setup for exposing vanishing gradients: does gradient
    signal from the end reach all the way back to the start?), runs
    forward+backward, and returns the gradient norm at every timestep.
    """
    rng = np.random.default_rng(seed)
    model = VanillaRNN(input_size, hidden_size, output_size, seed=seed)

    xs = []
    for _ in range(seq_len):
        vec = np.zeros(input_size)
        vec[rng.integers(input_size)] = 1.0
        xs.append(vec)

    # Loss only at the final timestep -- forces gradient to travel the full
    # sequence length to reach early timesteps.
    ys_target = [None] * (seq_len - 1) + [rng.integers(output_size)]

    ys, ps, hs = model.forward(xs)
    _, grad_norms_per_t = model.backward(xs, ys_target, ps, hs)
    return grad_norms_per_t


def plot_vanishing_gradients(seq_lengths=(5, 10, 15), save_path="outputs/gradient_norms.png"):
    plt.figure(figsize=(8, 5))
    for seq_len in seq_lengths:
        grad_norms = measure_grad_norms_vs_timestep(seq_len)
        timesteps = np.arange(seq_len)
        # normalize x-axis to "steps before the end" so curves are comparable
        steps_before_end = seq_len - 1 - timesteps
        plt.plot(steps_before_end, grad_norms, label=f"seq_len={seq_len}")

    plt.xlabel("timesteps before the loss position (0 = loss position itself)")
    plt.ylabel("||dL/dh_t|| (log scale)")
    # plt.yscale("log")
    plt.yscale("linear")
    plt.title("Vanilla RNN: gradient norm vs. distance from loss (BPTT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    plot_vanishing_gradients()

    # TODO for you (not code, analysis): once this plot works, extend the
    # same measurement to LSTMCellManual and GRUCellManual using PyTorch's
    # .register_hook() on the hidden/cell state at each timestep to capture
    # ||dL/dh_t|| and ||dL/dc_t|| the same way, then overlay all three
    # architectures on one plot. That overlay is one of the most convincing
    # single figures you can put in a thesis defense.
