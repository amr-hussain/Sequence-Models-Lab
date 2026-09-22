"""
Measure how a loss at the final timestep sends gradients backward through:
    1. a manual NumPy vanilla RNN
    2. a manual PyTorch LSTM

The plotted values are normalized by the gradient at the loss position. This
focuses the comparison on how much gradient survives as distance increases,
not on unrelated differences in output-layer scale.

Run:
    python analyze_gradients.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lstm_cell import LSTMCellManual
from models.gru_cell import GRUCellManual
from models.numpy_rnn import VanillaRNN


def normalize_to_loss_position(norms):
    """At the final timestep, relative gradient is 1.0."""
    norms = np.asarray(norms, dtype=np.float64)
    return norms / (norms[-1] + 1e-30)


def measure_rnn_gradients(seq_len, input_size=10, hidden_size=32,
                          output_size=10, seed=0):
    rng = np.random.default_rng(seed)
    model = VanillaRNN(input_size, hidden_size, output_size, seed=seed)

    xs = []
    for _ in range(seq_len):
        x_t = np.zeros(input_size)
        x_t[rng.integers(input_size)] = 1.0
        xs.append(x_t)

    # Only the final output contributes directly to the loss.
    targets = [None] * (seq_len - 1) + [rng.integers(output_size)]

    _, probabilities, hidden_states = model.forward(xs)
    _, gradient_norms = model.backward(
        xs, targets, probabilities, hidden_states
    )
    return np.asarray(gradient_norms)


def measure_lstm_gradients(seq_len, input_size=10, hidden_size=32,
                           output_size=10, seed=0):
    torch.manual_seed(seed)

    # Inputs are one-hot vectors, matching the NumPy RNN experiment. There is
    # deliberately no embedding layer here because we are testing recurrence.
    token_ids = torch.randint(0, input_size, (seq_len,))
    xs = F.one_hot(token_ids, num_classes=input_size).float()

    cell = LSTMCellManual(input_size, hidden_size)
    output_layer = nn.Linear(hidden_size, output_size)

    h = torch.zeros(1, hidden_size)
    c = torch.zeros(1, hidden_size)

    h_gradient_norms = [0.0] * seq_len
    c_gradient_norms = [0.0] * seq_len

    def save_norm(storage, timestep):
        # register_hook calls this function during backward and supplies the
        # gradient with respect to the tensor on which the hook was registered.
        def hook(gradient):
            storage[timestep] = gradient.norm().item()
        return hook

    for t in range(seq_len):
        h, c, _ = cell(xs[t].unsqueeze(0), (h, c))

        # h and c are tensors, not layers. They accept register_hook because
        # PyTorch recorded how autograd produced them.
        h.register_hook(save_norm(h_gradient_norms, t))
        c.register_hook(save_norm(c_gradient_norms, t))

    target = torch.randint(0, output_size, (1,))
    final_logits = output_layer(h)
    loss = F.cross_entropy(final_logits, target)
    loss.backward()

    return np.asarray(h_gradient_norms), np.asarray(c_gradient_norms)


def measure_gru_gradients(seq_len, input_size=10, hidden_size=32,
                          output_size=10, seed=0):
    torch.manual_seed(seed)

    token_ids = torch.randint(0, input_size, (seq_len,))
    xs = F.one_hot(token_ids, num_classes=input_size).float()

    cell = GRUCellManual(input_size, hidden_size)
    output_layer = nn.Linear(hidden_size, output_size)

    h = torch.zeros(1, hidden_size)

    h_gradient_norms = [0.0] * seq_len

    def save_norm(storage, timestep):
        def hook(gradient):
            storage[timestep] = gradient.norm().item()
        return hook

    for t in range(seq_len):
        h, _ = cell(xs[t].unsqueeze(0), h)

        h.register_hook(save_norm(h_gradient_norms, t))

    target = torch.randint(0, output_size, (1,))
    final_logits = output_layer(h)
    loss = F.cross_entropy(final_logits, target)
    loss.backward()

    return np.asarray(h_gradient_norms)


def average_over_seeds(measurement_function, seq_len, seeds):
    """Reduce the chance that one lucky/unlucky initialization dominates."""
    measurements = [measurement_function(seq_len, seed=seed) for seed in seeds]

    if isinstance(measurements[0], tuple):
        first = np.mean([m[0] for m in measurements], axis=0)
        second = np.mean([m[1] for m in measurements], axis=0)
        return first, second

    return np.mean(measurements, axis=0)


def plot_gradient_comparison(seq_len=40, seeds=range(5),
                             save_path="outputs/rnn_vs_lstm_gradients.png"):
    rnn = average_over_seeds(measure_rnn_gradients, seq_len, seeds)
    lstm_h, lstm_c = average_over_seeds(
        measure_lstm_gradients, seq_len, seeds
    )
    gru_h = average_over_seeds(measure_gru_gradients, seq_len, seeds)

    rnn = normalize_to_loss_position(rnn)
    lstm_h = normalize_to_loss_position(lstm_h)
    lstm_c = normalize_to_loss_position(lstm_c)
    gru_h = normalize_to_loss_position(gru_h)

    # Reverse the arrays as well as the x-axis. The plotted line now runs from
    # the loss position (distance 0) toward earlier timesteps.
    distance = np.arange(seq_len)

    plt.figure(figsize=(10, 6))
    plt.plot(distance, rnn[::-1], label="Vanilla RNN: relative ||dL/dh_t||")
    plt.plot(distance, lstm_h[::-1], label="LSTM: relative ||dL/dh_t||")
    plt.plot(distance, lstm_c[::-1], label="LSTM: relative ||dL/dc_t||")
    plt.plot(distance, gru_h[::-1], label="GRU: relative ||dL/dh_t||")

    plt.yscale("log")
    plt.xlabel("Steps before the final loss position")
    plt.ylabel("Gradient norm relative to the final timestep")
    plt.title(f"Gradient propagation through time, sequence length={seq_len}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    plot_gradient_comparison(seq_len=40)
