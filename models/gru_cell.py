"""
Manual GRU cell and sequence model.

This file intentionally does not use nn.GRU or nn.GRUCell. It is designed for
studying recurrent networks, gate behavior, gradient flow, and the copy task.

GRU equations used in this file
-------------------------------
    z_t = sigmoid(W_z([h_{t-1}, x_t]))
    r_t = sigmoid(W_r([h_{t-1}, x_t]))
    n_t = tanh(W_n([r_t * h_{t-1}, x_t]))
    h_t = (1 - z_t) * h_{t-1} + z_t * n_t

Gate convention
---------------
Under this convention, z_t is the amount of candidate information written:

    z_t near 0  -> preserve the previous hidden state
    z_t near 1  -> write the candidate state

Some books and libraries use the complementary convention. Either convention
can work, but equations, implementation, comments, and plots must agree.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


GateDict = Dict[str, Tensor]
DiagnosticsDict = Dict[str, Union[Tensor, List[Tensor]]]


class GRUCellManual(nn.Module):
    """A single GRU cell implemented with explicit PyTorch operations."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        # Each Linear layer includes both its weight matrix and bias vector.
        self.W_z = nn.Linear(concat_size, hidden_size)
        self.W_r = nn.Linear(concat_size, hidden_size)
        self.W_n = nn.Linear(concat_size, hidden_size)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tuple[Tensor, GateDict]:
        """
        Run one GRU timestep.

        Args:
            x_t:
                Current input with shape (batch, input_size).
            h_prev:
                Previous hidden state with shape (batch, hidden_size).

        Returns:
            h_t:
                New hidden state with shape (batch, hidden_size).
            gates:
                Dictionary containing z_t, r_t, and n_t. Each tensor has
                shape (batch, hidden_size).
        """
        self._validate_inputs(x_t, h_prev)

        # The update and reset gates inspect both the previous state and input.
        concat_zr = torch.cat([h_prev, x_t], dim=-1)
        z_t = torch.sigmoid(self.W_z(concat_zr))
        r_t = torch.sigmoid(self.W_r(concat_zr))

        # The reset gate controls which parts of h_prev are visible while the
        # candidate is being constructed. It acts before concatenation.
        concat_n = torch.cat([r_t * h_prev, x_t], dim=-1)
        n_t = torch.tanh(self.W_n(concat_n))

        # z_t is the candidate-write amount in this file's convention.
        h_t = (1.0 - z_t) * h_prev + z_t * n_t

        gates = {"z": z_t, "r": r_t, "n": n_t}
        return h_t, gates

    def _validate_inputs(self, x_t: Tensor, h_prev: Tensor) -> None:
        """Raise clear errors for common educational shape mistakes."""
        if x_t.ndim != 2:
            raise ValueError(
                "x_t must have shape (batch, input_size), "
                f"but received {tuple(x_t.shape)}"
            )
        if h_prev.ndim != 2:
            raise ValueError(
                "h_prev must have shape (batch, hidden_size), "
                f"but received {tuple(h_prev.shape)}"
            )
        if x_t.size(0) != h_prev.size(0):
            raise ValueError(
                "x_t and h_prev must have the same batch size, "
                f"but received {x_t.size(0)} and {h_prev.size(0)}"
            )
        if x_t.size(-1) != self.input_size:
            raise ValueError(
                f"Expected x_t feature size {self.input_size}, "
                f"but received {x_t.size(-1)}"
            )
        if h_prev.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"but received {h_prev.size(-1)}"
            )
        if x_t.device != h_prev.device:
            raise ValueError(
                "x_t and h_prev must be on the same device, "
                f"but received {x_t.device} and {h_prev.device}"
            )


class GRUSequenceModel(nn.Module):
    """Run GRUCellManual across a complete token sequence.

    The model produces one vocabulary prediction at every timestep, which is
    suitable for sequence-to-sequence copy-task targets.
    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_size <= 0:
            raise ValueError(f"embed_size must be positive, got {embed_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cell = GRUCellManual(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: Tensor,
        h: Optional[Tensor] = None,
        *,
        return_diagnostics: bool = True,
        retain_hidden_grads: bool = False,
        detach_diagnostics: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tensor, DiagnosticsDict],
    ]:
        """
        Run the GRU over a batch of token sequences.

        Args:
            x:
                Integer token IDs with shape (batch, seq_len).
            h:
                Optional initial hidden state with shape
                (batch, hidden_size). If omitted, it is initialized to zero.
            return_diagnostics:
                If True, also return hidden states and gates across time.
            retain_hidden_grads:
                If True, call retain_grad() on every h_t and return the
                individual hidden-state graph nodes. After loss.backward(),
                node.grad can be inspected to plot BPTT gradient retention.
                This option requires gradient tracking to be enabled.
            detach_diagnostics:
                If True, detach stacked diagnostic tensors from the graph.
                Use this for value-only visualization. It cannot be combined
                with retain_hidden_grads=True.

        Returns:
            Without diagnostics:
                logits, final_h

            With diagnostics:
                logits, final_h, diagnostics

                diagnostics["hidden_states"]:
                    Shape (batch, seq_len, hidden_size).
                diagnostics["z"]:
                    Shape (batch, seq_len, hidden_size).
                diagnostics["r"]:
                    Shape (batch, seq_len, hidden_size).
                diagnostics["n"]:
                    Shape (batch, seq_len, hidden_size).
                diagnostics["hidden_state_nodes"]:
                    A list containing each unstacked h_t. This key is present
                    only when retain_hidden_grads=True.
        """
        self._validate_sequence_input(x)

        if retain_hidden_grads and detach_diagnostics:
            raise ValueError(
                "retain_hidden_grads=True cannot be combined with "
                "detach_diagnostics=True"
            )
        if retain_hidden_grads and not torch.is_grad_enabled():
            raise RuntimeError(
                "retain_hidden_grads=True requires gradient tracking. "
                "Do not use it inside torch.no_grad()."
            )

        batch_size, seq_len = x.shape

        if h is None:
            # Match both the embedding's device and floating-point dtype.
            h = self.embed.weight.new_zeros(batch_size, self.hidden_size)
        else:
            self._validate_initial_hidden(h, batch_size)

        embedded = self.embed(x)

        logits_over_time: List[Tensor] = []
        hidden_over_time: List[Tensor] = []
        z_over_time: List[Tensor] = []
        r_over_time: List[Tensor] = []
        n_over_time: List[Tensor] = []

        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h, gates = self.cell(x_t, h)

            if retain_hidden_grads:
                h.retain_grad()

            logits_over_time.append(self.out(h))

            if return_diagnostics:
                hidden_over_time.append(h)
                z_over_time.append(gates["z"])
                r_over_time.append(gates["r"])
                n_over_time.append(gates["n"])

        logits = torch.stack(logits_over_time, dim=1)

        if not return_diagnostics:
            return logits, h

        hidden_states = torch.stack(hidden_over_time, dim=1)
        z_values = torch.stack(z_over_time, dim=1)
        r_values = torch.stack(r_over_time, dim=1)
        n_values = torch.stack(n_over_time, dim=1)

        if detach_diagnostics:
            hidden_states = hidden_states.detach()
            z_values = z_values.detach()
            r_values = r_values.detach()
            n_values = n_values.detach()

        diagnostics: DiagnosticsDict = {
            "hidden_states": hidden_states,
            "z": z_values,
            "r": r_values,
            "n": n_values,
        }

        if retain_hidden_grads:
            # Keep the individual non-leaf nodes because gradients retained on
            # h_t are available through these tensors after backward().
            diagnostics["hidden_state_nodes"] = hidden_over_time

        return logits, h, diagnostics

    def _validate_sequence_input(self, x: Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(
                "x must have shape (batch, seq_len), "
                f"but received {tuple(x.shape)}"
            )
        if x.size(1) == 0:
            raise ValueError("seq_len must be at least 1")
        if x.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "x must contain integer token IDs with dtype torch.int32 "
                f"or torch.int64, but received {x.dtype}"
            )

    def _validate_initial_hidden(self, h: Tensor, batch_size: int) -> None:
        expected_shape = (batch_size, self.hidden_size)
        if tuple(h.shape) != expected_shape:
            raise ValueError(
                f"Initial h must have shape {expected_shape}, "
                f"but received {tuple(h.shape)}"
            )
        if h.device != self.embed.weight.device:
            raise ValueError(
                "Initial h and the model must be on the same device, "
                f"but received {h.device} and {self.embed.weight.device}"
            )
        if h.dtype != self.embed.weight.dtype:
            raise TypeError(
                "Initial h and the model must use the same floating-point "
                f"dtype, but received {h.dtype} and {self.embed.weight.dtype}"
            )


def _run_smoke_tests() -> None:
    """Check shapes, value ranges, gradients, and basic device behavior."""
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 20
    vocab_size = 10
    embed_size = 16
    hidden_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUSequenceModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
    ).to(device)

    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )

    logits, final_h, diagnostics = model(
        x,
        return_diagnostics=True,
        retain_hidden_grads=True,
    )

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert final_h.shape == (batch_size, hidden_size)
    assert diagnostics["hidden_states"].shape == (
        batch_size,
        seq_len,
        hidden_size,
    )

    for name in ("z", "r", "n"):
        assert diagnostics[name].shape == (
            batch_size,
            seq_len,
            hidden_size,
        )

    assert torch.all((diagnostics["z"] >= 0) & (diagnostics["z"] <= 1))
    assert torch.all((diagnostics["r"] >= 0) & (diagnostics["r"] <= 1))
    assert torch.all((diagnostics["n"] >= -1) & (diagnostics["n"] <= 1))

    # A simple artificial loss verifies that gradients reach every stored h_t.
    loss = logits.square().mean()
    loss.backward()

    hidden_nodes = diagnostics["hidden_state_nodes"]
    assert len(hidden_nodes) == seq_len
    assert all(node.grad is not None for node in hidden_nodes)

    hidden_grad_norms = torch.stack(
        [node.grad.norm(dim=-1).mean() for node in hidden_nodes]
    )
    assert torch.isfinite(hidden_grad_norms).all()

    n_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"device: {device}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"final hidden shape: {tuple(final_h.shape)}")
    print(
        "hidden states shape: "
        f"{tuple(diagnostics['hidden_states'].shape)}"
    )
    print(f"GRU total params: {n_params:,}")
    print("All GRU smoke tests passed.")


if __name__ == "__main__":
    _run_smoke_tests()