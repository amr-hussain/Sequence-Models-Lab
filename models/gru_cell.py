"""
GRU cell, gates implemented BY HAND (no nn.GRU/nn.GRUCell).

Do this file AFTER lstm_cell.py. The instructive comparison is: GRU merges
LSTM's cell state and hidden state into one, and merges the forget+input
gates into a single "update gate". Implementing it right after LSTM should
feel like doing algebra on what you already built, not learning something new.

Equations:
    z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)               # update gate
    r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)               # reset gate
    n_t = tanh   (W_n @ [r_t * h_{t-1}, x_t] + b_n)         # candidate state
                                                              # note: r_t gates
                                                              # h_{t-1} BEFORE
                                                              # concatenation
    h_t = (1 - z_t) * h_{t-1} + z_t * n_t                   # interpolation,
                                                              # not addition —
                                                              # this is the key
                                                              # structural
                                                              # difference from
                                                              # LSTM's c_t update
"""
import torch
import torch.nn as nn


class GRUCellManual(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        self.W_z = nn.Linear(concat_size, hidden_size)
        self.W_r = nn.Linear(concat_size, hidden_size)
        self.W_n = nn.Linear(concat_size, hidden_size)

    def forward(self, x_t, h_prev):
        """
        Args:
            x_t:    (batch, input_size)
            h_prev: (batch, hidden_size)

        Returns:
            h_t: (batch, hidden_size)
            gates: dict with keys 'z','r','n'

        TODO(1): Implement the update gate, reset gate, candidate state, and
        interpolated output using the equations in the module docstring.

        Careful with step 3 (candidate state n_t): the reset gate r_t
        multiplies h_prev BEFORE it's concatenated with x_t — this is
        different from how f_t/i_t/o_t are applied in the LSTM, where the
        gate is applied to the cell state AFTER the linear layer, not before.
        Getting this order right is the easiest place to introduce a subtle
        bug, so pay attention to it.

        Steps:
            1. concat_zr = torch.cat([h_prev, x_t], dim=-1)
            2. z_t = torch.sigmoid(self.W_z(concat_zr))
            3. r_t = torch.sigmoid(self.W_r(concat_zr))
            4. concat_n = torch.cat([r_t * h_prev, x_t], dim=-1)
            5. n_t = torch.tanh(self.W_n(concat_n))
            6. h_t = (1 - z_t) * h_prev + z_t * n_t
        """
        # TODO(1): implement the gate computations described above
        raise NotImplementedError("Implement the GRU gates (see TODO(1) docstring)")

        gates = {"z": z_t, "r": r_t, "n": n_t}
        return h_t, gates


class GRUSequenceModel(nn.Module):
    """Runs GRUCellManual over a full sequence. Complete — mirrors
    LSTMSequenceModel in lstm_cell.py so the two are directly comparable."""

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cell = GRUCellManual(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        batch, seq_len = x.shape
        if h is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)

        embedded = self.embed(x)
        logits_list = []
        gates_over_time = []
        for t in range(seq_len):
            h, gates = self.cell(embedded[:, t, :], h)
            logits_list.append(self.out(h))
            gates_over_time.append(gates)

        logits = torch.stack(logits_list, dim=1)
        return logits, h, gates_over_time


if __name__ == "__main__":
    model = GRUSequenceModel(vocab_size=10, embed_size=16, hidden_size=32)
    x = torch.randint(0, 10, (4, 20))
    logits, h, gates = model(x)
    print("logits shape:", logits.shape)

    # Quick parameter-count comparison prompt for your writeup:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GRU total params: {n_params}")
