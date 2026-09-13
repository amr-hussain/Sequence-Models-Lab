"""
LSTM cell, gates implemented BY HAND (not nn.LSTM/nn.LSTMCell) using
PyTorch tensors. Backward pass is handled by autograd — by this point you've
already proven to yourself (via numpy_rnn.py) that you understand BPTT, so
here the goal shifts to understanding the GATING mechanism itself.

Do not import torch.nn.LSTM or torch.nn.LSTMCell anywhere in this file.
The whole point is that you write out W_f, W_i, W_o, W_c explicitly.

Equations (Gers et al. formulation, the standard one):
    f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)     # forget gate
    i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)     # input gate
    o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)     # output gate
    g_t = tanh   (W_c @ [h_{t-1}, x_t] + b_c)     # candidate cell state

    c_t = f_t * c_{t-1} + i_t * g_t               # <-- THE key line. Additive,
                                                    #     not multiplicative like
                                                    #     vanilla RNN's h_t update.
    h_t = o_t * tanh(c_t)

where [h_{t-1}, x_t] means concatenation along the feature dimension.

After you implement this, go back to analyze_gradients.py and compare
||dL/dc_t|| across timesteps to the vanilla RNN's ||dL/dh_t||. The forget
gate f_t sitting on a residual/additive path is why LSTM's gradient doesn't
have to shrink multiplicatively at every single step the way RNN's does.
"""
import torch
import torch.nn as nn


class LSTMCellManual(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        concat_size = input_size + hidden_size

        # One weight matrix + bias per gate. Combine [h, x] -> gate_size via
        # a single Linear layer each; this IS how real implementations do it,
        # it's just four separate ones instead of nn.LSTM's fused matrix.
        self.W_f = nn.Linear(concat_size, hidden_size)
        self.W_i = nn.Linear(concat_size, hidden_size)
        self.W_o = nn.Linear(concat_size, hidden_size)
        self.W_c = nn.Linear(concat_size, hidden_size)

    def forward(self, x_t, state):
        """
        Args:
            x_t:   (batch, input_size)
            state: tuple (h_prev, c_prev), each (batch, hidden_size)

        Returns:
            h_t, c_t: each (batch, hidden_size)
            gates: dict with keys 'f','i','o','g' holding the gate activations
                   at this timestep (batch, hidden_size) — keep these, you'll
                   want them later to plot gate activity over a sequence.

        TODO(1): Implement the four gates and the state updates using the
        equations in the module docstring.

        Steps:
            1. concat = torch.cat([h_prev, x_t], dim=-1)
            2. f_t = torch.sigmoid(self.W_f(concat))
            3. i_t = torch.sigmoid(self.W_i(concat))
            4. o_t = torch.sigmoid(self.W_o(concat))
            5. g_t = torch.tanh(self.W_c(concat))
            6. c_t = f_t * c_prev + i_t * g_t
            7. h_t = o_t * torch.tanh(c_t)
        """
        h_prev, c_prev = state

        # TODO(1): implement the gate computations described above
        raise NotImplementedError("Implement the LSTM gates (see TODO(1) docstring)")

        gates = {"f": f_t, "i": i_t, "o": o_t, "g": g_t}
        return h_t, c_t, gates


class LSTMSequenceModel(nn.Module):
    """Runs LSTMCellManual over a full sequence. This part is complete —
    focus your effort on the cell above."""

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.cell = LSTMCellManual(embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len) integer token ids
        Returns: logits (batch, seq_len, vocab_size), final_state, all_gates (list over t)
        """
        batch, seq_len = x.shape
        if state is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)
            c = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h, c = state

        embedded = self.embed(x)  # (batch, seq_len, embed_size)
        logits_list = []
        gates_over_time = []
        for t in range(seq_len):
            h, c, gates = self.cell(embedded[:, t, :], (h, c))
            logits_list.append(self.out(h))
            gates_over_time.append(gates)

        logits = torch.stack(logits_list, dim=1)  # (batch, seq_len, vocab_size)
        return logits, (h, c), gates_over_time


if __name__ == "__main__":
    model = LSTMSequenceModel(vocab_size=10, embed_size=16, hidden_size=32)
    x = torch.randint(0, 10, (4, 20))
    logits, state, gates = model(x)
    print("logits shape:", logits.shape)
    print("num timesteps of gate activations recorded:", len(gates))
