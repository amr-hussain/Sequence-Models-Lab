"""
Mamba block (selective state space model), implemented BY HAND. No
`mamba-ssm` package, no `causal_conv1d`, no `scan` from a library — the
recurrence below is the whole point of this file, write it yourself.

Do this file LAST, after attention.py. The instructive comparison chain:

    RNN/LSTM/GRU   -> a recurrence with a FIXED transition matrix W_hh
                      reused at every timestep (O(seq_len) compute)
    Transformer    -> no recurrence at all: every position looks at every
                      other position directly (O(seq_len^2) compute)
    Mamba          -> a recurrence whose TRANSITION PARAMETERS are computed
                      from the current input at every step: same O(seq_len)
                      family as RNN/LSTM/GRU, but "selective" the way
                      attention is. This is what makes it scale like a
                      recurrence while behaving like attention.

Continuous-time state space model (SSM):
    h'(t) = A h(t) + B(t) u(t)
    y(t)  = C(t) h(t)              (+ D u(t) as a skip connection)

Discretize with a per-timestep step size dt (zero-order hold, first-order):
    A_bar_t = exp(dt_t * A)        # A is a diagonal matrix here
    B_bar_t = dt_t * B_t
which turns the ODE into the recurrence you actually compute:
    h_t = A_bar_t * h_{t-1} + B_bar_t * u_t      # elementwise, d dims
    y_t = C_t * h_t                               # elementwise

Where does "selective" come in? dt_t, B_t and C_t are all functions of the
INPUT, produced by the three Linear projections below. Every timestep gets
its own transition — the model learns WHEN to keep vs. overwrite state from
the data, instead of one frozen matrix for the whole sequence like the RNN.

Why a diagonal A? With A diagonal, the per-step update is O(d) and the
whole scan is O(seq_len) — there is no d x d matrix to backprop through the
way there was with W_hh in numpy_rnn.py, and no seq_len x seq_len matrix the
way there is in attention. That's the O(n) vs O(n^2) win, made concrete once
you write the loop below and look at the shapes involved.
"""
import torch
import torch.nn as nn


class MambaBlockManual(nn.Module):
    """
    One selective SSM layer. Everything the recurrence needs is pre-projected
    once over the whole sequence, so the recurrence itself is a clean loop —
    same spirit as LSTMCellManual/GRUCellManual, which pre-concatenate and
    pre-project in a Linear per gate.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # in_proj writes the input twice: once for the SSM (u branch) and
        # once for a gating branch (z) combined afterward as silu(z) * out,
        # following the original Mamba architecture.
        self.in_proj = nn.Linear(d_model, 2 * d_model)

        # The fixed diagonal transition matrix, stored as ONE vector of
        # length d_model. Real Mamba stores the log and exponentiates for
        # numerical hygiene; an explicitly negative diagonal is equivalent
        # here and easier to read, so use self.A directly in the scan.
        self.A = nn.Parameter(-torch.rand(d_model))  # (d_model,) stable diag

        # The three input-dependent projections that make the recurrence
        # selective. Apply them to the whole sequence in one shot (batched
        # matmul), then slice per-timestep inside the loop.
        self.proj_dt = nn.Linear(d_model, d_model)   # -> softplus(dt)
        self.proj_B = nn.Linear(d_model, d_model)
        self.proj_C = nn.Linear(d_model, d_model)

        # Learned skip connection straight from input to output, one scalar
        # per channel. Outside the scan: y_out_t = y_t + D * u_t.
        self.D = nn.Parameter(torch.zeros(d_model))

        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()          # gate activation
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) — hidden states from the previous
               (embedding or Mamba) layer

        Returns:
            out: (batch, seq_len, d_model) — output of this block (the
                 LayerNorm'd residual, same shape as the input), so blocks
                 can be stacked just like TransformerBlock
            h_over_time: list of length seq_len, each (batch, d_model) —
                 the state vector at every step of the scan. Keep it for
                 the same reason you kept all hs in numpy_rnn.py: gradient
                 diagnostics and the "vanishing/exploding?" question.
        """
        batch, seq_len, _ = x.shape

        proj = self.in_proj(x)                 # (batch, seq_len, 2*d_model)
        u, z = proj.chunk(2, dim=-1)           # each (batch, seq_len, d_model)

        # --- the "selective" parameters, computed ONCE over the sequence ---
        # Each is a function of the input x, so every timestep gets a fresh
        # transition — this is the entire conceptual departure from the RNN.
        dt_all = torch.nn.functional.softplus(self.proj_dt(x))   # (batch, seq_len, d)
        B_all = self.proj_B(x)                                   # (batch, seq_len, d)
        C_all = self.proj_C(x)                                   # (batch, seq_len, d)

        # TODO(1): implement the selective scan (the S6 recurrence).
        #
        # Discretize + scan, per timestep t in 0..seq_len-1:
        #
        #   1. A_bar_t = torch.exp(dt_all[:, t, :] * self.A.unsqueeze(0))
        #               # (batch, d)   -- exp(dt * A), diagonal closed form
        #   2. B_bar_t = dt_all[:, t, :] * B_all[:, t, :]
        #               # (batch, d)   -- dt * B (first-order ZOH)
        #   3. h_t = A_bar_t * h_prev + B_bar_t * u[:, t, :]
        #               # (batch, d)   -- elementwise state update,
        #               #                the analogue of h_t=W_hh*h_{t-1}+... in
        #               #                numpy_rnn.py, but input-dependent
        #   4. y_t = C_all[:, t, :] * h_t
        #               # (batch, d)   -- elementwise readout
        #   5. store y_t into a list, and h_t into h_over_time (h_prev = h_t
        #      for the next step; initialize h_prev = zeros(batch, d_model,
        #      device=x.device), and note h_over_time[0] will be the state
        #      AFTER the first input, same convention as numpy_rnn.py).
        #
        # Then: y_stack = torch.stack(y_list, dim=1)  # (batch, seq_len, d)
        #       y_out   = y_stack + self.D * u        # skip the state entirely
        #       gated   = self.act(z) * y_out         # SiLU gated branch
        #       h       = self.norm(x + self.dropout(self.out_proj(gated)))
        #       return h, h_over_time
        #
        # Hint: you can vectorize steps 1-2 (compute A_bar and B_bar for ALL
        # timesteps at once, since dt_all and B_all are already batched), but
        # the h recurrence in step 3 must stay a loop over t — h_t depends on
        # h_{t-1}. That loop *is* the O(seq_len) scan; notice it is a single
        # elementwise line, nothing like W_hh @ h_{t-1}.
        raise NotImplementedError("Implement the Mamba selective scan (see TODO(1) docstring)")


class MambaSequenceModel(nn.Module):
    """
    Stacks MambaBlockManual layers into a full next-token-prediction LM so
    train.py can use it identically to LSTMSequenceModel / GRUSequenceModel /
    TransformerLM (same (logits, state, record) return convention).
    """

    def __init__(self, vocab_size, embed_size, hidden_size=None, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.ModuleList([
            MambaBlockManual(embed_size, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, state=None):
        """
        x: (batch, seq_len) integer token ids
        Returns:
            logits: (batch, seq_len, vocab_size)
            final_state: the hidden state after the last timestep of the
                    final block, kept for the uniform interface
            scan_record: list per timestep of the (batch, d_model) hidden
                    state from the final block — your gradient-analysis
                    hook for this architecture.
        """
        batch, seq_len = x.shape
        h = self.embed(x)  # (batch, seq_len, embed_size)

        scan_record = None
        final_state = None
        for block in self.blocks:
            h, h_over_time = block(h)
            scan_record = h_over_time      # state after each of the seq_len steps
            final_state = h_over_time[-1]  # state after the last timestep

        logits = self.out(self.norm(h))
        return logits, final_state, scan_record


if __name__ == "__main__":
    model = MambaSequenceModel(vocab_size=10, embed_size=16, hidden_size=16, num_layers=2)
    x = torch.randint(0, 10, (4, 20))
    logits, state, record = model(x)
    print("logits shape:", logits.shape)
    print("final state shape:", state.shape)
    print("scan record length (timesteps):", len(record))

    # Quick parameter-count comparison prompt for your writeup:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba total params: {n_params}  (LSTM is roughly twice this at the same hidden size)")