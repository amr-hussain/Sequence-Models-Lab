"""
Self-attention, built from the equation up. No nn.MultiheadAttention.

Do this file last. It's the biggest conceptual jump from the previous three:
RNN/LSTM/GRU all process the sequence step-by-step, carrying state forward.
Attention processes the WHOLE sequence at once and lets every position look
directly at every other position, with no recurrence at all.

Core equation (Vaswani et al. 2017):
    Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V

Shapes:
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    Q @ K^T -> (batch, seq_len, seq_len)   <- this IS the attention matrix,
                                              row i = "how much position i
                                              attends to every other position"
"""
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, num_heads, seq_len, d_k)
        K: (batch, num_heads, seq_len, d_k)
        V: (batch, num_heads, seq_len, d_v)
        mask: optional (seq_len, seq_len) or (batch, 1, seq_len, seq_len) boolean
              tensor; True/1 means "block this position" (used for causal
              masking in language modeling — position t must not see t+1..T)

    Returns:
        output: (batch, num_heads, seq_len, d_v)
        attn_weights: (batch, num_heads, seq_len, seq_len) -- SAVE this, you
              need it for visualize_attention.py

    TODO(1): Implement the equation above, plus masking and softmax.

    Steps:
        1. d_k = Q.shape[-1]
        2. scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
           -> why divide by sqrt(d_k)? Think about the variance of a dot
              product of two random d_k-dimensional vectors as d_k grows,
              and what happens to softmax when its inputs have large
              variance (hint: it saturates, gradients vanish). Write one
              sentence about this in your thesis notes once you've
              implemented it.
        3. if mask is not None: scores = scores.masked_fill(mask, float('-inf'))
        4. attn_weights = softmax(scores, dim=-1)
        5. output = attn_weights @ V
    """
    d_k = Q.shape[-1]

    # TODO(1): implement scaled dot-product attention as described above
    raise NotImplementedError("Implement scaled dot-product attention (see TODO(1) docstring)")

    return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.d_k = embed_size // num_heads

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, embed_size)
        Returns:
            out: (batch, seq_len, embed_size)
            attn_weights: (batch, num_heads, seq_len, seq_len)

        TODO(2): Project to Q/K/V, split into heads, run attention, merge
        heads back, project out.

        Steps:
            1. batch, seq_len, _ = x.shape
            2. Q = self.W_q(x); K = self.W_k(x); V = self.W_v(x)
               each currently (batch, seq_len, embed_size)
            3. reshape each to (batch, seq_len, num_heads, d_k) then
               .transpose(1, 2) -> (batch, num_heads, seq_len, d_k)
            4. out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
            5. merge heads: out.transpose(1, 2) -> (batch, seq_len, num_heads, d_k)
               -> .reshape(batch, seq_len, embed_size)  (use .contiguous() before
               reshape if needed)
            6. out = self.W_o(out)
        """
        batch, seq_len, _ = x.shape

        # TODO(2): implement multi-head attention as described above
        raise NotImplementedError("Implement multi-head attention (see TODO(2) docstring)")

        return out, attn_weights


def causal_mask(seq_len, device=None):
    """Standard upper-triangular causal mask for language modeling.
    Returns a (seq_len, seq_len) boolean tensor, True = blocked position."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al.), added to token embeddings
    BEFORE the first attention layer.

    Why this file exists at all: attention has NO notion of order built in —
    Attention(Q,K,V) is invariant to permuting the sequence dimension
    (verify this claim for yourself: if you permute rows of Q, K, V identically,
    what happens to the output? Try it in code with torch and a random
    permutation once you've built MultiHeadSelfAttention.) RNN/LSTM/GRU get
    order for free because they process tokens one at a time in sequence.
    Attention has to be told the order explicitly, and this module is how.
    """

    def __init__(self, embed_size, max_len=5000):
        super().__init__()

        # TODO(3): Build the (max_len, embed_size) positional encoding matrix.
        #
        # pe[pos, 2i]   = sin(pos / 10000^(2i/embed_size))
        # pe[pos, 2i+1] = cos(pos / 10000^(2i/embed_size))
        #
        # Steps:
        #   1. position = torch.arange(max_len).unsqueeze(1)              # (max_len, 1)
        #   2. div_term = torch.exp(torch.arange(0, embed_size, 2) *
        #                           (-math.log(10000.0) / embed_size))    # (embed_size/2,)
        #   3. pe = torch.zeros(max_len, embed_size)
        #   4. pe[:, 0::2] = torch.sin(position * div_term)
        #   5. pe[:, 1::2] = torch.cos(position * div_term)
        #   6. register as a buffer (not a parameter — it's fixed, not learned):
        #        self.register_buffer('pe', pe)
        raise NotImplementedError("Build the sinusoidal positional encoding (see TODO(3) docstring)")

    def forward(self, x):
        """x: (batch, seq_len, embed_size) -> adds positional encoding, same shape."""
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :].unsqueeze(0)


if __name__ == "__main__":
    x = torch.randn(2, 10, 32)
    mha = MultiHeadSelfAttention(embed_size=32, num_heads=4)
    mask = causal_mask(10)
    out, attn = mha(x, mask=mask)
    print("output shape:", out.shape)
    print("attn shape:", attn.shape)
