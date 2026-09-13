"""
Wraps your implemented cells (LSTM, GRU) and attention module into complete
sequence models comparable on identical tasks, plus a plain PyTorch RNN LM
(nn.RNN) as the "reference implementation" you can check numpy_rnn.py's
math against once you're confident in it. This file is complete — no TODOs.
Import from here in train.py.
"""
import torch
import torch.nn as nn

from models.lstm_cell import LSTMSequenceModel
from models.gru_cell import GRUSequenceModel
from models.attention import MultiHeadSelfAttention, PositionalEncoding, causal_mask
from models.mamba_block import MambaSequenceModel


class RNNBaseline(nn.Module):
    """Reference nn.RNN language model, for cross-checking your numpy_rnn.py."""

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True, nonlinearity="tanh")
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        out, h = self.rnn(e, h)
        logits = self.out(out)
        return logits, h, None  # None keeps the (logits, state, gates) interface uniform


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_size, num_heads)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_mult * embed_size),
            nn.GELU(),
            nn.Linear(ff_mult * embed_size, embed_size),
        )
        self.ln2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, mask=mask)
        x = self.ln1(x + self.dropout(attn_out))       # pre/post-norm residual, standard block
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x, attn_weights


class TransformerLM(nn.Module):
    """
    Single- or multi-layer decoder-only Transformer, causal-masked, for
    next-token prediction on identical data to the RNN/LSTM/GRU models.
    """

    def __init__(self, vocab_size, embed_size, num_heads=4, num_layers=2, max_len=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_enc = PositionalEncoding(embed_size, max_len=max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, state=None):
        """state is unused; kept for interface parity with the recurrent models."""
        batch, seq_len = x.shape
        mask = causal_mask(seq_len, device=x.device)

        h = self.embed(x)
        h = self.pos_enc(h)

        all_attn_weights = []
        for block in self.blocks:
            h, attn_weights = block(h, mask=mask)
            all_attn_weights.append(attn_weights)  # each (batch, num_heads, seq_len, seq_len)

        logits = self.out(h)
        return logits, None, all_attn_weights


def build_model(model_type, vocab_size, embed_size=32, hidden_size=64, num_heads=4, num_layers=2):
    """Factory so train.py can build any of the four with one call."""
    if model_type == "rnn":
        return RNNBaseline(vocab_size, embed_size, hidden_size)
    elif model_type == "lstm":
        return LSTMSequenceModel(vocab_size, embed_size, hidden_size)
    elif model_type == "gru":
        return GRUSequenceModel(vocab_size, embed_size, hidden_size)
    elif model_type == "transformer":
        return TransformerLM(vocab_size, embed_size, num_heads=num_heads, num_layers=num_layers)
    elif model_type == "mamba":
        return MambaSequenceModel(vocab_size, embed_size, hidden_size=hidden_size,
                                  num_layers=num_layers)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
