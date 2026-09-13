# Sequential Models Lab: RNN vs LSTM vs GRU vs Transformer vs Mamba

A hands-on comparative study of the five core sequence architectures on identical
data, built Coursera-lab style: the plumbing is done for you, the architectural
core is `# TODO` for you to derive and implement.

## Prerequisites (concepts, not tools)

Before you touch code, make sure you can do these on paper:

- **Linear algebra**: matrix-vector products, chain rule for matrix calculus
  (i.e. can you differentiate `y = Wx + b` w.r.t. `W`, `x`, `b`?)
- **Backpropagation**: the general algorithm, not just "autograd does it"
- **Basic probability**: softmax, cross-entropy loss, why CE is the right loss
  for next-token prediction
- **Calculus**: derivatives of tanh, sigmoid (you'll need d/dx tanh(x) = 1 - tanh²(x)
  and d/dx sigmoid(x) = sigmoid(x)(1 - sigmoid(x)) constantly)

You do NOT need to already know LSTM/GRU/Transformer internals — that's the point
of the lab. You do need to be comfortable reading and writing Python + NumPy.

## Environment setup

```bash
# 1. Create an isolated environment (conda or venv, either is fine)
python3 -m venv seqlab-env
source seqlab-env/bin/activate        # Windows: seqlab-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Sanity check
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A GPU is NOT required for this lab — all models here are small (the point is
understanding, not scale) and will train on CPU in minutes. If you have a GPU,
great, it'll just be faster.

## Project structure

```
seq_lab/
├── data/
│   ├── copy_task.py        # Synthetic long-range dependency task (COMPLETE)
│   └── shakespeare.py      # Tiny Shakespeare char-level loader (COMPLETE)
├── models/
│   ├── numpy_rnn.py        # Vanilla RNN, pure NumPy, forward+backward (TODO)
│   ├── lstm_cell.py        # LSTM cell, manual gates, PyTorch autograd (TODO)
│   ├── gru_cell.py         # GRU cell, manual gates, PyTorch autograd (TODO)
│   ├── attention.py        # Scaled dot-product + multi-head attention (TODO)
│   ├── mamba_block.py      # Selective state space model / Mamba (TODO)
│   └── sequence_models.py  # Wraps cells into full seq2seq/LM models (COMPLETE)
├── train.py                 # Generic trainer, works with all 4 models (COMPLETE)
├── analyze_gradients.py     # Gradient-norm-vs-timestep diagnostic (COMPLETE)
├── visualize_attention.py   # Attention heatmap plotting (COMPLETE)
└── requirements.txt
```

## Suggested order of work

1. `models/numpy_rnn.py` — implement forward pass, then manual BPTT. This is
   the most important file in the whole lab. Do not skip the backward pass.
2. Run `analyze_gradients.py` on the copy task with the vanilla RNN. Watch
   gradients vanish as sequence length grows. This is your baseline failure.
3. `models/lstm_cell.py` — implement the four gates. Re-run the gradient
   analysis. Compare.
4. `models/gru_cell.py` — implement the two gates, as a simplification of
   LSTM. Compare parameter count and behavior to LSTM.
5. `models/attention.py` — implement scaled dot-product attention from the
   equation up. Run `visualize_attention.py` on Shakespeare and look at what
   your model attends to.
6. `models/mamba_block.py` — implement the selective scan. This is the newest
   idea in the lab: a recurrence (like RNN/LSTM/GRU) whose transition
   parameters are computed FROM the input at each step, rather than fixed —
   it's the direct answer to attention's O(n²) cost, scaling O(n) instead.
7. Run the full comparative sweep in `train.py` across both datasets and all
   five architectures, and write up what you observe — that write-up IS your
   thesis chapter.

## Why Mamba is in this lab

Attention lets every position look at every other position directly, which
is powerful but costs O(seq_len²). Mamba keeps a recurrent hidden state
like RNN/LSTM/GRU (O(seq_len) cost) but makes the recurrence *selective* —
its effective transition parameters are a function of the current input,
computed fresh at every timestep, instead of one fixed matrix reused for
the whole sequence. It's the natural fifth point of comparison once you've
built the other four: same family of questions (how is information carried
forward? how expensive is it? does it fix vanishing gradients? does it
generalize to longer sequences?), one more architecture's answer.

## Where each TODO teaches you something specific

| File | TODO | What it teaches |
|---|---|---|
| `numpy_rnn.py` | forward step | the recurrence equation itself |
| `numpy_rnn.py` | backward (BPTT) | why gradients vanish/explode — you'll compute the repeated Jacobian product by hand |
| `lstm_cell.py` | 4 gates | why additive cell-state update fixes vanishing gradients |
| `gru_cell.py` | 2 gates | how GRU merges/simplifies LSTM's gating |
| `attention.py` | scaled dot-product | why attention replaces recurrence entirely; the role of the sqrt(d_k) scale |
| `attention.py` | positional encoding | what recurrence gives you "for free" that attention must add back |
| `mamba_block.py` | selective scan | how making a recurrence input-dependent gives attention-like selectivity at O(n) cost instead of O(n²) |
