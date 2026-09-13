"""
Validate a trained RNN on the copy task across a range of delay lengths.

The copy task (data/copy_task.py) measures long-range-dependency memory: the
model must hold k digits across delay_len blank steps and replay them at the
end. Only the final k target positions count toward accuracy (targets are
-100 everywhere else). Chance accuracy is 1/8 = 12.5%.

Usage:
    python rnn_validation/validate_copy.py [--ckpt outputs/rnn_copy.pt]
                                           [--k 8] [--delays 3 10 30 50]
                                           [--batch_size 256] [--seed 42]

Prints per-delay token accuracy on the target window.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.copy_task import generate_copy_batch, VOCAB_SIZE
from models.sequence_models import build_model


def eval_delay(model, k, delay_len, batch_size, seed):
    x, y = generate_copy_batch(batch_size, k=k, delay_len=delay_len, seed=seed)
    with torch.no_grad():
        logits, _, _ = model(x)
    preds = logits.argmax(-1)
    mask = y != -100  # only the final k positions carry a loss/target
    correct = (preds[mask] == y[mask])
    return correct.float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/rnn_copy.pt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--delays", type=int, nargs="+", default=[3, 10, 30, 50])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    model = build_model("rnn", VOCAB_SIZE).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    print(f"model: rnn  |  k={args.k}  |  ckpt: {args.ckpt}  |  chance = {1/args.k:.1%}")
    print(f"{'delay_len':>10}  {'token acc':>12}  {'seq_len':>8}")
    for delay_len in args.delays:
        seq_len = args.k + delay_len + 1 + args.k
        acc = eval_delay(model, args.k, delay_len, args.batch_size, args.seed)
        print(f"{delay_len:>10}  {acc:>12.1%}  {seq_len:>8}")


if __name__ == "__main__":
    main()