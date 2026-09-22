"""
Validate any trained sequence model on the copy task.

Examples:
    python rnn_validation/validate_copy.py \
        --model rnn \
        --ckpt outputs/rnn_copy_k5_delay4_seed42.pt \
        --k 5 --delays 4 8 16

    python rnn_validation/validate_copy.py \
        --model lstm \
        --ckpt outputs/lstm_copy_k5_delay4_seed42.pt \
        --k 5 --delays 4 8 16 --show_example

    python rnn_validation/validate_copy.py \
    --model gru \
    --ckpt outputs/gru_copy_k5_delay4_seed42.pt \
    --k 5 --delays 4 8 16 --show_example
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.copy_task import generate_copy_batch, VOCAB_SIZE
from models.sequence_models import build_model

IGNORE_INDEX = -100
NUM_DIGITS = 8  # Copy-task digits are tokens 1..8, regardless of k.


def format_target(values):
    """Show ignored positions as dots so the recall window is easy to see."""
    return ["." if value == IGNORE_INDEX else str(value) for value in values]


def evaluate_delay(model, device, k, delay_len, batch_size, seed, show_example):
    x, y = generate_copy_batch(
        batch_size=batch_size,
        k=k,
        delay_len=delay_len,
        seed=seed,
    )
    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        logits, _, _ = model(x)

    predictions = logits.argmax(dim=-1)
    mask = y != IGNORE_INDEX
    accuracy = (predictions[mask] == y[mask]).float().mean().item()

    if show_example:
        x0 = x[0].cpu().tolist()
        y0 = y[0].cpu().tolist()
        p0 = predictions[0].cpu().tolist()

        recall_start = k + delay_len + 1
        print(f"\nExample for delay_len={delay_len}")
        print("input:       ", x0)
        print("target:      ", format_target(y0))
        print("target tail: ", y0[recall_start:])
        print("pred tail:   ", p0[recall_start:])
        print(
            "correct:     ",
            ["OK" if p == t else "X" for p, t in zip(
                p0[recall_start:], y0[recall_start:]
            )],
        )

    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["rnn", "lstm", "gru", "transformer", "mamba"],
        default="rnn",
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--delays", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument(
        "--show_example",
        action="store_true",
        help="Print one human-readable input/target/prediction example per delay.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        args.model,
        VOCAB_SIZE,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
    ).to(device)

    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(
        f"model: {args.model}  |  k={args.k}  |  device={device}  |  "
        f"chance token accuracy={1 / NUM_DIGITS:.1%}"
    )
    print(f"{'delay_len':>10}  {'token acc':>12}  {'seq_len':>8}")

    for delay_len in args.delays:
        seq_len = args.k + delay_len + 1 + args.k
        accuracy = evaluate_delay(
            model=model,
            device=device,
            k=args.k,
            delay_len=delay_len,
            batch_size=args.batch_size,
            seed=args.seed,
            show_example=args.show_example,
        )
        print(f"{delay_len:>10}  {accuracy:>12.1%}  {seq_len:>8}")


if __name__ == "__main__":
    main()
