"""
Generic training loop, works identically for rnn/lstm/gru/transformer/mamba so
comparisons are apples-to-apples: same optimizer, same LR schedule, same
number of steps, same batch size, same (roughly) parameter budget.

Usage:
    python train.py --model lstm --task copy --delay_len 50
    python train.py --model transformer --task shakespeare

This file is complete. Your job was in models/*.py — run this once those
TODOs are filled in.
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.copy_task import CopyTaskDataset, VOCAB_SIZE as COPY_VOCAB
from data.shakespeare import ShakespeareCharDataset
from models.sequence_models import build_model


def get_dataloaders(task, batch_size, k=8, delay_len=50, seq_len=128):
    if task == "copy":
        train_ds = CopyTaskDataset(k=k, delay_len=delay_len, num_samples=20_000, seed=0)
        val_ds = CopyTaskDataset(k=k, delay_len=delay_len, num_samples=2_000, seed=999)
        vocab_size = COPY_VOCAB
    elif task == "shakespeare":
        train_ds = ShakespeareCharDataset(seq_len=seq_len, split="train")
        val_ds = ShakespeareCharDataset(seq_len=seq_len, split="val")
        vocab_size = train_ds.vocab.vocab_size
    else:
        raise ValueError(f"Unknown task: {task}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, vocab_size


def run_epoch(model, loader, optimizer, device, train=True, max_batches=None):
    model.train(train)
    total_loss, total_tokens, n_batches = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)  # (batch, seq_len, vocab_size)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=-100,  # copy task uses -100 for non-target positions
        )

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm", "gru", "transformer", "mamba"], required=True)
    parser.add_argument("--task", choices=["copy", "shakespeare"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=8, help="copy task: number of digits to remember")
    parser.add_argument("--delay_len", type=int, default=50, help="copy task: blank steps")
    parser.add_argument("--seq_len", type=int, default=128, help="shakespeare: chunk length")
    parser.add_argument("--max_batches_per_epoch", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs("outputs", exist_ok=True)  # checkpoints + plots land here

    train_loader, val_loader, vocab_size = get_dataloaders(
        args.task, args.batch_size, k=args.k, delay_len=args.delay_len, seq_len=args.seq_len
    )
    print(f"vocab_size: {vocab_size}")

    model = build_model(args.model, vocab_size, embed_size=args.embed_size,
                         hidden_size=args.hidden_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}  |  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True,
                                max_batches=args.max_batches_per_epoch)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False,
                              max_batches=args.max_batches_per_epoch // 4 or 1)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({dt:.1f}s)")

    torch.save(model.state_dict(), f"outputs/{args.model}_{args.task}.pt")
    print(f"saved to outputs/{args.model}_{args.task}.pt")


if __name__ == "__main__":
    main()
