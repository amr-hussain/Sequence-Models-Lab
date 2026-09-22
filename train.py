"""
Educational trainer for the Sequence Models Lab.

The important PyTorch training sequence stays visible:
    1. forward pass
    2. compute cross-entropy loss
    3. optimizer.zero_grad()
    4. loss.backward()
    5. gradient clipping
    6. optimizer.step()

Examples:
    python train.py --model rnn --task copy --k 5 --delay_len 4 --epochs 30 --seed 42
    python train.py --model lstm --task copy --k 5 --delay_len 20 --epochs 30 --seed 42
    python train.py --model transformer --task shakespeare --epochs 10 --seed 42
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.copy_task import CopyTaskDataset, VOCAB_SIZE as COPY_VOCAB
from data.shakespeare import ShakespeareCharDataset
from models.sequence_models import build_model

IGNORE_INDEX = -100


def set_seed(seed):
    """Make separate runs comparable by starting from the same randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Useful for a lab where reproducibility matters more than maximum speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(task, batch_size, seed, k=8, delay_len=50, seq_len=128):
    if task == "copy":
        train_ds = CopyTaskDataset(
            k=k, delay_len=delay_len, num_samples=20_000, seed=0
        )
        val_ds = CopyTaskDataset(
            k=k, delay_len=delay_len, num_samples=2_000, seed=999
        )
        vocab_size = COPY_VOCAB
    elif task == "shakespeare":
        train_ds = ShakespeareCharDataset(seq_len=seq_len, split="train")
        val_ds = ShakespeareCharDataset(seq_len=seq_len, split="val")
        vocab_size = train_ds.vocab.vocab_size
    else:
        raise ValueError(f"Unknown task: {task}")

    # This generator controls the order produced by shuffle=True.
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, vocab_size


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    """
    Run one epoch.

    optimizer is not None -> training mode and parameter updates.
    optimizer is None     -> validation mode, no gradients or updates.

    Accuracy is measured only at valid target positions. On the copy task,
    positions whose target is -100 are deliberately ignored.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    # During validation, disabling autograd saves memory and computation.
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # FORWARD PASS
            logits, _, _ = model(x)  # (batch, seq_len, vocab_size)

            # Cross-entropy expects (N, classes), so batch and time are merged.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            if training:
                # BACKWARD PASS AND PARAMETER UPDATE
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # Measure only positions that participate in the loss.
            mask = y != IGNORE_INDEX
            predictions = logits.argmax(dim=-1)
            valid_tokens = mask.sum().item()
            correct = ((predictions == y) & mask).sum().item()

            # loss.item() is the mean over valid tokens in this batch.
            total_loss += loss.item() * valid_tokens
            total_correct += correct
            total_tokens += valid_tokens
            n_batches += 1

            if max_batches is not None and n_batches >= max_batches:
                break

    mean_loss = total_loss / max(total_tokens, 1)
    accuracy = total_correct / max(total_tokens, 1)
    return mean_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["rnn", "lstm", "gru", "transformer", "mamba"],
        required=True,
    )
    parser.add_argument("--task", choices=["copy", "shakespeare"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--delay_len", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_batches_per_epoch", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  seed: {args.seed}")

    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader, vocab_size = get_dataloaders(
        task=args.task,
        batch_size=args.batch_size,
        seed=args.seed,
        k=args.k,
        delay_len=args.delay_len,
        seq_len=args.seq_len,
    )

    model = build_model(
        args.model,
        vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}  |  vocab_size: {vocab_size}  |  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_path = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_batches=args.max_batches_per_epoch,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            max_batches=max(1, args.max_batches_per_epoch // 4),
        )

        elapsed = time.time() - start

        print(
            f"epoch {epoch:>2}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.1%}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.1%}  "
            f"({elapsed:.1f}s)"
        )

        # Keep the best validation checkpoint, not merely the final epoch.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.task == "copy":
                run_name = (
                    f"{args.model}_copy_k{args.k}_delay{args.delay_len}"
                    f"_seed{args.seed}"
                )
            else:
                run_name = f"{args.model}_shakespeare_seed{args.seed}"

            best_path = f"outputs/{run_name}.pt"
            torch.save(model.state_dict(), best_path)

    print(f"best validation loss: {best_val_loss:.4f}")
    print(f"saved best weights to: {best_path}")


if __name__ == "__main__":
    main()
