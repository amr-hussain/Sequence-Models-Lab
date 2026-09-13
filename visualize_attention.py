"""
Visualize what a trained (or even untrained) Transformer attends to.
Run AFTER models/attention.py TODOs are implemented and (ideally) after
train.py has trained a transformer on shakespeare and saved a checkpoint.

Usage:
    python visualize_attention.py --checkpoint outputs/transformer_shakespeare.pt
"""
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from data.shakespeare import ShakespeareCharDataset
from models.sequence_models import TransformerLM


def plot_attention_heatmap(attn_weights, tokens, layer=0, head=0, save_path="outputs/attention_heatmap.png"):
    """
    attn_weights: (batch, num_heads, seq_len, seq_len) for one layer
    tokens: list of str, length seq_len -- the actual characters, for axis labels
    """
    weights = attn_weights[0, head].detach().cpu().numpy()  # (seq_len, seq_len)

    plt.figure(figsize=(8, 7))
    sns.heatmap(weights, xticklabels=tokens, yticklabels=tokens, cmap="viridis",
                square=True, cbar_kws={"label": "attention weight"})
    plt.xlabel("attending TO (key positions)")
    plt.ylabel("attending FROM (query positions)")
    plt.title(f"Self-attention, layer {layer}, head {head}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="path to a trained transformer .pt file; omit to use an untrained model")
    parser.add_argument("--seq_len", type=int, default=40)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--head", type=int, default=0)
    args = parser.parse_args()

    ds = ShakespeareCharDataset(seq_len=args.seq_len, split="val")
    model = TransformerLM(vocab_size=ds.vocab.vocab_size, embed_size=32, num_heads=4, num_layers=2)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"loaded weights from {args.checkpoint}")
    else:
        print("no checkpoint given -- visualizing an UNTRAINED model (attention will look near-random; "
              "train one first with train.py --model transformer --task shakespeare for a meaningful plot)")

    model.eval()
    x, y = ds[0]
    x = x.unsqueeze(0)  # (1, seq_len)
    tokens = list(ds.vocab.decode(x[0].tolist()))

    with torch.no_grad():
        _, _, all_attn_weights = model(x)

    plot_attention_heatmap(all_attn_weights[args.layer], tokens, layer=args.layer, head=args.head)


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()
