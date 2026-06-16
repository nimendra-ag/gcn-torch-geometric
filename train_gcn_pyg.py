"""
Graph Convolutional Network (Kipf & Welling, ICLR 2017), reimplemented with
PyTorch Geometric's built-in "GCNConv".

Faithful port of the original TensorFlow reference (https://github.com/tkipf/gcn):
same 2-layer architecture, renormalized propagation rule (paper Eq. 8), same
hyperparameters, weight decay on the first layer only, masked cross-entropy on
the public Planetoid splits, and the same early stopping rule.

Optionally reproduces Figure 1b of the paper: a t-SNE projection of the trained
model's hidden-layer activations, colored by class. Enable it with --tsne.

Run:
    python train_gcn_pyg.py --dataset cora            # train + report test acc
    python train_gcn_pyg.py --dataset cora --tsne     # also save the t-SNE plot

The --tsne option additionally requires scikit-learn and matplotlib.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# Reproducibility
def set_seed(seed: int) -> None:
    """Seed every RNG that affects training (matches `seed = 123` in train.py)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Model
class GCN(nn.Module):
    """Two-layer GCN matching the original models.GCN.

    forward() can optionally return the 16-dim hidden activations
    H = ReLU(conv1(X, A)) - the exact tensor the paper visualizes in Fig. 1b.

    GCNConv (normalize=True, add_self_loops=True) applies the renormalization
    trick on the fly, so we never build A_hat ourselves.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        bias: bool = False,   # original GraphConvolution defaults to bias=False
        cached: bool = True,  # transductive full-graph -> safe to cache A_hat
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(
            in_channels, hidden_channels,
            bias=bias, cached=cached, normalize=True, add_self_loops=True,
        )
        self.conv2 = GCNConv(
            hidden_channels, out_channels,
            bias=bias, cached=cached, normalize=True, add_self_loops=True,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor:
        # Dropout is applied to each layer's *input*, exactly as in layers.py.
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.conv1(x, edge_index, edge_weight))  # hidden activations [N, 16]
        if return_hidden:
            return h
        x = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight)  # raw logits


# Train / eval steps
def train_step(model: nn.Module, data, optimizer: torch.optim.Optimizer) -> float:
    """One full-batch gradient step on the training nodes."""
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    # Masked softmax cross-entropy == cross-entropy over the labeled subset.
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def evaluate(model: nn.Module, data, mask: Tensor) -> tuple[float, float]:
    """Return (loss, accuracy) on the nodes selected by mask."""
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[mask], data.y[mask])
    pred = logits.argmax(dim=1)
    acc = (pred[mask] == data.y[mask]).float().mean()
    return float(loss), float(acc)


# t-SNE of hidden activations (paper Fig. 1b) -- optional
def plot_tsne(model: nn.Module, data, perplexity: float, seed: int, out_path: str) -> None:
    """Project the trained model's hidden activations to 2D and save a scatter.

    Heavy/optional deps are imported lazily so the core training path doesn't
    depend on scikit-learn or matplotlib.
    """
    from sklearn.manifold import TSNE          # noqa: PLC0415  (lazy by design)
    import matplotlib
    matplotlib.use("Agg")                       # safe on headless machines
    import matplotlib.pyplot as plt

    model.eval()
    with torch.no_grad():
        hidden = model(data.x, data.edge_index, return_hidden=True).cpu().numpy()

    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(hidden)

    y = data.y.cpu().numpy()
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=12, alpha=0.85)
    plt.legend(*scatter.legend_elements(), title="Class", loc="best", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.title("t-SNE of GCN hidden-layer activations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved t-SNE figure to {out_path}")


# Main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GCN (Kipf & Welling 2017) in PyG.")
    p.add_argument("--dataset", default="cora",
                   choices=["cora", "citeseer", "pubmed"])
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stopping", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--root", default="data", help="Where to cache the dataset.")
    # t-SNE options
    p.add_argument("--tsne", action="store_true",
                   help="After training, save a t-SNE plot of hidden activations.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_out", default="tsne_gcn.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Planetoid 'public' split == the split used in the paper (20 train/class,
    # 500 val, 1000 test). NormalizeFeatures() == utils.preprocess_features.
    dataset = Planetoid(
        root=args.root,
        name=args.dataset.capitalize(),
        split="public",
        transform=T.NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    # Weight decay on the FIRST layer only - matches models.py.
    optimizer = torch.optim.Adam(
        [
            {"params": model.conv1.parameters(), "weight_decay": args.weight_decay},
            {"params": model.conv2.parameters(), "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
    )

    val_losses: list[float] = []
    for epoch in range(args.epochs):
        t = time.time()
        train_loss = train_step(model, data, optimizer)
        val_loss, val_acc = evaluate(model, data, data.val_mask)
        val_losses.append(val_loss)

        print(
            f"Epoch: {epoch + 1:04d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_acc={val_acc:.5f} "
            f"time={time.time() - t:.5f}"
        )

        # Original early stopping rule (train.py).
        if epoch > args.early_stopping:
            window = val_losses[-(args.early_stopping + 1):-1]
            if val_losses[-1] > sum(window) / len(window):
                print("Early stopping...")
                break

    test_loss, test_acc = evaluate(model, data, data.test_mask)
    print(f"Test set results: cost={test_loss:.5f} accuracy={test_acc:.5f}")

    if args.tsne:
        plot_tsne(model, data, args.perplexity, args.seed, args.tsne_out)


if __name__ == "__main__":
    main()