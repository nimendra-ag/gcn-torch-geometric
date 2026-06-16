# GCN in PyTorch Geometric

A faithful, modern reimplementation of the **Graph Convolutional Network (GCN)**
from:

> Thomas N. Kipf, Max Welling.
> **Semi-Supervised Classification with Graph Convolutional Networks.** ICLR 2017.
> Paper: [arXiv:1609.02907](http://arxiv.org/abs/1609.02907) ·
> Original code: [github.com/tkipf/gcn](https://github.com/tkipf/gcn)

The original 2016 reference implementation builds the graph-convolution layer,
the renormalized adjacency, sparse handling, and custom initialization **by hand
in TensorFlow**. This repository reproduces the same model using PyTorch
Geometric's built-in [`GCNConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html),
which performs the renormalization trick internally — so the same network is
expressed in a fraction of the code.

---

## What is reproduced from the paper

This repo implements the **core two-layer GCN for semi-supervised node
classification** exactly as described in the paper:

| Paper detail                                                 | Implementation                                                            |
|--------------------------------------------------------------|---------------------------------------------------------------------------|
| Renormalized propagation rule `Â = D̃⁻½(A+I)D̃⁻½` (Eq. 8)    | handled internally by `GCNConv` (`normalize=True`, `add_self_loops=True`) |
| Two-layer architecture: `C -> 16 -> F`                       | `GCNConv(in, 16)` -> ReLU -> `GCNConv(16, num_classes)`                   |
| Dropout 0.5 on each layer's input                            | `F.dropout(..., p=0.5)`                                                   |
| Row-normalized input features                                | `T.NormalizeFeatures()`                                                   |
| L2 weight decay `5e-4` on the **first layer only**           | per-layer optimizer parameter groups                                      |
| Adam, learning rate `0.01`, 200 epochs                       | matched                                                                   |
| Early stopping (window of 10 on validation loss)             | matched (same rule as `train.py`)                                         |
| Glorot weight initialization                                 | `GCNConv` default                                                         |
| Public Planetoid split (20 labels/class, 500 val, 1000 test) | `Planetoid(split="public")`                                               |
| Masked cross-entropy over labeled nodes only                 | `cross_entropy(out[train_mask], y[train_mask])`                           |
| Random seed 123                                              | matched                                                                   |

> **Scope :** this repository covers the main GCN model and its citation-network
> experiments (Cora / Citeseer / Pubmed). The paper's other components — the
> Chebyshev variant (`gcn_cheby`), the plain MLP baseline (`dense`), the NELL
> knowledge-graph and random-graph experiments, and the karate-club appendix —
> are **not** included here.

---

## Dataset

The model is trained on the **Cora** citation network (and optionally Citeseer
or Pubmed). Each node is a document with a sparse bag-of-words feature vector,
and edges are citation links.

| Dataset  | Nodes  | Features | Classes | Label rate |
|----------|--------|----------|---------|------------|
| Cora     | 2,708  | 1,433    | 7       | ~5.2%      |
| Citeseer | 3,327  | 3,703    | 6       | ~3.6%      |
| Pubmed   | 19,717 | 500      | 3       | ~0.3%      |

You do **not** need to download anything manually. PyTorch Geometric's
`Planetoid` loader downloads and caches the dataset automatically into `./data/`
on first run, using the exact `public` split from the paper.

---

## Requirements

```text
torch==2.12.0
torch_geometric==2.8.0
scipy==1.17.1
matplotlib==3.11.0
numpy==2.4.6
scikit-learn==1.9.0
```
Python Version - 3.12.13

Install:

```bash
pip install -r requirements.txt
```

## How to run

Train the GCN and report test accuracy:

```bash
python train_gcn_pyg.py
```

Choose a different dataset:

```bash
python train_gcn_pyg.py --dataset citeseer
python train_gcn_pyg.py --dataset pubmed
```

Expected result on Cora: **~81% test accuracy**, matching Table 2 of the paper.

All hyperparameters are exposed as command-line flags:

```bash
python train_gcn_pyg.py --help
```

(`--hidden`, `--learning_rate`, `--dropout`, `--weight_decay`, `--epochs`,
`--early_stopping`, `--seed`, ...)

---

## Optional: t-SNE visualization (paper Figure 1b)

The repository can also reproduce **Figure 1b** of the paper — a t-SNE
projection of the trained model's **hidden-layer activations**, colored by
document class. **This step is entirely optional** and only requires
`scikit-learn` and `matplotlib`.

```bash
python train_gcn_pyg.py --tsne
```

This trains the model, extracts the 16-dimensional hidden activations
`H = ReLU(conv1(X, A))` for all nodes, projects them to 2D with t-SNE, and saves
`tsne_gcn.png`. You should see the classes form well-separated clusters,
illustrating how the GCN organizes nodes by class using only ~5% of labels plus
the graph structure.

Customize the plot:

```bash
python train_gcn_pyg.py --tsne --perplexity 20 --tsne_out cora_embedding.png
```

> The plot will be qualitatively similar to the paper but not pixel-identical —
> t-SNE is stochastic and depends on the seed, perplexity, and initialization.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```
