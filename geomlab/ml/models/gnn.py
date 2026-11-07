"""GraphSAGE over the H3 adjacency graph (doc §6.2): message passing captures neighbourhood
effects (clusters, spillovers). Same type-embedding head as the MLP, so the ablation
"tabular → + graph" changes only the encoder.

Training is transductive on the full graph of all cells (non-candidate cells still pass
messages) with losses only on training rows. Inference is inductive: a freshly built
graph from new gold features can be scored by the same weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from geomlab.config import business_type_index
from geomlab.ml.data import CellData, Preprocessor, type_ids
from geomlab.ml.models.base import SuitabilityModel, expected_gradients
from geomlab.ml.models.mlp import pos_weight


class HexSAGE(nn.Module):
    def __init__(self, n_features: int, n_types: int, hidden: int, layers: int, emb_dim: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(n_features if i == 0 else hidden, hidden) for i in range(layers)])
        self.skip = nn.Linear(n_features, hidden)  # keeps the cell's own signal strong
        self.dropout = dropout
        self.type_emb = nn.Embedding(n_types, emb_dim)
        self.head = nn.Sequential(nn.Linear(hidden + emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            h = nn.functional.dropout(h, self.dropout, self.training)
        return h + self.skip(x)

    def forward(self, x, edge_index, nodes, types):
        h = self.encode(x, edge_index)
        return self.head(torch.cat([h[nodes], self.type_emb(types)], dim=1)).squeeze(1)


class GNNModel(SuitabilityModel):
    name = "gnn"

    def __init__(self, hidden=64, layers=2, type_embedding_dim=8, dropout=0.2, epochs=150, lr=5e-3,
                 weight_decay=5e-4):
        self.cfg = dict(hidden=hidden, layers=layers, emb_dim=type_embedding_dim, dropout=dropout)
        self.epochs, self.lr, self.wd = epochs, lr, weight_decay

    def fit(self, cells: CellData, rows: pd.DataFrame, seed: int) -> GNNModel:
        torch.manual_seed(seed)
        nodes = cells.nodes(rows)
        self.pre = Preprocessor().fit(cells.X[np.unique(nodes)])
        x = torch.from_numpy(self.pre.transform(cells.X))
        ei = torch.from_numpy(cells.edges)
        n_t, t_t = torch.from_numpy(nodes), torch.from_numpy(type_ids(rows))
        y = torch.from_numpy(rows["label"].to_numpy(dtype=np.float32))
        self.net = HexSAGE(x.shape[1], len(business_type_index()), **self.cfg)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(y.numpy()))
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss_fn(self.net(x, ei, n_t, t_t), y).backward()
            opt.step()
        self.net.eval()
        rng = np.random.default_rng(seed)
        uniq = np.unique(nodes)
        self.background_ = self.pre.transform(cells.X[rng.choice(uniq, size=min(200, len(uniq)), replace=False)])
        return self

    @torch.no_grad()
    def predict(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray:
        x = torch.from_numpy(self.pre.transform(cells.X))
        logits = self.net(x, torch.from_numpy(cells.edges), torch.from_numpy(cells.nodes(rows)),
                          torch.from_numpy(type_ids(rows)))
        return torch.sigmoid(logits).numpy().astype("float64")

    def attribute(self, cells: CellData, rows: pd.DataFrame, batch: int = 256) -> np.ndarray:
        """Expected gradients w.r.t. the target cell's OWN features, neighbours held at their
        observed values. Each target gets its own copy of its L-hop subgraph so targets in a
        batch do not interact."""
        X = self.pre.transform(cells.X)
        edges = cells.edges
        adj: dict[int, list[int]] = {}
        for s, d in edges.T:
            adj.setdefault(int(s), []).append(int(d))
        L = len(self.net.convs)

        def khop(n: int) -> list[int]:
            seen, frontier = {n}, [n]
            for _ in range(L):
                frontier = [m for f in frontier for m in adj.get(f, []) if m not in seen]
                seen.update(frontier)
            return [n] + sorted(seen - {n})

        nodes, types = cells.nodes(rows), type_ids(rows)
        out = np.zeros((len(rows), X.shape[1]))
        for start in range(0, len(rows), batch):
            sl = slice(start, start + batch)
            feats, src, dst, targets, offset = [], [], [], [], 0
            for n in nodes[sl]:
                sub = khop(int(n))
                local = {g: i + offset for i, g in enumerate(sub)}
                for g in sub:
                    for m in adj.get(g, []):
                        if m in local:
                            src.append(local[g])
                            dst.append(local[m])
                feats.append(X[sub])
                targets.append(offset)
                offset += len(sub)
            Xs = np.concatenate(feats).astype(np.float32)
            ei = torch.tensor([src, dst], dtype=torch.long)
            tg = torch.tensor(targets)
            tt = torch.from_numpy(types[sl])

            def f_grad(z: np.ndarray, Xs=Xs, ei=ei, tg=tg, tt=tt):
                full = torch.from_numpy(Xs.copy())
                zt = torch.from_numpy(z).requires_grad_(True)
                full = full.index_put((tg,), zt)
                o = self.net(full, ei, tg, tt)
                (g,) = torch.autograd.grad(o.sum(), zt)
                return o.detach().numpy(), g.numpy()

            out[sl] = expected_gradients(f_grad, X[nodes[sl]], self.background_)
        return out
