"""Multi-task MLP: cell features ⊕ learned business-type embedding → suitability logit
(doc §6.2 "one multi-task network for all business types")."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from geomlab.config import business_type_index
from geomlab.ml.data import CellData, Preprocessor, type_ids
from geomlab.ml.models.base import SuitabilityModel, expected_gradients


class TypeEmbeddingMLP(nn.Module):
    def __init__(self, n_features: int, n_types: int, hidden: list[int], emb_dim: int, dropout: float):
        super().__init__()
        self.type_emb = nn.Embedding(n_types, emb_dim)
        layers: list[nn.Module] = []
        d = n_features + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.type_emb(t)], dim=1)).squeeze(1)


def pos_weight(y: np.ndarray) -> torch.Tensor:
    p = max(y.mean(), 1e-3)
    return torch.tensor((1 - p) / p, dtype=torch.float32)


class MLPModel(SuitabilityModel):
    name = "mlp"

    def __init__(self, hidden=(128, 64), type_embedding_dim=8, dropout=0.2, epochs=60, lr=1e-3,
                 weight_decay=1e-4, batch_size=512):
        self.cfg = dict(hidden=list(hidden), emb_dim=type_embedding_dim, dropout=dropout)
        self.epochs, self.lr, self.wd, self.bs = epochs, lr, weight_decay, batch_size

    def fit(self, cells: CellData, rows: pd.DataFrame, seed: int) -> MLPModel:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        nodes = cells.nodes(rows)
        self.pre = Preprocessor().fit(cells.X[np.unique(nodes)])
        x = torch.from_numpy(self.pre.transform(cells.X[nodes]))
        t = torch.from_numpy(type_ids(rows))
        y = torch.from_numpy(rows["label"].to_numpy(dtype=np.float32))
        self.net = TypeEmbeddingMLP(x.shape[1], len(business_type_index()), **self.cfg)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(y.numpy()))
        self.net.train()
        for _ in range(self.epochs):
            perm = rng.permutation(len(y))
            for i in range(0, len(perm), self.bs):
                b = torch.from_numpy(perm[i : i + self.bs])
                opt.zero_grad()
                loss_fn(self.net(x[b], t[b]), y[b]).backward()
                opt.step()
        self.net.eval()
        self.background_ = self.pre.transform(cells.X[rng.choice(np.unique(nodes), size=min(200, len(np.unique(nodes))), replace=False)])
        return self

    @torch.no_grad()
    def predict(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray:
        x = torch.from_numpy(self.pre.transform(cells.X[cells.nodes(rows)]))
        return torch.sigmoid(self.net(x, torch.from_numpy(type_ids(rows)))).numpy().astype("float64")

    def attribute(self, cells: CellData, rows: pd.DataFrame) -> np.ndarray:
        x = self.pre.transform(cells.X[cells.nodes(rows)])
        t = torch.from_numpy(type_ids(rows))

        def f_grad(z: np.ndarray):
            zt = torch.from_numpy(z).requires_grad_(True)
            out = self.net(zt, t)
            (g,) = torch.autograd.grad(out.sum(), zt)
            return out.detach().numpy(), g.numpy()

        # attributions are w.r.t. standardised inputs; values stay additive in logit space
        return expected_gradients(f_grad, x, self.background_)
