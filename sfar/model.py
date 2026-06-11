from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class ViewEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, conv: str, dropout: float):
        super().__init__()
        self.conv = conv.lower()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * num_layers
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(self._make_layer(in_dim, out_dim))
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.projector = nn.Linear(hidden_dim, hidden_dim)

    def _make_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.conv == "gat":
            return GATConv(in_dim, out_dim, heads=1)
        if self.conv == "sage":
            return SAGEConv(in_dim, out_dim)
        if self.conv == "lin":
            return nn.Linear(in_dim, out_dim)
        return GCNConv(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        for i, layer in enumerate(self.layers):
            z = layer(z) if self.conv == "lin" else layer(z, edge_index)
            if i + 1 < len(self.layers):
                z = F.prelu(z, torch.ones(1, device=z.device))
                z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.norm(z)
        return z, self.projector(z)


class Predictor(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SFAR(nn.Module):
    def __init__(
        self,
        afp_dim: int,
        herp_dim: int,
        hidden_dim: int,
        num_layers: int,
        conv: str = "gcn",
        dropout: float = 0.2,
        fusion: str = "latents_afp",
    ):
        super().__init__()
        self.herp_encoder = ViewEncoder(herp_dim, hidden_dim, num_layers, conv, dropout)
        self.afp_encoder = ViewEncoder(afp_dim, hidden_dim, num_layers, conv, dropout)
        self.herp_predictor = Predictor(hidden_dim, dropout)
        self.afp_predictor = Predictor(hidden_dim, dropout)
        self.fusion = fusion

    def forward(self, edge_index: torch.Tensor, afp_features: torch.Tensor, herp_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        herp_z, herp_q = self.herp_encoder(herp_features, edge_index)
        afp_z, afp_q = self.afp_encoder(afp_features, edge_index)
        herp_p = self.herp_predictor(herp_q)
        afp_p = self.afp_predictor(afp_q)

        if self.fusion == "latents":
            z = torch.cat([herp_z, afp_z], dim=1)
        elif self.fusion == "herp_afp":
            z = torch.cat([herp_z, afp_features], dim=1)
        else:
            z = torch.cat([herp_z, afp_z, afp_features], dim=1)

        return {
            "z": F.normalize(z, p=2, dim=1),
            "herp_z": herp_z,
            "afp_z": afp_z,
            "herp_p": herp_p,
            "afp_p": afp_p,
        }

    def loss(self, edge_index: torch.Tensor, afp_features: torch.Tensor, herp_features: torch.Tensor) -> torch.Tensor:
        out = self.forward(edge_index, afp_features, herp_features)
        return 0.5 * (
            contrastive_loss(out["herp_p"], out["afp_z"].detach())
            + contrastive_loss(out["afp_p"], out["herp_z"].detach())
        )


def contrastive_loss(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    logits = query @ key.t() / temperature
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class GCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
