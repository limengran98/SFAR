from __future__ import annotations

import torch
from torch_geometric.utils import get_laplacian, remove_self_loops


def propagation_matrix(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, edge_weight = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
    edge_index, edge_weight = remove_self_loops(edge_index, -edge_weight)
    return torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()


class AdaptiveFeaturePropagation:
    def __init__(self, edge_index: torch.Tensor, features: torch.Tensor, known_nodes: torch.Tensor):
        self.edge_index = edge_index
        self.features = features
        self.known_nodes = known_nodes.long()
        self.num_nodes = features.size(0)
        self.adj = propagation_matrix(edge_index, self.num_nodes).to(features.device)
        self.out = torch.zeros_like(features)
        self.out[self.known_nodes] = features[self.known_nodes]
        self.known_init = self.out[self.known_nodes].clone()

    @torch.no_grad()
    def run(self, alpha: float, beta: float, iterations: int) -> torch.Tensor:
        out = self.out.clone()
        for _ in range(iterations):
            out = alpha * torch.sparse.mm(self.adj, out) + (1.0 - alpha) * out.mean(dim=0)
            out[self.known_nodes] = beta * out[self.known_nodes] + (1.0 - beta) * self.known_init
        return out
