import torch
from torch_sparse import SparseTensor
from collections import defaultdict

def to_edge_tensor(edge_index):
    """
    Convert an edge index tensor to a SparseTensor.
    """
    row, col = edge_index
    value = torch.ones(edge_index.size(1))
    return SparseTensor(row=row, col=col, value=value)

def validate_edges(edges):
    """
    Validate the edges of a graph with various criteria.
    """
    # No self-loops
    for src, dst in edges.t():
        if src.item() == dst.item():
            raise ValueError()

    # Each edge (a, b) appears only once.
    m = defaultdict(lambda: set())
    for src, dst in edges.t():
        src = src.item()
        dst = dst.item()
        if dst in m[src]:
            raise ValueError()
        m[src].add(dst)

    # Each pair (a, b) and (b, a) exists together.
    for src, neighbors in m.items():
        for dst in neighbors:
            if src not in m[dst]:
                raise ValueError()