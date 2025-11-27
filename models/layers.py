import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class UnitNorm(nn.Module):
    """
    Unit normalization of latent variables.
    """
    def __init__(self):
        super().__init__()

    def forward(self, vectors):
        valid_index = (vectors != 0).sum(1, keepdims=True) > 0
        vectors = torch.where(valid_index, vectors, torch.randn_like(vectors))
        return vectors / (vectors ** 2).sum(1, keepdims=True).sqrt()

class Normalize(nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, conv, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.conv = conv

        if conv == 'gat':
            self.layers.append(GATConv(input_dim, hidden_dim, heads=8))
            for _ in range(num_layers - 1):
                self.layers.append(GATConv(hidden_dim * 8, hidden_dim, heads=1))
        elif conv == 'lin':
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        elif conv == 'sage':
            self.layers.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        elif conv == 'hyb':
            self.layers.append(SAGEConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GATConv(hidden_dim, hidden_dim))
        else:
            self.layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            if self.conv == 'lin':
                z = conv(z)
            else:
                z = conv(z, edge_index, edge_weight)
        return z, self.projection_head(z)

# --- 下面是原 utils.py 中的分类器模型 ---

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

class Linear(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Linear, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x