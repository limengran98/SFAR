import copy
import torch
from torch import nn
from torch.nn import MSELoss
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

import GCL.augmentors as A
from GCL.models.contrast_model import BootstrapContrast
import GCL.losses as L



class UnitNorm(nn.Module):
    """
    Unit normalization of latent variables.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward(self, vectors):
        """
        Run forward propagation.
        """
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


class Encoder(nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.online_encoder1 = encoder1
        self.online_encoder2 = encoder2
        self.target_encoder1 = None
        self.target_encoder2 = None
        self.augmentor = augmentor
        self.predictor = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Linear(input_dim, hidden_dim)

    def get_target_encoder1(self):
        if self.target_encoder1 is None:
            self.target_encoder1 = copy.deepcopy(self.online_encoder1)
            for p in self.target_encoder1.parameters():
                p.requires_grad = False
        return self.target_encoder1

    def forward(self, x, edge_index, llmfeatures, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder1(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder1(llmfeatures, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder1()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder1()(llmfeatures, edge_index2, edge_weight2)

        xmlp = self.mlp(x)
        return h1, h2, h1_pred, h2_pred, h1_target, h2_target, x, xmlp

class SFAR(nn.Module):
    def __init__(self, device, llm_feature_dim, hidden_size=256, num_layers=2, conv='gcn', obs_features=None):
        super().__init__()
        self.conv = conv
        self.norm = UnitNorm()

        self.aug1 = A.PPRDiffusion()
        self.aug2 = A.PPRDiffusion()

        self.gconv1 = GConv(obs_features.shape[1], hidden_size, num_layers, conv=self.conv)
        self.gconv2 = GConv(llm_feature_dim, hidden_size, num_layers, conv=self.conv)

        self.encoder = Encoder(self.gconv1, self.gconv2, (self.aug1, self.aug2), obs_features.shape[1], hidden_size)
        self.contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)

    def forward(self, edge_index, x_feature, llmfeatures, for_loss=False):
        z1, z2, h1_pred, h2_pred, h1_target, h2_target, z3, z3mlp = self.encoder(x_feature, edge_index, llmfeatures, None)
        z = self.norm(torch.cat([z1, z2, z3], dim=1))
        loss = self.contrast_model(h1_pred=h1_pred, h2_pred=h2_pred,
                                   h1_target=h1_target.detach(), h2_target=h2_target.detach())

        x_hat = z3
        if for_loss:
            return loss
        return x_hat, z, z1, z2

    def to_losses(self, edge_index, x_features, llmfeatures, trn_nodes):
        c_loss = self.forward(edge_index, x_features, llmfeatures, for_loss=True)
        return c_loss


