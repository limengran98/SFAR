import torch
import torch.nn as nn
import GCL.augmentors as A
from GCL.models.contrast_model import BootstrapContrast
import GCL.losses as L

from .layers import UnitNorm, GConv
from .encoders import Encoder

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