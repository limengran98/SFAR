import torch
import torch.nn as nn
import copy

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