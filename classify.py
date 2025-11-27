import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
<<<<<<< HEAD
=======

>>>>>>> 70543e0977ea2ace31c5dd0564316b68dd06d64f
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

<<<<<<< HEAD
# Updated imports
from data.datasets import load_data_classify
from models.layers import Linear, GCN  # Moved from utils
=======

from data import load_data_classify

from utils import *
>>>>>>> 70543e0977ea2ace31c5dd0564316b68dd06d64f



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()

def classify_with_embeddings():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(args.data, 'embeddings')


    data = load_data_classify(args.data)
    data = data.to(device)

    y_all = data.y.to(device)
    edge_index = data.edge_index.to(device)
    num_classes = (data.y.max() + 1).item()
    
    z = torch.load(os.path.join(data_path, 'z.pt')).to(device)
    z1 = torch.load(os.path.join(data_path, 'z1.pt')).to(device)
    z2 = torch.load(os.path.join(data_path, 'z2.pt')).to(device)
    trn_nodes = torch.load(os.path.join(data_path, 'train_nodes.pt')).to(device)
    test_nodes = torch.load(os.path.join(data_path, 'test_nodes.pt')).to(device)

    print('----------------------MLP---------------------------')
    x_hat = z[test_nodes].to(device)
    y_subset = y_all[test_nodes]

    def train_mlp(feature, y_all, train_idx):
        cl_model.train()
        optimizer.zero_grad()
        out = cl_model(feature)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx])
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test_mlp(feature, y_all, test_idx):
        cl_model.eval()
        pred = cl_model(feature)
        pred_label = pred.argmax(dim=1)
        mask = test_idx
        return evaluate_metrics(y_all[mask], pred_label[mask])

    def evaluate_metrics(true, pred):
        f1 = f1_score(true.cpu().numpy(), pred.cpu().numpy(), average='macro')
        acc = accuracy_score(true.cpu().numpy(), pred.cpu().numpy())
        pre = precision_score(true.cpu().numpy(), pred.cpu().numpy(), average='macro')
        rec = recall_score(true.cpu().numpy(), pred.cpu().numpy(), average='macro')
        return acc, f1, pre, rec

    node_idx = shuffle(np.arange(x_hat.shape[0]), random_state=72)
    KF = KFold(n_splits=5)

    acc_list, f1_list, pre_list, rec_list = [], [], [], []

    for train_idx, test_idx in KF.split(node_idx):
        cl_model = Linear(x_hat.shape[1], 64, num_classes).to(device)
        optimizer = torch.optim.Adam(cl_model.parameters(), lr=1e-2)

        best_acc = best_f1 = best_pre = best_rec = 0
        for epoch in range(1001):
            loss = train_mlp(x_hat, y_subset, train_idx)
            acc, f1, pre, rec = test_mlp(x_hat, y_subset, test_idx)
            best_acc = max(best_acc, acc)
            best_f1 = max(best_f1, f1)
            best_pre = max(best_pre, pre)
            best_rec = max(best_rec, rec)

        acc_list.append(best_acc)
        f1_list.append(best_f1)
        pre_list.append(best_pre)
        rec_list.append(best_rec)

    print("-----------------") 
    print('Mean:\n  Macro_Pre: {:.2f}, Macro_Rec: {:.2f}, Macro_F1: {:.2f}, Acc: {:.2f}'.format(
        np.mean(pre_list) * 100, np.mean(rec_list) * 100, np.mean(f1_list) * 100, np.mean(acc_list) * 100))
    print('Std:\n  Macro_Pre: {:.2f}, Macro_Rec: {:.2f}, Macro_F1: {:.2f}, Acc: {:.2f}'.format(
        np.std(pre_list) * 100, np.std(rec_list) * 100, np.std(f1_list) * 100, np.std(acc_list) * 100))


    print('----------------------GCN---------------------------')
    edge_index, _ = subgraph(test_nodes, edge_index, relabel_nodes=True)
    edge_index = edge_index.to(device)

    x_hat = z[test_nodes].to(device)
    y_subset = y_all[test_nodes]
    node_idx = shuffle(np.arange(x_hat.shape[0]), random_state=72)
    KF = KFold(n_splits=5)

    acc_list, f1_list, pre_list, rec_list = [], [], [], []

    def train_gcn(feature, edge_index, y_all, train_idx):
        cl_model.train()
        optimizer.zero_grad()
        out = cl_model(feature, edge_index)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx])
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test_gcn(feature, edge_index, y_all, test_idx):
        cl_model.eval()
        pred = cl_model(feature, edge_index)
        pred_label = pred.argmax(dim=1)
        return evaluate_metrics(y_all[test_idx], pred_label[test_idx])

    for train_idx, test_idx in KF.split(node_idx):
        cl_model = GCN(x_hat.shape[1], 64, num_classes).to(device)
        optimizer = torch.optim.Adam(cl_model.parameters(), lr=1e-2)

        best_acc = best_f1 = best_pre = best_rec = 0
        for epoch in range(1001):
            loss = train_gcn(x_hat, edge_index, y_subset, train_idx)
            acc, f1, pre, rec = test_gcn(x_hat, edge_index, y_subset, test_idx)
            best_acc = max(best_acc, acc)
            best_f1 = max(best_f1, f1)
            best_pre = max(best_pre, pre)
            best_rec = max(best_rec, rec)

        acc_list.append(best_acc)
        f1_list.append(best_f1)
        pre_list.append(best_pre)
        rec_list.append(best_rec)

    print("-----------------") 
    print('Mean:\n  Macro_Pre: {:.2f}, Macro_Rec: {:.2f}, Macro_F1: {:.2f}, Acc: {:.2f}'.format(
        np.mean(pre_list) * 100, np.mean(rec_list) * 100, np.mean(f1_list) * 100, np.mean(acc_list) * 100))
    print('Std:\n  Macro_Pre: {:.2f}, Macro_Rec: {:.2f}, Macro_F1: {:.2f}, Acc: {:.2f}'.format(
        np.std(pre_list) * 100, np.std(rec_list) * 100, np.std(f1_list) * 100, np.std(acc_list) * 100))


if __name__ == '__main__':
    classify_with_embeddings()
