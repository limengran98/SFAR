import torch
import numpy as np
from sklearn.metrics import ndcg_score

@torch.no_grad()
def to_f1_score(input, target, epsilon=1e-8):
    assert (target < 0).int().sum() == 0
    tp = ((input > 0) & (target > 0)).sum()
    fp = ((input > 0) & (target == 0)).sum()
    fn = ((input <= 0) & (target > 0)).sum()
    return (tp / (tp + (fp + fn) / 2 + epsilon)).item()

@torch.no_grad()
def to_recall(input, target, k):
    pred = input.topk(k, dim=1, sorted=False)[1]
    row_index = torch.arange(target.size(0))
    target_list = []
    for i in range(k):
        target_list.append(target[row_index, pred[:, i]])
    num_pred = torch.stack(target_list, dim=1).sum(dim=1)
    num_true = target.sum(dim=1)
    return (num_pred[num_true > 0] / num_true[num_true > 0]).mean().item()

@torch.no_grad()
def to_ndcg(input, target, k, version='sat'):
    if version == 'base':
        return ndcg_score(target, input, k=k)
    elif version == 'sat':
        device = target.device
        target_sorted = torch.sort(target, dim=1, descending=True)[0]
        pred_index = torch.topk(input, k, sorted=True)[1]
        row_index = torch.arange(target.size(0))
        dcg = torch.zeros(target.size(0), device=device)
        for i in range(k):
            dcg += target[row_index, pred_index[:, i]] / np.log2(i + 2)
        idcg_divider = torch.log2(torch.arange(target.size(1), dtype=float, device=device) + 2)
        idcg = (target_sorted / idcg_divider).sum(dim=1)
        return (dcg[idcg > 0] / idcg[idcg > 0]).mean().item()
    else:
        raise ValueError(version)

@torch.no_grad()
def to_r2(input, target):
    a = ((input - target) ** 2).sum()
    b = ((target - target.mean(dim=0)) ** 2).sum()
    return (1 - a / b).item()

@torch.no_grad()
def to_rmse(input, target):
    return ((input - target) ** 2).mean(dim=1).sqrt().mean().item()