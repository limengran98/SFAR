from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch_geometric.utils import subgraph

from sfar.model import GCNClassifier, MLPClassifier
from sfar.utils import make_seed, set_seed


@torch.no_grad()
def recall_at_k(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    topk = pred.topk(k, dim=1, sorted=False).indices
    rows = torch.arange(target.size(0), device=target.device)
    hits = torch.stack([target[rows, topk[:, i]] for i in range(k)], dim=1).sum(dim=1)
    positives = target.sum(dim=1)
    valid = positives > 0
    return (hits[valid] / positives[valid]).mean().item()


@torch.no_grad()
def ndcg_at_k(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    device = target.device
    target_sorted = torch.sort(target, dim=1, descending=True).values
    pred_index = torch.topk(pred, k, sorted=True).indices
    rows = torch.arange(target.size(0), device=device)
    dcg = torch.zeros(target.size(0), device=device)
    for i in range(k):
        dcg += target[rows, pred_index[:, i]] / np.log2(i + 2)
    denom = torch.log2(torch.arange(target.size(1), dtype=torch.float32, device=device) + 2)
    idcg = (target_sorted / denom).sum(dim=1)
    valid = idcg > 0
    return (dcg[valid] / idcg[valid]).mean().item()


def reconstruction_scores(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    scores = {}
    for k in [10, 20, 50]:
        scores[f"Recall@{k}"] = recall_at_k(pred, target, k)
    for k in [10, 20, 50]:
        scores[f"nDCG@{k}"] = ndcg_at_k(pred, target, k)
    return scores


def node_classification(
    z: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    target_nodes: torch.Tensor,
    num_classes: int,
    classifier: str,
    hidden_size: int,
    dropout: float,
    lr: float,
    epochs: int,
    folds: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    features = z[target_nodes].to(device)
    labels = y[target_nodes].to(device)
    local_edge_index, _ = subgraph(target_nodes.to(device), edge_index.to(device), relabel_nodes=True)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    metrics: List[Dict[str, float]] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(np.arange(features.size(0)))):
        set_seed(make_seed(seed, classifier, fold_id))
        train_idx = torch.from_numpy(train_idx).long().to(device)
        test_idx = torch.from_numpy(test_idx).long().to(device)
        model = make_classifier(classifier, features.size(1), hidden_size, num_classes, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best = {"acc": 0.0}

        for _ in range(epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = forward_classifier(model, classifier, features, local_edge_index)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

            current = evaluate_classifier(model, classifier, features, local_edge_index, labels, test_idx)
            if current["acc"] > best["acc"]:
                best = current

        metrics.append(best)

    keys = metrics[0].keys()
    return {
        "mean": {key: float(np.mean([m[key] for m in metrics])) for key in keys},
        "std": {key: float(np.std([m[key] for m in metrics])) for key in keys},
        "folds": metrics,
    }


def make_classifier(name: str, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
    if name.lower() == "gcn":
        return GCNClassifier(input_dim, hidden_dim, num_classes, dropout=max(dropout, 0.2))
    return MLPClassifier(input_dim, hidden_dim, num_classes, dropout=dropout)


def forward_classifier(model, name: str, features: torch.Tensor, edge_index: torch.Tensor):
    if name.lower() == "gcn":
        return model(features, edge_index)
    return model(features)


@torch.no_grad()
def evaluate_classifier(model, name: str, features: torch.Tensor, edge_index: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor):
    model.eval()
    logits = forward_classifier(model, name, features, edge_index)
    pred = logits[idx].argmax(dim=1).cpu().numpy()
    true = labels[idx].cpu().numpy()
    return {"acc": accuracy_score(true, pred)}
