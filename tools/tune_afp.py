from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfar.data import load_graph, split_nodes
from sfar.evaluate import reconstruction_scores
from sfar.propagation import AdaptiveFeaturePropagation
from sfar.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Search AFP hyperparameters for feature reconstruction.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "amac", "amap"])
    parser.add_argument("--alphas", default="0.9,0.95,0.97,0.99,0.995,0.999")
    parser.add_argument("--betas", default="0.7,0.8,0.9,0.95,0.99")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    graph = load_graph(args.dataset, cfg["data_root"])
    data = graph.data
    split = split_nodes(
        data.y,
        args.dataset,
        cfg.get("cache_dir"),
        cfg["missing_rate"],
        cfg["seed"],
        cfg.get("validation_rate"),
    )
    train_nodes = split.train_nodes
    target_nodes = split.target_nodes
    val_nodes = split.val_nodes

    raw_features = data.x.float()
    masked = raw_features.clone()
    masked.index_fill_(0, target_nodes, 0.0)
    afp = AdaptiveFeaturePropagation(data.edge_index, masked, train_nodes)

    rows = []
    for alpha in alphas:
        for beta in betas:
            out = afp.out.clone()
            for iteration in range(1, args.max_iter + 1):
                out = alpha * torch.sparse.mm(afp.adj, out) + (1.0 - alpha) * out.mean(dim=0)
                out[afp.known_nodes] = beta * out[afp.known_nodes] + (1.0 - beta) * afp.known_init
                scores = reconstruction_scores(out[val_nodes].cpu(), raw_features[val_nodes])
                avg = sum(scores.values()) / len(scores)
                row = {
                    "dataset": args.dataset,
                    "alpha": alpha,
                    "beta": beta,
                    "iterations": iteration,
                    "avg": avg,
                }
                row.update(scores)
                rows.append(row)

    rows.sort(key=lambda r: r["avg"], reverse=True)
    for row in rows[: args.topk]:
        print(
            f"avg={row['avg'] * 100:.3f} alpha={row['alpha']} beta={row['beta']} "
            f"iter={row['iterations']} R10={row['Recall@10'] * 100:.2f} "
            f"R20={row['Recall@20'] * 100:.2f} R50={row['Recall@50'] * 100:.2f} "
            f"N10={row['nDCG@10'] * 100:.2f} N20={row['nDCG@20'] * 100:.2f} "
            f"N50={row['nDCG@50'] * 100:.2f}"
        )

    output = Path(args.output) if args.output else ROOT / "outputs" / f"afp_tune_{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "alpha", "beta", "iterations", "avg", "Recall@10", "Recall@20", "Recall@50", "nDCG@10", "nDCG@20", "nDCG@50"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved search results to {output}")


def parse_float_list(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()
