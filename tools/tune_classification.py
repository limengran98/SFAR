from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfar.data import load_graph, load_herp_features, paper_name, split_nodes
from sfar.evaluate import node_classification
from main import reconstruct_features, train_sfar
from sfar.utils import load_config, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Search light CKD settings for downstream node classification.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--datasets", default="cora,citeseer,amac,amap")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--missing-rate", type=float, default=None)
    parser.add_argument("--ckd-epochs", default="50,100,150")
    parser.add_argument("--ckd-lrs", default="0.001")
    parser.add_argument("--ckd-dropouts", default="0.1,0.2,0.3")
    parser.add_argument("--fusions", default="latents_afp,latents")
    parser.add_argument("--classifier-epochs", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    missing_rate = cfg.get("missing_rate", 0.6) if args.missing_rate is None else args.missing_rate
    device = select_device(args.device, args.gpu)
    set_seed(seed)

    datasets = parse_text_list(args.datasets)
    candidates = [
        {"epochs": epochs, "lr": lr, "dropout": dropout, "fusion": fusion}
        for epochs in parse_int_list(args.ckd_epochs)
        for lr in parse_float_list(args.ckd_lrs)
        for dropout in parse_float_list(args.ckd_dropouts)
        for fusion in parse_text_list(args.fusions)
    ]

    rows = []
    for dataset in datasets:
        graph = load_graph(dataset, cfg["data_root"])
        data = graph.data
        train_nodes, target_nodes = split_nodes(data.y, dataset, cfg.get("cache_dir"), missing_rate, seed)
        afp_features = reconstruct_features(dataset, cfg, data.x.float(), data.edge_index, train_nodes, target_nodes, device)
        herp_cfg = cfg["herp"]
        herp_features = load_herp_features(
            dataset,
            num_nodes=data.num_nodes,
            feature_dim=graph.feature_dim,
            cache_dir=cfg.get("cache_dir"),
            llm_emb_dir=herp_cfg.get("llm_emb_dir"),
            llm_model=herp_cfg["llm_model"],
            lm_model=herp_cfg["lm_model"],
            semantic_scale=herp_cfg.get("semantic_scale", 1.0),
        )

        for candidate in candidates:
            run_cfg = deepcopy(cfg)
            run_cfg["ckd"].update(candidate)
            run_cfg["ckd"]["log_interval"] = 0
            run_args = SimpleNamespace(ckd_epochs=None, classifier_epochs=args.classifier_epochs)
            z = train_sfar(dataset, data.edge_index, afp_features, herp_features, run_cfg, seed, device, run_args)

            clf_scores = {}
            for classifier in ["mlp", "gcn"]:
                clf_cfg = deepcopy(run_cfg["classifier"])
                if args.classifier_epochs is not None:
                    clf_cfg["epochs"] = args.classifier_epochs
                result = node_classification(
                    z=z.float(),
                    y=data.y,
                    edge_index=data.edge_index,
                    target_nodes=target_nodes,
                    num_classes=graph.num_classes,
                    classifier=classifier,
                    hidden_size=clf_cfg["hidden_size"],
                    dropout=clf_cfg["dropout"],
                    lr=clf_cfg["lr"],
                    epochs=clf_cfg["epochs"],
                    folds=clf_cfg["folds"],
                    seed=seed,
                    device=device,
                )
                clf_scores[classifier] = result["mean"]["acc"]

            row = {
                "dataset": paper_name(dataset),
                "epochs": candidate["epochs"],
                "lr": candidate["lr"],
                "dropout": candidate["dropout"],
                "fusion": candidate["fusion"],
                "mlp_acc": clf_scores["mlp"],
                "gcn_acc": clf_scores["gcn"],
                "avg_acc": (clf_scores["mlp"] + clf_scores["gcn"]) / 2.0,
            }
            rows.append(row)
            print(
                f"{row['dataset']} epochs={row['epochs']} lr={row['lr']} dropout={row['dropout']} "
                f"fusion={row['fusion']} MLP={row['mlp_acc'] * 100:.2f} "
                f"GCN={row['gcn_acc'] * 100:.2f} AVG={row['avg_acc'] * 100:.2f}",
                flush=True,
            )

    output = Path(args.output) if args.output else ROOT / "outputs" / "classification_tune.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "epochs", "lr", "dropout", "fusion", "mlp_acc", "gcn_acc", "avg_acc"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved search results to {output}")


def parse_text_list(text: str):
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_list(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()
