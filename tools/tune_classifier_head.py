from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfar.data import load_graph, load_herp_features, paper_name, split_nodes
from sfar.evaluate import node_classification
from main import reconstruct_features, train_sfar
from sfar.utils import load_config, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Tune classifier head settings with a fixed SFAR representation.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--dataset", required=True, choices=["cora", "citeseer", "amac", "amap"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--missing-rate", type=float, default=None)
    parser.add_argument("--ckd-epochs", type=int, default=None)
    parser.add_argument("--ckd-dropout", type=float, default=None)
    parser.add_argument("--fusion", default=None)
    parser.add_argument("--classifier-dropouts", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--classifier-lrs", default="0.005,0.01")
    parser.add_argument("--classifier-epochs", type=int, default=1000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    missing_rate = cfg.get("missing_rate", 0.6) if args.missing_rate is None else args.missing_rate
    device = select_device(args.device, args.gpu)
    set_seed(seed)

    if args.ckd_epochs is not None:
        cfg["ckd"]["epochs"] = args.ckd_epochs
    if args.ckd_dropout is not None:
        cfg["ckd"]["dropout"] = args.ckd_dropout
    if args.fusion is not None:
        cfg["ckd"]["fusion"] = args.fusion
    cfg["ckd"]["log_interval"] = 0

    graph = load_graph(args.dataset, cfg["data_root"])
    data = graph.data
    train_nodes, target_nodes = split_nodes(data.y, args.dataset, cfg.get("cache_dir"), missing_rate, seed)
    afp_features = reconstruct_features(args.dataset, cfg, data.x.float(), data.edge_index, train_nodes, target_nodes, device)
    herp_cfg = cfg["herp"]
    herp_features = load_herp_features(
        args.dataset,
        num_nodes=data.num_nodes,
        feature_dim=graph.feature_dim,
        cache_dir=cfg.get("cache_dir"),
        llm_emb_dir=herp_cfg.get("llm_emb_dir"),
        llm_model=herp_cfg["llm_model"],
        lm_model=herp_cfg["lm_model"],
        semantic_scale=herp_cfg.get("semantic_scale", 1.0),
    )
    run_args = SimpleNamespace(ckd_epochs=None, classifier_epochs=args.classifier_epochs)
    z = train_sfar(args.dataset, data.edge_index, afp_features, herp_features, cfg, seed, device, run_args)

    rows = []
    for dropout in parse_float_list(args.classifier_dropouts):
        for lr in parse_float_list(args.classifier_lrs):
            for classifier in ["mlp", "gcn"]:
                clf_cfg = deepcopy(cfg["classifier"])
                clf_cfg["dropout"] = dropout
                clf_cfg["lr"] = lr
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
                row = {
                    "dataset": paper_name(args.dataset),
                    "classifier": classifier.upper(),
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": args.classifier_epochs,
                    "acc": result["mean"]["acc"],
                    "std": result["std"]["acc"],
                }
                rows.append(row)
                print(
                    f"{row['dataset']} {row['classifier']} dropout={dropout} lr={lr} "
                    f"Acc={row['acc'] * 100:.2f} +/- {row['std'] * 100:.2f}",
                    flush=True,
                )

    output = Path(args.output) if args.output else ROOT / "outputs" / f"classifier_head_tune_{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "classifier", "dropout", "lr", "epochs", "acc", "std"]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved search results to {output}")


def parse_float_list(text: str):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


if __name__ == "__main__":
    main()
