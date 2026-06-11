from __future__ import annotations

import argparse

import torch

from sfar.data import load_cached_tensor, load_graph, load_herp_features, paper_name, split_nodes
from sfar.evaluate import node_classification, reconstruction_scores
from sfar.model import SFAR
from sfar.propagation import AdaptiveFeaturePropagation
from sfar.result_writer import ResultWriter
from sfar.utils import format_percent, load_config, make_seed, select_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="SFAR feature reconstruction and downstream node classification.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--dataset", default=None, help="Run one dataset, e.g., cora/citeseer/amac/amap.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index, e.g., --gpu 1. Overrides --device.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--missing-rate", type=float, default=None, help="Fraction of target nodes with missing features. Default: 0.6.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-tensors", action="store_true", default=None)
    parser.add_argument("--no-save-tensors", action="store_false", dest="save_tensors")
    parser.add_argument("--recompute-afp", action="store_true")
    parser.add_argument("--retrain-ckd", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--ckd-epochs", type=int, default=None)
    parser.add_argument("--classifier-epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    missing_rate = cfg.get("missing_rate", 0.6) if args.missing_rate is None else args.missing_rate
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("--missing-rate must be between 0 and 1.")
    cfg["missing_rate"] = missing_rate
    set_seed(seed)

    device = select_device(args.device, args.gpu)
    print(f"Using device: {device}")
    cache_dir = args.cache_dir or cfg.get("cache_dir")
    output_dir = args.output_dir or cfg.get("output_dir", "outputs")
    save_tensors = cfg.get("save_tensors", True) if args.save_tensors is None else args.save_tensors
    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    writer = ResultWriter(output_dir, args.run_name, vars(args), cfg)

    results = {}
    for dataset in datasets:
        print(f"\n========== {paper_name(dataset)} ==========")
        result = run_dataset(dataset, cfg, cache_dir, seed, missing_rate, device, args, writer, save_tensors)
        results[paper_name(dataset)] = result
        writer.save_dataset_result(dataset, result)

    writer.save_all_results(results)
    print(f"\nSaved run outputs under {writer.output_dir}/<dataset>/{writer.run_name}")


def run_dataset(
    dataset: str,
    cfg: dict,
    cache_dir: str | None,
    seed: int,
    missing_rate: float,
    device: torch.device,
    args,
    writer: ResultWriter,
    save_tensors: bool,
):
    graph = load_graph(dataset, cfg["data_root"])
    data = graph.data
    train_nodes, target_nodes = split_nodes(data.y, dataset, cache_dir, missing_rate, seed)
    tensor_paths = {}
    if save_tensors:
        tensor_paths["train_nodes"] = writer.save_tensor(dataset, "train_nodes.pt", train_nodes)
        tensor_paths["target_nodes"] = writer.save_tensor(dataset, "target_nodes.pt", target_nodes)

    raw_features = data.x.float()
    edge_index = data.edge_index
    print(f"Observed nodes: {train_nodes.numel()}, target nodes: {target_nodes.numel()}, missing rate: {target_nodes.numel() / data.num_nodes:.2%}")
    afp_features = reconstruct_features(dataset, cfg, raw_features, edge_index, train_nodes, target_nodes, device)
    if save_tensors:
        tensor_paths["x_feature"] = writer.save_tensor(dataset, "x_feature.pt", afp_features)

    rec = reconstruction_scores(afp_features[target_nodes].to(device), raw_features[target_nodes].to(device))
    print_reconstruction(rec)

    cls_results = {}
    if not args.skip_classification:
        z = None
        if cfg.get("use_cached_embeddings", True) and not args.retrain_ckd:
            z = load_cached_tensor(cache_dir, dataset, "z")
        if z is None:
            herp_cfg = cfg["herp"]
            herp_features = load_herp_features(
                dataset,
                num_nodes=data.num_nodes,
                feature_dim=graph.feature_dim,
                cache_dir=cache_dir,
                llm_emb_dir=herp_cfg.get("llm_emb_dir"),
                llm_model=herp_cfg["llm_model"],
                lm_model=herp_cfg["lm_model"],
                semantic_scale=herp_cfg.get("semantic_scale", 1.0),
            )
            z = train_sfar(dataset, edge_index, afp_features, herp_features, cfg, seed, device, args)
        if save_tensors:
            tensor_paths["z"] = writer.save_tensor(dataset, "z.pt", z)

        for classifier in ["mlp", "gcn"]:
            clf_cfg = get_classifier_config(dataset, classifier, cfg, args)
            result = node_classification(
                z=z.float(),
                y=data.y,
                edge_index=edge_index,
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
            cls_results[classifier.upper()] = result
            print_classification(classifier.upper(), result)

    return {
        "feature_reconstruction": rec,
        "node_classification": cls_results,
        "num_nodes": int(data.num_nodes),
        "feature_dim": graph.feature_dim,
        "target_nodes": int(target_nodes.numel()),
        "saved_tensors": tensor_paths,
    }


def reconstruct_features(
    dataset: str,
    cfg: dict,
    raw_features: torch.Tensor,
    edge_index: torch.Tensor,
    train_nodes: torch.Tensor,
    target_nodes: torch.Tensor,
    device: torch.device,
):
    raw_features = raw_features.float().to(device)
    edge_index = edge_index.to(device)
    train_nodes = train_nodes.to(device)
    target_nodes = target_nodes.to(device)
    masked = raw_features.clone()
    masked.index_fill_(0, target_nodes, 0.0)
    afp_cfg = get_afp_config(dataset, cfg)
    afp = AdaptiveFeaturePropagation(edge_index, masked, train_nodes)
    return afp.run(
        alpha=afp_cfg["alpha"],
        beta=afp_cfg["beta"],
        iterations=afp_cfg["iterations"],
    ).cpu()


def get_afp_config(dataset: str, cfg: dict) -> dict:
    afp = cfg["afp"]
    if "datasets" in afp:
        params = dict(afp.get("default", {}))
        params.update(afp["datasets"].get(dataset.lower(), {}))
        params.update(afp["datasets"].get(paper_name(dataset).lower(), {}))
        return params

    return {
        "alpha": afp.get("alpha", 0.99),
        "beta": afp.get("beta", 0.9),
        "iterations": afp.get("iterations", {}).get(dataset.lower(), 2),
    }


def get_ckd_config(dataset: str, cfg: dict, args) -> dict:
    ckd = cfg["ckd"]
    ckd_cfg = {key: value for key, value in ckd.items() if key != "datasets"}
    for key in dataset_config_keys(dataset):
        ckd_cfg.update(ckd.get("datasets", {}).get(key, {}))
    if args.ckd_epochs is not None:
        ckd_cfg["epochs"] = args.ckd_epochs
    return ckd_cfg


def get_classifier_config(dataset: str, classifier: str, cfg: dict, args) -> dict:
    classifier_cfg = cfg["classifier"]
    params = {key: value for key, value in classifier_cfg.items() if key != "datasets"}
    for key in dataset_config_keys(dataset):
        dataset_cfg = classifier_cfg.get("datasets", {}).get(key, {})
        params.update({name: value for name, value in dataset_cfg.items() if name.lower() not in ["mlp", "gcn"]})
        params.update(dataset_cfg.get(classifier.lower(), {}))
        params.update(dataset_cfg.get(classifier.upper(), {}))

    if args.classifier_epochs is not None:
        params["epochs"] = args.classifier_epochs
    return params


def dataset_config_keys(dataset: str) -> list[str]:
    return [dataset.lower(), paper_name(dataset).lower()]


def train_sfar(dataset: str, edge_index: torch.Tensor, afp_features: torch.Tensor, herp_features: torch.Tensor, cfg: dict, seed: int, device: torch.device, args):
    ckd_cfg = get_ckd_config(dataset, cfg, args)
    set_seed(make_seed(seed, dataset, "ckd"))

    model = SFAR(
        afp_dim=afp_features.size(1),
        herp_dim=herp_features.size(1),
        hidden_dim=ckd_cfg["hidden_size"],
        num_layers=ckd_cfg["num_layers"],
        conv=ckd_cfg["conv"],
        dropout=ckd_cfg["dropout"],
        fusion=ckd_cfg.get("fusion", "latents_afp"),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ckd_cfg["lr"])
    edge_index = edge_index.to(device)
    afp_features = afp_features.float().to(device)
    herp_features = herp_features.float().to(device)

    for epoch in range(ckd_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(edge_index, afp_features, herp_features)
        loss.backward()
        optimizer.step()
        log_interval = ckd_cfg.get("log_interval", 10)
        if log_interval and epoch % log_interval == 0:
            print(f"CKD epoch {epoch:04d}, loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        return model(edge_index, afp_features, herp_features)["z"].cpu()


def print_reconstruction(scores: dict) -> None:
    print("Feature reconstruction:")
    for key, value in scores.items():
        print(f"  {key}: {format_percent(value)}")


def print_classification(name: str, result: dict) -> None:
    mean = result["mean"]
    std = result["std"]
    print(f"{name} node classification:")
    print(f"  Acc: {format_percent(mean['acc'])} +/- {format_percent(std['acc'])}")


if __name__ == "__main__":
    main()
