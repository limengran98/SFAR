from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric import datasets


DATASET_ALIASES: Dict[str, str] = {
    "cora": "cora",
    "citeseer": "citeseer",
    "computer": "computers",
    "computers": "computers",
    "amac": "computers",
    "photo": "photo",
    "amap": "photo",
}

PAPER_NAMES: Dict[str, str] = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "computers": "Amac",
    "photo": "Amap",
}

CACHE_PREFIX: Dict[str, str] = {
    "cora": "cora",
    "citeseer": "citeseer",
    "computers": "computers",
    "photo": "photo",
}


@dataclass(frozen=True)
class GraphData:
    key: str
    paper_name: str
    data: object
    feature_dim: int
    num_classes: int


@dataclass(frozen=True)
class NodeSplit:
    train_nodes: torch.Tensor
    val_nodes: torch.Tensor
    test_nodes: torch.Tensor

    @property
    def target_nodes(self) -> torch.Tensor:
        return torch.cat([self.val_nodes, self.test_nodes], dim=0)


def canonical_dataset(name: str) -> str:
    key = name.lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unsupported dataset '{name}'. Use cora, citeseer, amac, or amap.")
    return DATASET_ALIASES[key]


def paper_name(name: str) -> str:
    return PAPER_NAMES[canonical_dataset(name)]


def cache_prefix(name: str) -> str:
    return CACHE_PREFIX[canonical_dataset(name)]


def load_graph(name: str, root: str) -> GraphData:
    key = canonical_dataset(name)
    if key == "cora":
        dataset = datasets.Planetoid(root, "Cora")
    elif key == "citeseer":
        dataset = datasets.Planetoid(root, "Citeseer")
    elif key == "computers":
        dataset = datasets.Amazon(root, "Computers")
    elif key == "photo":
        dataset = datasets.Amazon(root, "Photo")
    else:
        raise ValueError(key)

    data = dataset[0]
    return GraphData(
        key=key,
        paper_name=PAPER_NAMES[key],
        data=data,
        feature_dim=int(data.x.size(1)),
        num_classes=int(data.y.max().item() + 1),
    )


def split_nodes(
    y: torch.Tensor,
    name: str,
    cache_dir: str | os.PathLike[str] | None,
    missing_rate: float,
    seed: int,
    validation_rate: float | None = None,
) -> NodeSplit:
    indices = np.arange(y.numel())
    labels = y.cpu().numpy()
    train_nodes, target_nodes = train_test_split(
        indices,
        test_size=missing_rate,
        random_state=seed,
        stratify=labels,
    )
    if validation_rate is None:
        validation_rate = missing_rate / 6.0
    if not 0.0 < validation_rate < missing_rate:
        raise ValueError("validation_rate must be greater than 0 and smaller than missing_rate.")

    val_fraction = validation_rate / missing_rate
    val_nodes, test_nodes = train_test_split(
        target_nodes,
        train_size=val_fraction,
        random_state=seed,
        stratify=labels[target_nodes],
    )
    return NodeSplit(
        train_nodes=torch.from_numpy(train_nodes).long(),
        val_nodes=torch.from_numpy(val_nodes).long(),
        test_nodes=torch.from_numpy(test_nodes).long(),
    )


def load_cached_tensor(cache_dir: str | os.PathLike[str] | None, name: str, suffix: str):
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{cache_prefix(name)}{suffix}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def save_tensor(output_dir: str | os.PathLike[str], name: str, suffix: str, tensor: torch.Tensor) -> None:
    path = Path(output_dir) / cache_prefix(name) / f"{suffix}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.cpu(), path)


def load_herp_features(
    name: str,
    num_nodes: int,
    feature_dim: int,
    cache_dir: str | os.PathLike[str] | None,
    llm_emb_dir: str | os.PathLike[str] | None,
    llm_model: str,
    lm_model: str,
    semantic_scale: float,
) -> torch.Tensor:
    cached = load_cached_tensor(cache_dir, name, "llmfeatures")
    if cached is not None:
        return cached.float()
    if llm_emb_dir is None:
        raise FileNotFoundError("No cached HERP features found and llm_emb_dir is not set.")

    origin = _load_memmap_embedding(_embedding_path(Path(llm_emb_dir) / "Origin", lm_model), num_nodes, feature_dim)
    expert = _load_memmap_embedding(_embedding_path(Path(llm_emb_dir) / llm_model, lm_model), num_nodes, feature_dim)
    return origin + semantic_scale * expert


def _embedding_path(folder: Path, lm_model: str) -> Path:
    return folder / f"{lm_model}.emb"


def _load_memmap_embedding(path: Path, num_nodes: int, feature_dim: int) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.memmap(path, mode="r", dtype=np.float16)
    target_size = num_nodes * feature_dim
    factor = int(values.size // target_size)
    if factor <= 0:
        raise ValueError(f"{path} is too small for ({num_nodes}, {feature_dim}).")
    array = np.asarray(values[: factor * target_size], dtype=np.float32)
    array = array.reshape(target_size, factor).mean(axis=1)
    return torch.from_numpy(array.reshape(num_nodes, feature_dim)).float()
