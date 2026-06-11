from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TAPE-style .emb files into SFAR .pt HERP features.")
    parser.add_argument("--origin-emb", required=True, help="BERT embedding of original text.")
    parser.add_argument("--expert-emb", required=True, help="BERT embedding of LLM-generated explanation text.")
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--semantic-scale", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    origin = load_and_project(args.origin_emb, args.num_nodes, args.feature_dim)
    expert = load_and_project(args.expert_emb, args.num_nodes, args.feature_dim)
    out = origin + args.semantic_scale * expert
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved {tuple(out.shape)} HERP features to {args.output}")


def load_and_project(path: str, num_nodes: int, feature_dim: int) -> torch.Tensor:
    values = np.memmap(path, mode="r", dtype=np.float16)
    target_size = num_nodes * feature_dim
    factor = int(values.size // target_size)
    if factor <= 0:
        raise ValueError(f"{path} cannot be projected to ({num_nodes}, {feature_dim}).")
    array = np.asarray(values[: factor * target_size], dtype=np.float32)
    array = array.reshape(target_size, factor).mean(axis=1)
    return torch.from_numpy(array.reshape(num_nodes, feature_dim)).float()


if __name__ == "__main__":
    main()
