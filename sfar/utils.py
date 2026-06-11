from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def make_seed(base_seed: int, *parts: Any) -> int:
    text = "::".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str, gpu: int | None = None) -> torch.device:
    if gpu is not None:
        if gpu < 0:
            raise ValueError("--gpu must be a non-negative CUDA device index.")
        requested = f"cuda:{gpu}"

    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but CUDA is not available.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} was requested, but only {torch.cuda.device_count()} device(s) are visible.")
        torch.cuda.set_device(index)

    return device


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}"
