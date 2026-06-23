"""Reproducibility helpers for experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedState:
    """Summary of the reproducibility settings applied by ``set_seed``."""

    seed: int
    deterministic: bool
    benchmark: bool
    cuda_available: bool


def set_seed(
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
    benchmark: bool = False,
) -> SeedState:
    """Set random seeds for Python, NumPy, and PyTorch when available.

    Args:
        seed: Random seed value.
        deterministic: Whether to request deterministic PyTorch algorithms.
        benchmark: Value assigned to ``torch.backends.cudnn.benchmark``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    return SeedState(
        seed=seed,
        deterministic=deterministic,
        benchmark=benchmark,
        cuda_available=cuda_available,
    )


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED):
    """Return a seeded ``torch.Generator`` for DataLoader shuffling.

    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


if __name__ == "__main__":
    state = set_seed()
    print(state)
