"""Deterministic-enough seeding for training and evaluation."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    ``deterministic=True`` asks cuDNN / PyTorch for deterministic kernels.
    It can slow training down and a few ops may still be non-deterministic;
    the flag is recorded in every checkpoint and log so runs remain auditable.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a reproducible seed."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
