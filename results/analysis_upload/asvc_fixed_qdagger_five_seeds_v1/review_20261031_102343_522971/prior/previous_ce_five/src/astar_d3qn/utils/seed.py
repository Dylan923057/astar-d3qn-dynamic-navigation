from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed all generators and require deterministic training by default.

    Configure cuBLAS before any CUDA seeding or computation. Unsupported
    deterministic operators raise an error rather than silently falling back.
    """
    if deterministic_torch:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
