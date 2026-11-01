"""Reproducibility utilities for setting seeds and capturing environment metadata."""
import os
import random
import sys
import platform
import subprocess
import numpy as np
import torch
from typing import Dict, Any

def set_seed(seed: int = 42, deterministic: bool = False):
    """Sets random seeds across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_environment_info() -> Dict[str, Any]:
    """Captures software, hardware, and runtime environment details."""
    info = {
        "os": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "device_capability": torch.cuda.get_device_capability(0),
            "total_vram_mb": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        })
    return info
