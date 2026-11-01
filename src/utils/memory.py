"""GPU and system memory tracking utilities."""
import os
import psutil
import torch
from typing import Dict, Any

class MemoryTracker:
    """Tracks GPU and RAM allocation and peak usage."""
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.has_cuda = torch.cuda.is_available() and "cuda" in device

    def reset(self):
        """Reset peak memory stats."""
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Returns current and peak memory usage in MB."""
        stats = {
            "ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
            "ram_total_mb": psutil.virtual_memory().total / (1024 * 1024),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu_available": self.has_cuda
        }
        if self.has_cuda:
            stats.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
                "device_name": torch.cuda.get_device_name(0)
            })
        else:
            stats.update({
                "gpu_allocated_mb": 0.0,
                "gpu_reserved_mb": 0.0,
                "gpu_peak_allocated_mb": 0.0,
                "gpu_peak_reserved_mb": 0.0,
                "device_name": "CPU"
            })
        return stats

    def print_summary(self, prefix: str = ""):
        """Logs a formatted memory summary."""
        stats = self.get_stats()
        lead = f"[{prefix}] " if prefix else ""
        if self.has_cuda:
            print(
                f"{lead}GPU Memory: {stats['gpu_allocated_mb']:.1f} MB allocated, "
                f"Peak: {stats['gpu_peak_allocated_mb']:.1f} MB, "
                f"Reserved: {stats['gpu_reserved_mb']:.1f} MB | "
                f"RAM: {stats['ram_used_mb']:.1f}/{stats['ram_total_mb']:.1f} MB ({stats['ram_percent']}%)"
            )
        else:
            print(f"{lead}RAM: {stats['ram_used_mb']:.1f}/{stats['ram_total_mb']:.1f} MB ({stats['ram_percent']}%)")
