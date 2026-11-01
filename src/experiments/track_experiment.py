"""Experiment tracking and reproducibility metadata recorder."""
import os
import json
import time
from typing import Dict, Any, Optional
from src.utils.reproducibility import get_environment_info

class ExperimentTracker:
    """Records experiment configuration, hardware state, and progress logs locally."""
    def __init__(self, experiment_id: str, output_dir: str = "outputs/logs"):
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, f"{experiment_id}_meta.json")
        self.history: list[dict] = []
        
        self.metadata = {
            "experiment_id": experiment_id,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": get_environment_info(),
            "config": {},
            "metrics": {},
            "status": "initialized"
        }
        self.save()

    def set_config(self, config: Dict[str, Any]):
        self.metadata["config"] = config
        self.save()

    def log_step(self, step: int, metrics: Dict[str, Any]):
        entry = {
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        self.history.append(entry)

    def finish(self, final_metrics: Dict[str, Any], status: str = "completed"):
        self.metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["status"] = status
        self.metadata["final_metrics"] = final_metrics
        self.metadata["step_history"] = self.history
        self.save()

    def save(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
