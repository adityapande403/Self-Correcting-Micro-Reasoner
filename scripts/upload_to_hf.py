"""Uploads trained model artifacts and model card to Hugging Face Hub."""
import os
import sys
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logging import setup_logger

logger = setup_logger("upload_to_hf")

def main():
    repo_id = "Adityapande403/Self-Correcting-Micro-Reasoner"
    export_dir = "outputs/checkpoints/hf_export"
    
    logger.info(f"Connecting to Hugging Face Hub to upload to: {repo_id}")
    api = HfApi()

    user = api.whoami()["name"]
    logger.info(f"Authenticated as: {user}")

    logger.info(f"Uploading files from {export_dir} to {repo_id}...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=export_dir,
        repo_type="model",
        commit_message="Release: Self-Correcting Micro-Reasoner (SFT + GRPO-Scale + Consensus@8)"
    )
    logger.info(f"Successfully uploaded model and artifacts to https://huggingface.co/{repo_id}!")

if __name__ == "__main__":
    main()
