import os
import tempfile
from src.tools.extract_joints import extract_joints
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
import torch
from src.dataset import get_dataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from src.evaluation.motion_visualization import visualize_motion

def main():
    initialize(version_base=None, config_path="../config", job_name="test_video")
    cfg = compose(config_name="main")
    # Override dataset config for HumanML3D
    cfg.dataset = OmegaConf.load("config/dataset/humanml3d_tiny_final.yaml")
    cfg.conditioning.label = True
    cfg.conditioning.mask = False
    # Load dataset
    dataset = get_dataset(cfg.dataset, cfg.conditioning, stage="train")
    print(f"Loaded dataset with {len(dataset)} samples")
    output_dir = "test_reconstruction_videos"
    os.makedirs(output_dir, exist_ok=True)

    # Collect embeddings for checking
    embeddings = []
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        embeddings.append(item["label"])

    # Check if embeddings differ
    if len(embeddings) > 1:
        all_same = all(torch.allclose(embeddings[0], emb) for emb in embeddings[1:])
        if all_same:
            raise ValueError("All text embeddings are identical! This indicates a loading issue.")
        else:
            print("Text embeddings differ as expected across samples.")
    else:
        print("Only one sample found, skipping embedding check.")

    for i in range(min(5, len(dataset))):
        item = dataset[i]
        motions = item["image"].squeeze(0)  # [1, T, F]
        name = item.get("path", str(i)).replace("/", "_").replace(".npy", "")
        text = item.get("text", item["index"])
        video_path = os.path.join(output_dir, f"{name}.mp4")
        visualize_motion(motions, len(motions), video_path, text, device="cpu", fps=20)


if __name__ == "__main__":
    main() 
