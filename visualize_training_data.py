#!/usr/bin/env python3
"""
Script to visualize the Sudoku puzzles used in training for the ms_tiny experiment.
This helps identify which specific puzzles the model overfitted to.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf

from src.dataset import get_dataset
from src.type_extensions import ConditioningCfg


def load_sudoku_grids(dataset_root: str):
    """Load the raw Sudoku grids to get the actual puzzle layouts."""
    sudokus_path = Path(dataset_root) / "sudokus.npy"
    return np.load(sudokus_path)


def visualize_sudoku_grid(grid, title="Sudoku Grid"):
    """Visualize a 9x9 Sudoku grid."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    
    # Draw grid lines
    for i in range(10):
        linewidth = 2 if i % 3 == 0 else 1
        ax.axhline(y=i, color='black', linewidth=linewidth)
        ax.axvline(x=i, color='black', linewidth=linewidth)
    
    # Fill in numbers
    for i in range(9):
        for j in range(9):
            number = int(grid[i, j])
            ax.text(j + 0.5, 8.5 - i + 0.5, str(number), 
                   ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def main():
    # Load configuration
    cfg = OmegaConf.load('config/dataset/mnist_sudoku_tiny.yaml')
    conditioning = ConditioningCfg(mask=True, label=False)
    
    # Load dataset
    ds = get_dataset(cfg, conditioning, 'train')
    print(f"Training dataset has {len(ds)} samples")
    
    # Load raw Sudoku grids
    sudoku_grids = load_sudoku_grids(cfg.root)
    print(f"Total Sudoku grids available: {len(sudoku_grids)}")
    
    # Get the indices of training samples
    train_indices = [ds[i]['index'] for i in range(len(ds))]
    print(f"Training sample indices: {train_indices}")
    
    # Visualize each training Sudoku
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, sample_idx in enumerate(train_indices):
        # Get the Sudoku grid for this sample
        grid = sudoku_grids[sample_idx]
        
        # Create subplot
        ax = axes[i]
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        
        # Draw grid lines
        for j in range(10):
            linewidth = 2 if j % 3 == 0 else 1
            ax.axhline(y=j, color='black', linewidth=linewidth)
            ax.axvline(x=j, color='black', linewidth=linewidth)
        
        # Fill in numbers
        for row in range(9):
            for col in range(9):
                number = int(grid[row, col])
                ax.text(col + 0.5, 8.5 - row + 0.5, str(number), 
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_title(f'Training Sample {i}\n(Index {sample_idx})', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('training_sudokus.png', dpi=150, bbox_inches='tight')
    print("Saved training Sudoku puzzles to 'training_sudokus.png'")
    
    # Print the actual Sudoku grids as text
    print("\n" + "="*50)
    print("TRAINING SUDOKU PUZZLES (Text Format)")
    print("="*50)
    
    for i, sample_idx in enumerate(train_indices):
        grid = sudoku_grids[sample_idx]
        print(f"\nTraining Sample {i} (Index {sample_idx}):")
        print("-" * 25)
        for row in range(9):
            if row % 3 == 0 and row > 0:
                print("-" * 25)
            row_str = ""
            for col in range(9):
                if col % 3 == 0 and col > 0:
                    row_str += "| "
                row_str += f"{int(grid[row, col])} "
            print(row_str)
    
    # Save the training indices to a file for later reference
    with open('training_indices.txt', 'w') as f:
        f.write("Training sample indices for ms_tiny experiment:\n")
        f.write(f"Dataset: {cfg.root}\n")
        f.write(f"Total samples: {len(ds)}\n")
        f.write(f"Indices: {train_indices}\n")
        f.write(f"Subset size: {cfg.subset_size}\n")
        f.write(f"Top N: {cfg.top_n}\n")
    
    print(f"\nSaved training indices to 'training_indices.txt'")


if __name__ == "__main__":
    main() 