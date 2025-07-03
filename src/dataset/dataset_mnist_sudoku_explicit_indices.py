from dataclasses import dataclass
from typing import Literal, List

import torch

from .dataset_mnist_sudoku_9x9_lazy import DatasetMnistSudoku9x9Lazy, DatasetMnistSudoku9x9LazyCfg


@dataclass
class DatasetMnistSudokuExplicitIndicesCfg(DatasetMnistSudoku9x9LazyCfg):
    """Configuration for MNIST Sudoku dataset with explicit training and validation indices.
    
    This allows specifying exactly which data samples should be used for training and validation,
    rather than using a simple train/test split.
    """
    name: Literal["mnist_sudoku_explicit_indices"] = "mnist_sudoku_explicit_indices"
    train_indices: List[int] | None = None  # Explicit indices for training
    validation_indices: List[int] | None = None  # Explicit indices for validation


class DatasetMnistSudokuExplicitIndices(DatasetMnistSudoku9x9Lazy):
    """
    MNIST Sudoku dataset that allows explicit control over training and validation data indices.
    
    This dataset extends the lazy MNIST Sudoku dataset to support specifying
    exactly which data samples should be used for training and validation, rather than
    using a simple train/test split based on test_samples_num.
    """
    
    def __init__(self, cfg: DatasetMnistSudokuExplicitIndicesCfg, conditioning_cfg, stage):
        # Store training and validation indices if provided (before parent constructor)
        self.train_indices = set(cfg.train_indices) if cfg.train_indices else set()
        self.validation_indices = set(cfg.validation_indices) if cfg.validation_indices else set()
        
        # Temporarily change the name to match the base class expectation
        original_name = cfg.name
        cfg.name = "mnist_sudoku_lazy"
        
        super().__init__(cfg, conditioning_cfg, stage)
        
        # Restore the original name
        cfg.name = original_name
        
        # Override the sudoku grids based on stage and explicit indices
        if stage == "val" and self.validation_indices:
            # For validation, use only the specified validation indices
            all_sudoku_grids = self.get_all_sudoku_grids()
            selected_indices = [idx for idx in self.validation_indices if idx < len(all_sudoku_grids)]
            self.sudoku_grids = all_sudoku_grids[selected_indices]
            print(f"Validation dataset: using indices {selected_indices}, dataset size: {len(self.sudoku_grids)}")
        elif stage == "train":
            if self.train_indices:
                # For training, use only the specified training indices
                all_sudoku_grids = self.get_all_sudoku_grids()
                selected_indices = [idx for idx in self.train_indices if idx < len(all_sudoku_grids)]
                self.sudoku_grids = all_sudoku_grids[selected_indices]
                print(f"Training dataset: using indices {selected_indices}, dataset size: {len(self.sudoku_grids)}")
            elif self.validation_indices:
                # Fallback: exclude validation indices from all data
                all_sudoku_grids = self.get_all_sudoku_grids()
                train_indices = [idx for idx in range(len(all_sudoku_grids)) if idx not in self.validation_indices]
                self.sudoku_grids = all_sudoku_grids[train_indices]
                print(f"Training dataset (fallback): using indices {train_indices}, dataset size: {len(self.sudoku_grids)}")
    
    def get_all_sudoku_grids(self):
        """Get all sudoku grids without any train/test split."""
        import numpy as np
        all_sudoku_grids = np.load(self.sudokus_file_path)
        return torch.tensor(all_sudoku_grids)
    
    def get_raw_sudoku_grids(self):
        """Override to handle explicit training and validation indices."""
        if self.stage == "val" and self.validation_indices:
            # For validation, use only the specified validation indices
            all_sudoku_grids = self.get_all_sudoku_grids()
            selected_indices = [idx for idx in self.validation_indices if idx < len(all_sudoku_grids)]
            return all_sudoku_grids[selected_indices]
        elif self.stage == "train":
            if self.train_indices:
                # For training, use only the specified training indices
                all_sudoku_grids = self.get_all_sudoku_grids()
                selected_indices = [idx for idx in self.train_indices if idx < len(all_sudoku_grids)]
                return all_sudoku_grids[selected_indices]
            elif self.validation_indices:
                # Fallback: exclude validation indices from all data
                all_sudoku_grids = self.get_all_sudoku_grids()
                train_indices = [idx for idx in range(len(all_sudoku_grids)) if idx not in self.validation_indices]
                return all_sudoku_grids[train_indices]
        else:
            # Fall back to original behavior
            return super().get_raw_sudoku_grids() 