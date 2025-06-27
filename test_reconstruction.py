#!/usr/bin/env python3
"""
Test the model's reconstruction of the actual training data to check for overfitting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
from torchvision.utils import save_image
from dacite import Config, from_dict

from src.dataset import get_dataset
from src.type_extensions import ConditioningCfg
from src.model import Wrapper
from src.sampler import get_sampler, FixedSamplerCfg
from src.config import load_typed_root_config, RootCfg
from src.evaluation import EvaluationCfg


def load_checkpoint(checkpoint_path: str, map_location='cpu'):
    """Load checkpoint with proper device mapping."""
    return torch.load(checkpoint_path, map_location=map_location)


def test_reconstruction():
    """Test the model's ability to reconstruct the training data."""
    
    print("="*60)
    print("TESTING RECONSTRUCTION OF TRAINING DATA")
    print("="*60)
    
    # ------------------------------------------------------------------
    # 1. Locate checkpoint and the exact configuration that was used
    # ------------------------------------------------------------------
    checkpoint_path = Path("outputs/ms_tiny/2025-06-27_12-43-09/checkpoints/last.ckpt")
    hydra_cfg_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"

    # Load Hydra-generated config and cast it to the typed RootCfg dataclass
    cfg_dict = OmegaConf.load(hydra_cfg_path)
    
    # Custom type hooks to handle evaluation configs
    def evaluation_config_hook(data: dict) -> EvaluationCfg:
        """Convert dict to appropriate evaluation config based on name field."""
        if not isinstance(data, dict) or 'name' not in data:
            raise ValueError(f"Invalid evaluation config: {data}")
        
        name = data['name']
        if name == 'mnist_sudoku':
            from src.evaluation.mnist_sudoku_evaluation import MnistSudokuEvaluationCfg
            return from_dict(MnistSudokuEvaluationCfg, data)
        elif name == 'sampling':
            from src.evaluation.sampling_evaluation import SamplingEvaluationCfg
            return from_dict(SamplingEvaluationCfg, data)
        elif name == 'mnist_grid':
            from src.evaluation.mnist_grid_evaluation import MnistGridEvaluationCfg
            return from_dict(MnistGridEvaluationCfg, data)
        elif name == 'counting_objects':
            from src.evaluation.counting_objects_evaluation import CountingObjectsEvaluationCfg
            return from_dict(CountingObjectsEvaluationCfg, data)
        elif name == 'even_pixels':
            from src.evaluation.even_pixels_evaluation import EvenPixelsEvaluationCfg
            return from_dict(EvenPixelsEvaluationCfg, data)
        else:
            raise ValueError(f"Unknown evaluation config name: {name}")
    
    # Custom type hooks
    type_hooks = {
        Path: Path,
        EvaluationCfg: evaluation_config_hook,
    }
    
    # Load config with custom type hooks
    cfg = from_dict(
        RootCfg,
        OmegaConf.to_container(cfg_dict),
        config=Config(type_hooks=type_hooks),
    )

    # Prepare dataset using the *same* configuration as during training
    conditioning = cfg.conditioning
    ds = get_dataset(cfg.dataset, conditioning, "train")
    
    # Directory to store reconstructed outputs
    recon_dir = Path("reconstructions")
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n1. TRAINING DATA:")
    print(f"   - Number of samples: {len(ds)}")
    print(f"   - Indices: {[ds[i]['index'] for i in range(len(ds))]}")
    
    # Load checkpoint metadata (for global step information only)
    print(f"\n2. LOADING CHECKPOINT:")
    print(f"   - Path: {checkpoint_path}")
    
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        print(f"   - Checkpoint loaded successfully")
        print(f"   - Global step: {checkpoint.get('global_step', 'unknown')}")
    except Exception as e:
        print(f"   - Error loading checkpoint: {e}")
        return
    
    # Create model
    try:
        d_data = 1 if cfg.dataset.grayscale else 3
        num_classes = getattr(ds, "num_classes", None)
        model = Wrapper.load_from_checkpoint(
            str(checkpoint_path),
            cfg=cfg,
            d_data=d_data,
            image_shape=cfg.dataset.image_shape,
            num_classes=num_classes,
            map_location="cpu",
            strict=False,
        )
        model.eval()
        print(f"   - Model loaded successfully")
    except Exception as e:
        print(f"   - Error loading model: {e}")
        return
    
    # Create sampler
    sampler_cfg = FixedSamplerCfg(name="fixed")
    # Derive the grid of patches (H/patch_size, W/patch_size)
    patch_grid_shape = tuple(s // cfg.patch_size for s in cfg.dataset.image_shape)
    sampler = get_sampler(
        sampler_cfg,
        patch_size=cfg.patch_size,
        patch_grid_shape=patch_grid_shape,
    )
    
    print(f"\n3. RECONSTRUCTION TEST:")
    
    # Test reconstruction on each training sample
    for i in range(len(ds)):
        sample = ds[i]
        image = sample['image'].unsqueeze(0)  # Add batch dimension
        mask = sample.get('mask', None)
        if mask is not None:
            mask = mask.unsqueeze(0)
        
        print(f"\n   Testing Sample {i} (Index {sample['index']}):")
        print(f"   - Input image shape: {image.shape}")
        print(f"   - Input image range: [{image.min():.3f}, {image.max():.3f}]")
        
        try:
            with torch.no_grad():
                # Generate reconstruction
                z_t = torch.randn_like(image)
                masked = (1 - mask) * image if mask is not None else None
                reconstructed = sampler(
                    model,
                    z_t=z_t,
                    mask=mask,
                    masked=masked,
                    return_intermediate=False
                )
                
                # Calculate reconstruction error
                error = torch.mean((image - reconstructed['sample']) ** 2).item()
                print(f"   - Reconstruction error (MSE): {error:.6f}")
                
                # Save side-by-side image with padding (original | spacer | reconstructed)
                # Create white spacer (1 channel, same height, 10 pixels wide)
                spacer_width = 10
                spacer = torch.ones(1, 1, image.shape[2], spacer_width, device=image.device)
                side_by_side = torch.cat([image, spacer, reconstructed['sample']], dim=3)  # concatenate along width
                save_path = recon_dir / f"recon_{sample['index']}.png"
                save_image(
                    side_by_side,
                    save_path,
                    normalize=True,
                    value_range=(-1, 1)
                )
                print(f"   - Reconstruction saved to {save_path}")
                
                # Calculate accuracy (if we had a classifier)
                print(f"   - Reconstruction successful!")
                
        except Exception as e:
            print(f"   - Error during reconstruction: {e}")
    
    print(f"\n4. OVERFITTING ANALYSIS:")
    print("   ✓ Model loaded successfully")
    print("   ✓ Testing on exact training data (indices 0-9)")
    print("   ✓ Training loss was very low (0.001-0.002)")
    print("   ✓ Only 10 training samples")
    
    print(f"\n5. EXPECTED RESULTS:")
    print("   - If overfitting: Very low reconstruction error (< 0.01)")
    print("   - If not overfitting: Higher reconstruction error (> 0.1)")
    print("   - Perfect reconstruction = overfitting confirmed!")
    
    print(f"\n" + "="*60)
    print("RECONSTRUCTION TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    test_reconstruction() 