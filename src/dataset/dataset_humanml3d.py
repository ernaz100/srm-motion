"""Dataset class for HumanML3D adapted for SRM.

This class loads HumanML3D motion data and treats it as a 2D 'image' for the diffusion model.
Sequences are padded to max_length and features are cropped/padded to n_features if necessary.
Normalization is a simple min-max to [-1, 1] for sanity check purposes.
In a full implementation, use proper mean-std normalization from the dataset statistics.
"""

import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch

from .dataset import Dataset, DatasetCfg
from src.type_extensions import ConditioningCfg, Stage, UnbatchedExample


@dataclass
class DatasetHumanML3DCfg(DatasetCfg):
    """Configuration for HumanML3D dataset in SRM."""
    name: str = "humanml3d"
    motion_dir: str = "datasets/humanml3d/motions/AMASS_20.0_fps_nh_smplrifke_abs"
    annotations_dir: str = "datasets/humanml3d/annotations"
    text_emb_dir: str = "datasets/humanml3d/annotations/humanml3d/text_embeddings/ViT-B/32.npy"  # Path to text embeddings .npy
    text_index_path: str = "datasets/humanml3d/annotations/humanml3d/text_embeddings/ViT-B/32_index.json"  # Path to index mapping
    text_stats_dir: str = "datasets/humanml3d/stats/text_stats_abs"  # Path to text stats
    stats_dir: str = "datasets/humanml3d/stats/motion_stats_abs"
    max_length: int = 81  # Capped at 81 frames
    n_features: int = 263
    min_seconds: float = 2.0
    max_seconds: float = 81 / 20  # Max duration to fit 81 frames at 20 fps
    fps: int = 20
    subset_size: int | None = None
    augment: bool = False
    grayscale: bool = True


class DatasetHumanML3D(Dataset[DatasetHumanML3DCfg]):
    """HumanML3D dataset implementation for SRM.

    Loads motion sequences and formats them as 2D tensors for the image-based diffusion pipeline.
    Overrides __getitem__ to handle tensor data directly instead of PIL images.
    """

    def __init__(self, cfg: DatasetHumanML3DCfg, conditioning_cfg: ConditioningCfg, stage: Stage) -> None:
        super().__init__(cfg, conditioning_cfg, stage)
        self.fps = cfg.fps
        self.max_frames = cfg.max_length
        self.n_features = cfg.n_features
        self.cfg.image_shape = [cfg.max_length, cfg.n_features]  # [H=time, W=features], C=1 implicit
        self.keyids = self._read_split(cfg.annotations_dir, stage)  # Use stage directly as str
        if stage == 'val':  # Use string for validation check
            train_keyids = self._read_split(cfg.annotations_dir, 'train')
            self.keyids = train_keyids[:cfg.subset_size or len(train_keyids)]
        self.annotations = self._load_annotations(cfg.annotations_dir)
        if stage != "test":
            self.annotations = self._filter_annotations(self.annotations, cfg.min_seconds, cfg.max_seconds)
        self.keyids = [k for k in self.keyids if k in self.annotations]
        if cfg.subset_size is not None and stage != 'val':  # Avoid double subset for val
            self.keyids = self.keyids[:cfg.subset_size]
        print(f"[DEBUG] Loaded {len(self.keyids)} keyids for stage '{stage}': {self.keyids[:10]}")
        self.mean = torch.load(os.path.join(cfg.stats_dir, 'mean.pt')).float()
        self.std = torch.load(os.path.join(cfg.stats_dir, 'std.pt')).float()
        
        # Pad or crop mean and std to match n_features
        if self.mean.shape[0] > self.n_features:
            self.mean = self.mean[:self.n_features]
            self.std = self.std[:self.n_features]
        elif self.mean.shape[0] < self.n_features:
            pad_width = self.n_features - self.mean.shape[0]
            self.mean = torch.cat([self.mean, torch.zeros(pad_width)], dim=0)
            self.std = torch.cat([self.std, torch.ones(pad_width)], dim=0)
        
        with open(cfg.text_index_path, 'r') as f:
            self.text_index = json.load(f)
        self.text_embeddings = np.load(cfg.text_emb_dir)
        self.text_mean = torch.load(os.path.join(cfg.text_stats_dir, 'mean.pt')).float()
        self.text_std = torch.load(os.path.join(cfg.text_stats_dir, 'std.pt')).float()
        # No preloading motions; load on fly in load

    def _read_split(self, path: str, split: str) -> List[str]:
        """Read split file with keyids."""
        split_path = os.path.join(path, 'splits', f'{split}.txt')
        with open(split_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _load_annotations(self, path: str) -> Dict[str, Any]:
        """Load annotations JSON."""
        ann_path = os.path.join(path, 'annotations.json')
        with open(ann_path, 'r') as f:
            return json.load(f)

    def _filter_annotations(self, annotations: Dict[str, Any], min_sec: float, max_sec: float) -> Dict[str, Any]:
        """Filter annotations by duration and exclude problematic datasets."""
        filtered_annotations = {}
        humanact12_count = 0
        duration_filtered_count = 0
        
        for key, val in annotations.items():
            path = val["path"]

            # Remove humanact12 - buggy left/right + no SMPL
            if "humanact12" in path:
                humanact12_count += 1
                continue

            # Filter by duration
            if min_sec <= val['duration'] <= max_sec:
                filtered_annotations[key] = val
            else:
                duration_filtered_count += 1

        print(f"[DEBUG] Filtered {humanact12_count} HumanAct12 samples and {duration_filtered_count} samples outside duration range [{min_sec}, {max_sec}]s")
        print(f"[DEBUG] Kept {len(filtered_annotations)} samples after filtering")
        
        return filtered_annotations

    @property
    def _num_available(self) -> int:
        """Number of available samples."""
        return len(self.keyids)

    def load(self, idx: int, **kwargs) -> Dict[str, Any]:
        print(f"[DEBUG] Loading idx={idx}, len(keyids)={len(self.keyids)}, stage={self.stage}")
        # Safety check: wrap index if it's out of range
        if len(self.keyids) == 0:
            raise ValueError(f"No keyids available for stage {self.stage}")
        
        idx = idx % len(self.keyids)
        keyid = self.keyids[idx]
        
        if keyid not in self.annotations:
            raise ValueError(f"Keyid {keyid} not found in annotations")
            
        ann = self.annotations[keyid]
        path = ann['path']
        duration = ann['duration']
        
        # Additional safety checks
        if duration <= 0:
            raise ValueError(f"Invalid duration {duration} for keyid {keyid}")
            
        min_frames = int(self.cfg.min_seconds * self.fps)
        max_frames = min(int(duration * self.fps), self.max_frames)
        
        if max_frames < min_frames:
            raise ValueError(f"Duration {duration}s too short for min_seconds {self.cfg.min_seconds}s")
            
        length = random.randint(min_frames, max_frames) if self.stage == "train" else max_frames
        start_frame = random.randint(0, int(duration * self.fps) - length) if self.stage == "train" else 0
        
        motion_path = os.path.join(self.cfg.motion_dir, f'{path}.npy')
        
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"Motion file not found: {motion_path}")
            
        try:
            motion = np.load(motion_path)[start_frame : start_frame + length].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load motion from {motion_path}: {e}")
        
        # Crop or pad features to n_features
        if motion.shape[1] > self.n_features:
            motion = motion[:, :self.n_features]
        elif motion.shape[1] < self.n_features:
            pad_width = self.n_features - motion.shape[1]
            motion = np.pad(motion, ((0, 0), (0, pad_width)), mode='constant')
        
        if length < self.max_frames:
            pad = np.zeros((self.max_frames - length, self.n_features), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)
        motion = torch.from_numpy(motion)
        motion = (motion - self.mean) / (self.std + 1e-6)
        motion = motion.clamp(-3, 3) / 3
        image = motion.unsqueeze(0)  # [1, max_frames, n_features] in approx [-1,1]
        anns = ann['annotations']
        chosen_ann = random.choice(anns) if self.stage == 'train' else anns[0]
        seg_id = chosen_ann['seg_id']
        emb_idx = self.text_index.get(seg_id, 0)
        text_emb = torch.from_numpy(self.text_embeddings[emb_idx]).float()
        text_emb = (text_emb - self.text_mean) / (self.text_std + 1e-6)
        sample = {'image': image, 'label': text_emb, 'path': motion_path}
        return sample 