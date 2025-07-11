from dataclasses import dataclass, field
from math import ceil
from typing import Iterator, Literal, NotRequired, TypeVar, Optional

from jaxtyping import Float, Int64
import numpy as np
import torch
from torch import Tensor

from ..dataset import Dataset
from .evaluation import Evaluation, EvaluationCfg, UnbatchedExample, BatchedExample
from ..model import Wrapper
from ..sampler import FixedSamplerCfg, get_sampler, Sampler, SamplerCfg
from .types import EvaluationOutput

import tempfile
import os

from .motion_visualization import recover_from_ric, plot_3d_motion
from .motion_recovery import recover_from_smplrifke
from src.type_extensions import SamplingOutput


class UnbatchedSamplingExample(UnbatchedExample, total=False):
    index: int
    # image: NotRequired[Float[Tensor, "channel height width"]]
    image: Optional[torch.Tensor]
    label: int
    # mask: NotRequired[Float[Tensor, "1 height width"]]
    mask: Optional[torch.Tensor]


class BatchedSamplingExample(BatchedExample, total=False):
    index: Int64[Tensor, "batch"]
    # image: NotRequired[Float[Tensor, "batch channel height width"]]
    image: Optional[torch.Tensor]
    # label: NotRequired[Int64[Tensor, "batch"]]
    label: Optional[torch.Tensor]
    # mask: NotRequired[Float[Tensor, "batch 1 height width"]]
    mask: Optional[torch.Tensor]
    
    
@dataclass
class SamplingEvaluationCfg(EvaluationCfg):
    name: Literal["sampling"] = "sampling"
    samplers: dict[str, SamplerCfg] = field(
        default_factory=lambda: {"fixed": FixedSamplerCfg("fixed")}
    )
    num_samples: int | None = 1
    dataset_indices: list[int] | None = None
    sampling_video: bool = True
    visualize_time: bool = True
    visualize_sigma: bool = True
    visualize_x: bool = True


T = TypeVar("T", bound=SamplingEvaluationCfg)


class SamplingEvaluation(Evaluation[T, UnbatchedSamplingExample, BatchedSamplingExample]):
    samplers: dict[str, Sampler]

    def __init__(
        self,
        cfg: SamplingEvaluationCfg,
        tag: str,
        dataset: Dataset,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False
    ) -> None:
        assert cfg.samplers, "At least one sampler config must be provided"
        super().__init__(cfg, tag, dataset, deterministic)
        # only used if there are conditionings from the dataset
        self.chunk_size = ceil(len(dataset) / len(self)) if self.cfg.dataset_indices is None else 1
        dependency_matrix = dataset.get_dependency_matrix(patch_grid_shape) if patch_grid_shape is not None else None
        self.samplers = {
            k: get_sampler(c, patch_size, patch_grid_shape, dependency_matrix) 
            for k, c in self.cfg.samplers.items()
        }   
            
    @torch.no_grad()
    def evaluate(
        self, 
        model: Wrapper, 
        batch: BatchedSamplingExample,
        return_sample: bool = True
    ) -> Iterator[EvaluationOutput]:
        return_intermediate = self.cfg.sampling_video and return_sample
        return_time = return_intermediate and self.cfg.visualize_time
        return_sigma = return_intermediate and self.cfg.visualize_sigma
        return_x = return_intermediate and self.cfg.visualize_x
        
        device = model.device
        z_t = (torch.empty if self.deterministic else torch.randn)(
            (len(batch["name"]), model.d_data, *self.dataset.cfg.image_shape), 
            device=device
        )
        if self.deterministic:
            for i, index in enumerate(batch["index"]):
                z_t[i] = torch.randn(
                    z_t[i].shape, 
                    generator=torch.Generator(device).manual_seed(index.item()), 
                    device=device
                )
        
        masked = (1-batch["mask"]) * batch["image"] if "mask" in batch else None
        for key, sampler in self.samplers.items():
            sample = sampler.__call__(
                model,
                z_t=z_t,
                label=batch.get("label", None),
                mask=batch.get("mask", None),
                masked=masked,
                return_intermediate=return_intermediate, 
                return_time=return_time,
                return_sigma=return_sigma,
                return_x=return_x
            )
            out = EvaluationOutput(key=key, names=batch["name"], sample=sample)
            if masked is not None:
                out["masked"] = masked
            yield out
                
    
    def __getitem__(self, idx: int, **kwargs) -> UnbatchedSamplingExample:
        if self.cfg.dataset_indices is not None:
            idx = self.cfg.dataset_indices[0] + idx
        if self.dataset.conditioning_cfg.label or self.dataset.conditioning_cfg.mask or kwargs:
            start = idx * self.chunk_size
            end = min(start + self.chunk_size, len(self.dataset))
            sample_idx = np.random.default_rng(idx).integers(start, end).item() \
                if self.deterministic else np.random.randint(start, end)
            sample: UnbatchedSamplingExample = self.dataset.__getitem__(sample_idx, **kwargs)
            sample["name"] = str(sample_idx)
        else:
            sample = dict(index=idx, name=str(idx))
        return sample

    
    def __len__(self) -> int:
        if self.cfg.dataset_indices is not None:
            return self.cfg.dataset_indices[1] - self.cfg.dataset_indices[0]
        if self.cfg.num_samples is not None:
            return self.cfg.num_samples
        return len(self.dataset)

    def log_sample(
        self,
        model: Wrapper,
        sample: SamplingOutput,
        key: str,
        names: list[str],
        # masked: Float[Tensor, "batch channel height width"] | None = None,
        masked: Optional[torch.Tensor] = None,
        num_log: int | None = None
    ) -> None:
        if 'humanml3d' in self.dataset.cfg.name:
            s = slice(num_log)
            motions = sample["sample"][s]
            # Unnormalize the motion data using mean and std
            motions = motions * self.dataset.std.to(motions.device) + self.dataset.mean.to(motions.device)
            motions = motions.cpu().numpy().squeeze(1)  # [batch, time, features]
            step = model.step_tracker.get_step()
            for i, (mot, name) in enumerate(zip(motions, names[s])):
                # Recover joint positions from SMPL-RIFKE absolute motion data
                smpldata = recover_from_smplrifke(mot, fps=self.dataset.cfg.fps, abs_root=True)
                joints = smpldata["joints"]
                
                # Use NamedTemporaryFile to ensure the file is actually created
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    print(f"Creating video for {name} at {temp_path}")
                    plot_3d_motion(joints, temp_path, title=name)
                    
                    # Verify the file was created before trying to log it
                    if os.path.exists(temp_path):
                        file_size = os.path.getsize(temp_path)
                        print(f"Video file created successfully at {temp_path} (size: {file_size} bytes)")
                        if file_size > 0:
                            model.logger.log_video(f"{key}/video", [temp_path], step=step, caption=[name], fps=[self.cfg.fps], format=['mp4'])
                        else:
                            print(f"Warning: Video file is empty at {temp_path}")
                    else:
                        print(f"Warning: Video file was not created at {temp_path}")
                        
                except Exception as e:
                    print(f"Error creating video for {name}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Cleanup: remove the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print(f"Cleaned up temporary file: {temp_path}")
        else:
            pass # The original code had this else block, but it was empty. Keeping it as is.
