from typing import Sequence

from .denoiser import Denoiser
from .unet import UNetDenoiser, UNetDenoiserCfg
from .transformer import TransformerDenoiser, TransformerDenoiserCfg
from src.type_extensions import ConditioningCfg


DENOISERS = {
    "unet": UNetDenoiser,
    "transformer": TransformerDenoiser
}

DenoiserCfg = UNetDenoiserCfg | TransformerDenoiserCfg


def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    d_in: int,
    d_out: int,
    image_shape: Sequence[int],
    num_classes: int | None = None,
    conditioning_cfg: ConditioningCfg = None,
    learn_variance: bool = False,
    learn_sigma: bool = False
) -> Denoiser:
    denoiser: Denoiser = DENOISERS[denoiser_cfg.name](
        denoiser_cfg, d_in, d_out, image_shape, num_classes, conditioning_cfg=conditioning_cfg,
        learn_variance=learn_variance, learn_sigma=learn_sigma
    )
    denoiser.init_weights()
    denoiser.freeze()
    return denoiser
