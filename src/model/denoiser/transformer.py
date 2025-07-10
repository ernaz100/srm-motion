"""Transformer Denoiser for sequence data."""

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64

from .denoiser import Denoiser, DenoiserCfg
from .embedding import get_embedding, EmbeddingCfg, EmbeddingSinusodialCfg
from .class_embedding import get_class_embedding, ClassEmbeddingCfg, ClassEmbeddingParametersCfg
from src.type_extensions import ConditioningCfg

@dataclass
class TransformerDenoiserCfg(DenoiserCfg):
    name: str = "transformer"
    num_layers: int = 6
    num_heads: int = 8
    d_model: int = 512
    dim_feedforward: int = 2048
    dropout: float = 0.1
    time_embedding: EmbeddingCfg = field(default_factory=EmbeddingSinusodialCfg)
    class_embedding: ClassEmbeddingCfg = field(default_factory=ClassEmbeddingParametersCfg)

class TransformerDenoiser(Denoiser[TransformerDenoiserCfg]):
    def __init__(
        self,
        cfg: TransformerDenoiserCfg,
        d_in: int,
        d_out: int,
        image_shape: Sequence[int],
        num_classes: int | None = None,
        conditioning_cfg: ConditioningCfg = None
    ) -> None:
        super().__init__(cfg, d_in, d_out, image_shape, num_classes, conditioning_cfg)
        self.seq_len = image_shape[0]  # Assuming height = time, width = features
        self.d_features = image_shape[1]
        self.d_model = cfg.d_model
        self.input_proj = nn.Linear(d_in, self.d_model)
        self.output_proj = nn.Linear(self.d_model, d_out)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.num_layers,
        )
        # Add projection for continuous label conditioning
        if conditioning_cfg and conditioning_cfg.label and num_classes is None:
            self.label_proj = nn.Linear(512, self.d_model)  # Project text embedding to d_model
        else:
            self.label_proj = None

    @property
    def d_c(self) -> int:
        return self.cfg.d_model

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        label: Tensor | None = None,
        c_cat: Tensor | None = None,
        sample: bool = False,
        use_ema: bool = True
    ):
        # x: [batch, time, d_in, seq_len] or [batch, time, d_in, height, width]
        if x.dim() == 5:
            batch, num_times, d_in, height, width = x.shape
            seq_len = height * width
            x = x.permute(0, 1, 3, 4, 2).reshape(batch * num_times, seq_len, d_in)
        elif x.dim() == 4:
            batch, num_times, d_in, seq_len = x.shape
            x = x.reshape(batch * num_times, seq_len, d_in)
        else:
            raise ValueError(f"Unexpected input shape for x: {x.shape}")
        # Project motion features to d_model
        x = self.input_proj(x)
        # Conditioning
        emb = self.embed_conditioning(label) if label is not None else None
        if emb is not None:
            if self.label_proj is not None:
                emb = self.label_proj(emb)
            x = x + emb.unsqueeze(1)  # Broadcast emb to sequence
        x = self.transformer(x)
        x = self.output_proj(x)
        x = x.view(batch, num_times, self.d_out, height, width)
        return x

# TODO: Fix reshaping logic to properly handle motion sequences. 