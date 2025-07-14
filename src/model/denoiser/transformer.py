"""Transformer Denoiser for sequence data."""

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn, Tensor
from jaxtyping import Float, Int64

import math  # For positional encodings

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
        self.n_frames = image_shape[0]  # height = n_frames (time)
        self.d_features = image_shape[1]  # width = n_features per frame
        self.d_model = cfg.d_model
        self.input_proj = nn.Linear(self.d_features, self.d_model)  # Project features per frame
        self.output_proj = nn.Linear(self.d_model, d_out * self.d_features)  # Project back to d_out per feature
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
        # Time embedding (per frame)
        self.time_emb = nn.Linear(1, self.d_model)  # Simple projection for t
        # Positional encodings
        self.register_buffer('pos_enc', self._get_positional_encodings(self.n_frames, self.d_model))
        # Text projection (for conditioning as prefix)
        if conditioning_cfg and conditioning_cfg.label and num_classes is None:
            self.text_proj = nn.Linear(512, self.d_model)  # Project text emb to d_model
        else:
            self.text_proj = None

    def _get_positional_encodings(self, seq_len: int, d_model: int) -> Tensor:
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1e4) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]

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
        # x: [batch, time, d_in=1, n_frames, n_features] -> squeeze channel, treat as sequence
        batch, num_times, d_in, n_frames, n_features = x.shape
        assert d_in == 1, "Assuming single channel for sequences"
        x = x.squeeze(2)  # [batch, time, n_frames, n_features]
        x = x.reshape(batch * num_times, n_frames, n_features)  # [batch*time, seq_len=n_frames, d_features]
        
        # t: Average over spatial dimensions (height, width) first, then reshape to per-sample
        t = t.mean(dim=(3, 4))  # [batch, time, 1] - average over height=81, width=205
        t = t.reshape(batch * num_times, 1)  # [batch*time, 1]
        t = t.expand(-1, n_frames)  # Broadcast to [batch*time, n_frames]
        t = t.unsqueeze(-1)  # [batch*time, n_frames, 1]
        t_emb = self.time_emb(t).squeeze(2)  # [batch*time, n_frames, d_model]
        
        # Project x and add t_emb + pos_enc
        x = self.input_proj(x) + t_emb + self.pos_enc[:, :n_frames]
        
        # Text as prefix (repeat for time if needed)
        if label is not None and self.text_proj is not None:
            text_token = label.unsqueeze(1)  # [batch, 1, d_model]
            text_token = text_token.unsqueeze(1).expand(-1, num_times, -1, -1)  # [batch, time, 1, d_model]
            text_token = text_token.reshape(batch * num_times, 1, self.d_model)
            x = torch.cat([text_token, x], dim=1)  # [batch*time, 1 + n_frames, d_model]
        
        # Transformer forward
        x = self.transformer(x)
        
        # Remove prefix, project back
        if label is not None:
            x = x[:, 1:]
        x = self.output_proj(x)  # [batch*time, n_frames, d_out * n_features]
        x = x.view(batch * num_times, n_frames, self.d_out, n_features)
        return x.view(batch, num_times, self.d_out, n_frames, n_features) 