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
        conditioning_cfg: ConditioningCfg = None,
        learn_variance: bool = False,
        learn_sigma: bool = False
    ) -> None:
        super().__init__(cfg, d_in, d_out, image_shape, num_classes, conditioning_cfg)
        self.n_frames = image_shape[0]  # height = n_frames (time)
        self.d_features = image_shape[1]  # width = n_features per frame
        self.d_model = cfg.d_model
        self.d_data = d_in  # Use d_in as d_data assuming single channel input
        self.input_proj = nn.Linear(self.d_features, self.d_model)  # Project features per frame
        self.mean_proj = nn.Linear(self.d_model, self.d_data * self.d_features)  # Project to mean
        if learn_variance:
            self.variance_proj = nn.Linear(self.d_model, self.d_data * self.d_features)
        if learn_sigma:
            self.sigma_proj = nn.Linear(self.d_model, 1 * self.d_features)  # 1 channel for logvar
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.num_layers // 2,
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Simple temporal pooling
        self.unpool = nn.Upsample(scale_factor=2, mode='nearest')  # Upsample back
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.num_layers // 2,
        )
        # Time embedding (per frame)
        self.time_embedding = get_embedding(cfg.time_embedding, self.d_model)  # Use sinusoidal from config
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
        
        # t: Average over feature dimension only (dim=4, width=n_features) to get per-frame t
        t = t.mean(dim=4, keepdim=True)  # [batch, num_times, 1, n_frames, 1] - average over features
        t = t.reshape(batch * num_times, n_frames, 1)  # [batch*time, n_frames, 1]
        t_emb = self.time_embedding.forward(t).squeeze(2)  # [batch*time, n_frames, d_model]
        
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
        x = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # Pool sequence dim
        x = self.decoder(x, x)  # Self-decode
        x = self.unpool(x.permute(0, 2, 1)).permute(0, 2, 1)  # Unpool
        
        # Remove prefix
        if label is not None:
            x = x[:, 1:]
        
        # Projections
        predictions = [self.mean_proj(x)]
        if hasattr(self, 'variance_proj'):
            predictions.append(torch.sigmoid(self.variance_proj(x)))
        if hasattr(self, 'sigma_proj'):
            predictions.append(self.sigma_proj(x))
        pred = torch.cat(predictions, dim=-1)  # [batch*time, n_frames, d_out * n_features]
        
        pred = pred.view(batch * num_times, n_frames, self.d_out, n_features)
        return pred.view(batch, num_times, self.d_out, n_frames, n_features) 