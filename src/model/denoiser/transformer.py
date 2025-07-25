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
from src.misc.nn_module_tools import constant_init

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
        
        # Improved input projection with proper initialization
        self.input_proj = nn.Linear(self.d_features, self.d_model)
        
        # Improved output projection with zero initialization for final layer
        self.output_proj = nn.Linear(self.d_model, d_out * self.d_features)
        
        # Enhanced transformer with better configuration
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
                activation='gelu',  # Use GELU for better performance
            ),
            num_layers=cfg.num_layers,
            norm=nn.LayerNorm(self.d_model),  # Add layer norm
        )
        
        # Enhanced time embedding using the same embedding system as UNet
        # This will use the time_embedding from the parent class
        self.time_proj = nn.Linear(self.d_t, self.d_model)
        
        # Positional encodings
        self.register_buffer('pos_enc', self._get_positional_encodings(self.n_frames, self.d_model))
        
        # Text projection (for conditioning as prefix)
        if conditioning_cfg and conditioning_cfg.label and num_classes is None:
            self.text_proj = nn.Linear(512, self.d_model)  # Project text emb to d_model
        else:
            self.text_proj = None
            
        # Layer normalization for input
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Initialize weights properly
        self.init_weights()

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

    def init_weights(self) -> None:
        """Initialize weights with proper scaling and zero initialization for output layer."""
        super().init_weights()
        
        # Initialize input projection with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        
        # Initialize time projection
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        
        # Initialize output projection with zero weights (like UNet's final conv)
        constant_init(self.output_proj, 0)
        
        # Initialize text projection if it exists
        if self.text_proj is not None:
            nn.init.xavier_uniform_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        label: Tensor | None = None,
        c_cat: Tensor | None = None,
        sample: bool = False,
        use_ema: bool = True
    ):
        # x: [batch, num_times, channels=1, num_frames, num_features]
        # Treat as sequences: remove channel dim assumption, handle as [batch, num_times, num_frames, num_features]
        batch, num_times, channels, num_frames, num_features = x.shape
        original_num_features = num_features  # Track for output reshape
        if channels > 1:
            raise ValueError("Multi-channel inputs detected. For sequence data, disable mask conditioning in config as it's not supported yet.")
        # Proceed with original logic, no flattening needed
        x = x.reshape(batch * num_times, num_frames, num_features)  # [batch*num_times, num_frames, num_features]

        # Time embedding: handle potential per-frame t
        # t shape: [batch, num_times, 1, num_frames, num_features] - flatten to embed per position
        try:
            t_flat = t.reshape(batch * num_times, num_frames * num_features)  # Flatten spatial for embedding if needed
            t_emb = self.time_embedding(t_flat)  # Assume embedding handles [batch*num_times, num_frames*num_features]
            t_emb = self.time_proj(t_emb)  # Project
            t_emb = t_emb.view(batch * num_times, num_frames, num_features, self.d_model)  # Reshape back
            t_emb = t_emb.mean(dim=2)  # Average over features if multi-dim
        except Exception as e:
            print(f"Shape mismatch in time embedding: t.shape={t.shape}, error={e}")
            raise
        
        # Ensure t_emb is [batch*num_times, num_frames, d_model]
        if t_emb.dim() != 3 or t_emb.shape[1:] != (num_frames, self.d_model):
            raise ValueError(f"Unexpected t_emb shape: {t_emb.shape}")
        
        x = self.input_proj(x)  # Uses original d_features
        x = x + t_emb + self.pos_enc[:, :num_frames]
        x = self.input_norm(x)
        
        # Text conditioning as prefix
        if label is not None and self.text_proj is not None:
            text_token = self.text_proj(label)  # [batch, d_model]
            text_token = text_token.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, d_model]
            text_token = text_token.expand(-1, num_times, 1, -1).reshape(batch * num_times, 1, self.d_model)
            x = torch.cat([text_token, x], dim=1)  # Prefix
        
        # Transformer
        x = self.transformer(x)
        
        # Remove prefix
        if label is not None:
            x = x[:, 1:]
        
        # Output proj, assuming d_out is for original num_features
        x = self.output_proj(x)  # [batch*num_times, num_frames, d_out * original_num_features]
        x = x.view(batch * num_times, num_frames, self.d_out, original_num_features)
        return x.view(batch, num_times, self.d_out, num_frames, original_num_features) 