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
    use_causal_mask: bool = False  # Enable causal masking for better temporal modeling
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
        # Improved time embedding with multi-layer MLP
        self.time_emb = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Frame-wise positional embedding for better temporal modeling
        self.frame_pos_emb = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Positional encodings (learned parameters for better adaptation)
        self.pos_enc = nn.Parameter(self._get_positional_encodings(self.n_frames, self.d_model))
        
        # Text projection (for conditioning as prefix)
        if conditioning_cfg and conditioning_cfg.label and num_classes is None:
            self.text_proj = nn.Sequential(
                nn.Linear(512, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, self.d_model)
            )
        else:
            self.text_proj = None
            
        # Layer normalization for better training stability
        self.input_norm = nn.LayerNorm(self.d_model)
        self.output_norm = nn.LayerNorm(self.d_model)

    def _get_positional_encodings(self, seq_len: int, d_model: int) -> Tensor:
        """Create improved positional encodings for motion sequences."""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [seq_len, d_model]
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor | None:
        """Create causal mask for temporal consistency."""
        if not self.cfg.use_causal_mask:
            return None
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

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
        
        # Process timesteps - take mean over spatial dimensions
        t = t.mean(dim=(3, 4))  # [batch, time, 1] - average over height, width
        t = t.reshape(batch * num_times, 1)  # [batch*time, 1]
        
        # Global time embedding (same for all frames)
        global_t_emb = self.time_emb(t)  # [batch*time, d_model]
        global_t_emb = global_t_emb.unsqueeze(1).expand(-1, n_frames, -1)  # [batch*time, n_frames, d_model]
        
        # Frame-wise positional embedding for better temporal modeling
        frame_positions = torch.arange(n_frames, device=x.device).float().unsqueeze(0).expand(batch * num_times, -1)
        frame_positions = frame_positions.unsqueeze(-1)  # [batch*time, n_frames, 1]
        frame_pos_emb = self.frame_pos_emb(frame_positions)  # [batch*time, n_frames, d_model]
        
        # Project input features
        x = self.input_proj(x)  # [batch*time, n_frames, d_model]
        
        # Add all embeddings
        x = x + global_t_emb + frame_pos_emb + self.pos_enc[:n_frames].unsqueeze(0)
        
        # Apply input normalization
        x = self.input_norm(x)
        
        # Text conditioning as prefix token
        if label is not None and self.text_proj is not None:
            text_emb = self.text_proj(label)  # [batch, d_model]
            text_emb = text_emb.unsqueeze(1).expand(-1, num_times, -1)  # [batch, time, d_model]
            text_emb = text_emb.reshape(batch * num_times, 1, self.d_model)  # [batch*time, 1, d_model]
            x = torch.cat([text_emb, x], dim=1)  # [batch*time, 1 + n_frames, d_model]
        
        # Create causal mask for temporal consistency (optional)
        seq_len = x.size(1)
        causal_mask = self._create_causal_mask(seq_len, x.device)
        
        # Apply transformer with optional causal masking
        x = self.transformer(x, mask=causal_mask)
        
        # Remove text prefix if present
        if label is not None and self.text_proj is not None:
            x = x[:, 1:]  # Remove first token (text)
        
        # Apply output normalization
        x = self.output_norm(x)
        
        # Project to output
        x = self.output_proj(x)  # [batch*time, n_frames, d_out * n_features]
        x = x.view(batch * num_times, n_frames, self.d_out, n_features)
        return x.view(batch, num_times, self.d_out, n_frames, n_features) 