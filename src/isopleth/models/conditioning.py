"""Parameter and physical lead-time conditioning modules for neural operators.

Embeds physical parameters (viscosity, gravity, reaction rates) and continuous
lead time dt using sinusoidal Fourier embeddings and Feature-wise Linear Modulation (FiLM).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar physical parameters and lead time."""

    def __init__(self, embed_dim: int = 64, max_freq: float = 1000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input tensor of shape [...] into [..., embed_dim]."""
        # Shape: [..., 1] * [half_dim] -> [..., half_dim]
        angles = x.unsqueeze(-1) * self.freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return emb

class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.
    
    Applies affine modulation: y = gamma * x + beta
    where gamma and beta are conditioned on physical parameters and lead time.
    """

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, 2 * feature_dim),
        )
        # Initialize gamma centered at 1 and beta near 0 with small weights for gradient flow
        nn.init.normal_(self.condition_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.condition_mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Modulate feature tensor x with conditioning vector cond.
        
        Args:
            x: Tensor of shape [B, C, ...].
            cond: Tensor of shape [B, condition_dim].
        """
        # cond_params: [B, 2 * C]
        cond_params = self.condition_mlp(cond)
        gamma, beta = torch.chunk(cond_params, 2, dim=-1)
        # Reshape gamma and beta to broadcast across spatial dimensions
        ndim = x.ndim
        view_shape = [-1, self.feature_dim] + [1] * (ndim - 2)
        gamma = gamma.view(*view_shape) + 1.0  # Centered at 1.0
        beta = beta.view(*view_shape)
        return gamma * x + beta

class PhysicalConditioningEncoder(nn.Module):
    """Encodes physical parameters and lead time dt into a unified conditioning vector."""

    def __init__(
        self,
        num_physical_params: int = 2,
        embed_dim: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.param_embed = nn.Sequential(
            nn.Linear(num_physical_params, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.time_embed = FourierEmbedding(embed_dim=embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        parameters: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        """Encode parameters and dt into conditioning latent.
        
        Args:
            parameters: Tensor [B, num_physical_params]
            dt: Tensor [B] or scalar float
        Returns:
            latent: Tensor [B, out_dim]
        """
        B = parameters.shape[0]
        if isinstance(dt, (int, float)):
            dt_tensor = torch.full((B,), float(dt), dtype=parameters.dtype, device=parameters.device)
        elif dt.ndim == 0:
            dt_tensor = dt.repeat(B)
        else:
            dt_tensor = dt

        p_latent = self.param_embed(parameters)
        t_emb = self.time_embed(dt_tensor)
        t_latent = self.time_mlp(t_emb)

        combined = torch.cat([p_latent, t_latent], dim=-1)
        return self.fusion(combined)
