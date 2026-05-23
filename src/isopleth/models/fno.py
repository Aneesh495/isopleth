"""Original Fourier Neural Operator (FNO) baseline implementation for 1D and 2D systems.

Implements spectral convolution blocks with truncated complex modes, continuous
coordinate lifting, and residual spatial skips.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.models.conditioning import FiLMBlock, PhysicalConditioningEncoder

class SpectralConv1d(nn.Module):
    """1D Fourier spectral convolution layer with complex weight parameterization."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral convolution: out = irfft(weights * rfft(x)).
        
        Args:
            x: Tensor of shape [B, in_channels, nx].
        """
        B, C, nx = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        # Truncate to active modes
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        active_modes = min(self.modes, x_ft.shape[-1])

        # Complex matrix multiply: [B, C_in, k] x [C_in, C_out, k] -> [B, C_out, k]
        out_ft[:, :, :active_modes] = torch.einsum(
            "bix,iox->box",
            x_ft[:, :, :active_modes],
            self.weights[:, :, :active_modes],
        )

        return torch.fft.irfft(out_ft, n=nx, dim=-1)

class SpectralConv2d(nn.Module):
    """2D Fourier spectral convolution layer with complex weight parameterization."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        # Weights for top-left (positive ky) and bottom-left (negative ky) quadrants
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 2D spectral convolution."""
        B, C, ny, nx = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        out_ft = torch.zeros(
            B, self.out_channels, ny, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        act_y = min(self.modes_y, ny // 2)
        act_x = min(self.modes_x, x_ft.shape[-1])

        # Top-left corner (positive frequencies)
        out_ft[:, :, :act_y, :act_x] = torch.einsum(
            "biyx,ioyx->boyx",
            x_ft[:, :, :act_y, :act_x],
            self.weights1[:, :, :act_y, :act_x],
        )
        # Bottom-left corner (negative frequencies)
        out_ft[:, :, -act_y:, :act_x] = torch.einsum(
            "biyx,ioyx->boyx",
            x_ft[:, :, -act_y:, :act_x],
            self.weights2[:, :, :act_y, :act_x],
        )

        return torch.fft.irfft2(out_ft, s=(ny, nx), dim=(-2, -1))

class FNO1D(nn.Module):
    """1D Fourier Neural Operator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        modes: int = 16,
        num_layers: int = 4,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.condition_dim = condition_dim

        # Lifting: in_channels + 1 (spatial coordinate x) -> hidden_channels
        self.lift = nn.Linear(in_channels + 1, hidden_channels)

        self.spectral_layers = nn.ModuleList([
            SpectralConv1d(hidden_channels, hidden_channels, modes=modes)
            for _ in range(num_layers)
        ])
        self.skip_layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(num_layers)
        ])
        self.films = nn.ModuleList([
            FiLMBlock(hidden_channels, condition_dim)
            for _ in range(num_layers)
        ])

        # Projection: hidden_channels -> out_channels
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape [B, in_channels, nx] or [B, nx].
            cond: Optional conditioning latent [B, condition_dim].
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        B, C, nx = x.shape

        # Append normalized coordinate grid [0, 1]
        grid = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype)
        grid = grid.view(1, 1, nx).repeat(B, 1, 1)
        x_with_grid = torch.cat([x, grid], dim=1)

        # Lift: [B, C+1, nx] -> [B, nx, C+1] -> [B, nx, hidden] -> [B, hidden, nx]
        h = self.lift(x_with_grid.permute(0, 2, 1)).permute(0, 2, 1)

        for spec, skip, film in zip(self.spectral_layers, self.skip_layers, self.films):
            h_spec = spec(h)
            h_skip = skip(h)
            h = F.gelu(h_spec + h_skip)
            if cond is not None:
                h = film(h, cond)

        # Project: [B, hidden, nx] -> [B, nx, hidden] -> [B, nx, out] -> [B, out, nx]
        out = self.projection(h.permute(0, 2, 1)).permute(0, 2, 1)
        return out.squeeze(1) if self.out_channels == 1 else out

class FNO2D(nn.Module):
    """2D Fourier Neural Operator."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 48,
        modes_x: int = 12,
        modes_y: int = 12,
        num_layers: int = 4,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Lifting: in_channels + 2 (coordinates x, y) -> hidden_channels
        self.lift = nn.Linear(in_channels + 2, hidden_channels)

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(hidden_channels, hidden_channels, modes_x=modes_x, modes_y=modes_y)
            for _ in range(num_layers)
        ])
        self.skip_layers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
            for _ in range(num_layers)
        ])
        self.films = nn.ModuleList([
            FiLMBlock(hidden_channels, condition_dim)
            for _ in range(num_layers)
        ])

        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, 96),
            nn.GELU(),
            nn.Linear(96, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape [B, C, ny, nx].
            cond: Optional conditioning latent [B, condition_dim].
        """
        B, C, ny, nx = x.shape

        # Coordinate grids
        gx = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0.0, 1.0, ny, device=x.device, dtype=x.dtype)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")
        coord = torch.stack([GX, GY], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        x_with_grid = torch.cat([x, coord], dim=1)

        # Lift
        h = self.lift(x_with_grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for spec, skip, film in zip(self.spectral_layers, self.skip_layers, self.films):
            h_spec = spec(h)
            h_skip = skip(h)
            h = F.gelu(h_spec + h_skip)
            if cond is not None:
                h = film(h, cond)

        out = self.projection(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


FourierNeuralOperator1D = FNO1D
FourierNeuralOperator2D = FNO2D
