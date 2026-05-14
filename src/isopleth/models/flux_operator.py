"""Multiscale Face-Flux Neural Operator (MFFNO) for conservation-aware dynamics (I10).

Combines multiscale spectral Fourier representations and local convolutional feature paths
with an antisymmetric interface flux head, discrete divergence, and explicit source accounting.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.data.contracts import BoundaryCondition
from isopleth.models.conditioning import FiLMBlock, PhysicalConditioningEncoder
from isopleth.models.fno import SpectralConv1d, SpectralConv2d
from isopleth.models.interfaces import DiscreteFluxDivergence, InterfaceFluxHead1D, InterfaceFluxHead2D

class MultiscaleFaceFluxOperator1D(nn.Module):
    """Primary 1D Multiscale Face-Flux Neural Operator."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 64,
        modes: int = 16,
        num_layers: int = 3,
        condition_dim: int = 128,
        use_coarse_numerical_prior: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_coarse_numerical_prior = use_coarse_numerical_prior

        # Lift layer
        self.lift = nn.Linear(in_channels + 1, hidden_channels)

        # Spectral branch layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv1d(hidden_channels, hidden_channels, modes=modes)
            for _ in range(num_layers)
        ])
        # Local spatial branch layers
        self.local_layers = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular")
            for _ in range(num_layers)
        ])
        # Feature modulation
        self.films = nn.ModuleList([
            FiLMBlock(hidden_channels, condition_dim)
            for _ in range(num_layers)
        ])

        # Interface flux head
        self.flux_head = InterfaceFluxHead1D(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )
        self.divergence = DiscreteFluxDivergence(spatial_dim=1)

    def forward(
        self,
        x: torch.Tensor,
        dx: float,
        dt: float | torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        source_terms: Optional[torch.Tensor] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute single-step conservative operator forward pass.
        
        Args:
            x: Current state [B, in_channels, nx] or [B, nx].
            dx: Spatial grid spacing.
            dt: Lead time step.
            cond: Optional conditioning latent [B, condition_dim].
            source_terms: Optional explicit source terms [B, out_channels, nx].
            boundary: Boundary condition semantics.
            
        Returns:
            (x_next, face_fluxes):
                x_next: Updated state [B, in_channels, nx].
                face_fluxes: Predicted interface fluxes.
        """
        is_flat = (x.ndim == 2)
        if is_flat:
            x = x.unsqueeze(1)
        B, C, nx = x.shape

        # Append normalized coordinate grid
        grid = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype).view(1, 1, nx).repeat(B, 1, 1)
        h = self.lift(torch.cat([x, grid], dim=1).permute(0, 2, 1)).permute(0, 2, 1)

        # Multiscale fusion through layers
        for spec, local, film in zip(self.spectral_layers, self.local_layers, self.films):
            h_spec = spec(h)
            h_local = local(h)
            h = F.gelu(h_spec + h_local)
            if cond is not None:
                h = film(h, cond)

        # Predict interface fluxes
        flux_faces = self.flux_head(h, boundary=boundary)

        # Optional coarse numerical flux baseline
        if self.use_coarse_numerical_prior:
            # First order Rusanov flux as prior
            u_L = x
            u_R = torch.roll(x, shifts=-1, dims=-1) if boundary == BoundaryCondition.PERIODIC else torch.cat([x, x[..., -1:]], dim=-1)
            f_prior = 0.5 * (0.5 * u_L**2 + 0.5 * u_R**2) - 0.5 * torch.maximum(torch.abs(u_L), torch.abs(u_R)) * (u_R - u_L)
            flux_faces = f_prior + flux_faces

        # Discrete divergence: -dF/dx
        div_flux = self.divergence.forward_1d(flux_faces, dx=dx, boundary=boundary)

        # Time integration: x^{n+1} = x^n + dt * (-div F) + dt * S
        if isinstance(dt, torch.Tensor):
            dt_step = dt.view(-1, 1, 1)
        else:
            dt_step = float(dt)

        dx_dt = div_flux
        if source_terms is not None:
            dx_dt = dx_dt + source_terms

        x_next = x + dt_step * dx_dt

        out_next = x_next.squeeze(1) if is_flat and self.out_channels == 1 else x_next
        return out_next, flux_faces

class MultiscaleFaceFluxOperator2D(nn.Module):
    """Primary 2D Multiscale Face-Flux Neural Operator."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_channels: int = 48,
        modes_x: int = 12,
        modes_y: int = 12,
        num_layers: int = 3,
        condition_dim: int = 128,
        use_coarse_numerical_prior: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_coarse_numerical_prior = use_coarse_numerical_prior

        # Lift layer: in_channels + 2 (x, y coords) -> hidden_channels
        self.lift = nn.Linear(in_channels + 2, hidden_channels)

        # Spectral branch layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(hidden_channels, hidden_channels, modes_x=modes_x, modes_y=modes_y)
            for _ in range(num_layers)
        ])
        # Local spatial branch layers
        self.local_layers = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular")
            for _ in range(num_layers)
        ])
        self.films = nn.ModuleList([
            FiLMBlock(hidden_channels, condition_dim)
            for _ in range(num_layers)
        ])

        # Interface flux head
        self.flux_head = InterfaceFluxHead2D(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )
        self.divergence = DiscreteFluxDivergence(spatial_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        dx: float,
        dy: float,
        dt: float | torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        source_terms: Optional[torch.Tensor] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Execute single-step 2D conservative operator forward pass.
        
        Args:
            x: Current state [B, C, ny, nx].
            dx: Grid spacing along x.
            dy: Grid spacing along y.
            dt: Lead time step.
            cond: Optional conditioning latent [B, condition_dim].
            source_terms: Optional source terms [B, C, ny, nx].
        """
        B, C, ny, nx = x.shape

        # Continuous coordinate grids
        gx = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0.0, 1.0, ny, device=x.device, dtype=x.dtype)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")
        coord = torch.stack([GX, GY], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        h = self.lift(torch.cat([x, coord], dim=1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for spec, local, film in zip(self.spectral_layers, self.local_layers, self.films):
            h_spec = spec(h)
            h_local = local(h)
            h = F.gelu(h_spec + h_local)
            if cond is not None:
                h = film(h, cond)

        flux_x, flux_y = self.flux_head(h, boundary=boundary)

        # Discrete divergence: -(dFx/dx + dFy/dy)
        div_flux = self.divergence.forward_2d(flux_x, flux_y, dx=dx, dy=dy, boundary=boundary)

        if isinstance(dt, torch.Tensor):
            dt_step = dt.view(-1, 1, 1, 1)
        else:
            dt_step = float(dt)

        dx_dt = div_flux
        if source_terms is not None:
            dx_dt = dx_dt + source_terms

        x_next = x + dt_step * dx_dt
        return x_next, (flux_x, flux_y)
