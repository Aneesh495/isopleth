"""Antisymmetric and shared interface flux heads with exact discrete divergence (I11).

Formulates neural operator predictions as discrete interface fluxes rather than unconstrained
cell updates. The shared face-flux ownership guarantees that discrete divergence sums to zero
telescopically across the computational domain, enforcing exact mass and momentum conservation.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.data.contracts import BoundaryCondition

class InterfaceFluxHead1D(nn.Module):
    """Predicts staggered interface fluxes F_{i+1/2} for 1D conservation laws."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        # Input to interface flux is concatenated left and right cell representations: 2 * in_channels
        self.flux_net = nn.Sequential(
            nn.Conv1d(2 * in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> torch.Tensor:
        """Compute interface fluxes F_{i+1/2}.
        
        Args:
            features: Tensor of shape [B, C, nx] at cell centers.
            boundary: Boundary condition semantics.
            
        Returns:
            fluxes: Tensor of shape [B, out_channels, nx] (periodic) or [B, out_channels, nx + 1].
        """
        if boundary == BoundaryCondition.PERIODIC:
            feat_L = features
            feat_R = torch.roll(features, shifts=-1, dims=-1)
            # Concatenate features at interface: [B, 2*C, nx]
            interface_repr = torch.cat([feat_L, feat_R], dim=1)
            return self.flux_net(interface_repr)
        else:
            feat_pad = torch.cat([features[..., :1], features, features[..., -1:]], dim=-1)
            feat_L = feat_pad[..., :-1]
            feat_R = feat_pad[..., 1:]
            interface_repr = torch.cat([feat_L, feat_R], dim=1)
            return self.flux_net(interface_repr)

class InterfaceFluxHead2D(nn.Module):
    """Predicts staggered zonal (F_x) and meridional (F_y) interface fluxes for 2D conservation laws."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.flux_x_net = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self.flux_y_net = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute interface fluxes F_x and F_y.
        
        Args:
            features: Tensor of shape [B, C, ny, nx] at cell centers.
            
        Returns:
            flux_x: Tensor of shape [B, out_channels, ny, nx] at x-interfaces.
            flux_y: Tensor of shape [B, out_channels, ny, nx] at y-interfaces.
        """
        if boundary == BoundaryCondition.PERIODIC:
            # x-faces (i+1/2, j)
            fx_L = features
            fx_R = torch.roll(features, shifts=-1, dims=-1)
            fx_input = torch.cat([fx_L, fx_R], dim=1)
            flux_x = self.flux_x_net(fx_input)

            # y-faces (i, j+1/2)
            fy_L = features
            fy_R = torch.roll(features, shifts=-1, dims=-2)
            fy_input = torch.cat([fy_L, fy_R], dim=1)
            flux_y = self.flux_y_net(fy_input)

            return flux_x, flux_y
        else:
            # Closed walls
            fx_pad = torch.cat([features[..., :, :1], features, features[..., :, -1:]], dim=-1)
            fx_L = fx_pad[..., :, :-1]
            fx_R = fx_pad[..., :, 1:]
            fx_input = torch.cat([fx_L, fx_R], dim=1)
            flux_x = self.flux_x_net(fx_input)
            # Enforce zero mass flux through wall
            flux_x[..., :, 0] = 0.0
            flux_x[..., :, -1] = 0.0

            fy_pad = torch.cat([features[..., :1, :], features, features[..., -1:, :]], dim=-2)
            fy_L = fy_pad[..., :-1, :]
            fy_R = fy_pad[..., 1:, :]
            fy_input = torch.cat([fy_L, fy_R], dim=1)
            flux_y = self.flux_y_net(fy_input)
            flux_y[..., 0, :] = 0.0
            flux_y[..., -1, :] = 0.0

            return flux_x, flux_y

class DiscreteFluxDivergence(nn.Module):
    """Computes exact discrete divergence of face fluxes."""

    def __init__(self, spatial_dim: int = 1) -> None:
        super().__init__()
        self.spatial_dim = spatial_dim

    def forward_1d(
        self,
        flux_faces: torch.Tensor,
        dx: float,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> torch.Tensor:
        """Compute -(F_{i+1/2} - F_{i-1/2}) / dx."""
        if boundary == BoundaryCondition.PERIODIC:
            # F_{i-1/2} is roll(flux_faces, 1)
            div = (flux_faces - torch.roll(flux_faces, shifts=1, dims=-1)) / dx
        else:
            div = (flux_faces[..., 1:] - flux_faces[..., :-1]) / dx
        return -div

    def forward_2d(
        self,
        flux_x: torch.Tensor,
        flux_y: torch.Tensor,
        dx: float,
        dy: float,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> torch.Tensor:
        """Compute -(dFx/dx + dFy/dy)."""
        if boundary == BoundaryCondition.PERIODIC:
            div_x = (flux_x - torch.roll(flux_x, shifts=1, dims=-1)) / dx
            div_y = (flux_y - torch.roll(flux_y, shifts=1, dims=-2)) / dy
        else:
            div_x = (flux_x[..., :, 1:] - flux_x[..., :, :-1]) / dx
            div_y = (flux_y[..., 1:, :] - flux_y[..., :-1, :]) / dy
        return -(div_x + div_y)
