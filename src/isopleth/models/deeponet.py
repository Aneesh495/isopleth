"""Deep Operator Network (DeepONet) architectures.

Implements unconstrained DeepONet and conservative flux-projected DeepONet
for 1D and 2D conservation law benchmarks. DeepONet decomposes the solution
operator into Branch network (encoding discretized input fields) and Trunk
network (evaluating continuous spatio-temporal coordinates).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from isopleth.models.interfaces import DiscreteFluxDivergence, InterfaceFluxHead1D


class BranchMLP(nn.Module):
    """Multi-layer perceptron encoding sensor evaluations of the input field."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: List[nn.Module] = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(act_cls())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input sensor values, shape (B, in_features).

        Returns:
            Latent representations, shape (B, out_features).
        """
        return self.net(x)


class TrunkMLP(nn.Module):
    """Multi-layer perceptron encoding continuous spatio-temporal coordinates."""

    def __init__(
        self,
        coord_dim: int,
        out_features: int,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        layers: List[nn.Module] = []
        curr_dim = coord_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(act_cls())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            coords: Query spatial coordinates, shape (N_pts, coord_dim) or (B, N_pts, coord_dim).

        Returns:
            Spatial basis functions, shape (N_pts, out_features) or (B, N_pts, out_features).
        """
        return self.net(coords)


class DeepONet1D(nn.Module):
    """Standard unconstrained DeepONet for 1D fields."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_sensors: int = 64,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_sensors = num_sensors
        self.latent_dim = latent_dim

        # Branch net takes flattened sensor grid
        self.branch = BranchMLP(
            in_features=in_channels * num_sensors,
            out_features=out_channels * latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        # Trunk net takes 1D coordinate x in [0, 1]
        self.trunk = TrunkMLP(
            coord_dim=1,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        u_sensors: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate DeepONet operator.

        Args:
            u_sensors: Sensor measurements, shape (B, C_in, N_sensors) or (B, C_in * N_sensors).
            query_coords: Query points in [0, 1], shape (N_pts, 1) or (B, N_pts, 1).
                         If None, uniform grid over [0, 1] matching num_sensors is used.

        Returns:
            Reconstructed state predictions, shape (B, C_out, N_pts).
        """
        batch_size = u_sensors.shape[0]
        if u_sensors.ndim == 3:
            u_flat = u_sensors.reshape(batch_size, -1)
        else:
            u_flat = u_sensors

        if query_coords is None:
            query_coords = torch.linspace(
                0.0, 1.0, self.num_sensors, device=u_sensors.device, dtype=u_sensors.dtype
            ).unsqueeze(-1)  # (N, 1)

        b_out = self.branch(u_flat)  # (B, C_out * latent_dim)
        b_out = b_out.view(batch_size, self.out_channels, self.latent_dim)

        if query_coords.ndim == 2:
            # Shared query points across batch: (N_pts, 1)
            t_out = self.trunk(query_coords)  # (N_pts, latent_dim)
            # Dot product: (B, C_out, latent_dim) x (N_pts, latent_dim) -> (B, C_out, N_pts)
            out = torch.einsum("bcp,np->bcn", b_out, t_out)
        else:
            # Batch-specific query points: (B, N_pts, 1)
            t_out = self.trunk(query_coords)  # (B, N_pts, latent_dim)
            out = torch.einsum("bcp,bnp->bcn", b_out, t_out)

        out = out + self.bias.view(1, self.out_channels, 1)
        return out


class ConservativeFluxDeepONet1D(nn.Module):
    """Conservative Flux-projected DeepONet for 1D conservation laws.

    Instead of predicting the solution field directly, this network predicts
    face-centered numerical fluxes F_{i+1/2} and updates cell averages via
    exact discrete divergence:
        u^{n+1}_i = u^n_i - (dt / dx) * (F_{i+1/2} - F_{i-1/2})
    guaranteeing machine-precision conservation under periodic boundary conditions.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_cells: int = 64,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_cells = num_cells
        self.latent_dim = latent_dim

        self.branch = BranchMLP(
            in_features=in_channels * num_cells,
            out_features=out_channels * latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        # Trunk evaluates face-centered coordinates x_{i+1/2}
        self.trunk = TrunkMLP(
            coord_dim=1,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.flux_bias = nn.Parameter(torch.zeros(out_channels))

        # Face coordinates on periodic grid [0, 1]
        face_coords = (
            torch.linspace(0.0, 1.0, num_cells + 1)[:-1] + (0.5 / num_cells)
        ).unsqueeze(-1)
        self.register_buffer("face_coords", face_coords)

    def forward(
        self,
        u_cells: torch.Tensor,
        dt: float = 0.01,
        dx: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting conservative next step and face fluxes.

        Args:
            u_cells: Cell-centered current state, shape (B, C_in, N_cells).
            dt: Timestep size.
            dx: Cell width. If None, 1.0 / num_cells is used.

        Returns:
            Tuple of (u_next, face_fluxes):
            - u_next: Next state, shape (B, C_out, N_cells).
            - face_fluxes: Predicted numerical fluxes, shape (B, C_out, N_cells).
        """
        batch_size, channels, n_cells = u_cells.shape
        if dx is None:
            dx = 1.0 / float(n_cells)

        u_flat = u_cells.reshape(batch_size, -1)
        b_out = self.branch(u_flat).view(batch_size, self.out_channels, self.latent_dim)

        # Trunk evaluated at face coordinates
        faces = self.face_coords.to(device=u_cells.device, dtype=u_cells.dtype)
        t_out = self.trunk(faces)  # (N_cells, latent_dim)

        # Fluxes: (B, C_out, N_cells)
        fluxes = torch.einsum("bcp,np->bcn", b_out, t_out) + self.flux_bias.view(
            1, self.out_channels, 1
        )

        # Discrete divergence with periodic boundary: F_{i+1/2} - F_{i-1/2}
        flux_right = fluxes
        flux_left = torch.roll(fluxes, shifts=1, dims=-1)
        div = (flux_right - flux_left) / float(dx)

        u_next = u_cells[:, : self.out_channels, :] - float(dt) * div
        return u_next, fluxes


class DeepONet2D(nn.Module):
    """Standard unconstrained DeepONet for 2D spatial fields."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        grid_size: Tuple[int, int] = (32, 32),
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        total_sensors = in_channels * grid_size[0] * grid_size[1]
        self.branch = BranchMLP(
            in_features=total_sensors,
            out_features=out_channels * latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        # Trunk takes 2D coordinates (x, y)
        self.trunk = TrunkMLP(
            coord_dim=2,
            out_features=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Default coordinate grid
        x = torch.linspace(0.0, 1.0, grid_size[0])
        y = torch.linspace(0.0, 1.0, grid_size[1])
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        coords_2d = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)
        self.register_buffer("coords_2d", coords_2d)

    def forward(
        self,
        u_grid: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for 2D DeepONet.

        Args:
            u_grid: Input grid states, shape (B, C_in, H, W).
            query_coords: Query points, shape (N_pts, 2) or (B, N_pts, 2).
                         If None, standard HxW grid coordinates are used.

        Returns:
            Predicted values, shape (B, C_out, H, W) or (B, C_out, N_pts).
        """
        batch_size, c_in, h, w = u_grid.shape
        u_flat = u_grid.reshape(batch_size, -1)

        b_out = self.branch(u_flat).view(batch_size, self.out_channels, self.latent_dim)

        if query_coords is None:
            pts = self.coords_2d.to(device=u_grid.device, dtype=u_grid.dtype)
            t_out = self.trunk(pts)  # (H*W, latent_dim)
            out_flat = torch.einsum("bcp,np->bcn", b_out, t_out)
            out = out_flat.view(batch_size, self.out_channels, h, w)
            out = out + self.bias.view(1, self.out_channels, 1, 1)
        else:
            if query_coords.ndim == 2:
                t_out = self.trunk(query_coords)
                out = torch.einsum("bcp,np->bcn", b_out, t_out)
            else:
                t_out = self.trunk(query_coords)
                out = torch.einsum("bcp,bnp->bcn", b_out, t_out)
            out = out + self.bias.view(1, self.out_channels, 1)

        return out
