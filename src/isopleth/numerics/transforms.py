"""Discrete cell-face operators, divergence stencils, and conservative resolution transforms.

Defines conservative restriction and prolongation operators that strictly preserve
the discrete physical integral:
    Integral(u) = sum_{cells} u_i * dx_i
as well as discrete divergence stencils mapping face-centered fluxes to cell centers.
"""

from __future__ import annotations

import math
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

def discrete_divergence_1d(
    flux_faces: torch.Tensor,
    dx: float,
) -> torch.Tensor:
    """Compute discrete divergence of 1D face-centered fluxes.
    
    Args:
        flux_faces: Tensor of shape [..., nx + 1] representing fluxes at interfaces i + 1/2.
        dx: Cell spacing.
        
    Returns:
        div_flux: Tensor of shape [..., nx] at cell centers: (F_{i+1/2} - F_{i-1/2}) / dx.
    """
    if flux_faces.shape[-1] < 2:
        raise ValueError(f"Face flux tensor requires at least 2 points along last axis, got {flux_faces.shape[-1]}")
    return (flux_faces[..., 1:] - flux_faces[..., :-1]) / dx

def discrete_divergence_2d(
    flux_x_faces: torch.Tensor,
    flux_y_faces: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute discrete divergence of 2D face-centered fluxes.
    
    Args:
        flux_x_faces: Tensor of shape [..., ny, nx + 1] at x-faces (i + 1/2, j).
        flux_y_faces: Tensor of shape [..., ny + 1, nx] at y-faces (i, j + 1/2).
        dx: Cell spacing along x.
        dy: Cell spacing along y.
        
    Returns:
        div_flux: Tensor of shape [..., ny, nx] at cell centers.
    """
    dF_dx = (flux_x_faces[..., :, 1:] - flux_x_faces[..., :, :-1]) / dx
    dG_dy = (flux_y_faces[..., 1:, :] - flux_y_faces[..., :-1, :]) / dy
    return dF_dx + dG_dy

def conservative_restriction_1d(
    u_fine: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """Conservatively restrict 1D cell-centered field by integer coarsening factor.
    
    Guarantees sum(u_coarse * dx_coarse) == sum(u_fine * dx_fine) exactly.
    """
    if factor <= 0:
        raise ValueError(f"Coarsening factor must be positive, got {factor}")
    nx_fine = u_fine.shape[-1]
    if nx_fine % factor != 0:
        raise ValueError(f"Fine grid size {nx_fine} must be divisible by factor {factor}")
    
    # Reshape and average: (..., nx_coarse, factor)
    orig_shape = u_fine.shape[:-1]
    nx_coarse = nx_fine // factor
    reshaped = u_fine.view(*orig_shape, nx_coarse, factor)
    return reshaped.mean(dim=-1)

def conservative_restriction_2d(
    u_fine: torch.Tensor,
    factor: int = 2,
) -> torch.Tensor:
    """Conservatively restrict 2D cell-centered field by integer coarsening factor.
    
    Guarantees sum(u_coarse * dx_coarse * dy_coarse) == sum(u_fine * dx_fine * dy_fine).
    """
    if factor <= 0:
        raise ValueError(f"Coarsening factor must be positive, got {factor}")
    ny_fine, nx_fine = u_fine.shape[-2], u_fine.shape[-1]
    if ny_fine % factor != 0 or nx_fine % factor != 0:
        raise ValueError(
            f"Fine grid size ({ny_fine}, {nx_fine}) must be divisible by factor {factor}"
        )
    
    orig_shape = u_fine.shape[:-2]
    ny_coarse = ny_fine // factor
    nx_coarse = nx_fine // factor
    reshaped = u_fine.view(*orig_shape, ny_coarse, factor, nx_coarse, factor)
    return reshaped.mean(dim=(-3, -1))

def conservative_prolongation_1d(
    u_coarse: torch.Tensor,
    factor: int = 2,
    order: int = 1,
) -> torch.Tensor:
    """Conservatively prolong 1D cell-centered field to a finer grid.
    
    Order 0/1 piecewise constant replication:
    Each coarse cell of value c is replicated into `factor` subcells of value c.
    This guarantees exact conservation:
        sum_{subcells} c * (dx / factor) = factor * c * (dx / factor) = c * dx.
    Order 2 uses slope-limited linear reconstruction with zero-mean subcell deviation.
    """
    if factor <= 0:
        raise ValueError(f"Refinement factor must be positive, got {factor}")
    
    if order == 1:
        # Exact piece-wise constant replication
        orig_shape = u_coarse.shape[:-1]
        nx_coarse = u_coarse.shape[-1]
        expanded = u_coarse.unsqueeze(-1).repeat_interleave(factor, dim=-1)
        return expanded.view(*orig_shape, nx_coarse * factor)
    
    elif order == 2:
        # Conservative linear reconstruction with minmod slope
        nx_coarse = u_coarse.shape[-1]
        orig_shape = u_coarse.shape[:-1]
        # Periodic boundary extension for slopes
        u_padded = torch.cat([u_coarse[..., -1:], u_coarse, u_coarse[..., :1]], dim=-1)
        slope_fwd = u_padded[..., 2:] - u_padded[..., 1:-1]
        slope_bwd = u_padded[..., 1:-1] - u_padded[..., :-2]
        
        # Minmod slope limiter
        minmod_slope = torch.where(
            slope_fwd * slope_bwd > 0.0,
            torch.sign(slope_fwd) * torch.minimum(torch.abs(slope_fwd), torch.abs(slope_bwd)),
            torch.zeros_like(slope_fwd),
        )
        
        # Subcell offsets centered at 0 with zero mean: sum delta_x_k = 0
        offsets = torch.linspace(-0.5 + 0.5 / factor, 0.5 - 0.5 / factor, factor, device=u_coarse.device)
        # u_{i, k} = u_i + s_i * offset_k
        u_fine_cells = u_coarse.unsqueeze(-1) + minmod_slope.unsqueeze(-1) * offsets
        return u_fine_cells.view(*orig_shape, nx_coarse * factor)
    else:
        raise ValueError(f"Supported prolongation orders are 1 and 2, got {order}")

def conservative_prolongation_2d(
    u_coarse: torch.Tensor,
    factor: int = 2,
    order: int = 1,
) -> torch.Tensor:
    """Conservatively prolong 2D cell-centered field to a finer grid.
    
    Guarantees sum_{fine} u_fine * dA_fine == sum_{coarse} u_coarse * dA_coarse.
    """
    if factor <= 0:
        raise ValueError(f"Refinement factor must be positive, got {factor}")
    
    orig_shape = u_coarse.shape[:-2]
    ny_coarse, nx_coarse = u_coarse.shape[-2], u_coarse.shape[-1]
    
    if order == 1:
        # Piecewise constant block replication
        rep = u_coarse.unsqueeze(-1).unsqueeze(-3)
        rep = rep.repeat_interleave(factor, dim=-1).repeat_interleave(factor, dim=-3)
        return rep.view(*orig_shape, ny_coarse * factor, nx_coarse * factor)
    
    elif order == 2:
        # Conservative bilinear reconstruction with minmod slopes in both directions
        u_pad_x = torch.cat([u_coarse[..., :, -1:], u_coarse, u_coarse[..., :, :1]], dim=-1)
        u_pad = torch.cat([u_pad_x[..., -1:, :], u_pad_x, u_pad_x[..., :1, :]], dim=-2)
        
        # Slopes along x
        sx_fwd = u_pad[..., 1:-1, 2:] - u_pad[..., 1:-1, 1:-1]
        sx_bwd = u_pad[..., 1:-1, 1:-1] - u_pad[..., 1:-1, :-2]
        sx = torch.where(
            sx_fwd * sx_bwd > 0.0,
            torch.sign(sx_fwd) * torch.minimum(torch.abs(sx_fwd), torch.abs(sx_bwd)),
            torch.zeros_like(sx_fwd),
        )
        
        # Slopes along y
        sy_fwd = u_pad[..., 2:, 1:-1] - u_pad[..., 1:-1, 1:-1]
        sy_bwd = u_pad[..., 1:-1, 1:-1] - u_pad[..., :-2, 1:-1]
        sy = torch.where(
            sy_fwd * sy_bwd > 0.0,
            torch.sign(sy_fwd) * torch.minimum(torch.abs(sy_fwd), torch.abs(sy_bwd)),
            torch.zeros_like(sy_fwd),
        )
        
        offsets = torch.linspace(-0.5 + 0.5 / factor, 0.5 - 0.5 / factor, factor, device=u_coarse.device)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        
        u_expanded = u_coarse.unsqueeze(-1).unsqueeze(-1)
        sx_expanded = sx.unsqueeze(-1).unsqueeze(-1)
        sy_expanded = sy.unsqueeze(-1).unsqueeze(-1)
        
        fine_blocks = u_expanded + sx_expanded * ox + sy_expanded * oy
        # Permute and reshape to (..., ny_coarse * factor, nx_coarse * factor)
        fine_blocks = fine_blocks.permute(*range(len(orig_shape)), -4, -2, -3, -1)
        return fine_blocks.reshape(*orig_shape, ny_coarse * factor, nx_coarse * factor)
    else:
        raise ValueError(f"Supported prolongation orders are 1 and 2, got {order}")
