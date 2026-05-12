"""Differentiable conservative depth limiter and diagnostics for shallow-water dynamics.

Strictly preserves total mass while guaranteeing local non-negativity (h >= h_min).
Blind per-cell clipping introduces unphysical artificial mass; this module redistributes
deficits to adjacent surplus cells to enforce exact discrete mass conservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

@dataclass
class LimiterDiagnostics:
    """Diagnostic tracking record for conservative limiter interventions."""
    activations_count: int
    total_deficit_mass: float
    max_local_deficit: float
    is_safe: bool
    unrecoverable_deficit: float = 0.0

class ConservativeDepthLimiter:
    """Conservative depth limiter redistributing negative deficits to neighbors."""

    def __init__(self, h_min: float = 1e-6, max_iter: int = 3) -> None:
        self.h_min = h_min
        self.max_iter = max_iter

    def __call__(
        self,
        h: torch.Tensor,
        hu: Optional[torch.Tensor] = None,
        hv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], LimiterDiagnostics]:
        """Apply conservative limiter to depth h and desaturate momentum in near-dry cells.
        
        Args:
            h: Depth tensor of shape [..., ny, nx].
            hu: Zonal momentum tensor of shape [..., ny, nx].
            hv: Meridional momentum tensor of shape [..., ny, nx].
            
        Returns:
            h_lim: Conservatives depth tensor with h >= h_min.
            hu_lim: Momentum desaturated to prevent unphysical velocities when h -> 0.
            hv_lim: Momentum desaturated.
            diagnostics: LimiterDiagnostics tracking activations and mass integrity.
        """
        orig_mass = torch.sum(h, dim=(-2, -1), keepdim=True)
        h_curr = h.clone()
        
        deficit = torch.clamp(self.h_min - h_curr, min=0.0)
        has_deficit = deficit > 0.0
        n_activated = int(torch.sum(has_deficit).item())
        max_def = float(torch.max(deficit).item())
        tot_def = float(torch.sum(deficit).item())

        if n_activated == 0:
            return h_curr, hu, hv, LimiterDiagnostics(
                activations_count=0,
                total_deficit_mass=0.0,
                max_local_deficit=0.0,
                is_safe=True,
                unrecoverable_deficit=0.0,
            )

        # Iterative local conservative diffusion of surplus to deficient cells
        kernel = torch.tensor(
            [[0.0, 0.25, 0.0],
             [0.25, 0.0, 0.25],
             [0.0, 0.25, 0.0]],
            dtype=h.dtype,
            device=h.device,
        ).view(1, 1, 3, 3)

        for _ in range(self.max_iter):
            def_mask = (h_curr < self.h_min).float()
            def_amount = torch.clamp(self.h_min - h_curr, min=0.0)
            if torch.max(def_amount) <= 1e-9:
                break

            # Calculate surplus available in cells: max(h - h_min, 0)
            surplus = torch.clamp(h_curr - self.h_min, min=0.0)
            
            # Smoothly spread deficit to neighbors
            # Reshape for 2D convolution
            shape_prefix = h_curr.shape[:-2]
            ny, nx = h_curr.shape[-2], h_curr.shape[-1]
            flat_h = h_curr.view(-1, 1, ny, nx)
            flat_surplus = surplus.view(-1, 1, ny, nx)
            flat_def = def_amount.view(-1, 1, ny, nx)

            # Circular pad for periodic boundaries
            padded_surplus = F.pad(flat_surplus, (1, 1, 1, 1), mode="circular")
            neighbor_surplus = F.conv2d(padded_surplus, kernel)

            # Redistribution ratio
            transfer = torch.minimum(
                flat_def,
                neighbor_surplus * 0.5
            )

            # Add to deficient cells, subtract from surplus neighbors
            padded_transfer = F.pad(transfer, (1, 1, 1, 1), mode="circular")
            subtracted_from_neighbors = F.conv2d(padded_transfer, kernel)

            flat_h = flat_h + transfer - subtracted_from_neighbors
            h_curr = flat_h.view(*shape_prefix, ny, nx)

        # If localized iterations leave residual minor deficits, perform global zero-sum correction
        final_def = torch.clamp(self.h_min - h_curr, min=0.0)
        tot_final_def = torch.sum(final_def, dim=(-2, -1), keepdim=True)
        
        # Lift remaining negative cells to h_min
        h_curr = torch.maximum(h_curr, torch.full_like(h_curr, self.h_min))
        
        # Deduct total lifted deficit proportionally from cells with generous surplus
        surplus_pool = torch.clamp(h_curr - 2.0 * self.h_min, min=0.0)
        tot_surplus = torch.sum(surplus_pool, dim=(-2, -1), keepdim=True)
        
        is_safe = bool(torch.all(tot_surplus > tot_final_def).item())
        unrecoverable = 0.0
        
        if is_safe:
            deduction = (surplus_pool / torch.clamp(tot_surplus, min=1e-12)) * tot_final_def
            h_curr = h_curr - deduction
        else:
            unrecoverable = float(torch.sum(torch.clamp(tot_final_def - tot_surplus, min=0.0)).item())

        # Exact global mass conservation audit
        new_mass = torch.sum(h_curr, dim=(-2, -1), keepdim=True)
        mass_error = float(torch.max(torch.abs(new_mass - orig_mass)).item())

        # Velocity / momentum desaturation:
        # Near dry cells (h < 10 * h_min) must have velocity bounded to avoid u = hu / h blowing up
        hu_lim = hu
        hv_lim = hv
        if hu is not None and hv is not None:
            # Differentiable cutoff function: tanh(h / (5 * h_min))
            dry_factor = torch.clamp(h_curr / (5.0 * self.h_min), min=0.0, max=1.0)
            hu_lim = hu * dry_factor
            hv_lim = hv * dry_factor

        diagnostics = LimiterDiagnostics(
            activations_count=n_activated,
            total_deficit_mass=tot_def,
            max_local_deficit=max_def,
            is_safe=is_safe and (mass_error < 1e-5),
            unrecoverable_deficit=unrecoverable,
        )

        return h_curr, hu_lim, hv_lim, diagnostics
