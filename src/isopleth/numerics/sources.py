"""Source term evaluation, Runge-Kutta integration, and physical balance auditing.

Validates exact physical accounting:
    Mass(t_end) - Mass(t_start) = Integral(Source * dA) * dt - Integral(F_boundary * dl) * dt
to float64 tolerance 1e-9 for closed and periodic domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import torch

@dataclass
class BalanceAuditReport:
    """Report detailing discrete mass conservation and source balance."""
    initial_integral: float
    final_integral: float
    integrated_source: float
    integrated_boundary_flux: float
    absolute_residual: float
    relative_residual: float
    is_balanced: bool
    tolerance: float

def compute_spatial_integral(
    field: torch.Tensor,
    cell_measures: Tuple[float, ...],
) -> torch.Tensor:
    """Compute discrete spatial integral sum(u * cell_volume).
    
    Args:
        field: Tensor of shape [..., nx] (1D) or [..., ny, nx] (2D).
        cell_measures: Tuple (dx,) or (dx, dy).
        
    Returns:
        integral: Tensor of shape [...] containing the total integral.
    """
    if len(cell_measures) == 1:
        dx = cell_measures[0]
        return torch.sum(field, dim=-1) * dx
    elif len(cell_measures) == 2:
        dx, dy = cell_measures[0], cell_measures[1]
        return torch.sum(field, dim=(-2, -1)) * (dx * dy)
    else:
        raise ValueError(f"Unsupported spatial dimension: {len(cell_measures)}")

class BalanceAuditor:
    """Audits discrete conservation laws with explicit boundary and source accounting."""

    def __init__(self, cell_measures: Tuple[float, ...], tolerance_float64: float = 1e-9) -> None:
        self.cell_measures = cell_measures
        self.tolerance = tolerance_float64

    def audit(
        self,
        u_initial: torch.Tensor,
        u_final: torch.Tensor,
        total_source_integrated: float = 0.0,
        total_boundary_flux_integrated: float = 0.0,
    ) -> BalanceAuditReport:
        """Audit physical balance across an arbitrary time horizon."""
        m_init = float(compute_spatial_integral(u_initial, self.cell_measures).item())
        m_final = float(compute_spatial_integral(u_final, self.cell_measures).item())

        delta_m = m_final - m_init
        expected_change = total_source_integrated - total_boundary_flux_integrated
        abs_res = abs(delta_m - expected_change)
        denom = max(1.0, abs(m_init))
        rel_res = abs_res / denom

        return BalanceAuditReport(
            initial_integral=m_init,
            final_integral=m_final,
            integrated_source=total_source_integrated,
            integrated_boundary_flux=total_boundary_flux_integrated,
            absolute_residual=abs_res,
            relative_residual=rel_res,
            is_balanced=(rel_res <= self.tolerance),
            tolerance=self.tolerance,
        )

def gray_scott_reaction_sources(
    u: torch.Tensor,
    v: torch.Tensor,
    f: float,
    k: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute audited Gray-Scott reaction rates.
    
    du/dt = -u * v^2 + f * (1 - u)
    dv/dt = +u * v^2 - (f + k) * v  (audited positive production term)
    """
    uv2 = u * (v**2)
    s_u = -uv2 + f * (1.0 - u)
    s_v = +uv2 - (f + k) * v
    return s_u, s_v

def shallow_water_bottom_friction(
    h: torch.Tensor,
    hu: torch.Tensor,
    hv: torch.Tensor,
    friction_coeff: float = 0.001,
    h_min: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute quadratic bottom friction drag: S_f = - c_f * u * |u|."""
    h_safe = torch.clamp(h, min=h_min)
    u_vel = hu / h_safe
    v_vel = hv / h_safe
    vel_mag = torch.sqrt(u_vel**2 + v_vel**2 + 1e-12)

    drag_factor = friction_coeff * vel_mag
    s_hu = -drag_factor * hu
    s_hv = -drag_factor * hv
    return s_hu, s_hv

def shallow_water_coriolis_source(
    hu: torch.Tensor,
    hv: torch.Tensor,
    f_coriolis: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Coriolis acceleration source: S = (+f * hv, -f * hu)."""
    s_hu = f_coriolis * hv
    s_hv = -f_coriolis * hu
    return s_hu, s_hv

def integrate_source_rk4(
    state: torch.Tensor,
    source_fn: Callable[[torch.Tensor], torch.Tensor],
    dt: float,
) -> torch.Tensor:
    """Explicit 4th-order Runge-Kutta integrator for stiff/non-stiff source terms."""
    k1 = source_fn(state)
    k2 = source_fn(state + 0.5 * dt * k1)
    k3 = source_fn(state + 0.5 * dt * k2)
    k4 = source_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
