"""Original conservative finite-volume reference solver for 1D viscous Burgers equation.

Solves:
    du/dt + d(0.5 * u^2)/dx = nu * d^2 u / dx^2
using high-order TVD reconstruction (minmod, van Leer, superbee), Godunov/Rusanov numerical fluxes,
conservative interface diffusion, and explicit SSP-RK2/RK3 time stepping.
Rejects any timestep violating the CFL stability contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from isopleth.data.contracts import (
    BoundaryCondition,
    GridContract,
    PhysicalFamily,
    TrajectoryBatch,
    TrajectoryProvenance,
)

class FluxScheme(str, Enum):
    GODUNOV = "godunov"
    RUSANOV = "rusanov"
    ROE = "roe"

class SlopeLimiter(str, Enum):
    MINMOD = "minmod"
    VAN_LEER = "van_leer"
    SUPERBEE = "superbee"
    NONE = "none"

@dataclass
class BurgersSolverConfig:
    viscosity: float = 0.01
    cfl: float = 0.45
    flux_scheme: FluxScheme = FluxScheme.GODUNOV
    limiter: SlopeLimiter = SlopeLimiter.VAN_LEER
    time_integrator: str = "ssp_rk2"  # 'ssp_rk2' or 'euler'
    adaptive_cfl: bool = False

class Burgers1DSolver:
    """Production-grade conservative finite-volume solver for 1D viscous Burgers flow."""

    def __init__(self, config: Optional[BurgersSolverConfig] = None) -> None:
        self.config = config or BurgersSolverConfig()

    def check_cfl_stability(self, u: torch.Tensor, dx: float, dt: float) -> Tuple[bool, float]:
        """Verify that dt satisfies advective and viscous stability criteria.
        
        Returns:
            (is_stable, max_allowable_dt)
        """
        max_speed = float(torch.max(torch.abs(u)).item())
        if max_speed <= 1e-12:
            dt_adv = 1e6
        else:
            dt_adv = self.config.cfl * (dx / max_speed)

        if self.config.viscosity > 0.0:
            dt_diff = 0.45 * (dx**2) / self.config.viscosity
        else:
            dt_diff = 1e6

        max_allowable_dt = min(dt_adv, dt_diff)
        is_stable = dt <= (max_allowable_dt * 1.0001)
        return is_stable, max_allowable_dt

    def reconstruct_interface_states(
        self,
        u: torch.Tensor,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute left and right state reconstructions (u_L, u_R) at interfaces i + 1/2.
        
        Args:
            u: Cell-centered values [..., nx].
        Returns:
            u_L: Left state at faces [..., nx + 1] (or nx for periodic wrap).
            u_R: Right state at faces [..., nx + 1].
        """
        nx = u.shape[-1]
        if self.config.limiter == SlopeLimiter.NONE:
            # Piecewise constant (1st order)
            if boundary == BoundaryCondition.PERIODIC:
                u_L = u  # u_i
                u_R = torch.roll(u, shifts=-1, dims=-1)  # u_{i+1}
                return u_L, u_R
            else:
                u_L_face = torch.cat([u[..., :1], u], dim=-1)
                u_R_face = torch.cat([u, u[..., -1:]], dim=-1)
                return u_L_face, u_R_face

        # TVD slope limiter
        if boundary == BoundaryCondition.PERIODIC:
            u_pad = torch.cat([u[..., -1:], u, u[..., :1]], dim=-1)
        else:
            u_pad = torch.cat([u[..., :1], u, u[..., -1:]], dim=-1)

        # Neighbor differences
        delta_L = u_pad[..., 1:-1] - u_pad[..., :-2]    # u_i - u_{i-1}
        delta_R = u_pad[..., 2:] - u_pad[..., 1:-1]    # u_{i+1} - u_i

        eps = 1e-12
        r = torch.where(
            torch.abs(delta_R) > eps,
            delta_L / (delta_R + eps),
            torch.zeros_like(delta_L),
        )

        if self.config.limiter == SlopeLimiter.MINMOD:
            phi = torch.maximum(torch.zeros_like(r), torch.minimum(r, torch.ones_like(r)))
        elif self.config.limiter == SlopeLimiter.VAN_LEER:
            phi = (r + torch.abs(r)) / (1.0 + torch.abs(r) + eps)
        elif self.config.limiter == SlopeLimiter.SUPERBEE:
            t1 = torch.minimum(2.0 * r, torch.ones_like(r))
            t2 = torch.minimum(r, torch.full_like(r, 2.0))
            phi = torch.maximum(torch.zeros_like(r), torch.maximum(t1, t2))
        else:
            phi = torch.zeros_like(r)

        slopes = phi * delta_R  # Slope delta_i

        # Cell interface reconstructions:
        # At right boundary of cell i: u_{i+1/2}^- = u_i + 0.5 * slope_i
        # At left boundary of cell i: u_{i-1/2}^+ = u_i - 0.5 * slope_i
        u_face_minus = u + 0.5 * slopes
        u_face_plus = u - 0.5 * slopes

        if boundary == BoundaryCondition.PERIODIC:
            u_L = u_face_minus
            u_R = torch.roll(u_face_plus, shifts=-1, dims=-1)
            return u_L, u_R
        else:
            u_L = torch.cat([u_face_plus[..., :1], u_face_minus], dim=-1)
            u_R = torch.cat([u_face_plus, u_face_minus[..., -1:]], dim=-1)
            return u_L, u_R

    def compute_advective_flux(
        self,
        u_L: torch.Tensor,
        u_R: torch.Tensor,
    ) -> torch.Tensor:
        """Compute numerical advective flux across interfaces."""
        flux_fn = lambda v: 0.5 * (v**2)

        if self.config.flux_scheme == FluxScheme.RUSANOV:
            f_L = flux_fn(u_L)
            f_R = flux_fn(u_R)
            c_max = torch.maximum(torch.abs(u_L), torch.abs(u_R))
            return 0.5 * (f_L + f_R) - 0.5 * c_max * (u_R - u_L)

        elif self.config.flux_scheme == FluxScheme.GODUNOV:
            # Exact Godunov flux for Burgers f(u) = u^2 / 2
            f_rarefaction = torch.where(
                u_L >= 0.0,
                flux_fn(u_L),
                torch.where(u_R <= 0.0, flux_fn(u_R), torch.zeros_like(u_L)),
            )
            shock_speed = 0.5 * (u_L + u_R)
            f_shock = torch.where(shock_speed >= 0.0, flux_fn(u_L), flux_fn(u_R))
            return torch.where(u_L <= u_R, f_rarefaction, f_shock)

        elif self.config.flux_scheme == FluxScheme.ROE:
            f_L = flux_fn(u_L)
            f_R = flux_fn(u_R)
            a_roe = 0.5 * (u_L + u_R)
            delta_sonic = 0.1 * torch.maximum(torch.abs(u_L), torch.abs(u_R))
            abs_a = torch.where(
                torch.abs(a_roe) >= delta_sonic,
                torch.abs(a_roe),
                0.5 * ((a_roe**2) / (delta_sonic + 1e-12) + delta_sonic),
            )
            return 0.5 * (f_L + f_R) - 0.5 * abs_a * (u_R - u_L)

        else:
            raise ValueError(f"Unknown flux scheme: {self.config.flux_scheme}")

    def compute_viscous_flux(
        self,
        u: torch.Tensor,
        dx: float,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> torch.Tensor:
        """Compute viscous interface diffusion flux: D_{i+1/2} = nu * (u_{i+1} - u_i) / dx."""
        if self.config.viscosity <= 0.0:
            return torch.zeros_like(u)

        if boundary == BoundaryCondition.PERIODIC:
            diff = torch.roll(u, shifts=-1, dims=-1) - u
            return self.config.viscosity * diff / dx
        else:
            diff_interior = (u[..., 1:] - u[..., :-1]) / dx
            d_left = diff_interior[..., :1]
            d_right = diff_interior[..., -1:]
            return self.config.viscosity * torch.cat([d_left, diff_interior, d_right], dim=-1)

    def compute_rhs(
        self,
        u: torch.Tensor,
        dx: float,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RHS: - d(F_total) / dx, returning (du_dt, total_face_fluxes)."""
        u_L, u_R = self.reconstruct_interface_states(u, boundary=boundary)
        f_adv = self.compute_advective_flux(u_L, u_R)
        f_visc = self.compute_viscous_flux(u, dx, boundary=boundary)
        f_total = f_adv - f_visc

        if boundary == BoundaryCondition.PERIODIC:
            # For cell i: du/dt = -(F_{i+1/2} - F_{i-1/2}) / dx
            # f_total is at i+1/2, so F_{i-1/2} is roll(f_total, 1)
            du_dt = -(f_total - torch.roll(f_total, shifts=1, dims=-1)) / dx
        else:
            du_dt = -(f_total[..., 1:] - f_total[..., :-1]) / dx
        return du_dt, f_total

    def step(
        self,
        u: torch.Tensor,
        dx: float,
        dt: float,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> torch.Tensor:
        """Advance state by one time step dt."""
        is_stable, max_dt = self.check_cfl_stability(u, dx, dt)
        if not is_stable:
            raise ValueError(
                f"Requested time step {dt:.6e} violates CFL stability limit {max_dt:.6e} (cfl={self.config.cfl})"
            )

        if self.config.time_integrator == "euler":
            du_dt, _ = self.compute_rhs(u, dx, boundary=boundary)
            return u + dt * du_dt
        elif self.config.time_integrator == "ssp_rk2":
            # 2-stage Strong Stability Preserving Runge-Kutta
            du1, _ = self.compute_rhs(u, dx, boundary=boundary)
            u1 = u + dt * du1
            du2, _ = self.compute_rhs(u1, dx, boundary=boundary)
            return 0.5 * u + 0.5 * (u1 + dt * du2)
        else:
            raise ValueError(f"Unsupported time integrator: {self.config.time_integrator}")

    def solve_trajectory(
        self,
        u_initial: torch.Tensor,
        grid: GridContract,
        seed: int = 42,
    ) -> TrajectoryBatch:
        """Solve and record complete discrete trajectory."""
        dx = grid.cell_measures[0]
        dt = grid.time_step
        total_steps = grid.total_steps
        boundary = grid.boundary_conditions[0]

        trajectory = [u_initial.clone()]
        curr_u = u_initial.clone()

        for _ in range(total_steps):
            curr_u = self.step(curr_u, dx=dx, dt=dt, boundary=boundary)
            trajectory.append(curr_u.clone())

        if u_initial.ndim == 1:
            val_tensor = torch.stack(trajectory, dim=0).unsqueeze(0)  # [1, T+1, nx]
        else:
            val_tensor = torch.stack(trajectory, dim=1)  # [B, T+1, nx]
        times = torch.linspace(0.0, grid.time_horizon, total_steps + 1, dtype=torch.float32)
        B = u_initial.shape[0] if u_initial.ndim > 1 else 1

        provenance = [
            TrajectoryProvenance(
                trajectory_id=f"burgers_{seed}_{i}",
                family=PhysicalFamily.BURGERS,
                generator_name="Burgers1DSolver",
                generator_version="1.0.0",
                solver_accuracy_order=2 if self.config.limiter != SlopeLimiter.NONE else 1,
                seed=seed + i,
                parameters={"viscosity": self.config.viscosity, "cfl": self.config.cfl},
                creation_timestamp="2026-10-09T00:00:00Z",
                float_precision="float64" if u_initial.dtype == torch.float64 else "float32",
            )
            for i in range(B)
        ]

        return TrajectoryBatch(
            family=PhysicalFamily.BURGERS,
            grid=grid,
            field_names=("u",),
            values=val_tensor.float(),
            times=times,
            provenance=provenance,
        )
