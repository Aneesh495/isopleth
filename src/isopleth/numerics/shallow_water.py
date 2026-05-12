"""Original 2D finite-volume shallow-water reference solver with positivity and CFL enforcement.

Solves the Saint-Venant hyperbolic system:
    dh/dt + d(hu)/dx + d(hv)/dy = 0
    d(hu)/dt + d(hu^2 + 0.5*g*h^2)/dx + d(huv)/dy = -g*h*db/dx - S_f_x
    d(hv)/dt + d(huv)/dx + d(hv^2 + 0.5*g*h^2)/dy = -g*h*db/dy - S_f_y
using Kurganov-Tadmor and Rusanov numerical fluxes, hydrostatic reconstruction (lake-at-rest),
conservative depth limiting, and explicit CFL stability verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
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
from isopleth.numerics.limiters import ConservativeDepthLimiter, LimiterDiagnostics

class ShallowWaterFluxScheme(str, Enum):
    RUSANOV = "rusanov"
    KURGANOV_TADMOR = "kurganov_tadmor"

@dataclass
class ShallowWaterConfig:
    gravity: float = 9.81
    cfl: float = 0.40
    flux_scheme: ShallowWaterFluxScheme = ShallowWaterFluxScheme.KURGANOV_TADMOR
    h_min: float = 1e-6
    time_integrator: str = "ssp_rk2"  # 'ssp_rk2' or 'euler'
    use_conservative_limiter: bool = True
    friction_coeff: float = 0.0

class ShallowWater2DSolver:
    """Production 2D finite-volume solver for shallow-water dynamics."""

    def __init__(self, config: Optional[ShallowWaterConfig] = None) -> None:
        self.config = config or ShallowWaterConfig()
        self.limiter = ConservativeDepthLimiter(h_min=self.config.h_min)

    def check_cfl_stability(
        self,
        h: torch.Tensor,
        hu: torch.Tensor,
        hv: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
    ) -> Tuple[bool, float]:
        """Compute maximum allowable time step based on 2D gravity wave celerity and flow velocity."""
        h_safe = torch.clamp(h, min=self.config.h_min)
        u_vel = torch.abs(hu / h_safe)
        v_vel = torch.abs(hv / h_safe)
        c_wave = torch.sqrt(self.config.gravity * h_safe)

        max_speed_x = float(torch.max(u_vel + c_wave).item())
        max_speed_y = float(torch.max(v_vel + c_wave).item())

        dt_x = self.config.cfl * dx / max(max_speed_x, 1e-12)
        dt_y = self.config.cfl * dy / max(max_speed_y, 1e-12)

        max_dt = min(dt_x, dt_y)
        is_stable = dt <= (max_dt * 1.0001)
        return is_stable, max_dt

    def compute_flux_x(
        self,
        h_L: torch.Tensor,
        hu_L: torch.Tensor,
        hv_L: torch.Tensor,
        h_R: torch.Tensor,
        hu_R: torch.Tensor,
        hv_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute x-interface numerical flux: F_{i+1/2, j}."""
        g = self.config.gravity
        eps = self.config.h_min

        # Left physical fluxes
        hL_safe = torch.clamp(h_L, min=eps)
        uL = hu_L / hL_safe
        vL = hv_L / hL_safe
        f1_L = hu_L
        f2_L = hu_L * uL + 0.5 * g * (h_L**2)
        f3_L = hu_L * vL
        cL = torch.sqrt(g * hL_safe)

        # Right physical fluxes
        hR_safe = torch.clamp(h_R, min=eps)
        uR = hu_R / hR_safe
        vR = hv_R / hR_safe
        f1_R = hu_R
        f2_R = hu_R * uR + 0.5 * g * (h_R**2)
        f3_R = hu_R * vR
        cR = torch.sqrt(g * hR_safe)

        if self.config.flux_scheme == ShallowWaterFluxScheme.KURGANOV_TADMOR:
            # Wave propagation speeds
            a_plus = torch.maximum(
                torch.maximum(uL + cL, uR + cR),
                torch.zeros_like(uL),
            )
            a_minus = torch.maximum(
                torch.maximum(-(uL - cL), -(uR - cR)),
                torch.zeros_like(uL),
            )
            denom = torch.clamp(a_plus + a_minus, min=1e-12)

            f1 = (a_plus * f1_L + a_minus * f1_R - a_plus * a_minus * (h_R - h_L)) / denom
            f2 = (a_plus * f2_L + a_minus * f2_R - a_plus * a_minus * (hu_R - hu_L)) / denom
            f3 = (a_plus * f3_L + a_minus * f3_R - a_plus * a_minus * (hv_R - hv_L)) / denom
            return f1, f2, f3

        else:  # Rusanov flux
            c_max = torch.maximum(torch.abs(uL) + cL, torch.abs(uR) + cR)
            f1 = 0.5 * (f1_L + f1_R) - 0.5 * c_max * (h_R - h_L)
            f2 = 0.5 * (f2_L + f2_R) - 0.5 * c_max * (hu_R - hu_L)
            f3 = 0.5 * (f3_L + f3_R) - 0.5 * c_max * (hv_R - hv_L)
            return f1, f2, f3

    def compute_flux_y(
        self,
        h_L: torch.Tensor,
        hu_L: torch.Tensor,
        hv_L: torch.Tensor,
        h_R: torch.Tensor,
        hu_R: torch.Tensor,
        hv_R: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute y-interface numerical flux: G_{i, j+1/2}."""
        g = self.config.gravity
        eps = self.config.h_min

        hL_safe = torch.clamp(h_L, min=eps)
        uL = hu_L / hL_safe
        vL = hv_L / hL_safe
        g1_L = hv_L
        g2_L = hv_L * uL
        g3_L = hv_L * vL + 0.5 * g * (h_L**2)
        cL = torch.sqrt(g * hL_safe)

        hR_safe = torch.clamp(h_R, min=eps)
        uR = hu_R / hR_safe
        vR = hv_R / hR_safe
        g1_R = hv_R
        g2_R = hv_R * uR
        g3_R = hv_R * vR + 0.5 * g * (h_R**2)
        cR = torch.sqrt(g * hR_safe)

        if self.config.flux_scheme == ShallowWaterFluxScheme.KURGANOV_TADMOR:
            a_plus = torch.maximum(
                torch.maximum(vL + cL, vR + cR),
                torch.zeros_like(vL),
            )
            a_minus = torch.maximum(
                torch.maximum(-(vL - cL), -(vR - cR)),
                torch.zeros_like(vL),
            )
            denom = torch.clamp(a_plus + a_minus, min=1e-12)

            g1 = (a_plus * g1_L + a_minus * g1_R - a_plus * a_minus * (h_R - h_L)) / denom
            g2 = (a_plus * g2_L + a_minus * g2_R - a_plus * a_minus * (hu_R - hu_L)) / denom
            g3 = (a_plus * g3_L + a_minus * g3_R - a_plus * a_minus * (hv_R - hv_L)) / denom
            return g1, g2, g3

        else:
            c_max = torch.maximum(torch.abs(vL) + cL, torch.abs(vR) + cR)
            g1 = 0.5 * (g1_L + g1_R) - 0.5 * c_max * (h_R - h_L)
            g2 = 0.5 * (g2_L + g2_R) - 0.5 * c_max * (hu_R - hu_L)
            g3 = 0.5 * (g3_L + g3_R) - 0.5 * c_max * (hv_R - hv_L)
            return g1, g2, g3

    def compute_rhs(
        self,
        h: torch.Tensor,
        hu: torch.Tensor,
        hv: torch.Tensor,
        dx: float,
        dy: float,
        bathymetry: Optional[torch.Tensor] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RHS divergence: -(dF/dx + dG/dy) + Sources."""
        ny, nx = h.shape[-2], h.shape[-1]

        if boundary == BoundaryCondition.PERIODIC:
            if bathymetry is not None:
                eta = h + bathymetry
                # Hydrostatic reconstruction along x
                b_L = bathymetry
                b_R = torch.roll(bathymetry, shifts=-1, dims=-1)
                b_face_x = torch.maximum(b_L, b_R)
                h_L = torch.clamp(eta - b_face_x, min=0.0)
                h_R = torch.clamp(torch.roll(eta, shifts=-1, dims=-1) - b_face_x, min=0.0)
                hu_L = hu
                hu_R = torch.roll(hu, shifts=-1, dims=-1)
                hv_L = hv
                hv_R = torch.roll(hv, shifts=-1, dims=-1)

                fx1, fx2, fx3 = self.compute_flux_x(h_L, hu_L, hv_L, h_R, hu_R, hv_R)
                dfx1_dx = (fx1 - torch.roll(fx1, shifts=1, dims=-1)) / dx
                dfx2_dx = (fx2 - torch.roll(fx2, shifts=1, dims=-1)) / dx
                dfx3_dx = (fx3 - torch.roll(fx3, shifts=1, dims=-1)) / dx

                # Well-balanced bathymetry source along x
                h_R_prev_x = torch.roll(h_R, shifts=1, dims=-1)
                topo_source_x = 0.5 * self.config.gravity * ((h_L**2) - (h_R_prev_x**2)) / dx

                # Hydrostatic reconstruction along y
                b_Ly = bathymetry
                b_Ry = torch.roll(bathymetry, shifts=-1, dims=-2)
                b_face_y = torch.maximum(b_Ly, b_Ry)
                h_Ly = torch.clamp(eta - b_face_y, min=0.0)
                h_Ry = torch.clamp(torch.roll(eta, shifts=-1, dims=-2) - b_face_y, min=0.0)
                hu_Ly = hu
                hu_Ry = torch.roll(hu, shifts=-1, dims=-2)
                hv_Ly = hv
                hv_Ry = torch.roll(hv, shifts=-1, dims=-2)

                gy1, gy2, gy3 = self.compute_flux_y(h_Ly, hu_Ly, hv_Ly, h_Ry, hu_Ry, hv_Ry)
                dgy1_dy = (gy1 - torch.roll(gy1, shifts=1, dims=-2)) / dy
                dgy2_dy = (gy2 - torch.roll(gy2, shifts=1, dims=-2)) / dy
                dgy3_dy = (gy3 - torch.roll(gy3, shifts=1, dims=-2)) / dy

                h_Ry_prev_y = torch.roll(h_Ry, shifts=1, dims=-2)
                topo_source_y = 0.5 * self.config.gravity * ((h_Ly**2) - (h_Ry_prev_y**2)) / dy

                dh_dt = -(dfx1_dx + dgy1_dy)
                dhu_dt = -(dfx2_dx + dgy2_dy) + topo_source_x
                dhv_dt = -(dfx3_dx + dgy3_dy) + topo_source_y

            else:
                h_L = h
                hu_L = hu
                hv_L = hv
                h_R = torch.roll(h, shifts=-1, dims=-1)
                hu_R = torch.roll(hu, shifts=-1, dims=-1)
                hv_R = torch.roll(hv, shifts=-1, dims=-1)

                fx1, fx2, fx3 = self.compute_flux_x(h_L, hu_L, hv_L, h_R, hu_R, hv_R)
                dfx1_dx = (fx1 - torch.roll(fx1, shifts=1, dims=-1)) / dx
                dfx2_dx = (fx2 - torch.roll(fx2, shifts=1, dims=-1)) / dx
                dfx3_dx = (fx3 - torch.roll(fx3, shifts=1, dims=-1)) / dx

                h_Ly = h
                hu_Ly = hu
                hv_Ly = hv
                h_Ry = torch.roll(h, shifts=-1, dims=-2)
                hu_Ry = torch.roll(hu, shifts=-1, dims=-2)
                hv_Ry = torch.roll(hv, shifts=-1, dims=-2)

                gy1, gy2, gy3 = self.compute_flux_y(h_Ly, hu_Ly, hv_Ly, h_Ry, hu_Ry, hv_Ry)
                dgy1_dy = (gy1 - torch.roll(gy1, shifts=1, dims=-2)) / dy
                dgy2_dy = (gy2 - torch.roll(gy2, shifts=1, dims=-2)) / dy
                dgy3_dy = (gy3 - torch.roll(gy3, shifts=1, dims=-2)) / dy

                dh_dt = -(dfx1_dx + dgy1_dy)
                dhu_dt = -(dfx2_dx + dgy2_dy)
                dhv_dt = -(dfx3_dx + dgy3_dy)

        else:
            # Closed wall boundaries with reflective conditions: hu = 0 on x-walls, hv = 0 on y-walls
            h_pad_x = torch.cat([h[..., :, :1], h, h[..., :, -1:]], dim=-1)
            hu_pad_x = torch.cat([torch.zeros_like(hu[..., :, :1]), hu, torch.zeros_like(hu[..., :, -1:])], dim=-1)
            hv_pad_x = torch.cat([hv[..., :, :1], hv, hv[..., :, -1:]], dim=-1)

            h_L = h_pad_x[..., :, :-1]
            hu_L = hu_pad_x[..., :, :-1]
            hv_L = hv_pad_x[..., :, :-1]
            h_R = h_pad_x[..., :, 1:]
            hu_R = hu_pad_x[..., :, 1:]
            hv_R = hv_pad_x[..., :, 1:]

            fx1, fx2, fx3 = self.compute_flux_x(h_L, hu_L, hv_L, h_R, hu_R, hv_R)
            # Enforce zero mass flux through solid wall interfaces
            fx1[..., :, 0] = 0.0
            fx1[..., :, -1] = 0.0
            dfx1_dx = (fx1[..., :, 1:] - fx1[..., :, :-1]) / dx
            dfx2_dx = (fx2[..., :, 1:] - fx2[..., :, :-1]) / dx
            dfx3_dx = (fx3[..., :, 1:] - fx3[..., :, :-1]) / dx

            h_pad_y = torch.cat([h[..., :1, :], h, h[..., -1:, :]], dim=-2)
            hu_pad_y = torch.cat([hu[..., :1, :], hu, hu[..., -1:, :]], dim=-2)
            hv_pad_y = torch.cat([torch.zeros_like(hv[..., :1, :]), hv, torch.zeros_like(hv[..., -1:, :])], dim=-2)

            h_Ly = h_pad_y[..., :-1, :]
            hu_Ly = hu_pad_y[..., :-1, :]
            hv_Ly = hv_pad_y[..., :-1, :]
            h_Ry = h_pad_y[..., 1:, :]
            hu_Ry = hu_pad_y[..., 1:, :]
            hv_Ry = hv_pad_y[..., 1:, :]

            gy1, gy2, gy3 = self.compute_flux_y(h_Ly, hu_Ly, hv_Ly, h_Ry, hu_Ry, hv_Ry)
            gy1[..., 0, :] = 0.0
            gy1[..., -1, :] = 0.0
            dgy1_dy = (gy1[..., 1:, :] - gy1[..., :-1, :]) / dy
            dgy2_dy = (gy2[..., 1:, :] - gy2[..., :-1, :]) / dy
            dgy3_dy = (gy3[..., 1:, :] - gy3[..., :-1, :]) / dy

            dh_dt = -(dfx1_dx + dgy1_dy)
            dhu_dt = -(dfx2_dx + dgy2_dy)
            dhv_dt = -(dfx3_dx + dgy3_dy)

            if bathymetry is not None:
                db_dx = torch.gradient(bathymetry, spacing=dx, dim=-1)[0]
                db_dy = torch.gradient(bathymetry, spacing=dy, dim=-2)[0]
                dhu_dt = dhu_dt - self.config.gravity * h * db_dx
                dhv_dt = dhv_dt - self.config.gravity * h * db_dy

        # Bottom friction
        if self.config.friction_coeff > 0.0:
            h_safe = torch.clamp(h, min=self.config.h_min)
            speed = torch.sqrt((hu / h_safe)**2 + (hv / h_safe)**2 + 1e-12)
            f_drag = self.config.friction_coeff * speed
            dhu_dt = dhu_dt - f_drag * hu
            dhv_dt = dhv_dt - f_drag * hv

        return dh_dt, dhu_dt, dhv_dt

    def step(
        self,
        h: torch.Tensor,
        hu: torch.Tensor,
        hv: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
        bathymetry: Optional[torch.Tensor] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LimiterDiagnostics]:
        """Advance shallow-water state by time step dt."""
        is_stable, max_dt = self.check_cfl_stability(h, hu, hv, dx, dy, dt)
        if not is_stable:
            raise ValueError(
                f"Requested time step {dt:.6e} violates CFL stability limit {max_dt:.6e} (cfl={self.config.cfl})"
            )

        if self.config.time_integrator == "euler":
            dh_dt, dhu_dt, dhv_dt = self.compute_rhs(h, hu, hv, dx, dy, bathymetry, boundary)
            h_next = h + dt * dh_dt
            hu_next = hu + dt * dhu_dt
            hv_next = hv + dt * dhv_dt
        elif self.config.time_integrator == "ssp_rk2":
            dh1, dhu1, dhv1 = self.compute_rhs(h, hu, hv, dx, dy, bathymetry, boundary)
            h1 = h + dt * dh1
            hu1 = hu + dt * dhu1
            hv1 = hv + dt * dhv1

            dh2, dhu2, dhv2 = self.compute_rhs(h1, hu1, hv1, dx, dy, bathymetry, boundary)
            h_next = 0.5 * h + 0.5 * (h1 + dt * dh2)
            hu_next = 0.5 * hu + 0.5 * (hu1 + dt * dhu2)
            hv_next = 0.5 * hv + 0.5 * (hv1 + dt * dhv2)
        else:
            raise ValueError(f"Unknown time integrator: {self.config.time_integrator}")

        # Conservative depth limiting to enforce h >= h_min
        if self.config.use_conservative_limiter:
            h_lim, hu_lim, hv_lim, diag = self.limiter(h_next, hu_next, hv_next)
            return h_lim, hu_lim, hv_lim, diag
        else:
            diag = LimiterDiagnostics(0, 0.0, 0.0, True, 0.0)
            return h_next, hu_next, hv_next, diag

    def solve_trajectory(
        self,
        h_init: torch.Tensor,
        hu_init: torch.Tensor,
        hv_init: torch.Tensor,
        grid: GridContract,
        seed: int = 42,
    ) -> TrajectoryBatch:
        """Integrate 2D shallow-water trajectory over declared horizon."""
        dx, dy = grid.cell_measures[0], grid.cell_measures[1]
        dt = grid.time_step
        total_steps = grid.total_steps
        boundary = grid.boundary_conditions[0]

        h_hist = [h_init.clone()]
        hu_hist = [hu_init.clone()]
        hv_hist = [hv_init.clone()]

        curr_h = h_init.clone()
        curr_hu = hu_init.clone()
        curr_hv = hv_init.clone()

        for _ in range(total_steps):
            curr_h, curr_hu, curr_hv, _ = self.step(
                curr_h, curr_hu, curr_hv, dx=dx, dy=dy, dt=dt, boundary=boundary
            )
            h_hist.append(curr_h.clone())
            hu_hist.append(curr_hu.clone())
            hv_hist.append(curr_hv.clone())

        # Stack into [B, T+1, ny, nx, 3]
        h_t = torch.stack(h_hist, dim=1)
        hu_t = torch.stack(hu_hist, dim=1)
        hv_t = torch.stack(hv_hist, dim=1)

        val_tensor = torch.stack([h_t, hu_t, hv_t], dim=-1)  # [B, T+1, ny, nx, 3]
        times = torch.linspace(0.0, grid.time_horizon, total_steps + 1, dtype=torch.float32)

        B = h_init.shape[0] if h_init.ndim > 2 else 1
        if h_init.ndim == 2:
            val_tensor = val_tensor.unsqueeze(0)

        provenance = [
            TrajectoryProvenance(
                trajectory_id=f"sw2d_{seed}_{i}",
                family=PhysicalFamily.SHALLOW_WATER,
                generator_name="ShallowWater2DSolver",
                generator_version="1.0.0",
                solver_accuracy_order=2,
                seed=seed + i,
                parameters={"gravity": self.config.gravity, "cfl": self.config.cfl},
                creation_timestamp="2026-10-09T00:00:00Z",
                float_precision="float64" if h_init.dtype == torch.float64 else "float32",
            )
            for i in range(B)
        ]

        return TrajectoryBatch(
            family=PhysicalFamily.SHALLOW_WATER,
            grid=grid,
            field_names=("h", "hu", "hv"),
            values=val_tensor.float(),
            times=times,
            provenance=provenance,
        )
