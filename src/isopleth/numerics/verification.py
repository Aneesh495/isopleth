"""Independently checked manufactured solutions, grid refinement, and convergence verification.

Provides Method of Manufactured Solutions (MMS) for 1D Burgers and 2D Shallow Water,
Experimental Order of Convergence (EOC) calculators, and float64 mass conservation audits
enforcing residual thresholds strictly below 1e-9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch

from isopleth.data.contracts import BoundaryCondition, GridContract
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig, FluxScheme, SlopeLimiter
from isopleth.numerics.shallow_water import ShallowWater2DSolver, ShallowWaterConfig
from isopleth.numerics.sources import BalanceAuditor, compute_spatial_integral

@dataclass
class ConvergenceEntry:
    resolution: int
    dx: float
    dt: float
    l1_error: float
    l2_error: float
    linf_error: float
    eoc: Optional[float] = None

@dataclass
class ConvergenceTable:
    scheme_name: str
    target_order: int
    entries: List[ConvergenceEntry]

    @property
    def final_eoc(self) -> float:
        if len(self.entries) < 2 or self.entries[-1].eoc is None:
            return 0.0
        return self.entries[-1].eoc

class BurgersMMS:
    """Method of Manufactured Solutions for 1D viscous Burgers equation.
    
    Exact solution: u(x, t) = sin(2*pi*(x - c*t))
    Analytical source:
        S(x, t) = -2*pi*c*cos(theta) + pi*sin(2*theta) + 4*pi^2*nu*sin(theta)
    where theta = 2*pi*(x - c*t).
    """

    def __init__(self, c_speed: float = 0.5, viscosity: float = 0.01) -> None:
        self.c = c_speed
        self.nu = viscosity

    def exact_solution(self, x: torch.Tensor, t: float) -> torch.Tensor:
        theta = 2.0 * math.pi * (x - self.c * t)
        return torch.sin(theta)

    def analytical_source(self, x: torch.Tensor, t: float) -> torch.Tensor:
        theta = 2.0 * math.pi * (x - self.c * t)
        term_time = -2.0 * math.pi * self.c * torch.cos(theta)
        term_adv = math.pi * torch.sin(2.0 * theta)
        term_diff = 4.0 * (math.pi**2) * self.nu * torch.sin(theta)
        return term_time + term_adv + term_diff

    def run_refinement_study(
        self,
        resolutions: Tuple[int, ...] = (32, 64, 128),
        t_final: float = 0.1,
    ) -> ConvergenceTable:
        """Execute grid refinement study and compute Experimental Order of Convergence."""
        entries = []
        prev_err = None
        prev_dx = None

        solver = Burgers1DSolver(
            BurgersSolverConfig(
                viscosity=self.nu,
                cfl=0.35,
                flux_scheme=FluxScheme.GODUNOV,
                limiter=SlopeLimiter.VAN_LEER,
                time_integrator="ssp_rk2",
            )
        )

        for nx in resolutions:
            dx = 1.0 / nx
            x = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
            u = self.exact_solution(x, t=0.0)

            # Choose stable dt proportional to dx
            dt = 0.2 * dx
            n_steps = int(math.ceil(t_final / dt))
            actual_dt = t_final / n_steps

            curr_t = 0.0
            for _ in range(n_steps):
                # RK2 step with explicit MMS source
                s1 = self.analytical_source(x, curr_t)
                du1, _ = solver.compute_rhs(u, dx=dx, boundary=BoundaryCondition.PERIODIC)
                u1 = u + actual_dt * (du1 + s1)

                s2 = self.analytical_source(x, curr_t + actual_dt)
                du2, _ = solver.compute_rhs(u1, dx=dx, boundary=BoundaryCondition.PERIODIC)
                u = 0.5 * u + 0.5 * (u1 + actual_dt * (du2 + s2))
                curr_t += actual_dt

            u_exact = self.exact_solution(x, t=t_final)
            err = torch.abs(u - u_exact)
            l1 = float(torch.mean(err).item())
            l2 = float(torch.sqrt(torch.mean(err**2)).item())
            linf = float(torch.max(err).item())

            eoc = None
            if prev_err is not None and prev_dx is not None:
                eoc = math.log(err.norm(p=2).item() / prev_err) / math.log(dx / prev_dx)

            entries.append(
                ConvergenceEntry(
                    resolution=nx,
                    dx=dx,
                    dt=actual_dt,
                    l1_error=l1,
                    l2_error=l2,
                    linf_error=linf,
                    eoc=eoc,
                )
            )
            prev_err = float(err.norm(p=2).item())
            prev_dx = dx

        return ConvergenceTable(
            scheme_name="Burgers1D_TVD_RK2",
            target_order=2,
            entries=entries,
        )

class ShallowWaterVerification:
    """Verification suite for 2D Shallow Water solver."""

    @staticmethod
    def verify_lake_at_rest(
        resolution: Tuple[int, int] = (32, 32),
        steps: int = 50,
        dt: float = 0.005,
    ) -> Tuple[bool, float, float]:
        """Test lake-at-rest well-balanced hydrostatic equilibrium over submerged seamount.
        
        Initial condition:
            b(x, y) = 0.25 * exp(-40 * ((x-0.5)^2 + (y-0.5)^2))
            eta = h + b = 1.0 (constant surface elevation)
            u = 0, v = 0
            
        Returns:
            (passed, max_elevation_residual, max_velocity_residual)
        """
        ny, nx = resolution
        dx = 1.0 / nx
        dy = 1.0 / ny
        xc = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
        yc = torch.linspace(0.5 * dy, 1.0 - 0.5 * dy, ny, dtype=torch.float64)
        Y, X = torch.meshgrid(yc, xc, indexing="ij")

        # Topography seamount
        b = 0.25 * torch.exp(-40.0 * ((X - 0.5)**2 + (Y - 0.5)**2))
        eta_target = 1.0
        h = eta_target - b  # h + b = 1.0 exactly
        hu = torch.zeros_like(h)
        hv = torch.zeros_like(h)

        solver = ShallowWater2DSolver(
            ShallowWaterConfig(
                gravity=9.81,
                cfl=0.35,
                use_conservative_limiter=False,
            )
        )

        curr_h = h.clone()
        curr_hu = hu.clone()
        curr_hv = hv.clone()

        for _ in range(steps):
            curr_h, curr_hu, curr_hv, _ = solver.step(
                curr_h, curr_hu, curr_hv,
                dx=dx, dy=dy, dt=dt,
                bathymetry=b,
                boundary=BoundaryCondition.PERIODIC,
            )

        eta_final = curr_h + b
        eta_res = float(torch.max(torch.abs(eta_final - eta_target)).item())
        vel_res = float(torch.max(torch.sqrt(curr_hu**2 + curr_hv**2)).item())

        passed = (eta_res < 1e-10) and (vel_res < 1e-10)
        return passed, eta_res, vel_res

    @staticmethod
    def verify_mass_conservation_float64(
        steps: int = 50,
        dt: float = 0.005,
    ) -> Tuple[bool, float]:
        """Verify float64 mass conservation strictly to tolerance 1e-9 on 2D periodic grid."""
        ny, nx = 32, 32
        dx = 1.0 / nx
        dy = 1.0 / ny
        xc = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
        yc = torch.linspace(0.5 * dy, 1.0 - 0.5 * dy, ny, dtype=torch.float64)
        Y, X = torch.meshgrid(yc, xc, indexing="ij")

        # Smooth perturbation
        h = 1.0 + 0.1 * torch.sin(2.0 * math.pi * X) * torch.cos(2.0 * math.pi * Y)
        hu = 0.05 * torch.sin(2.0 * math.pi * Y)
        hv = -0.05 * torch.cos(2.0 * math.pi * X)

        solver = ShallowWater2DSolver(
            ShallowWaterConfig(
                gravity=9.81,
                cfl=0.35,
                use_conservative_limiter=True,
            )
        )

        auditor = BalanceAuditor(cell_measures=(dx, dy), tolerance_float64=1e-9)
        h_init = h.clone()
        curr_h = h.clone()
        curr_hu = hu.clone()
        curr_hv = hv.clone()

        for _ in range(steps):
            curr_h, curr_hu, curr_hv, _ = solver.step(
                curr_h, curr_hu, curr_hv,
                dx=dx, dy=dy, dt=dt,
                boundary=BoundaryCondition.PERIODIC,
            )

        report = auditor.audit(h_init, curr_h)
        return report.is_balanced, report.relative_residual
