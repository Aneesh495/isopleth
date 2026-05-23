"""Cross-resolution evaluation using continuous coordinates and independent numerical references (I17).

Evaluates learned neural operators on held-out finer discretizations (e.g. 64 -> 128 cells)
without image interpolation, comparing against independently integrated fine numerical solutions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from isopleth.data.contracts import BoundaryCondition, GridContract, PhysicalFamily
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig
from isopleth.numerics.shallow_water import ShallowWater2DSolver, ShallowWaterConfig
from isopleth.rollout.runner import AutoregressiveRolloutRunner, RolloutResult

@dataclass
class CrossResolutionReport:
    """Outcome report for zero-shot cross-resolution generalization."""
    family: str
    coarse_resolution: Tuple[int, ...]
    fine_resolution: Tuple[int, ...]
    relative_l2_error: float
    linf_error: float
    coarse_mass_error: float
    fine_mass_error: float
    fine_steps_completed: int
    is_stable: bool

class CrossResolutionEvaluator:
    """Evaluates neural operators on refined spatial grids against independent numerical references."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def evaluate_burgers_transfer(
        self,
        model: nn.Module,
        continuous_ic_fn: Callable[[torch.Tensor], torch.Tensor],
        coarse_nx: int = 64,
        fine_nx: int = 128,
        viscosity: float = 0.01,
        steps: int = 20,
        dt: float = 0.005,
    ) -> CrossResolutionReport:
        """Evaluate 1D Burgers operator transfer from 64 to 128 grid."""
        runner = AutoregressiveRolloutRunner(model=model, device=self.device)

        dx_fine = 1.0 / fine_nx
        xf = torch.linspace(0.5 * dx_fine, 1.0 - 0.5 * dx_fine, fine_nx, dtype=torch.float32)
        u_init_fine = continuous_ic_fn(xf)

        # 1. Learned operator rollout directly on fine 128 grid with physical dx_fine
        rollout = runner.rollout_1d(
            x_init=u_init_fine,
            dx=dx_fine,
            dt=dt,
            steps=steps,
            boundary=BoundaryCondition.PERIODIC,
        )

        # 2. Independent fine numerical reference solver on 128 grid
        ref_solver = Burgers1DSolver(BurgersSolverConfig(viscosity=viscosity, cfl=0.4))
        u_ref = u_init_fine.clone()
        for _ in range(steps):
            u_ref = ref_solver.step(u_ref, dx=dx_fine, dt=dt, boundary=BoundaryCondition.PERIODIC)

        u_pred_final = rollout.trajectory[-1]
        err = torch.abs(u_pred_final - u_ref)
        rel_l2 = float((torch.norm(u_pred_final - u_ref) / (torch.norm(u_ref) + 1e-12)).item())
        linf = float(torch.max(err).item())

        return CrossResolutionReport(
            family="burgers_1d",
            coarse_resolution=(coarse_nx,),
            fine_resolution=(fine_nx,),
            relative_l2_error=rel_l2,
            linf_error=linf,
            coarse_mass_error=0.0,
            fine_mass_error=rollout.max_mass_error,
            fine_steps_completed=rollout.completed_steps,
            is_stable=rollout.is_stable,
        )

    def evaluate_shallow_water_transfer(
        self,
        model: nn.Module,
        continuous_ic_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        coarse_res: Tuple[int, int] = (32, 32),
        fine_res: Tuple[int, int] = (64, 64),
        steps: int = 15,
        dt: float = 0.002,
    ) -> CrossResolutionReport:
        """Evaluate 2D Shallow Water operator transfer to finer grid."""
        runner = AutoregressiveRolloutRunner(model=model, device=self.device)

        ny_f, nx_f = fine_res
        dx_f, dy_f = 1.0 / nx_f, 1.0 / ny_f
        xc = torch.linspace(0.5 * dx_f, 1.0 - 0.5 * dx_f, nx_f, dtype=torch.float32)
        yc = torch.linspace(0.5 * dy_f, 1.0 - 0.5 * dy_f, ny_f, dtype=torch.float32)
        Y, X = torch.meshgrid(yc, xc, indexing="ij")

        h_init, hu_init, hv_init = continuous_ic_fn(X, Y)
        state_init_fine = torch.stack([h_init, hu_init, hv_init], dim=0)

        # 1. Learned rollout on fine grid
        rollout = runner.rollout_2d(
            x_init=state_init_fine,
            dx=dx_f,
            dy=dy_f,
            dt=dt,
            steps=steps,
            boundary=BoundaryCondition.PERIODIC,
        )

        # 2. Independent fine reference solver
        solver = ShallowWater2DSolver(ShallowWaterConfig(gravity=9.81, cfl=0.35))
        h_ref = h_init.clone()
        hu_ref = hu_init.clone()
        hv_ref = hv_init.clone()
        for _ in range(steps):
            h_ref, hu_ref, hv_ref, _ = solver.step(h_ref, hu_ref, hv_ref, dx=dx_f, dy=dy_f, dt=dt)

        ref_final = torch.stack([h_ref, hu_ref, hv_ref], dim=0)
        pred_final = rollout.trajectory[0, -1]  # [C, ny, nx]

        err = torch.abs(pred_final - ref_final)
        rel_l2 = float((torch.norm(pred_final - ref_final) / (torch.norm(ref_final) + 1e-12)).item())
        linf = float(torch.max(err).item())

        return CrossResolutionReport(
            family="shallow_water_2d",
            coarse_resolution=coarse_res,
            fine_resolution=fine_res,
            relative_l2_error=rel_l2,
            linf_error=linf,
            coarse_mass_error=0.0,
            fine_mass_error=rollout.max_mass_error,
            fine_steps_completed=rollout.completed_steps,
            is_stable=rollout.is_stable,
        )


CrossResolutionTransferEngine = CrossResolutionEvaluator
