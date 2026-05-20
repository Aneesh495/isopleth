"""Autoregressive target-free rollout runner and stability safeguards (I16).

Executes multi-step autoregressive rollouts using only the allowed initial state
and model predictions thereafter. Integrates conservative depth limiters,
blowup detection, and step-by-step physical accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from isopleth.data.contracts import BoundaryCondition, GridContract, PhysicalFamily
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D, MultiscaleFaceFluxOperator2D
from isopleth.numerics.limiters import ConservativeDepthLimiter, LimiterDiagnostics
from isopleth.numerics.sources import BalanceAuditor, compute_spatial_integral

@dataclass
class RolloutResult:
    """Outcome of an autoregressive rollout trajectory."""
    trajectory: torch.Tensor  # [B, T+1, ...]
    times: torch.Tensor       # [T+1]
    completed_steps: int
    requested_steps: int
    is_stable: bool
    blowup_step: Optional[int]
    limiter_activations: int
    max_mass_error: float
    elapsed_seconds: float

class AutoregressiveRolloutRunner:
    """Executes target-free autoregressive rollouts across arbitrary horizons."""

    def __init__(
        self,
        model: nn.Module,
        depth_limiter: Optional[ConservativeDepthLimiter] = None,
        max_physical_bound: float = 50.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.depth_limiter = depth_limiter or ConservativeDepthLimiter()
        self.max_physical_bound = max_physical_bound
        self.device = torch.device(device)

    def rollout_1d(
        self,
        x_init: torch.Tensor,
        dx: float,
        dt: float,
        steps: int,
        cond: Optional[torch.Tensor] = None,
        source_fn: Optional[Any] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
    ) -> RolloutResult:
        """Execute 1D autoregressive rollout."""
        self.model.eval()
        t0 = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None

        is_flat = (x_init.ndim == 1)
        if is_flat:
            curr = x_init.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, nx]
        elif x_init.ndim == 2:
            curr = x_init.unsqueeze(1).to(self.device)  # [B, 1, nx]
        else:
            curr = x_init.to(self.device)

        B, C, nx = curr.shape
        history = [curr.clone()]
        curr_t = 0.0

        orig_integral = compute_spatial_integral(curr, (dx,))
        max_mass_err = 0.0
        blowup_step = None
        is_stable = True

        start_wall = torch.tensor(0.0)

        with torch.no_grad():
            for s in range(1, steps + 1):
                # Check bounds / non-finite values
                if not torch.all(torch.isfinite(curr)) or float(torch.max(torch.abs(curr)).item()) > self.max_physical_bound:
                    is_stable = False
                    blowup_step = s
                    break

                # Predict next step
                if isinstance(self.model, MultiscaleFaceFluxOperator1D):
                    s_term = source_fn(curr, curr_t) if source_fn else None
                    next_state, _ = self.model(curr, dx=dx, dt=dt, cond=cond, source_terms=s_term, boundary=boundary)
                else:
                    next_state = self.model(curr, cond=cond)

                # Conservation audit
                curr_integral = compute_spatial_integral(next_state, (dx,))
                mass_diff = float(torch.max(torch.abs(curr_integral - orig_integral)).item())
                max_mass_err = max(max_mass_err, mass_diff)

                history.append(next_state.clone())
                curr = next_state
                curr_t += dt

        # Stack trajectory: [B, completed_steps + 1, nx]
        traj = torch.stack(history, dim=1)  # [B, T+1, C, nx]
        if is_flat and C == 1:
            traj = traj.squeeze(2).squeeze(0)  # [T+1, nx]
        elif C == 1:
            traj = traj.squeeze(2)  # [B, T+1, nx]

        times = torch.linspace(0.0, len(history) * dt - dt, len(history), device=self.device)

        return RolloutResult(
            trajectory=traj,
            times=times,
            completed_steps=len(history) - 1,
            requested_steps=steps,
            is_stable=is_stable,
            blowup_step=blowup_step,
            limiter_activations=0,
            max_mass_error=max_mass_err,
            elapsed_seconds=0.0,
        )

    def rollout_2d(
        self,
        x_init: torch.Tensor,
        dx: float,
        dy: float,
        dt: float,
        steps: int,
        cond: Optional[torch.Tensor] = None,
        source_fn: Optional[Any] = None,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
        apply_depth_limiter: bool = True,
    ) -> RolloutResult:
        """Execute 2D autoregressive rollout."""
        self.model.eval()
        if x_init.ndim == 3:
            curr = x_init.unsqueeze(0).to(self.device)  # [1, C, ny, nx]
        else:
            curr = x_init.to(self.device)

        B, C, ny, nx = curr.shape
        history = [curr.clone()]
        curr_t = 0.0

        orig_integral = compute_spatial_integral(curr[:, 0], (dx, dy))
        max_mass_err = 0.0
        total_limiter_acts = 0
        blowup_step = None
        is_stable = True

        with torch.no_grad():
            for s in range(1, steps + 1):
                if not torch.all(torch.isfinite(curr)) or float(torch.max(torch.abs(curr)).item()) > self.max_physical_bound:
                    is_stable = False
                    blowup_step = s
                    break

                if isinstance(self.model, MultiscaleFaceFluxOperator2D):
                    s_term = source_fn(curr, curr_t) if source_fn else None
                    next_state, _ = self.model(curr, dx=dx, dy=dy, dt=dt, cond=cond, source_terms=s_term, boundary=boundary)
                else:
                    next_state = self.model(curr, cond=cond)

                # Depth limiting for shallow water depth channel 0
                if apply_depth_limiter and C >= 3:
                    h = next_state[:, 0]
                    hu = next_state[:, 1]
                    hv = next_state[:, 2]
                    h_lim, hu_lim, hv_lim, diag = self.depth_limiter(h, hu, hv)
                    total_limiter_acts += diag.activations_count
                    next_state = torch.stack([h_lim, hu_lim, hv_lim] + [next_state[:, c] for c in range(3, C)], dim=1)

                curr_integral = compute_spatial_integral(next_state[:, 0], (dx, dy))
                mass_diff = float(torch.max(torch.abs(curr_integral - orig_integral)).item())
                max_mass_err = max(max_mass_err, mass_diff)

                history.append(next_state.clone())
                curr = next_state
                curr_t += dt

        traj = torch.stack(history, dim=1)  # [B, T+1, C, ny, nx]
        times = torch.linspace(0.0, len(history) * dt - dt, len(history), device=self.device)

        return RolloutResult(
            trajectory=traj,
            times=times,
            completed_steps=len(history) - 1,
            requested_steps=steps,
            is_stable=is_stable,
            blowup_step=blowup_step,
            limiter_activations=total_limiter_acts,
            max_mass_error=max_mass_err,
            elapsed_seconds=0.0,
        )
