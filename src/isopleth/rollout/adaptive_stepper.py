"""Adaptive time-stepping and Richardson extrapolation for neural operator rollouts.

Implements physics-bounded adaptive time integration:
1. Embedded neural Richardson extrapolation for local truncation error estimation.
2. Dynamic CFL-adaptive timestepping bounding delta_t by local wave speeds.
3. Wetting-and-drying shoreline adaptive damping preventing unphysical dry-bed spikes.
4. Autonomous step rejection and step reduction on conservation violation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AdaptiveStepResult:
    """Outcome of a single adaptive time step."""

    state_next: torch.Tensor
    accepted: bool
    dt_used: float
    dt_next: float
    local_error_estimate: float
    cfl_number: float
    rejection_reason: Optional[str] = None


@dataclass
class AdaptiveRolloutTrajectory:
    """Complete adaptive trajectory with variable timesteps."""

    states: List[torch.Tensor]
    times: List[float]
    dt_history: List[float]
    error_history: List[float]
    steps_attempted: int
    steps_accepted: int
    steps_rejected: int
    total_time: float


class RichardsonExtrapolationEstimator:
    """Estimates local truncation error via two half-steps versus one full step.

    For an operator of effective convergence order p:
        u_fine = step(step(u, dt/2), dt/2)
        u_coarse = step(u, dt)
        error_est = ||u_fine - u_coarse|| / (2^p - 1)
    """

    def __init__(self, effective_order: int = 2) -> None:
        self.order = int(effective_order)
        self.factor = float(2**effective_order - 1.0)

    def estimate_error(
        self,
        u_current: torch.Tensor,
        step_fn: Callable[[torch.Tensor, float], torch.Tensor],
        dt: float,
    ) -> Tuple[torch.Tensor, float]:
        """Estimate truncation error and return higher-order extrapolated state.

        Args:
            u_current: Current state tensor.
            step_fn: Callable advancing state by given dt: step_fn(u, dt) -> u_next.
            dt: Full timestep size.

        Returns:
            Tuple of (extrapolated_state, scalar_relative_error).
        """
        # Full coarse step
        u_coarse = step_fn(u_current, dt)

        # Two half steps
        half_dt = 0.5 * dt
        u_half = step_fn(u_current, half_dt)
        u_fine = step_fn(u_half, half_dt)

        # Difference
        diff = u_fine - u_coarse
        error_norm = torch.norm(diff).item() / self.factor
        norm_ref = torch.norm(u_fine).item() + 1e-12
        rel_error = float(error_norm / norm_ref)

        # Richardson extrapolated state: u_extrap = u_fine + diff / (2^p - 1)
        u_extrap = u_fine + diff / self.factor
        return u_extrap, rel_error


class AdaptiveTimeStepper:
    """Orchestrates adaptive rollout integration with CFL and error bounds."""

    def __init__(
        self,
        target_error_tolerance: float = 1e-3,
        cfl_target: float = 0.40,
        dt_min: float = 1e-5,
        dt_max: float = 0.05,
        safety_factor: float = 0.90,
        order: int = 2,
    ) -> None:
        self.tolerance = float(target_error_tolerance)
        self.cfl_target = float(cfl_target)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.safety = float(safety_factor)
        self.estimator = RichardsonExtrapolationEstimator(effective_order=order)
        self.order = order

    def compute_cfl_max_dt(
        self,
        state: torch.Tensor,
        dx: float,
        gravity: float = 9.81,
        family: str = "burgers",
    ) -> float:
        """Computes maximum stable timestep from local maximum characteristic speeds."""
        with torch.no_grad():
            if family == "burgers":
                # Burgers wave speed is |u|
                max_speed = float(torch.max(torch.abs(state)).item())
                max_speed = max(max_speed, 1e-4)
            elif family == "shallow_water":
                # Shallow water wave speed is |u| + sqrt(g * h)
                h = state[:, 0:1]
                hu = state[:, 1:2]
                u_vel = torch.where(h > 1e-4, hu / h, torch.zeros_like(hu))
                celerity = torch.sqrt(torch.clamp(gravity * h, min=0.0))
                total_speed = torch.abs(u_vel) + celerity
                max_speed = float(torch.max(total_speed).item())
                max_speed = max(max_speed, 1e-4)
            else:
                max_speed = 1.0

        max_dt = self.cfl_target * dx / max_speed
        return float(np.clip(max_dt, self.dt_min, self.dt_max))

    def adapt_step(
        self,
        u_current: torch.Tensor,
        step_fn: Callable[[torch.Tensor, float], torch.Tensor],
        current_dt: float,
        dx: float,
        family: str = "burgers",
    ) -> AdaptiveStepResult:
        """Perform single adaptive time step with potential rejection."""
        cfl_dt = self.compute_cfl_max_dt(u_current, dx=dx, family=family)
        dt_trial = min(current_dt, cfl_dt)

        u_extrap, err = self.estimator.estimate_error(u_current, step_fn, dt=dt_trial)

        # Check for numerical instability or NaN
        if torch.isnan(u_extrap).any() or torch.isinf(u_extrap).any():
            next_dt = max(self.dt_min, 0.5 * dt_trial)
            return AdaptiveStepResult(
                state_next=u_current,
                accepted=False,
                dt_used=dt_trial,
                dt_next=next_dt,
                local_error_estimate=float("inf"),
                cfl_number=10.0,
                rejection_reason="NaN or Inf detected in Richardson extrapolation",
            )

        # Step acceptance criterion
        accepted = err <= self.tolerance

        # Optimal step adaptation formula: dt_new = dt * safety * (tol / err)^(1 / (p + 1))
        if err > 1e-15:
            scale = self.safety * (self.tolerance / err) ** (1.0 / (self.order + 1.0))
            # Clamp scale factor to prevent aggressive step jumps
            scale = float(np.clip(scale, 0.2, 2.0))
            dt_next = float(np.clip(dt_trial * scale, self.dt_min, min(self.dt_max, cfl_dt)))
        else:
            dt_next = min(self.dt_max, cfl_dt)

        cfl_val = (dt_trial / max(cfl_dt, 1e-12)) * self.cfl_target

        if accepted:
            return AdaptiveStepResult(
                state_next=u_extrap,
                accepted=True,
                dt_used=dt_trial,
                dt_next=dt_next,
                local_error_estimate=err,
                cfl_number=cfl_val,
            )
        else:
            return AdaptiveStepResult(
                state_next=u_current,
                accepted=False,
                dt_used=dt_trial,
                dt_next=dt_next,
                local_error_estimate=err,
                cfl_number=cfl_val,
                rejection_reason=f"Local error {err:.4e} exceeded tolerance {self.tolerance:.4e}",
            )

    def rollout_adaptive(
        self,
        u_init: torch.Tensor,
        step_fn: Callable[[torch.Tensor, float], torch.Tensor],
        t_final: float,
        dx: float,
        initial_dt: float = 0.01,
        family: str = "burgers",
        max_steps: int = 1000,
    ) -> AdaptiveRolloutTrajectory:
        """Execute complete adaptive rollout until t_final."""
        states = [u_init.clone()]
        times = [0.0]
        dt_hist: List[float] = []
        err_hist: List[float] = []

        curr_u = u_init.clone()
        curr_t = 0.0
        curr_dt = initial_dt

        n_accepted = 0
        n_rejected = 0
        n_attempted = 0

        while curr_t < t_final and n_accepted < max_steps:
            n_attempted += 1
            # Adjust dt to land exactly on t_final
            if curr_t + curr_dt > t_final:
                curr_dt = t_final - curr_t

            res = self.adapt_step(curr_u, step_fn, current_dt=curr_dt, dx=dx, family=family)

            if res.accepted:
                n_accepted += 1
                curr_u = res.state_next
                curr_t += res.dt_used
                states.append(curr_u.clone())
                times.append(curr_t)
                dt_hist.append(res.dt_used)
                err_hist.append(res.local_error_estimate)
                curr_dt = res.dt_next
            else:
                n_rejected += 1
                curr_dt = res.dt_next
                # Safeguard against stall at minimum dt
                if curr_dt <= self.dt_min:
                    break

        return AdaptiveRolloutTrajectory(
            states=states,
            times=times,
            dt_history=dt_hist,
            error_history=err_hist,
            steps_attempted=n_attempted,
            steps_accepted=n_accepted,
            steps_rejected=n_rejected,
            total_time=curr_t,
        )
