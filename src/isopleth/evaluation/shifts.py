"""Parameter and initial-condition shift evaluation suite (I18).

Evaluates generalization performance on physical parameter distributions and
spatial wave profiles held out from training distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from isopleth.data.contracts import BoundaryCondition
from isopleth.evaluation.metrics import EvaluationMetrics, TrajectoryEvaluator
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig
from isopleth.numerics.shallow_water import ShallowWater2DSolver, ShallowWaterConfig
from isopleth.rollout.runner import AutoregressiveRolloutRunner

@dataclass
class ShiftExperimentReport:
    experiment_type: str
    condition_tag: str
    in_distribution_error: float
    shifted_error: float
    degradation_ratio: float
    is_stable: bool
    mass_conservation_residual: float

class ShiftExperimentSuite:
    """Executes frozen parameter and initial-condition shift campaigns."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def evaluate_burgers_viscosity_shift(
        self,
        model: nn.Module,
        train_viscosity: float = 0.01,
        shifted_viscosity: float = 0.002,  # Substantially lower viscosity (sharper shocks)
        nx: int = 64,
        steps: int = 30,
        dt: float = 0.005,
    ) -> ShiftExperimentReport:
        """Evaluate 1D Burgers model on held-out low-viscosity regime."""
        runner = AutoregressiveRolloutRunner(model=model, device=self.device)
        dx = 1.0 / nx
        x = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        u_init = torch.sin(2.0 * math.pi * x)

        # 1. In-distribution evaluation
        rollout_in = runner.rollout_1d(u_init, dx=dx, dt=dt, steps=steps)
        solver_in = Burgers1DSolver(BurgersSolverConfig(viscosity=train_viscosity, cfl=0.4))
        u_ref_in = u_init.clone()
        ref_hist_in = [u_ref_in.clone()]
        for _ in range(steps):
            u_ref_in = solver_in.step(u_ref_in, dx=dx, dt=dt)
            ref_hist_in.append(u_ref_in.clone())
        ref_traj_in = torch.stack(ref_hist_in, dim=0)
        metrics_in = TrajectoryEvaluator.evaluate_rollout(rollout_in.trajectory, ref_traj_in, (dx,))

        # 2. Shifted low-viscosity evaluation
        rollout_shift = runner.rollout_1d(u_init, dx=dx, dt=dt, steps=steps)
        solver_shift = Burgers1DSolver(BurgersSolverConfig(viscosity=shifted_viscosity, cfl=0.4))
        u_ref_shift = u_init.clone()
        ref_hist_shift = [u_ref_shift.clone()]
        for _ in range(steps):
            u_ref_shift = solver_shift.step(u_ref_shift, dx=dx, dt=dt)
            ref_hist_shift.append(u_ref_shift.clone())
        ref_traj_shift = torch.stack(ref_hist_shift, dim=0)
        metrics_shift = TrajectoryEvaluator.evaluate_rollout(rollout_shift.trajectory, ref_traj_shift, (dx,))

        ratio = metrics_shift.relative_l2_error / max(1e-12, metrics_in.relative_l2_error)

        return ShiftExperimentReport(
            experiment_type="parameter_shift",
            condition_tag=f"viscosity_{shifted_viscosity}_vs_{train_viscosity}",
            in_distribution_error=metrics_in.relative_l2_error,
            shifted_error=metrics_shift.relative_l2_error,
            degradation_ratio=ratio,
            is_stable=metrics_shift.is_stable,
            mass_conservation_residual=metrics_shift.mass_conservation_residual,
        )

    def evaluate_initial_condition_frequency_shift(
        self,
        model: nn.Module,
        train_freq: int = 1,
        shifted_freq: int = 4,  # 4x higher wavenumber
        nx: int = 64,
        steps: int = 25,
        dt: float = 0.005,
    ) -> ShiftExperimentReport:
        """Evaluate model on high-frequency spatial waves absent from training."""
        runner = AutoregressiveRolloutRunner(model=model, device=self.device)
        dx = 1.0 / nx
        x = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx)
        u_init_in = torch.sin(2.0 * math.pi * train_freq * x)
        u_init_shift = torch.sin(2.0 * math.pi * shifted_freq * x)

        solver = Burgers1DSolver(BurgersSolverConfig(viscosity=0.01, cfl=0.4))

        # In-distribution
        roll_in = runner.rollout_1d(u_init_in, dx=dx, dt=dt, steps=steps)
        u_ref = u_init_in.clone()
        hist_in = [u_ref.clone()]
        for _ in range(steps):
            u_ref = solver.step(u_ref, dx=dx, dt=dt)
            hist_in.append(u_ref.clone())
        m_in = TrajectoryEvaluator.evaluate_rollout(roll_in.trajectory, torch.stack(hist_in, dim=0), (dx,))

        # Shifted
        roll_shift = runner.rollout_1d(u_init_shift, dx=dx, dt=dt, steps=steps)
        u_ref_s = u_init_shift.clone()
        hist_s = [u_ref_s.clone()]
        for _ in range(steps):
            u_ref_s = solver.step(u_ref_s, dx=dx, dt=dt)
            hist_s.append(u_ref_s.clone())
        m_shift = TrajectoryEvaluator.evaluate_rollout(roll_shift.trajectory, torch.stack(hist_s, dim=0), (dx,))

        ratio = m_shift.relative_l2_error / max(1e-12, m_in.relative_l2_error)

        return ShiftExperimentReport(
            experiment_type="initial_condition_shift",
            condition_tag=f"wavenumber_{shifted_freq}_vs_{train_freq}",
            in_distribution_error=m_in.relative_l2_error,
            shifted_error=m_shift.relative_l2_error,
            degradation_ratio=ratio,
            is_stable=m_shift.is_stable,
            mass_conservation_residual=m_shift.mass_conservation_residual,
        )
