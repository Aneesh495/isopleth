"""Tests for adaptive time-stepping, Richardson extrapolation, and long-horizon auditor."""

import pytest
import torch

from isopleth.rollout.adaptive_stepper import (
    AdaptiveTimeStepper,
    RichardsonExtrapolationEstimator,
)
from isopleth.rollout.long_horizon_auditor import LongHorizonConservationAuditor


def test_richardson_extrapolation_estimator():
    estimator = RichardsonExtrapolationEstimator(effective_order=2)

    # Simple linear decay step: u_next = u - dt * u
    def step_fn(u: torch.Tensor, dt: float) -> torch.Tensor:
        return u * (1.0 - dt)

    u0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    u_extrap, err = estimator.estimate_error(u0, step_fn, dt=0.01)

    assert u_extrap.shape == (2,)
    assert err > 0.0
    assert err < 1e-2


def test_adaptive_time_stepper():
    stepper = AdaptiveTimeStepper(target_error_tolerance=1e-3, cfl_target=0.4, dt_min=1e-4, dt_max=0.05)

    def step_fn(u: torch.Tensor, dt: float) -> torch.Tensor:
        return u - dt * 0.1 * u

    u0 = torch.ones(1, 1, 32)
    traj = stepper.rollout_adaptive(u0, step_fn, t_final=0.05, dx=1.0 / 32, initial_dt=0.01)

    assert traj.steps_accepted > 0
    assert traj.total_time >= 0.049


def test_long_horizon_conservation_auditor():
    auditor = LongHorizonConservationAuditor(mass_drift_tolerance=1e-4)

    # 10 steps of constant state (exact conservation)
    u_clean = [torch.sin(torch.linspace(0, 3.14, 32)) for _ in range(10)]
    rep = auditor.audit_trajectory_1d(u_clean, dx=0.03, dt=0.01)

    assert rep.passed_all_invariants
    assert rep.max_mass_drift < 1e-6
    assert not rep.gibbs_overshoot_detected
