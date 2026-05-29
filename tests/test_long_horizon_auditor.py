"""Tests for long-horizon stability and physical invariant audit engine."""

from __future__ import annotations

import pytest
import torch

from isopleth.rollout.long_horizon_auditor import (
    LongHorizonAuditReport,
    LongHorizonConservationAuditor,
)


def test_long_horizon_conservation_auditor_clean_trajectory() -> None:
    auditor = LongHorizonConservationAuditor(
        mass_drift_tolerance=1e-3,
        allowable_tv_growth_factor=1.2,
        enforce_monotonic_energy_decay=True,
    )

    # 10 steps of decaying smooth sine wave
    dx = 1.0 / 64.0
    x = torch.linspace(0.0, 1.0, 64)[:-1]
    trajectory = []
    for step in range(10):
        decay = float(torch.exp(torch.tensor(-0.1 * step)).item())
        u = decay * torch.sin(2.0 * 3.14159 * x)
        trajectory.append(u)

    traj_tensor = torch.stack(trajectory, dim=0)
    report = auditor.audit_trajectory_1d(traj_tensor, dx=dx, dt=0.01)

    assert isinstance(report, LongHorizonAuditReport)
    assert report.steps_audited == 10
    assert report.monotonic_energy_dissipation is True
    assert report.passed_all_invariants is True
    assert "mass" in report.invariant_time_series


def test_long_horizon_conservation_auditor_detects_gibbs_growth() -> None:
    auditor = LongHorizonConservationAuditor(
        mass_drift_tolerance=1.0,
        allowable_tv_growth_factor=1.5,
    )

    dx = 1.0 / 32.0
    u0 = torch.zeros(32)
    u0[10:20] = 1.0  # Square wave

    # Create oscillatory state with huge TV
    u1 = u0.clone()
    u1[::2] += 2.0  # High-frequency alternating spike

    report = auditor.audit_trajectory_1d([u0, u1], dx=dx, dt=0.01)
    assert report.gibbs_overshoot_detected is True
