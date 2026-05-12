"""Tests for source terms, Gray-Scott kinetics, and balance auditing (I12)."""

import pytest
import torch

from isopleth.numerics.sources import (
    BalanceAuditor,
    compute_spatial_integral,
    gray_scott_reaction_sources,
    integrate_source_rk4,
    shallow_water_bottom_friction,
    shallow_water_coriolis_source,
)

def test_gray_scott_reaction_sources_positive_production():
    u = torch.tensor([1.0, 0.5])
    v = torch.tensor([0.2, 0.25])
    f = 0.04
    k = 0.06

    s_u, s_v = gray_scott_reaction_sources(u, v, f=f, k=k)

    # u is consumed by reaction: - u * v^2
    # v is produced by reaction: + u * v^2
    # For index 0: uv^2 = 1.0 * (0.04) = 0.04
    # s_u = -0.04 + 0.04 * (1.0 - 1.0) = -0.04
    # s_v = +0.04 - (0.04 + 0.06) * 0.2 = +0.04 - 0.02 = +0.02
    assert torch.isclose(s_u[0], torch.tensor(-0.04))
    assert torch.isclose(s_v[0], torch.tensor(0.02))

def test_balance_auditor():
    auditor = BalanceAuditor(cell_measures=(0.1,), tolerance_float64=1e-9)
    u_init = torch.ones(10, dtype=torch.float64)  # Mass = 10 * 0.1 = 1.0
    u_final = torch.ones(10, dtype=torch.float64) * 1.2  # Mass = 1.2
    # Expected mass increase = 0.2
    report = auditor.audit(u_init, u_final, total_source_integrated=0.2)
    assert report.is_balanced is True
    assert report.relative_residual < 1e-12

    # Unbalanced scenario
    report_unbal = auditor.audit(u_init, u_final, total_source_integrated=0.0)
    assert report_unbal.is_balanced is False
    assert abs(report_unbal.relative_residual - 0.2) < 1e-12

def test_rk4_source_integrator():
    # Test linear decay: dy/dt = -2 y, exact: y(t) = y0 * exp(-2 dt)
    y0 = torch.tensor([1.0])
    dt = 0.1
    y_rk4 = integrate_source_rk4(y0, lambda y: -2.0 * y, dt=dt)
    y_exact = torch.exp(torch.tensor([-0.2]))
    assert torch.allclose(y_rk4, y_exact, atol=1e-5)
