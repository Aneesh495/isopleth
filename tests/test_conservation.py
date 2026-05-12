"""Tests for float64 mass conservation strict audit (1e-9 tolerance)."""

import pytest
import math
import torch

from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig
from isopleth.numerics.shallow_water import ShallowWater2DSolver, ShallowWaterConfig
from isopleth.numerics.sources import BalanceAuditor
from isopleth.data.contracts import BoundaryCondition

def test_burgers_float64_conservation():
    nx = 64
    dx = 1.0 / nx
    xc = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
    u_init = 1.0 + 0.5 * torch.sin(2.0 * math.pi * xc)

    solver = Burgers1DSolver(
        BurgersSolverConfig(
            viscosity=0.005,
            cfl=0.35,
            time_integrator="ssp_rk2",
        )
    )

    auditor = BalanceAuditor(cell_measures=(dx,), tolerance_float64=1e-9)
    u = u_init.clone()
    dt = 0.002
    for _ in range(50):
        u = solver.step(u, dx=dx, dt=dt, boundary=BoundaryCondition.PERIODIC)

    report = auditor.audit(u_init, u)
    assert report.is_balanced is True
    assert report.relative_residual < 1e-9

def test_shallow_water_float64_conservation():
    ny, nx = 32, 32
    dx, dy = 1.0 / nx, 1.0 / ny
    xc = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
    yc = torch.linspace(0.5 * dy, 1.0 - 0.5 * dy, ny, dtype=torch.float64)
    Y, X = torch.meshgrid(yc, xc, indexing="ij")

    h_init = 1.0 + 0.1 * torch.sin(2.0 * math.pi * X) * torch.cos(2.0 * math.pi * Y)
    hu_init = 0.02 * torch.sin(2.0 * math.pi * Y)
    hv_init = -0.02 * torch.cos(2.0 * math.pi * X)

    solver = ShallowWater2DSolver(
        ShallowWaterConfig(
            gravity=9.81,
            cfl=0.35,
            use_conservative_limiter=True,
        )
    )

    auditor = BalanceAuditor(cell_measures=(dx, dy), tolerance_float64=1e-9)
    h, hu, hv = h_init.clone(), hu_init.clone(), hv_init.clone()
    dt = 0.001
    for _ in range(50):
        h, hu, hv, diag = solver.step(h, hu, hv, dx=dx, dy=dy, dt=dt)
        assert diag.is_safe is True

    report = auditor.audit(h_init, h)
    assert report.is_balanced is True
    assert report.relative_residual < 1e-9
