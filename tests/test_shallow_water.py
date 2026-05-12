"""Tests for 2D shallow water solver with positivity and CFL enforcement (I05)."""

import pytest
import math
import torch

from isopleth.data.contracts import BoundaryCondition, GridContract, PhysicalFamily
from isopleth.numerics.shallow_water import (
    ShallowWater2DSolver,
    ShallowWaterConfig,
    ShallowWaterFluxScheme,
)

def test_shallow_water_cfl_enforcement():
    solver = ShallowWater2DSolver(ShallowWaterConfig(cfl=0.35))
    ny, nx = 32, 32
    dx, dy = 1.0 / nx, 1.0 / ny
    h = torch.ones(ny, nx) * 1.0
    hu = torch.zeros(ny, nx)
    hv = torch.zeros(ny, nx)

    # c = sqrt(9.81 * 1.0) ~ 3.132 m/s
    # max dt ~ 0.35 * (1/32) / 3.132 ~ 0.00349 s
    is_stable, max_dt = solver.check_cfl_stability(h, hu, hv, dx=dx, dy=dy, dt=0.002)
    assert is_stable is True

    # dt=0.01 exceeds max_dt and must raise error
    with pytest.raises(ValueError, match="violates CFL stability limit"):
        solver.step(h, hu, hv, dx=dx, dy=dy, dt=0.01)

def test_shallow_water_schemes_mass_conservation():
    ny, nx = 24, 24
    dx, dy = 1.0 / nx, 1.0 / ny
    xc = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
    yc = torch.linspace(0.5 * dy, 1.0 - 0.5 * dy, ny, dtype=torch.float64)
    Y, X = torch.meshgrid(yc, xc, indexing="ij")

    # Smooth depth perturbation
    h_init = 1.0 + 0.1 * torch.sin(2.0 * math.pi * X) * torch.cos(2.0 * math.pi * Y)
    hu_init = 0.02 * torch.sin(2.0 * math.pi * Y)
    hv_init = -0.02 * torch.cos(2.0 * math.pi * X)

    for scheme in [ShallowWaterFluxScheme.KURGANOV_TADMOR, ShallowWaterFluxScheme.RUSANOV]:
        solver = ShallowWater2DSolver(
            ShallowWaterConfig(
                gravity=9.81,
                cfl=0.35,
                flux_scheme=scheme,
                use_conservative_limiter=True,
            )
        )
        h, hu, hv = h_init.clone(), hu_init.clone(), hv_init.clone()
        dt = 0.001
        for _ in range(15):
            h, hu, hv, diag = solver.step(h, hu, hv, dx=dx, dy=dy, dt=dt)
            assert diag.is_safe is True

        init_mass = torch.sum(h_init * dx * dy)
        final_mass = torch.sum(h * dx * dy)
        assert torch.abs(final_mass - init_mass) < 1e-10

def test_shallow_water_closed_walls():
    ny, nx = 20, 20
    dx, dy = 1.0 / nx, 1.0 / ny
    h = torch.ones(ny, nx, dtype=torch.float64) * 1.0
    # Impinging flow towards walls
    hu = torch.ones(ny, nx, dtype=torch.float64) * 0.1
    hv = torch.ones(ny, nx, dtype=torch.float64) * 0.1

    solver = ShallowWater2DSolver(ShallowWaterConfig(cfl=0.3))
    dt = 0.001
    h_next, hu_next, hv_next, _ = solver.step(
        h, hu, hv, dx=dx, dy=dy, dt=dt, boundary=BoundaryCondition.CLOSED_SLIP
    )
    # Mass must still be conserved with closed wall boundaries
    assert torch.abs(torch.sum(h_next) - torch.sum(h)) < 1e-10
