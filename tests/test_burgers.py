"""Tests for conservative finite-volume Burgers solver (I04)."""

import pytest
import math
import torch

from isopleth.data.contracts import BoundaryCondition, GridContract, PhysicalFamily
from isopleth.numerics.burgers import (
    Burgers1DSolver,
    BurgersSolverConfig,
    FluxScheme,
    SlopeLimiter,
)

def test_burgers_cfl_enforcement():
    solver = Burgers1DSolver(BurgersSolverConfig(cfl=0.4))
    nx = 64
    dx = 1.0 / nx
    u = torch.ones(nx) * 2.0  # speed 2.0

    # Max allowable dt: 0.4 * (1/64) / 2.0 = 0.003125
    is_stable, max_dt = solver.check_cfl_stability(u, dx=dx, dt=0.003)
    assert is_stable is True

    # dt=0.01 exceeds max_dt and must be rejected
    is_unstable, _ = solver.check_cfl_stability(u, dx=dx, dt=0.01)
    assert is_unstable is False

    with pytest.raises(ValueError, match="violates CFL stability limit"):
        solver.step(u, dx=dx, dt=0.01)

def test_burgers_flux_schemes_conservation():
    nx = 64
    dx = 1.0 / nx
    x = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, nx, dtype=torch.float64)
    # Smooth initial condition with zero mean or non-zero mean
    u_init = torch.sin(2.0 * math.pi * x)

    for scheme in [FluxScheme.GODUNOV, FluxScheme.RUSANOV, FluxScheme.ROE]:
        for limiter in [SlopeLimiter.NONE, SlopeLimiter.MINMOD, SlopeLimiter.VAN_LEER, SlopeLimiter.SUPERBEE]:
            solver = Burgers1DSolver(
                BurgersSolverConfig(
                    viscosity=0.005,
                    flux_scheme=scheme,
                    limiter=limiter,
                    time_integrator="ssp_rk2",
                )
            )
            u = u_init.clone()
            dt = 0.002
            for _ in range(20):
                u = solver.step(u, dx=dx, dt=dt, boundary=BoundaryCondition.PERIODIC)

            # Mass must be conserved: sum(u * dx) == sum(u_init * dx)
            mass_init = torch.sum(u_init * dx)
            mass_final = torch.sum(u * dx)
            assert torch.abs(mass_final - mass_init) < 1e-10

def test_burgers_trajectory_batch():
    grid = GridContract(
        spatial_dim=1,
        resolution=(32,),
        domain_bounds=((0.0, 1.0),),
        boundary_conditions=(BoundaryCondition.PERIODIC,),
        time_step=0.005,
        time_horizon=0.05,
    )
    solver = Burgers1DSolver(BurgersSolverConfig(viscosity=0.01))
    x = torch.linspace(0.5 / 32, 1.0 - 0.5 / 32, 32)
    u_init = torch.sin(2.0 * math.pi * x)

    batch = solver.solve_trajectory(u_init, grid=grid, seed=10)
    assert batch.values.shape == (1, 11, 32)
    assert batch.times.shape == (11,)
    assert batch.provenance[0].family == PhysicalFamily.BURGERS
