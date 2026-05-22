"""Tests for numerical solver inverse baseline and 2D interpolation baselines."""

import numpy as np
import pytest
import torch

from isopleth.inverse import (
    NumericalSolverInverseBaseline,
    ObservationRecord,
    PriorMeanBaseline,
    SparseObservationOperator,
    SpatialInterpolationBaseline,
    evaluate_reconstruction_accuracy,
)
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig


def test_2d_spatial_interpolation_baseline() -> None:
    """Verifies that 2D spatial interpolation reconstructs smooth fields from sparse points."""
    ny, nx = 16, 16
    channels = 2
    # Create smooth test field
    y = torch.linspace(0, 1, ny).view(ny, 1).repeat(1, nx)
    x = torch.linspace(0, 1, nx).view(1, nx).repeat(ny, 1)
    field_c0 = torch.sin(2 * np.pi * x) * torch.cos(2 * np.pi * y)
    field_c1 = torch.cos(2 * np.pi * x) * torch.sin(2 * np.pi * y)
    field = torch.stack([field_c0, field_c1], dim=0).unsqueeze(0)  # [1, 2, 16, 16]

    obs_op = SparseObservationOperator(spatial_fraction=0.30, temporal_stride=2, noise_std=0.0, seed=77)
    times = torch.linspace(0, 0.1, 4)
    traj = field.squeeze(0).unsqueeze(0).repeat(4, 1, 1, 1)  # [4, 2, 16, 16]
    obs = obs_op.observe_2d(traj, times)

    interp_baseline = SpatialInterpolationBaseline(method="linear")
    reconstructed = interp_baseline.reconstruct_2d(obs, shape=(ny, nx), channels=channels)

    assert reconstructed.shape == (1, channels, ny, nx)
    # Check that reconstructed field is finite and non-empty
    assert not torch.isnan(reconstructed).any()
    metrics = evaluate_reconstruction_accuracy(reconstructed, field)
    # Relative error should be reasonably bounded on smooth field
    assert metrics["rel_l2_initial"] < 0.85


def test_numerical_solver_inverse_burgers_baseline() -> None:
    """Verifies numerical solver finite-difference optimization baseline on 1D Burgers."""
    nx = 32
    solver_config = BurgersSolverConfig(viscosity=0.01)
    solver = Burgers1DSolver(solver_config)

    x = torch.linspace(0, 1, nx)
    u0_true = torch.sin(2 * np.pi * x).view(1, 1, nx)

    obs_op = SparseObservationOperator(spatial_fraction=0.35, temporal_stride=2, noise_std=0.001, seed=55)
    u_curr = u0_true.squeeze().clone()
    dx = 1.0 / nx
    dt = 0.01
    traj_list = [u_curr.numpy().copy()]
    for _ in range(4):
        u_curr = solver.step(u_curr, dx=dx, dt=dt)
        traj_list.append(u_curr.numpy().copy())
    traj_np = np.array(traj_list)
    obs = obs_op.observe_1d(torch.tensor(traj_np, dtype=torch.float32), torch.linspace(0, 0.04, 5))

    baseline_solver = NumericalSolverInverseBaseline(
        solver_config=solver_config,
        nx=nx,
        max_iterations=10,
        learning_rate=0.02,
    )
    reconstructed, metrics = baseline_solver.reconstruct_burgers(
        observation=obs,
        num_steps=4,
        dt=dt,
    )

    assert reconstructed.shape == (1, 1, nx)
    assert metrics.iterations == 10
    assert metrics.wall_clock_seconds > 0.0
