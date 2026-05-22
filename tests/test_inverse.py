"""Tests for sparse observation operators, differentiable inverse reconstruction, and baselines."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from isopleth.inverse import (
    DifferentiableInverseReconstructor,
    FiniteDifferenceCheckResult,
    GradientVerificationSuite,
    InverseConfig,
    NumericalSolverInverseBaseline,
    ObservationRecord,
    PriorMeanBaseline,
    SparseObservationOperator,
    SpatialInterpolationBaseline,
    SurrogateGradientDistortionAnalyzer,
    evaluate_reconstruction_accuracy,
)
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.numerics.burgers import Burgers1DSolver, BurgersSolverConfig


def test_sparse_observation_operator_1d_and_2d() -> None:
    """Tests observation operator subsampling and noise corruption."""
    obs_op = SparseObservationOperator(spatial_fraction=0.25, temporal_stride=2, noise_std=0.01, seed=42)

    # 1D test: [T=11, nx=32]
    traj_1d = torch.sin(torch.linspace(0, 2 * np.pi, 32)).unsqueeze(0).repeat(11, 1)
    times_1d = torch.linspace(0, 1.0, 11)
    rec_1d = obs_op.observe_1d(traj_1d, times_1d)

    assert rec_1d.sensor_indices.ndim == 1
    assert len(rec_1d.sensor_indices) == 8  # 32 * 0.25
    assert rec_1d.observed_values.shape[0] == 6  # 11 steps with stride 2 -> 6 steps
    assert rec_1d.mask.shape == traj_1d.shape
    assert rec_1d.mask.sum() == 6 * 8

    # 2D test: [T=11, C=2, ny=16, nx=16]
    traj_2d = torch.zeros(11, 2, 16, 16)
    times_2d = torch.linspace(0, 1.0, 11)
    rec_2d = obs_op.observe_2d(traj_2d, times_2d)

    expected_sensors = int(round(16 * 16 * 0.25))
    assert len(rec_2d.sensor_indices) == expected_sensors
    assert rec_2d.observed_values.shape == (6, 2, expected_sensors)


def test_prior_mean_and_interpolation_baselines() -> None:
    """Tests non-surrogate inverse baselines on 1D sinusoidal profile."""
    nx = 64
    x = torch.linspace(0, 2 * np.pi, nx, dtype=torch.float32)
    u_true = torch.sin(x).view(1, 1, nx)

    obs_op = SparseObservationOperator(spatial_fraction=0.20, temporal_stride=5, noise_std=0.0, seed=123)
    times = torch.linspace(0, 1.0, 10)
    traj = u_true.squeeze(1).repeat(10, 1)  # [10, nx]
    obs = obs_op.observe_1d(traj, times)

    # Prior mean baseline
    prior_model = PriorMeanBaseline()
    u_mean = prior_model.reconstruct_1d(obs, nx=nx)
    assert u_mean.shape == (1, 1, nx)

    # Spatial interpolation baseline
    interp_model = SpatialInterpolationBaseline(method="linear")
    u_interp = interp_model.reconstruct_1d(obs, nx=nx)
    assert u_interp.shape == (1, 1, nx)

    err_mean = evaluate_reconstruction_accuracy(u_mean, u_true)
    err_interp = evaluate_reconstruction_accuracy(u_interp, u_true)

    # Interpolation should outperform a naive flat mean
    assert err_interp["rel_l2_initial"] < err_mean["rel_l2_initial"]


def test_differentiable_inverse_reconstruction_1d() -> None:
    """Verifies optimization loop convergence through neural surrogate."""
    nx = 32
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=8)

    # Ground truth initial state
    x = torch.linspace(0, 1, nx)
    u0_true = torch.sin(2 * np.pi * x).view(1, 1, nx)

    # Synthetic observation record
    obs_op = SparseObservationOperator(spatial_fraction=0.5, temporal_stride=2, noise_std=0.001, seed=99)
    dx = 1.0 / nx
    dt = 0.01
    # Rollout model for 4 steps to create observations
    with torch.no_grad():
        step1, _ = model(u0_true, dx=dx, dt=dt)
        step2, _ = model(step1, dx=dx, dt=dt)
        step3, _ = model(step2, dx=dx, dt=dt)
        step4, _ = model(step3, dx=dx, dt=dt)
        traj = torch.stack([u0_true, step1, step2, step3, step4], dim=1).squeeze(2).squeeze(0)  # [5, nx]

    times = torch.linspace(0, 0.04, 5)
    obs = obs_op.observe_1d(traj, times)

    config = InverseConfig(
        max_iterations=15,
        learning_rate=0.05,
        optimizer_type="adam",
        weight_observation=1.0,
        weight_smoothness_h1=0.001,
    )
    reconstructor = DifferentiableInverseReconstructor(model, config)

    # Run reconstruction with initial guess of zeros
    guess = torch.zeros_like(u0_true)
    result = reconstructor.reconstruct(
        observation=obs,
        num_rollout_steps=4,
        initial_guess=guess,
        dt=0.05,
    )

    assert result.iterations_run > 0
    assert result.final_loss < result.initial_loss
    assert result.reconstructed_initial_state.shape == u0_true.shape


def test_float64_central_finite_difference_check() -> None:
    """Verifies autograd against central finite difference to high precision in float64."""
    def sample_objective(w: torch.Tensor) -> torch.Tensor:
        # Non-linear smooth test function: sum(w^3 - 2 * sin(w))
        return torch.sum(w ** 3 - 2.0 * torch.sin(w))

    w_test = torch.tensor([0.5, -1.2, 2.1, 0.05], dtype=torch.float64)
    result = GradientVerificationSuite.check_central_finite_difference(
        objective_fn=sample_objective,
        x=w_test,
        epsilon=1e-6,
        tolerance=1e-5,
    )

    assert result.passed
    assert result.relative_l2_error < 1e-5


def test_surrogate_gradient_distortion_analyzer() -> None:
    """Verifies that distortion analyzer correctly evaluates cosine similarity and magnitude ratio."""
    nx = 32
    solver_config = BurgersSolverConfig(viscosity=0.01)
    solver = Burgers1DSolver(solver_config)
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=8)

    analyzer = SurrogateGradientDistortionAnalyzer(surrogate_model=model, numerical_solver=solver)

    u0 = torch.sin(2 * np.pi * torch.linspace(0, 1, nx)).view(1, 1, nx)
    obs_op = SparseObservationOperator(spatial_fraction=0.25, temporal_stride=2, noise_std=0.01, seed=42)
    u_curr = u0.squeeze().clone()
    dx = 1.0 / nx
    dt = 0.01
    traj_list = [u_curr.numpy().copy()]
    for _ in range(5):
        u_curr = solver.step(u_curr, dx=dx, dt=dt)
        traj_list.append(u_curr.numpy().copy())
    traj_np = np.array(traj_list)
    obs = obs_op.observe_1d(torch.tensor(traj_np, dtype=torch.float32), torch.linspace(0, 0.05, 6))

    distortion = analyzer.analyze_distortion(u0_initial=u0, observation=obs, num_steps=5, dt=dt)

    assert -1.0 <= distortion.cosine_similarity <= 1.0
    assert distortion.magnitude_ratio >= 0.0
    assert len(distortion.spectral_energy_surrogate) > 0
