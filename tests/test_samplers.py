"""Tests for continuous and discrete random field initial condition samplers."""

import numpy as np
import pytest
import torch

from isopleth.data.samplers import (
    GaussianRandomField1D,
    GaussianRandomField2D,
    GrayScottSeedSampler,
    KarhunenLoeveExpander1D,
    RiemannDamBreakSampler,
    ShallowWaterTopographySampler,
    sample_initial_conditions_torch,
)


def test_grf_1d_continuous_and_grid():
    sampler = GaussianRandomField1D(tau=3.0, alpha=2.5, amplitude=1.0, seed=42)

    # Continuous evaluation
    x_coords = np.linspace(0.0, 1.0, 64)
    continuous_field = sampler.sample_continuous(x_coords, num_modes=32)
    assert continuous_field.shape == (64,)
    assert not np.isnan(continuous_field).any()

    # Grid evaluation
    grid_fields = sampler.sample_grid(n_cells=64, num_samples=4)
    assert grid_fields.shape == (4, 64)
    assert not np.isnan(grid_fields).any()


def test_grf_2d_grid():
    sampler = GaussianRandomField2D(tau=3.0, alpha=2.5, amplitude=1.0, seed=42)
    fields = sampler.sample_grid(nx=32, ny=32, num_samples=3)
    assert fields.shape == (3, 32, 32)
    assert not np.isnan(fields).any()


def test_karhunen_loeve_1d():
    kl_fourier = KarhunenLoeveExpander1D(num_modes=16, decay_rate=2.0, basis_type="fourier", seed=42)
    x = np.linspace(0.0, 1.0, 50)
    samples = kl_fourier.sample(x, num_samples=5)
    assert samples.shape == (5, 50)

    kl_chebyshev = KarhunenLoeveExpander1D(num_modes=16, decay_rate=2.0, basis_type="chebyshev", seed=42)
    samples_cheb = kl_chebyshev.sample(x, num_samples=5)
    assert samples_cheb.shape == (5, 50)


def test_riemann_dambreak_sampler():
    sampler = RiemannDamBreakSampler(seed=42)

    # 1D
    h, hu = sampler.sample_1d(n_cells=64, num_samples=3)
    assert h.shape == (3, 64)
    assert hu.shape == (3, 64)
    assert (h > 0.0).all()

    # 2D radial
    h_2d, hu_2d, hv_2d = sampler.sample_2d_radial(nx=32, ny=32, num_samples=2)
    assert h_2d.shape == (2, 32, 32)
    assert (h_2d > 0.0).all()
    assert hu_2d.shape == (2, 32, 32)
    assert hv_2d.shape == (2, 32, 32)


def test_topography_sampler():
    sampler = ShallowWaterTopographySampler(seed=42)
    b_1d = sampler.sample_1d_seamount(n_cells=64)
    assert b_1d.shape == (64,)
    assert (b_1d >= 0.0).all()

    b_2d = sampler.sample_2d_seamount(nx=32, ny=32)
    assert b_2d.shape == (32, 32)
    assert (b_2d >= 0.0).all()


def test_gray_scott_sampler():
    sampler = GrayScottSeedSampler(seed=42)
    u, v = sampler.sample_central_square(nx=32, ny=32, num_samples=2)
    assert u.shape == (2, 32, 32)
    assert v.shape == (2, 32, 32)
    assert (u >= 0.0).all() and (u <= 1.0).all()
    assert (v >= 0.0).all() and (v <= 1.0).all()

    u_multi, v_multi = sampler.sample_multi_spot(nx=32, ny=32, num_samples=2)
    assert u_multi.shape == (2, 32, 32)
    assert v_multi.shape == (2, 32, 32)


def test_sample_initial_conditions_torch():
    t_grf1d = sample_initial_conditions_torch("grf_1d", resolution=64, num_samples=3)
    assert t_grf1d.shape == (3, 1, 64)
    assert isinstance(t_grf1d, torch.Tensor)

    t_grf2d = sample_initial_conditions_torch("grf_2d", resolution=(32, 32), num_samples=2)
    assert t_grf2d.shape == (2, 1, 32, 32)

    t_dam1d = sample_initial_conditions_torch("dambreak_1d", resolution=64, num_samples=2)
    assert t_dam1d.shape == (2, 2, 64)

    t_dam2d = sample_initial_conditions_torch("dambreak_2d", resolution=(32, 32), num_samples=2)
    assert t_dam2d.shape == (2, 3, 32, 32)
