"""Tests for discrete transforms, prolongation, and restriction (I07)."""

import pytest
import torch

from isopleth.numerics.transforms import (
    conservative_prolongation_1d,
    conservative_prolongation_2d,
    conservative_restriction_1d,
    conservative_restriction_2d,
    discrete_divergence_1d,
    discrete_divergence_2d,
)

def test_discrete_divergence_1d():
    # Constant flux -> divergence is zero
    flux = torch.ones(1, 65) * 3.5
    div = discrete_divergence_1d(flux, dx=0.01)
    assert div.shape == (1, 64)
    assert torch.allclose(div, torch.zeros_like(div))

    # Linear flux F(x) = 2x -> div = 2
    x_faces = torch.linspace(0.0, 1.0, 65)
    flux_lin = 2.0 * x_faces
    div_lin = discrete_divergence_1d(flux_lin, dx=1.0 / 64)
    assert torch.allclose(div_lin, torch.full_like(div_lin, 2.0), atol=1e-5)

def test_conservative_restriction_prolongation_1d():
    # Fine grid 64 -> coarse 32
    nx_fine = 64
    dx_fine = 1.0 / nx_fine
    u_fine = torch.randn(1, nx_fine, dtype=torch.float64)
    int_fine = torch.sum(u_fine * dx_fine)

    u_coarse = conservative_restriction_1d(u_fine, factor=2)
    assert u_coarse.shape == (1, 32)
    dx_coarse = 1.0 / 32
    int_coarse = torch.sum(u_coarse * dx_coarse)
    # Exact integral equality
    assert torch.abs(int_coarse - int_fine) < 1e-12

    # Prolongation order 1 (piecewise constant)
    u_prolong1 = conservative_prolongation_1d(u_coarse, factor=2, order=1)
    assert u_prolong1.shape == (1, 64)
    int_prolong1 = torch.sum(u_prolong1 * dx_fine)
    assert torch.abs(int_prolong1 - int_coarse) < 1e-12

    # Prolongation order 2 (linear TVD)
    u_prolong2 = conservative_prolongation_1d(u_coarse, factor=2, order=2)
    assert u_prolong2.shape == (1, 64)
    int_prolong2 = torch.sum(u_prolong2 * dx_fine)
    assert torch.abs(int_prolong2 - int_coarse) < 1e-12

def test_conservative_restriction_prolongation_2d():
    ny_fine, nx_fine = 32, 32
    dA_fine = (1.0 / 32) * (1.0 / 32)
    u_fine = torch.randn(1, ny_fine, nx_fine, dtype=torch.float64)
    int_fine = torch.sum(u_fine * dA_fine)

    u_coarse = conservative_restriction_2d(u_fine, factor=2)
    assert u_coarse.shape == (1, 16, 16)
    dA_coarse = (1.0 / 16) * (1.0 / 16)
    int_coarse = torch.sum(u_coarse * dA_coarse)
    assert torch.abs(int_coarse - int_fine) < 1e-12

    u_prolong1 = conservative_prolongation_2d(u_coarse, factor=2, order=1)
    assert u_prolong1.shape == (1, 32, 32)
    assert torch.abs(torch.sum(u_prolong1 * dA_fine) - int_coarse) < 1e-12

    u_prolong2 = conservative_prolongation_2d(u_coarse, factor=2, order=2)
    assert u_prolong2.shape == (1, 32, 32)
    assert torch.abs(torch.sum(u_prolong2 * dA_fine) - int_coarse) < 1e-12
