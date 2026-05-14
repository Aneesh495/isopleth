"""Tests for Multiscale Face-Flux Neural Operator (I10)."""

import pytest
import torch

from isopleth.models.flux_operator import (
    MultiscaleFaceFluxOperator1D,
    MultiscaleFaceFluxOperator2D,
)
from isopleth.data.contracts import BoundaryCondition

def test_mffno_1d_exact_discrete_conservation():
    model = MultiscaleFaceFluxOperator1D(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        modes=8,
        num_layers=2,
    )
    nx = 64
    dx = 1.0 / nx
    dt = 0.01

    x = torch.randn(2, 1, nx)
    orig_integral = torch.sum(x * dx, dim=-1)

    # Forward pass without sources
    x_next, fluxes = model(x, dx=dx, dt=dt, boundary=BoundaryCondition.PERIODIC)
    new_integral = torch.sum(x_next * dx, dim=-1)

    # Exact conservation: Integral(x_next) == Integral(x)
    assert torch.allclose(new_integral, orig_integral, atol=1e-5)

def test_mffno_1d_gradients_in_all_branches():
    model = MultiscaleFaceFluxOperator1D(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        modes=8,
        num_layers=2,
    )
    x = torch.randn(2, 1, 32, requires_grad=True)
    cond = torch.randn(2, 128, requires_grad=True)
    x_next, _ = model(x, dx=0.03, dt=0.01, cond=cond)

    loss = torch.sum(x_next**2)
    loss.backward()

    # Ensure gradients propagate to spectral and local branches
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients"
        grad_norm = float(param.grad.norm().item())
        assert grad_norm > 0.0, f"Parameter {name} had zero gradient norm"

def test_mffno_2d_exact_discrete_conservation():
    model = MultiscaleFaceFluxOperator2D(
        in_channels=3,
        out_channels=3,
        hidden_channels=24,
        modes_x=6,
        modes_y=6,
        num_layers=2,
    )
    ny, nx = 24, 24
    dx, dy = 1.0 / nx, 1.0 / ny
    dt = 0.005

    x = torch.randn(2, 3, ny, nx)
    orig_integral = torch.sum(x * (dx * dy), dim=(-2, -1))

    x_next, _ = model(x, dx=dx, dy=dy, dt=dt, boundary=BoundaryCondition.PERIODIC)
    new_integral = torch.sum(x_next * (dx * dy), dim=(-2, -1))

    assert torch.allclose(new_integral, orig_integral, atol=1e-5)
