"""Tests for interface flux heads and discrete divergence (I11)."""

import pytest
import torch

from isopleth.data.contracts import BoundaryCondition
from isopleth.models.interfaces import (
    DiscreteFluxDivergence,
    InterfaceFluxHead1D,
    InterfaceFluxHead2D,
)

def test_interface_flux_head_1d_conservation():
    head = InterfaceFluxHead1D(in_channels=16, out_channels=1, hidden_channels=32)
    div_op = DiscreteFluxDivergence(spatial_dim=1)
    
    features = torch.randn(2, 16, 64)
    fluxes = head(features, boundary=BoundaryCondition.PERIODIC)
    assert fluxes.shape == (2, 1, 64)

    # Discrete divergence
    div = div_op.forward_1d(fluxes, dx=0.01, boundary=BoundaryCondition.PERIODIC)
    assert div.shape == (2, 1, 64)

    # Telescoping sum must be exactly 0 across periodic domain
    total_div = torch.sum(div, dim=-1)
    assert torch.allclose(total_div, torch.zeros_like(total_div), atol=1e-5)

def test_interface_flux_head_2d_conservation():
    head = InterfaceFluxHead2D(in_channels=16, out_channels=3, hidden_channels=32)
    div_op = DiscreteFluxDivergence(spatial_dim=2)

    features = torch.randn(2, 16, 32, 32)
    flux_x, flux_y = head(features, boundary=BoundaryCondition.PERIODIC)
    assert flux_x.shape == (2, 3, 32, 32)
    assert flux_y.shape == (2, 3, 32, 32)

    div = div_op.forward_2d(flux_x, flux_y, dx=0.01, dy=0.01, boundary=BoundaryCondition.PERIODIC)
    assert div.shape == (2, 3, 32, 32)

    # Telescoping physical mass integral in 2D must sum to 0
    total_mass_change = torch.sum(div * (0.01 * 0.01), dim=(-2, -1))
    assert torch.allclose(total_mass_change, torch.zeros_like(total_mass_change), atol=1e-6)
