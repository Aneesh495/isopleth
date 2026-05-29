"""Tests for DeepONet operator architectures and conservative flux variant."""

import pytest
import torch

from isopleth.models.deeponet import (
    BranchMLP,
    ConservativeFluxDeepONet1D,
    DeepONet1D,
    DeepONet2D,
    TrunkMLP,
)


def test_branch_and_trunk_mlp():
    b_net = BranchMLP(in_features=32, out_features=64, hidden_dims=(64, 64))
    x_sensor = torch.randn(4, 32)
    b_out = b_net(x_sensor)
    assert b_out.shape == (4, 64)

    t_net = TrunkMLP(coord_dim=1, out_features=64, hidden_dims=(64, 64))
    coords = torch.linspace(0.0, 1.0, 50).unsqueeze(-1)
    t_out = t_net(coords)
    assert t_out.shape == (50, 64)


def test_deeponet_1d_forward():
    model = DeepONet1D(
        in_channels=1,
        out_channels=1,
        num_sensors=32,
        latent_dim=48,
        hidden_dims=(32, 32),
    )
    u_sensor = torch.randn(5, 1, 32)
    out = model(u_sensor)
    assert out.shape == (5, 1, 32)

    # Arbitrary query points
    query_pts = torch.rand(17, 1)
    out_arbitrary = model(u_sensor, query_coords=query_pts)
    assert out_arbitrary.shape == (5, 1, 17)


def test_conservative_flux_deeponet_1d_conservation():
    model = ConservativeFluxDeepONet1D(
        in_channels=1,
        out_channels=1,
        num_cells=32,
        latent_dim=32,
        hidden_dims=(32, 32),
    )
    u_in = torch.randn(4, 1, 32, dtype=torch.float64)
    model = model.to(dtype=torch.float64)

    u_next, fluxes = model(u_in, dt=0.01, dx=1.0 / 32)
    assert u_next.shape == (4, 1, 32)
    assert fluxes.shape == (4, 1, 32)

    # Check exact telescopic conservation: mean(u_next) == mean(u_in) within double precision
    mass_in = u_in.sum(dim=-1)
    mass_out = u_next.sum(dim=-1)
    assert torch.allclose(mass_in, mass_out, atol=1e-12)


def test_deeponet_2d_forward():
    model = DeepONet2D(
        in_channels=2,
        out_channels=2,
        grid_size=(16, 16),
        latent_dim=32,
        hidden_dims=(32, 32),
    )
    u_grid = torch.randn(3, 2, 16, 16)
    out = model(u_grid)
    assert out.shape == (3, 2, 16, 16)
