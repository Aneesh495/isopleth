"""Tests for physics-informed optimizers and projection operators."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from isopleth.training.optimizers import (
    ConservationConstrainedOptimizer,
    ProjectedGradientDescent,
    SecondOrderLBFGSWrapper,
    StochasticWeightAveragingPDE,
)


def test_projected_gradient_descent_projections() -> None:
    # 1. Depth positivity
    depth = torch.tensor([[-0.5, 1.2, -0.01]])
    proj_depth = ProjectedGradientDescent.project_positive_depth(depth, h_min=1e-3, depth_channel=0)
    assert proj_depth[0, 0] >= 1e-3
    assert proj_depth[0, 1] == pytest.approx(1.2)

    # 2. Species bounds [0, 1]
    species = torch.tensor([[[-0.2, 1.5], [0.8, -0.1]]])
    proj_species = ProjectedGradientDescent.project_species_bounds(species)
    assert torch.all(proj_species >= 0.0)
    assert torch.all(proj_species <= 1.0)

    # 3. Zero-divergence flux projection
    fluxes = torch.tensor([1.0, 2.0, 3.0])
    proj_flux = ProjectedGradientDescent.project_zero_divergence_1d(fluxes)
    assert torch.sum(proj_flux).item() == pytest.approx(0.0, abs=1e-7)


def test_conservation_constrained_optimizer() -> None:
    # Target parameter to optimize with mass preservation
    target = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    initial_mass = torch.sum(target).item()
    base_opt = torch.optim.SGD([target], lr=0.1)
    optimizer = ConservationConstrainedOptimizer(base_opt, target_tensor=target, preserve_mass=True)

    # Take step with loss that would otherwise alter mass
    loss = torch.sum((target - 5.0) ** 2)
    loss.backward()
    optimizer.step()

    final_mass = torch.sum(target).item()
    assert final_mass == pytest.approx(initial_mass, abs=1e-6)


def test_stochastic_weight_averaging_pde() -> None:
    model = nn.Linear(4, 4)
    swa = StochasticWeightAveragingPDE(model)

    # Simulate 2 training epochs with weight updates
    with torch.no_grad():
        model.weight.fill_(1.0)
    swa.update()

    with torch.no_grad():
        model.weight.fill_(3.0)
    swa.update()

    swa.apply_averaged_weights()
    # Average of 1.0 and 3.0 is 2.0
    assert torch.allclose(model.weight, torch.full_like(model.weight, 2.0))


def test_second_order_lbfgs_wrapper() -> None:
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    wrapper = SecondOrderLBFGSWrapper([x], learning_rate=0.5, max_iter=10)

    def closure() -> torch.Tensor:
        wrapper.optimizer.zero_grad()
        loss = torch.sum(x**2)
        loss.backward()
        return loss

    wrapper.step(closure)
    assert torch.norm(x).item() < 0.5
