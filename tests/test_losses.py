"""Tests for physical-unit multi-component loss functions."""

import pytest
import torch

from isopleth.training.losses import PhysicalLoss

def test_physical_loss_components_1d():
    loss_fn = PhysicalLoss()
    pred = torch.randn(2, 64)
    target = pred.clone()  # Exact match
    init = pred.clone()

    comp_zero = loss_fn(pred, target, init, cell_measures=(0.01,))
    assert comp_zero.total_loss.item() < 1e-6
    assert comp_zero.mse_loss.item() < 1e-6
    assert comp_zero.h1_loss.item() < 1e-6
    assert comp_zero.spectral_loss.item() < 1e-6

    # Perturbed prediction
    pred_bad = pred + 0.5
    comp_bad = loss_fn(pred_bad, target, init, cell_measures=(0.01,))
    assert comp_bad.total_loss.item() > 0.0
    assert comp_bad.conservation_penalty.item() > 0.0

def test_physical_loss_components_2d():
    loss_fn = PhysicalLoss()
    pred = torch.randn(2, 3, 16, 16)
    target = pred.clone()
    init = pred.clone()

    comp = loss_fn(pred, target, init, cell_measures=(0.01, 0.01))
    assert comp.total_loss.item() < 1e-6
