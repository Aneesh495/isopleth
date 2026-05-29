"""Tests for Evidential Deep Learning and NIG uncertainty decomposition."""

from __future__ import annotations

import pytest
import torch

from isopleth.uncertainty.evidential import (
    EvidentialHead1D,
    EvidentialHead2D,
    EvidentialLoss,
    EvidentialPrediction,
    EvidentialUncertaintyEstimator,
)


def test_evidential_head_1d() -> None:
    head = EvidentialHead1D(in_channels=16, out_channels=1)
    x = torch.randn(2, 16, 64)
    mu, v, alpha, beta = head(x)

    assert mu.shape == (2, 1, 64)
    assert v.shape == (2, 1, 64)
    assert alpha.shape == (2, 1, 64)
    assert beta.shape == (2, 1, 64)
    # Support checks
    assert torch.all(v > 0.0)
    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 0.0)


def test_evidential_head_2d() -> None:
    head = EvidentialHead2D(in_channels=8, out_channels=2)
    x = torch.randn(2, 8, 16, 16)
    mu, v, alpha, beta = head(x)

    assert mu.shape == (2, 2, 16, 16)
    assert torch.all(v > 0.0)
    assert torch.all(alpha > 1.0)
    assert torch.all(beta > 0.0)


def test_evidential_loss_and_uncertainty() -> None:
    loss_fn = EvidentialLoss(reg_coeff=0.01)
    mu = torch.zeros(2, 1, 32)
    v = torch.ones(2, 1, 32)
    alpha = torch.full((2, 1, 32), 2.0)
    beta = torch.ones(2, 1, 32)
    y_true = torch.randn(2, 1, 32)

    loss = loss_fn(mu, v, alpha, beta, y_true)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    pred = EvidentialUncertaintyEstimator.estimate_uncertainty(mu, v, alpha, beta)
    assert isinstance(pred, EvidentialPrediction)
    assert torch.all(pred.aleatoric_variance > 0.0)
    assert torch.all(pred.epistemic_variance > 0.0)
    assert torch.all(pred.total_variance > pred.aleatoric_variance)
