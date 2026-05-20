"""Tests for evaluation metrics suite (I20)."""

import pytest
import torch

from isopleth.evaluation.metrics import TrajectoryEvaluator

def test_spatial_and_spectral_metrics():
    target = torch.randn(20, 32)
    pred = target.clone()  # Perfect prediction

    metrics = TrajectoryEvaluator.evaluate_rollout(
        pred_trajectory=pred,
        target_trajectory=target,
        cell_measures=(1.0 / 32,),
    )

    assert metrics.relative_l2_error < 1e-6
    assert metrics.l1_error < 1e-6
    assert metrics.spectral_l1_mismatch < 1e-6
    assert metrics.is_stable is True
    assert metrics.blowup_event is False

def test_perturbed_trajectory_metrics():
    target = torch.randn(20, 32)
    pred = target + 0.1  # Shifted

    metrics = TrajectoryEvaluator.evaluate_rollout(
        pred_trajectory=pred,
        target_trajectory=target,
        cell_measures=(1.0 / 32,),
    )

    assert metrics.relative_l2_error > 0.0
    assert metrics.l1_error > 0.0
