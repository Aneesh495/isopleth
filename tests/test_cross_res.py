"""Tests for zero-shot cross-resolution transfer (I17)."""

import pytest
import math
import torch

from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.rollout.cross_resolution import CrossResolutionEvaluator

def test_cross_resolution_burgers_transfer():
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=8, num_layers=1)
    evaluator = CrossResolutionEvaluator()

    # Continuous analytical initial condition
    continuous_ic = lambda x: torch.sin(2.0 * math.pi * x)

    report = evaluator.evaluate_burgers_transfer(
        model=model,
        continuous_ic_fn=continuous_ic,
        coarse_nx=32,
        fine_nx=64,
        steps=5,
        dt=0.005,
    )

    assert report.family == "burgers_1d"
    assert report.coarse_resolution == (32,)
    assert report.fine_resolution == (64,)
    assert report.fine_steps_completed == 5
    assert report.is_stable is True
    assert report.fine_mass_error < 1e-4
