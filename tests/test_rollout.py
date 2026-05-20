"""Tests for target-free autoregressive rollout runner (I16)."""

import pytest
import math
import torch

from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D, MultiscaleFaceFluxOperator2D
from isopleth.rollout.runner import AutoregressiveRolloutRunner

def test_rollout_1d_target_free():
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    runner = AutoregressiveRolloutRunner(model=model)

    x_init = torch.sin(2.0 * math.pi * torch.linspace(0, 1, 32))
    res = runner.rollout_1d(x_init, dx=1.0 / 32, dt=0.01, steps=10)

    assert res.completed_steps == 10
    assert res.trajectory.shape == (11, 32)
    assert res.is_stable is True
    # Mass conservation in 1D operator rollout
    assert res.max_mass_error < 1e-4

def test_rollout_2d_target_free():
    model = MultiscaleFaceFluxOperator2D(in_channels=3, out_channels=3, hidden_channels=16, modes_x=4, modes_y=4, num_layers=1)
    runner = AutoregressiveRolloutRunner(model=model)

    x_init = torch.ones(3, 16, 16)
    res = runner.rollout_2d(x_init, dx=1.0 / 16, dy=1.0 / 16, dt=0.005, steps=5)

    assert res.completed_steps == 5
    assert res.trajectory.shape == (1, 6, 3, 16, 16)
    assert res.is_stable is True
