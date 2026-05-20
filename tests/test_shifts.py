"""Tests for parameter and initial condition shift suite (I18)."""

import pytest
import torch

from isopleth.evaluation.shifts import ShiftExperimentSuite
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D

def test_parameter_viscosity_shift():
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    suite = ShiftExperimentSuite()

    report = suite.evaluate_burgers_viscosity_shift(
        model=model,
        train_viscosity=0.01,
        shifted_viscosity=0.005,
        nx=32,
        steps=5,
        dt=0.005,
    )

    assert report.experiment_type == "parameter_shift"
    assert report.is_stable is True
    assert report.shifted_error >= 0.0

def test_initial_condition_frequency_shift():
    model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    suite = ShiftExperimentSuite()

    report = suite.evaluate_initial_condition_frequency_shift(
        model=model,
        train_freq=1,
        shifted_freq=3,
        nx=32,
        steps=5,
        dt=0.005,
    )

    assert report.experiment_type == "initial_condition_shift"
    assert report.is_stable is True
