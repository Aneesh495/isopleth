"""Tests for ensemble uncertainty and conformal trajectory calibration (I19)."""

import pytest
import torch

from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.uncertainty.calibration import ConformalTrajectoryCalibrator
from isopleth.uncertainty.ensembles import TrajectoryEnsemble

def test_ensemble_prediction_and_variance():
    m1 = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    m2 = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16, modes=4, num_layers=1)
    ensemble = TrajectoryEnsemble([m1, m2])

    x_init = torch.sin(2.0 * 3.14159 * torch.linspace(0, 1, 32))
    pred = ensemble.predict_1d(x_init, dx=1.0 / 32, dt=0.01, steps=5)

    assert pred.mean_trajectory.shape == (6, 32)
    assert pred.variance_trajectory.shape == (6, 32)
    assert torch.all(pred.variance_trajectory >= 0.0)

def test_conformal_calibrator_fitting_and_intervals():
    calibrator = ConformalTrajectoryCalibrator(target_coverage=0.90)

    # Synthetic calibration predictions and ground truth
    preds = []
    truths = []
    for _ in range(10):
        mean = torch.randn(6, 32)
        var = torch.abs(torch.randn(6, 32)) * 0.1 + 0.01
        truth = mean + torch.randn(6, 32) * 0.1
        from isopleth.uncertainty.ensembles import EnsemblePrediction
        preds.append(EnsemblePrediction(mean, var, [mean, mean], torch.linspace(0, 0.05, 6), True))
        truths.append(truth)

    q = calibrator.fit_calibration_scores(preds, truths)
    assert q > 0.0

    lower, upper = calibrator.predict_intervals(preds[0])
    assert torch.all(upper >= lower)

    report = calibrator.evaluate_test_coverage(preds, truths)
    assert report.observed_coverage >= 0.0
    assert report.observed_coverage <= 1.0
    assert report.average_band_width > 0.0
