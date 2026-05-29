"""Tests for Conformal Risk Control and distribution-free risk limits."""

from __future__ import annotations

import numpy as np
import pytest

from isopleth.uncertainty.conformal_risk import (
    ConformalRiskController,
    ConservationRiskAuditor,
    RiskCalibrationReport,
    RiskControlConfig,
)


def test_conformal_risk_controller_calibration() -> None:
    config = RiskControlConfig(target_risk_level=0.10, delta=0.05, bound_type="hoeffding")
    controller = ConformalRiskController(config=config)

    # Synthetic calibration losses: monotonic decreasing with lambda
    lambdas = np.linspace(0.01, 0.5, 50)
    rng = np.random.default_rng(42)
    # Losses decrease as lambda increases
    base_curve = np.exp(-10.0 * lambdas)[None, :]
    calib_losses = np.clip(base_curve + rng.normal(0.0, 0.01, size=(200, 50)), 0.0, 1.0)

    calibrated_lambda = controller.calibrate_threshold(
        calib_losses, lambdas, is_monotonic_decreasing=True
    )
    assert 0.01 <= calibrated_lambda <= 0.5

    # Test evaluation
    test_losses = np.clip(base_curve + rng.normal(0.0, 0.01, size=(100, 50)), 0.0, 1.0)
    report = controller.evaluate_test_risk(
        test_losses, calibrated_lambda, lambdas, calibration_losses=calib_losses
    )
    assert isinstance(report, RiskCalibrationReport)
    assert report.sample_size_calib == 200
    assert report.sample_size_test == 100


def test_conservation_risk_auditor_loss_matrices() -> None:
    drifts = [0.01, 0.05, 0.12, 0.002]
    thresholds = [0.01, 0.05, 0.10]
    loss_mat = ConservationRiskAuditor.build_mass_drift_loss_matrix(drifts, thresholds)
    assert loss_mat.shape == (4, 3)
    assert np.all((loss_mat == 0.0) | (loss_mat == 1.0))

    errors = [0.05, 0.15, 0.30]
    tolerances = [0.1, 0.2]
    bounded_loss = ConservationRiskAuditor.build_bounded_l2_loss_matrix(errors, tolerances)
    assert bounded_loss.shape == (3, 2)
    assert np.all(bounded_loss >= 0.0)
